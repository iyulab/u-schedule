//! Priority-dispatching scheduler.
//!
//! # Algorithm
//!
//! Operation-level, non-delay dispatching (an event-driven simulation: a
//! decision is taken whenever a resource could start work, among the work
//! waiting for it then):
//!
//! 1. Every unfinished task offers its next activity. Each is placed at its
//!    earliest feasible start on eligible resources
//!    ([`ResourceRequirement::accepts`](crate::models::ResourceRequirement::accepts):
//!    a listed candidate, or one of the requirement's type, with the skills),
//!    honoring calendars, capacities and sequence-dependent setups.
//! 2. The decision time is the earliest of those starts; the offers that can
//!    start then compete.
//! 3. The rule (or task priority, without a rule engine) picks one of them,
//!    judged on a [`SchedulingContext`] filled for that decision: the time,
//!    each task's imminent activity, work left, arrival in the queue, the
//!    queue at its next resource, the average imminent operation of the
//!    competitors, and planned resource utilization.
//! 4. The pick is booked, and the step repeats until no activity is left.
//!
//! An activity that holds no resource takes its time as soon as its task
//! reaches it — within its own [calendar](crate::models::Activity::calendar),
//! if it has one — and is reported in [`Schedule::time_only`]; it competes
//! for nothing.
//!
//! A resource never idles while work it can start waits — no slot is held
//! for a task that arrives later, however much the rule prefers it. On
//! Taillard's job-shop instances this beats holding slots (active,
//! Giffler–Thompson schedules) for every rule measured; see
//! `examples/taillard.rs`.
//!
//! # Complexity
//! O(A · n · P) where A = activities, n = tasks, P = one placement
//! (candidate resources × timeline query).
//!
//! # Reference
//! Pinedo (2016), "Scheduling", Ch. 4: Priority Dispatching; Kolisch (1996),
//! "Serial and parallel resource-constrained project scheduling methods
//! revisited", *EJOR* 90(2) — the parallel (non-delay) scheme with priority
//! rules.

use std::collections::HashMap;

use super::timeline::ResourceTimeline;
use crate::dispatching::policy::SharedPolicy;
use crate::dispatching::{rules, Decision, RuleEngine, RulePolicy, SchedulingContext};
use crate::models::{
    transport_between, Activity, Assignment, Resource, Schedule, Task, TimeOnlySpan,
    TransitionMatrixCollection,
};
use crate::Problem;

/// Safety bound for the serial-SGS fixed-point search.
const MAX_FIXED_POINT_ITERS: usize = 10_000;

/// Input container for scheduling.
#[derive(Debug, Clone)]
pub struct ScheduleRequest {
    /// The tasks to schedule and the resources available to them.
    pub problem: Problem,
    /// Schedule start time (ms).
    pub start_time_ms: i64,
    /// Sequence-dependent setup time matrices.
    pub transition_matrices: TransitionMatrixCollection,
    /// Additional constraints. The greedy scheduler does not yet place
    /// against these — they are validated post-hoc, so any breach shows
    /// up honestly in `Schedule::violations`.
    pub constraints: Vec<crate::models::Constraint>,
}

impl ScheduleRequest {
    /// Creates a new schedule request.
    pub fn new(problem: Problem) -> Self {
        Self {
            problem,
            start_time_ms: 0,
            transition_matrices: TransitionMatrixCollection::new(),
            constraints: Vec::new(),
        }
    }

    /// Sets the schedule start time.
    pub fn with_start_time(mut self, start_time_ms: i64) -> Self {
        self.start_time_ms = start_time_ms;
        self
    }

    /// Sets transition matrices.
    pub fn with_transition_matrices(mut self, matrices: TransitionMatrixCollection) -> Self {
        self.transition_matrices = matrices;
        self
    }

    /// Sets additional constraints (validated post-hoc).
    pub fn with_constraints(mut self, constraints: Vec<crate::models::Constraint>) -> Self {
        self.constraints = constraints;
        self
    }
}

/// Priority-dispatching scheduler.
///
/// At each decision point — the earliest time some waiting activity can
/// start — picks one of the activities that can start then, by dispatching
/// rule (or task priority), and books it on the earliest-available eligible
/// resource. Supports
/// sequence-dependent setup times via transition matrices.
///
/// # Example
///
/// ```
/// use u_schedule::scheduler::{SimpleScheduler, ScheduleRequest};
/// use u_schedule::models::{Task, Resource, Activity, ActivityDuration, ResourceRequirement};
/// use u_schedule::Problem;
///
/// let tasks = vec![
///     Task::new("J1").with_activity(
///         Activity::new("O1", "J1", 0)
///             .with_duration(ActivityDuration::fixed(1000))
///             .with_requirement(
///                 ResourceRequirement::new("Machine")
///                     .with_candidates(vec!["M1".into()])
///             )
///     ),
/// ];
/// let resources = vec![Resource::new("M1", "Machine")];
/// let problem = Problem::new(tasks, resources).expect("valid input");
/// let request = ScheduleRequest::new(problem);
///
/// let scheduler = SimpleScheduler::new();
/// let schedule = scheduler.schedule_request(&request);
/// assert_eq!(schedule.assignment_count(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct SimpleScheduler {
    transition_matrices: TransitionMatrixCollection,
    rules: Rules,
    fixed: Vec<Assignment>,
}

/// What decides each conflict.
#[derive(Debug, Clone)]
enum Rules {
    /// The higher `Task::priority`.
    Priority,
    /// One engine for every decision.
    Engine(RuleEngine),
    /// An engine chosen for each decision.
    Policy(SharedPolicy),
}

impl SimpleScheduler {
    /// Creates a new scheduler.
    pub fn new() -> Self {
        Self {
            transition_matrices: TransitionMatrixCollection::new(),
            rules: Rules::Priority,
            fixed: Vec::new(),
        }
    }

    /// Sets transition matrices.
    pub fn with_transition_matrices(mut self, matrices: TransitionMatrixCollection) -> Self {
        self.transition_matrices = matrices;
        self
    }

    /// Sets the rule engine that decides each conflict.
    ///
    /// Without one, the higher `Task::priority` wins (ties: input order).
    /// Replaces a [`with_rule_policy`](Self::with_rule_policy).
    pub fn with_rule_engine(mut self, engine: RuleEngine) -> Self {
        self.rules = Rules::Engine(engine);
        self
    }

    /// Chooses the rule engine for each conflict from that decision's state
    /// ([`RulePolicy`] — a rule for the start of the horizon and another
    /// once tasks run late, say). Replaces a
    /// [`with_rule_engine`](Self::with_rule_engine).
    pub fn with_rule_policy(mut self, policy: impl RulePolicy + 'static) -> Self {
        self.rules = Rules::Policy(SharedPolicy(std::sync::Arc::new(policy)));
        self
    }

    /// Seeds the schedule with fixed (pinned) assignments.
    ///
    /// Each pin is an externally-decided `Assignment` — a specific
    /// (`activity_id`, `resource_id`, `[start_ms, end_ms)`) hold. Seeded
    /// activities are honored verbatim: the scheduler
    ///
    /// 1. **pre-books** each pin's resource timeline, so SGS-scheduled
    ///    activities steer clear of the reserved interval;
    /// 2. **skips** the pinned activity in the SGS loop and emits the pin
    ///    unchanged into the result;
    /// 3. advances the intra-task cursor to the pin's `end_ms`, so later
    ///    activities of the same task start no earlier than the pin ends;
    /// 4. **never silently moves** a pin: if a pin itself is infeasible
    ///    (double-booked resource, outside a calendar, over capacity), the
    ///    self-annotating feasibility pass reports it in
    ///    [`Schedule::violations`] rather than relocating it.
    ///
    /// # Multi-requirement / quantity semantics
    ///
    /// A pin is per-(activity, resource). An activity that holds several
    /// resources simultaneously (multiple requirements, or `quantity > 1`)
    /// is pinned by supplying one `Assignment` **per held resource**, all
    /// sharing the `activity_id`. An activity that appears in the fixed set
    /// is treated **all-or-nothing**: it is skipped by SGS entirely, so the
    /// caller must fully specify every resource hold. A partially pinned
    /// activity leaves its un-pinned requirements unfilled, honestly
    /// reported as [`ViolationType::RequirementUnfilled`] — the scheduler
    /// does not fill them in.
    ///
    /// Pins are **opaque reservations**: their setup is assumed baked into
    /// the given interval, so they do not participate in
    /// `TransitionMatrix` changeover accounting.
    ///
    /// # Known limitation: setup underestimation right after a pin
    ///
    /// A pinned activity never updates `last_category` (that update only
    /// happens for SGS-placed activities on a resource with a matrix — see
    /// `schedule()`). So the **next SGS-placed activity on the same
    /// resource** computes its changeover against whatever category was
    /// current *before* the pin, not against the pin's own task category.
    /// If the pin's task differs from that pre-pin category, the following
    /// activity's setup is **understated** (it should reflect a changeover
    /// *from* the pin, but doesn't). This is a deliberate, documented gap —
    /// not a bug to work around here. Changeover-aware pin accounting
    /// (treating a pin as a `last_category` update) is a known limitation;
    /// see the solver enforcement matrix footnote in `crate` docs.
    ///
    /// [`ViolationType::RequirementUnfilled`]: crate::models::ViolationType::RequirementUnfilled
    pub fn with_fixed_assignments(mut self, fixed: Vec<Assignment>) -> Self {
        self.fixed = fixed;
        self
    }

    /// Schedules tasks on resources.
    ///
    /// # Algorithm (non-delay dispatching)
    /// 1. Every unfinished task offers its next activity, placed via
    ///    fixed-point search at the earliest common start where **every**
    ///    resource requirement can hold `quantity` resources simultaneously
    ///    — honoring calendars and capacities through [`ResourceTimeline`].
    /// 2. The earliest start among the offers is the decision time; the
    ///    offers that start then compete.
    /// 3. The rule engine (or task priority) picks one of them on a
    ///    [`SchedulingContext`] filled for this decision; ties go to input
    ///    order. The pick is booked; the rest are offered again.
    /// 4. The occupied span is `max(setup) + process + teardown`, where a
    ///    resource's setup comes from its transition matrix if one is
    ///    defined, else from `ActivityDuration::setup_ms`.
    ///
    /// Activities whose requirements cannot be met (too few eligible
    /// resources -- listed, or of the requirement's type when none are
    /// listed, and holding the required skills -- or a calendar that can
    /// never contain the duration) are left unassigned;
    /// the self-annotating feasibility check reports them as
    /// `RequirementUnfilled` rather than silently shrinking the schedule.
    ///
    /// # Reference
    /// Kolisch (1996), parallel schedule generation; Kolisch & Hartmann
    /// (1999) for the resource-feasibility check.
    pub fn schedule(&self, problem: &Problem, start_time_ms: i64) -> Schedule {
        let (tasks, resources) = (problem.tasks(), problem.resources());
        let mut schedule = Schedule::new();
        let mut timelines: HashMap<String, ResourceTimeline> = resources
            .iter()
            .map(|r| (r.id.clone(), ResourceTimeline::new(r)))
            .collect();
        let mut last_category: HashMap<String, String> = HashMap::new();
        // The working time of each activity with a calendar of its own, as a
        // timeline nothing books.
        let own: HashMap<&str, ResourceTimeline> = tasks
            .iter()
            .flat_map(|t| &t.activities)
            .filter_map(|a| Some((a.id.as_str(), own_timeline(a)?)))
            .collect();

        // Fixed-assignment seeding: index pins by activity, pre-book their
        // resources, and emit them verbatim. Pins are opaque reservations —
        // feasibility (below) validates them; we never relocate them.
        let mut fixed_by_activity: HashMap<&str, Vec<&Assignment>> = HashMap::new();
        for pin in &self.fixed {
            fixed_by_activity
                .entry(pin.activity_id.as_str())
                .or_default()
                .push(pin);
            // A pin of one of the problem's activities takes that activity's
            // usage; an external booking, one unit.
            let usage = resources
                .iter()
                .find(|r| r.id == pin.resource_id)
                .and_then(|r| {
                    tasks
                        .iter()
                        .flat_map(|t| &t.activities)
                        .find(|a| a.id == pin.activity_id)
                        .map(|a| a.usage_on(r))
                })
                .unwrap_or(1);
            if let Some(tl) = timelines.get_mut(&pin.resource_id) {
                tl.book(pin.start_ms, pin.end_ms, usage); // 선점 예약 (자원 미상이면 feasibility가 보고)
            }
            schedule.add_assignment(pin.clone());
        }

        let by_priority = RuleEngine::new().with_rule(rules::Priority);

        // Per task: the next activity to place, and the earliest it may start.
        let mut cursor: Vec<usize> = vec![0; tasks.len()];
        let mut ready: Vec<i64> = tasks
            .iter()
            .map(|t| t.release_time.unwrap_or(start_time_ms).max(start_time_ms))
            .collect();
        // Per task: the resources its last placed activity ran on, which the
        // work moves on from (`Resource::transport_ms`).
        let mut came_from: Vec<Vec<String>> = vec![Vec::new(); tasks.len()];

        // The resources each activity may run on, by task and step (WINQ).
        let eligible: Vec<Vec<Vec<&str>>> = tasks
            .iter()
            .map(|t| {
                t.activities
                    .iter()
                    .map(|a| a.eligible_resources(resources))
                    .collect()
            })
            .collect();
        // Per task: its offer, kept until a booking can change it, and the
        // resources whose booking can (`Self::read_by`).
        let mut kept: Vec<Option<(Placement, Vec<&str>)>> = vec![None; tasks.len()];

        loop {
            let mut offers: Vec<(usize, Placement)> = Vec::new();
            for (i, task) in tasks.iter().enumerate() {
                if let Some((placement, _)) = &kept[i] {
                    offers.push((i, placement.clone()));
                    continue;
                }
                while let Some(activity) = task.activities.get(cursor[i]) {
                    if let Some(pins) = fixed_by_activity.get(activity.id.as_str()) {
                        // 고정 activity: SGS 스킵, 후속 커서를 pin end 이후로 전진
                        let pin_end = pins.iter().map(|p| p.end_ms).max().unwrap_or(ready[i]);
                        ready[i] = ready[i].max(pin_end);
                        came_from[i] = pins.iter().map(|p| p.resource_id.clone()).collect();
                        cursor[i] += 1;
                        continue;
                    }
                    if activity.resource_requirements.is_empty() {
                        // It holds nothing, so nothing competes for it: it
                        // runs when the task reaches it. The work stays where
                        // it was (`came_from`). Never placed: the feasibility
                        // check reports it.
                        let from = activity.release_time.map_or(ready[i], |r| ready[i].max(r));
                        if let Some((start, end)) =
                            time_only_span(activity, own.get(activity.id.as_str()), from)
                        {
                            schedule.time_only.push(
                                TimeOnlySpan::new(&activity.id, &task.id, start, end)
                                    .with_sequence(cursor[i] as i32 + 1),
                            );
                            ready[i] = end;
                        }
                        cursor[i] += 1;
                        continue;
                    }
                    match self.place_activity(
                        activity,
                        task,
                        &timelines,
                        own.get(activity.id.as_str()),
                        &last_category,
                        resources,
                        ready[i],
                        &came_from[i],
                    ) {
                        Some(placement) => {
                            let read = self.read_by(activity, &placement, resources);
                            kept[i] = Some((placement.clone(), read));
                            offers.push((i, placement));
                            break;
                        }
                        // 충족 불가 — feasibility 검사가 RequirementUnfilled로 보고
                        None => cursor[i] += 1,
                    }
                }
            }
            if offers.is_empty() {
                break;
            }

            // Non-delay: decide at the earliest start, among what can start then.
            let now = offers
                .iter()
                .map(|(_, p)| p.start_ms)
                .min()
                .expect("offers is non-empty");
            let conflict: Vec<usize> = offers
                .iter()
                .enumerate()
                .filter(|(_, (_, p))| p.start_ms == now)
                .map(|(k, _)| k)
                .collect();
            let context = decision_context(
                tasks,
                &eligible,
                &timelines,
                &fixed_by_activity,
                &cursor,
                &ready,
                &offers,
                &conflict,
                start_time_ms,
            );
            let candidates: Vec<usize> = conflict.iter().map(|&k| offers[k].0).collect();
            let per_decision;
            let engine = match &self.rules {
                Rules::Priority => &by_priority,
                Rules::Engine(engine) => engine,
                Rules::Policy(policy) => {
                    let waiting: Vec<usize> = offers.iter().map(|(i, _)| *i).collect();
                    per_decision = policy.0.rules_for(&Decision {
                        tasks,
                        waiting: &waiting,
                        competing: &candidates,
                        context: &context,
                    });
                    &per_decision
                }
            };
            let chosen = engine
                .best_of(tasks, &candidates, &context)
                .expect("the conflict set is non-empty");
            let k = offers
                .iter()
                .position(|(i, _)| *i == chosen)
                .expect("the chosen task made an offer");
            let (i, placement) = offers.swap_remove(k);
            let (task, activity) = (&tasks[i], &tasks[i].activities[cursor[i]]);

            for (resource_id, setup_ms, usage) in &placement.selections {
                let tl = timelines
                    .get_mut(resource_id)
                    .expect("selection references known resource");
                tl.book(placement.start_ms, placement.end_ms, *usage);
                let companion = placement.companions.contains(resource_id);
                schedule.add_assignment(
                    Assignment::new(
                        &activity.id,
                        &task.id,
                        resource_id,
                        placement.start_ms,
                        placement.end_ms,
                    )
                    .with_setup(*setup_ms),
                );
                if !companion && self.transition_matrices.has_matrix(resource_id) {
                    last_category.insert(resource_id.clone(), task.category.clone());
                }
            }
            // The offers that read a resource just booked are made again.
            kept[i] = None;
            for offer in kept.iter_mut() {
                if offer.as_ref().is_some_and(|(_, read)| {
                    read.iter()
                        .any(|r| placement.selections.iter().any(|(s, _, _)| s == r))
                }) {
                    *offer = None;
                }
            }
            ready[i] = placement.end_ms;
            came_from[i] = placement
                .selections
                .iter()
                .map(|(rid, _, _)| rid.clone())
                .collect();
            cursor[i] += 1;
        }

        let input = super::feasibility::FeasibilityInput {
            tasks,
            resources,
            constraints: &[],
        };
        super::feasibility::annotate_schedule(&mut schedule, &input);
        schedule
    }

    /// The resources a booking on which can change `placement`, the offer of
    /// `activity`: the ones it holds, and the candidates with a transition
    /// matrix. A booking only takes time from a resource, so a candidate the
    /// offer passed over can only get later — except that a matrix charges a
    /// set-up from the category booked last, which a booking changes.
    fn read_by<'r>(
        &self,
        activity: &'r Activity,
        placement: &Placement,
        resources: &'r [Resource],
    ) -> Vec<&'r str> {
        let mut read: Vec<&str> = Vec::new();
        for r in activity
            .resource_requirements
            .iter()
            .flat_map(|req| req.eligible(resources))
        {
            let held = placement.selections.iter().any(|(s, _, _)| s == &r.id);
            if held || self.transition_matrices.has_matrix(&r.id) {
                for id in std::iter::once(&r.id).chain(&r.companions) {
                    if !read.contains(&id.as_str()) {
                        read.push(id);
                    }
                }
            }
        }
        read
    }

    /// Finds the earliest common start for an activity across all its
    /// resource requirements (serial-SGS fixed point), from `ready` on --
    /// and on each resource, no earlier than the work arrives there from the
    /// resources its task's previous activity ran on (`came_from`, by
    /// [`Resource::transport_ms`]).
    #[allow(clippy::too_many_arguments)]
    fn place_activity(
        &self,
        activity: &Activity,
        task: &Task,
        timelines: &HashMap<String, ResourceTimeline>,
        own: Option<&ResourceTimeline>,
        last_category: &HashMap<String, String>,
        resources: &[Resource],
        ready: i64,
        came_from: &[String],
    ) -> Option<Placement> {
        let ready = activity.release_time.map_or(ready, |r| ready.max(r));
        let lookup = |id: &str| resources.iter().find(|r| r.id == id);
        let moves = came_from
            .iter()
            .any(|id| lookup(id).is_some_and(|r| !r.transport_ms.is_empty()));
        let arrival = |rid: &str| -> i64 {
            if moves {
                ready + transport_between(lookup, came_from.iter().map(String::as_str), &[rid])
            } else {
                ready
            }
        };
        // The setup a resource charges for this activity, given the activity's
        // own setup on the resources it runs on.
        let setup_for = |resource_id: &str, own_setup_ms: i64| -> i64 {
            self.transition_matrices.setup_ms(
                resource_id,
                last_category.get(resource_id).map(String::as_str),
                &task.category,
                own_setup_ms,
            )
        };

        // Phase 1: 고정점 — 모든 requirement가 t에서 시작 가능한 최소 t.
        // Each candidate is judged on its own time there; at the same start
        // the one that processes faster is taken (non-delay first, then the
        // shorter processing).
        let mut t = ready;
        let mut selections: Vec<(String, i32)> = Vec::new();
        for _ in 0..MAX_FIXED_POINT_ITERS {
            selections.clear();
            let mut t_next = t;
            for req in &activity.resource_requirements {
                let mut fits: Vec<(i64, i64, &str)> = req
                    .eligible(resources)
                    .map(|r| r.id.as_str())
                    .filter_map(|rid| {
                        let d = activity.duration_on([rid]);
                        let span = setup_for(rid, d.setup_ms) + d.process_ms + d.teardown_ms;
                        let tl = timelines.get(rid)?;
                        let from = t.max(arrival(rid));
                        let fit = if activity.resumable {
                            tl.earliest_resumable(from, span, req.usage)
                                .map(|(fit, _)| fit)
                        } else {
                            tl.earliest_fit(from, span, req.usage)
                        };
                        // With its companions free over the same span.
                        let fit = with_companions(
                            fit?,
                            span,
                            (rid, req.usage),
                            resources,
                            timelines,
                            activity.resumable,
                        )?;
                        Some((fit, d.process_ms, rid))
                    })
                    .collect();
                // 동시각이면 짧은 처리, 그다음 자원 선언 순서 (stable sort)
                fits.sort_by_key(|&(fit, process, _)| (fit, process));
                if (fits.len() as i32) < req.quantity {
                    return None; // 후보 부족 — 영원히 충족 불가
                }
                for &(fit, _, rid) in fits.iter().take(req.quantity as usize) {
                    t_next = t_next.max(fit);
                    selections.push((rid.to_string(), req.usage));
                }
            }
            if t_next == t {
                break;
            }
            t = t_next;
        }

        // Phase 2: the time on the chosen resources together, with a common
        // span D (the largest setup), re-checked on every resource.
        let d = activity.duration_on(selections.iter().map(|(rid, _)| rid.as_str()));
        let selections: Vec<(String, i64, i32)> = selections
            .into_iter()
            .map(|(rid, usage)| {
                let setup = setup_for(&rid, d.setup_ms);
                (rid, setup, usage)
            })
            .collect();
        let max_setup = selections.iter().map(|&(_, s, _)| s).max().unwrap_or(0);
        let d = max_setup + d.process_ms + d.teardown_ms;
        // The resources chosen hold their companions for the same span; a
        // companion has no setup of its own and does not change the time.
        let mut companions: Vec<String> = Vec::new();
        for c in selections
            .iter()
            .flat_map(|(rid, _, _)| companions_of(rid, resources))
        {
            if selections.iter().all(|(s, _, _)| s != c) && !companions.contains(c) {
                companions.push(c.clone());
            }
        }
        let mut selections = selections;
        selections.extend(companions.iter().map(|c| (c.clone(), 0, 1)));
        if activity.resumable {
            return Self::place_resumable(timelines, own, selections, companions, t, d);
        }
        for _ in 0..MAX_FIXED_POINT_ITERS {
            let mut t_next = t;
            for (rid, _, usage) in &selections {
                let tl = timelines.get(rid).expect("selected resource has timeline");
                // None: 캘린더가 D를 영원히 못 담음
                let fit = tl.earliest_fit(t, d, *usage)?;
                t_next = t_next.max(fit);
            }
            // Within the activity's own working time.
            if let Some(own) = own {
                t_next = t_next.max(own.earliest_fit(t, d, 1)?);
            }
            if t_next == t {
                return Some(Placement {
                    start_ms: t,
                    end_ms: t + d,
                    selections,
                    companions,
                });
            }
            t = t_next;
        }
        None // 고정점 미수렴 (방어적 상한 — property test가 실질 커버)
    }

    /// The earliest `[start, end)` from `t` on in which `work_ms` of work
    /// runs while every selected resource is in working time, pausing
    /// otherwise, with every selected resource free throughout
    /// ([`Activity::resumable`]).
    fn place_resumable(
        timelines: &HashMap<String, ResourceTimeline>,
        own: Option<&ResourceTimeline>,
        selections: Vec<(String, i64, i32)>,
        companions: Vec<String>,
        t: i64,
        work_ms: i64,
    ) -> Option<Placement> {
        let selected: Vec<(&ResourceTimeline, i32)> = selections
            .iter()
            .map(|(rid, _, usage)| {
                (
                    timelines.get(rid).expect("selected resource has timeline"),
                    *usage,
                )
            })
            .collect();
        // The activity's own working time joins its resources'.
        let calendars: Vec<&crate::models::Calendar> = selected
            .iter()
            .map(|(tl, _)| *tl)
            .chain(own)
            .filter_map(ResourceTimeline::calendar)
            .collect();
        // Laid out once: every candidate start is tried against it.
        let joint = crate::models::JointTime::new(&calendars, t);
        if joint.is_empty() {
            return None; // they never work together
        }
        let mut candidates: Vec<i64> = selected
            .iter()
            .map(|(tl, _)| *tl)
            .chain(own)
            .flat_map(|tl| tl.start_candidates(t))
            .chain(joint.starts_from(t))
            .collect();
        candidates.sort_unstable();
        candidates.dedup();
        candidates.into_iter().find_map(|start| {
            let end = joint.work_end(start, work_ms)?;
            selected
                .iter()
                .all(|(tl, usage)| tl.is_free(start, end, *usage))
                .then(|| Placement {
                    start_ms: start,
                    end_ms: end,
                    selections: selections.clone(),
                    companions: companions.clone(),
                })
        })
    }

    /// Schedules from a request.
    ///
    /// Re-annotates with the request's constraints so `TimeWindow`,
    /// `Synchronize`, `NoOverlap`, tightened `Capacity`, and cross-task
    /// `Precedence` breaches are reported in `Schedule::violations`.
    pub fn schedule_request(&self, request: &ScheduleRequest) -> Schedule {
        let scheduler = Self {
            transition_matrices: request.transition_matrices.clone(),
            rules: self.rules.clone(),
            fixed: self.fixed.clone(),
        };
        let mut schedule = scheduler.schedule(&request.problem, request.start_time_ms);
        let input = super::feasibility::FeasibilityInput {
            tasks: request.problem.tasks(),
            resources: request.problem.resources(),
            constraints: &request.constraints,
        };
        super::feasibility::annotate_schedule(&mut schedule, &input);
        schedule
    }
}

/// The working time of `activity`'s own calendar as a timeline nothing books,
/// if it has one.
fn own_timeline(activity: &Activity) -> Option<ResourceTimeline> {
    let calendar = activity.calendar.clone()?;
    Some(ResourceTimeline::new(
        &Resource::new(&activity.id, "").with_calendar(calendar),
    ))
}

/// When `activity`, which holds no resource, runs from `from` on: its whole
/// time at once, within its own working time (`own`) — pausing outside it
/// when the activity is resumable. `None` if that time never holds it.
fn time_only_span(
    activity: &Activity,
    own: Option<&ResourceTimeline>,
    from: i64,
) -> Option<(i64, i64)> {
    let work = activity.duration.total_ms();
    match own {
        None => Some((from, from + work)),
        Some(own) if activity.resumable => own.earliest_resumable(from, work, 1),
        Some(own) => own.earliest_fit(from, work, 1).map(|t| (t, t + work)),
    }
}

/// The state a rule judges one decision on.
///
/// Every task with an offer is waiting; for each one competing now: its
/// imminent activity, the work left on it, when it joined the queue, and how
/// many other waiting tasks want a resource its next activity can use
/// (WINQ). The average imminent operation
/// is over the conflict set (ATC's `p̄`). Utilization is booked time over the
/// planned span so far, per unit of capacity (LPUL).
#[allow(clippy::too_many_arguments)]
fn decision_context(
    tasks: &[Task],
    eligible: &[Vec<Vec<&str>>],
    timelines: &HashMap<String, ResourceTimeline>,
    fixed: &HashMap<&str, Vec<&Assignment>>,
    cursor: &[usize],
    ready: &[i64],
    offers: &[(usize, Placement)],
    conflict: &[usize],
    start_time_ms: i64,
) -> SchedulingContext {
    let time = conflict
        .iter()
        .map(|&k| offers[k].1.start_ms)
        .min()
        .expect("the conflict set is non-empty");
    let mut context = SchedulingContext::at_time(time);

    let wanted: Vec<(usize, &[&str])> = offers
        .iter()
        .map(|&(i, _)| (i, eligible[i][cursor[i]].as_slice()))
        .collect();
    // The rules judge the competing tasks only, so their entries are the
    // ones filled.
    for &k in conflict {
        let i = offers[k].0;
        let (task, c) = (&tasks[i], cursor[i]);
        let remaining: i64 = task.activities[c..]
            .iter()
            .map(|a| a.duration.total_ms())
            .sum();
        let next_queue = task.activities[c + 1..]
            .iter()
            .position(|a| !a.resource_requirements.is_empty() && !fixed.contains_key(a.id.as_str()))
            .map(|n| {
                let usable = &eligible[i][c + 1 + n];
                wanted
                    .iter()
                    .filter(|(j, rs)| *j != i && rs.iter().any(|r| usable.contains(r)))
                    .count()
            })
            .unwrap_or(0);
        context.imminent_activity.insert(task.id.clone(), c);
        context.remaining_work.insert(task.id.clone(), remaining);
        context.arrival_times.insert(task.id.clone(), ready[i]);
        context
            .next_queue_length
            .insert(task.id.clone(), next_queue);
    }

    let competing: i64 = conflict
        .iter()
        .map(|&k| {
            let i = offers[k].0;
            tasks[i].activities[cursor[i]].duration.total_ms()
        })
        .sum();
    context.average_processing_time = Some(competing as f64 / conflict.len() as f64);

    let span = timelines
        .values()
        .filter_map(ResourceTimeline::latest_end)
        .max()
        .map_or(0, |end| end - start_time_ms);
    for (id, tl) in timelines {
        let load = if span > 0 {
            (tl.booked_ms() as f64 / (f64::from(tl.capacity()) * span as f64)).min(1.0)
        } else {
            0.0
        };
        context.resource_utilization.insert(id.clone(), load);
    }
    context
}

impl Default for SimpleScheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// A resolved activity placement: common interval plus per-resource setup.
#[derive(Clone)]
struct Placement {
    start_ms: i64,
    end_ms: i64,
    selections: Vec<(String, i64, i32)>, // (resource_id, setup_ms, usage) — companions included
    /// The selections held only as companions of another.
    companions: Vec<String>,
}

/// The companions `resource_id` declares ([`Resource::companions`]).
fn companions_of<'a>(resource_id: &str, resources: &'a [Resource]) -> &'a [String] {
    resources
        .iter()
        .find(|r| r.id == resource_id)
        .map_or(&[], |r| r.companions.as_slice())
}

/// The earliest start from `fit` on at which `resource_id`'s companions are
/// also free (non-resumable: in working time) for `span` -- a unit each --
/// and the resource itself still fits `usage` -- or `None` if they never are.
fn with_companions(
    fit: i64,
    span: i64,
    (resource_id, usage): (&str, i32),
    resources: &[Resource],
    timelines: &HashMap<String, ResourceTimeline>,
    resumable: bool,
) -> Option<i64> {
    let companions = companions_of(resource_id, resources);
    if companions.is_empty() {
        return Some(fit);
    }
    let group: Vec<(&ResourceTimeline, i32)> = std::iter::once((resource_id, usage))
        .chain(companions.iter().map(|c| (c.as_str(), 1)))
        .map(|(id, units)| timelines.get(id).map(|tl| (tl, units)))
        .collect::<Option<_>>()?;
    let mut t = fit;
    for _ in 0..MAX_FIXED_POINT_ITERS {
        let mut next = t;
        for (tl, units) in &group {
            let f = if resumable {
                tl.earliest_resumable(t, span, *units).map(|(f, _)| f)
            } else {
                tl.earliest_fit(t, span, *units)
            }?;
            next = next.max(f);
        }
        if next == t {
            return Some(t);
        }
        t = next;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatching::rules;
    use crate::models::{
        Activity, ActivityDuration, Resource, ResourceRequirement, TransitionMatrix,
    };

    /// A checked problem, for tests whose input is valid by construction.
    fn valid(tasks: &[Task], resources: &[Resource]) -> crate::Problem {
        crate::Problem::new(tasks.to_vec(), resources.to_vec()).expect("a valid problem")
    }

    /// A one-activity task on any of `machines`, 60 s of processing that runs
    /// at `speeds` on them.
    fn sped_task(id: &str, machines: &[(&str, f64)]) -> Task {
        let mut activity = Activity::new(format!("{id}-1"), id, 0)
            .with_process_time(60_000)
            .with_requirement(
                ResourceRequirement::new("Machine")
                    .with_candidates(machines.iter().map(|(m, _)| m.to_string()).collect()),
            );
        for &(m, speed) in machines {
            activity = activity.with_speed_on(m, speed);
        }
        Task::new(id).with_activity(activity)
    }

    fn on(id: &str, machine: &str, ms: i64) -> Task {
        Task::new(id).with_activity(
            Activity::new(format!("{id}-1"), id, 0)
                .with_process_time(ms)
                .with_requirement(
                    ResourceRequirement::new("Machine").with_candidates(vec![machine.into()]),
                ),
        )
    }

    #[test]
    fn test_a_resource_holds_its_companion_for_the_whole_span() {
        // The robot works in its cell: running on the robot occupies the cell.
        let tasks = vec![on("A", "ROBOT", 30_000), on("B", "CELL", 20_000)];
        let resources = vec![
            Resource::new("ROBOT", "Machine").with_companion("CELL"),
            Resource::new("CELL", "Machine"),
        ];

        let schedule = SimpleScheduler::new().schedule(&valid(&tasks, &resources), 0);

        assert!(schedule.violations.is_empty(), "{:?}", schedule.violations);
        let spans: Vec<(&str, &str, i64, i64)> = schedule
            .assignments
            .iter()
            .map(|a| {
                (
                    a.activity_id.as_str(),
                    a.resource_id.as_str(),
                    a.start_ms,
                    a.end_ms,
                )
            })
            .collect();
        assert!(spans.contains(&("A-1", "ROBOT", 0, 30_000)));
        assert!(
            spans.contains(&("A-1", "CELL", 0, 30_000)),
            "the companion is booked: {spans:?}"
        );
        assert!(spans.contains(&("B-1", "CELL", 30_000, 50_000)));
    }

    #[test]
    fn test_a_companion_in_use_holds_back_its_resource() {
        // The cell is busy first; the robot cannot start until it is free.
        let tasks = vec![on("A", "CELL", 20_000), on("B", "ROBOT", 10_000)];
        let resources = vec![
            Resource::new("CELL", "Machine"),
            Resource::new("ROBOT", "Machine").with_companion("CELL"),
        ];

        let schedule = SimpleScheduler::new().schedule(&valid(&tasks, &resources), 0);

        assert!(schedule.violations.is_empty(), "{:?}", schedule.violations);
        let b = schedule.assignment_for_activity("B-1").expect("placed");
        assert_eq!((b.start_ms, b.end_ms), (20_000, 30_000));
    }

    #[test]
    fn test_feasibility_requires_the_companion_and_counts_it_as_no_unit() {
        let tasks = vec![on("A", "ROBOT", 30_000)];
        let resources = vec![
            Resource::new("ROBOT", "Machine").with_companion("CELL"),
            Resource::new("CELL", "Machine"),
        ];
        let input = super::super::feasibility::FeasibilityInput {
            tasks: &tasks,
            resources: &resources,
            constraints: &[],
        };
        let mut with = Schedule::new();
        with.add_assignment(Assignment::new("A-1", "A", "ROBOT", 0, 30_000));
        with.add_assignment(Assignment::new("A-1", "A", "CELL", 0, 30_000));
        assert!(super::super::feasibility::check_schedule(&with, &input).is_empty());
        let mut without = Schedule::new();
        without.add_assignment(Assignment::new("A-1", "A", "ROBOT", 0, 30_000));
        let found = super::super::feasibility::check_schedule(&without, &input);
        assert!(
            found
                .iter()
                .any(|v| v.message.contains("without its companion 'CELL'")),
            "{found:?}"
        );
    }

    #[test]
    fn test_a_companion_must_be_another_known_resource() {
        let resources = vec![
            Resource::new("ROBOT", "Machine").with_companion("NOPE"),
            Resource::new("CELL", "Machine").with_companion("CELL"),
        ];
        let errors = crate::Problem::new(vec![on("A", "CELL", 1)], resources).expect_err("refused");
        assert_eq!(
            errors
                .iter()
                .filter(|e| e.kind.code() == "invalid_resource_reference")
                .count(),
            2,
            "{errors:?}"
        );
    }

    #[test]
    fn test_the_first_activity_on_a_resource_takes_the_initial_setup() {
        let task = |id: &str, category: &str| {
            Task::new(id).with_category(category).with_activity(
                Activity::new(format!("{id}-1"), id, 0)
                    .with_process_time(1_000)
                    .with_requirement(ResourceRequirement::new("Machine")),
            )
        };
        let tasks = vec![task("A", "red"), task("B", "blue")];
        let resources = vec![Resource::new("M", "Machine")];
        let mut matrix = TransitionMatrix::new("m", "M")
            .with_default(300)
            .with_initial(500);
        matrix.set_initial("blue", 50);
        let matrices = TransitionMatrixCollection::new().with_matrix(matrix);

        let schedule = SimpleScheduler::new()
            .with_transition_matrices(matrices)
            .schedule(&valid(&tasks, &resources), 0);

        let at = |a: &str| schedule.assignment_for_activity(a).expect("placed");
        assert_eq!((at("A-1").setup_ms, at("A-1").end_ms), (500, 1_500)); // from the initial state
        assert_eq!((at("B-1").setup_ms, at("B-1").end_ms), (300, 2_800)); // red → blue: the default
    }

    const HOUR: i64 = 3_600_000;

    /// 08–12 and 13–17 on each of `days` days from 0.
    fn day_shifts(id: &str, days: i64) -> crate::models::Calendar {
        (0..days).fold(crate::models::Calendar::new(id), |c, d| {
            let day = d * 24 * HOUR;
            c.with_window(day + 8 * HOUR, day + 12 * HOUR)
                .with_window(day + 13 * HOUR, day + 17 * HOUR)
        })
    }

    fn one_activity(id: &str, hours: i64, resumable: bool) -> Task {
        let mut a = Activity::new(format!("{id}-1"), id, 0)
            .with_process_time(hours * HOUR)
            .with_requirement(ResourceRequirement::new("Machine"));
        if resumable {
            a = a.resumable();
        }
        Task::new(id).with_activity(a)
    }

    #[test]
    fn test_resumable_work_pauses_over_a_break_and_holds_the_resource() {
        let resources = vec![Resource::new("M", "Machine").with_calendar(day_shifts("c", 3))];
        let tasks = vec![one_activity("A", 3, true), one_activity("B", 6, true)];

        let schedule = SimpleScheduler::new().schedule(&valid(&tasks, &resources), 11 * HOUR);

        assert!(schedule.violations.is_empty(), "{:?}", schedule.violations);
        let at = |a: &str| {
            let x = schedule.assignment_for_activity(a).expect("placed");
            (x.start_ms, x.end_ms)
        };
        // 11–12, lunch, 13–15.
        assert_eq!(at("A-1"), (11 * HOUR, 15 * HOUR));
        // 15–17, night, next day 08–12.
        assert_eq!(at("B-1"), (15 * HOUR, (24 + 12) * HOUR));
    }

    #[test]
    fn test_work_that_is_not_resumable_waits_for_a_stretch_long_enough() {
        let resources = vec![Resource::new("M", "Machine").with_calendar(day_shifts("c", 3))];
        let tasks = vec![one_activity("A", 3, false)];

        let schedule = SimpleScheduler::new().schedule(&valid(&tasks, &resources), 11 * HOUR);

        let a = schedule.assignment_for_activity("A-1").expect("placed");
        assert_eq!((a.start_ms, a.end_ms), (13 * HOUR, 16 * HOUR));
    }

    #[test]
    fn test_resumable_work_runs_only_while_every_resource_works() {
        // The machine runs around the clock; the operator works 08–12, 13–17.
        let tasks = vec![Task::new("A").with_activity(
            Activity::new("A-1", "A", 0)
                .with_process_time(5 * HOUR)
                .with_requirement(ResourceRequirement::new("Machine"))
                .with_requirement(ResourceRequirement::new("Operator"))
                .resumable(),
        )];
        let resources = vec![
            Resource::new("M", "Machine"),
            Resource::new("OP", "Operator").with_calendar(day_shifts("c", 2)),
        ];

        let schedule = SimpleScheduler::new().schedule(&valid(&tasks, &resources), 0);

        assert!(schedule.violations.is_empty(), "{:?}", schedule.violations);
        for a in &schedule.assignments {
            assert_eq!(
                (a.start_ms, a.end_ms),
                (8 * HOUR, 14 * HOUR),
                "{}",
                a.resource_id
            );
        }
    }

    #[test]
    fn test_feasibility_counts_the_working_time_of_resumable_work() {
        let tasks = vec![one_activity("A", 3, true)];
        let resources = vec![Resource::new("M", "Machine").with_calendar(day_shifts("c", 1))];
        let input = super::super::feasibility::FeasibilityInput {
            tasks: &tasks,
            resources: &resources,
            constraints: &[],
        };
        let check = |start: i64, end: i64| {
            let mut s = Schedule::new();
            s.add_assignment(Assignment::new("A-1", "A", "M", start, end));
            super::super::feasibility::check_schedule(&s, &input)
        };
        assert!(check(11 * HOUR, 15 * HOUR).is_empty());
        // Only 2 h of working time in 11–14.
        assert!(!check(11 * HOUR, 14 * HOUR).is_empty());
        // Starts in the lunch break.
        assert!(!check(12 * HOUR, 16 * HOUR).is_empty());
    }

    #[test]
    fn test_speed_divides_processing_and_breaks_a_tie_at_the_same_start() {
        let machines = [("SLOW", 0.5), ("FAST", 2.0)];
        let tasks: Vec<Task> = ["A", "B", "C"]
            .iter()
            .map(|id| sped_task(id, &machines))
            .collect();
        let resources = vec![
            Resource::new("SLOW", "Machine"),
            Resource::new("FAST", "Machine"),
        ];

        let schedule = SimpleScheduler::new().schedule(&valid(&tasks, &resources), 0);

        assert!(schedule.violations.is_empty(), "{:?}", schedule.violations);
        let at = |a: &str| schedule.assignment_for_activity(a).expect("placed");
        // Both free at 0: the faster takes the first; the slow one is not left
        // idle (non-delay) and takes the second; the third waits for FAST.
        assert_eq!(
            (at("A-1").resource_id.as_str(), at("A-1").end_ms),
            ("FAST", 30_000)
        );
        assert_eq!(
            (at("B-1").resource_id.as_str(), at("B-1").end_ms),
            ("SLOW", 120_000)
        );
        assert_eq!(
            (
                at("C-1").resource_id.as_str(),
                at("C-1").start_ms,
                at("C-1").end_ms
            ),
            ("FAST", 30_000, 60_000)
        );
    }

    #[test]
    fn test_a_duration_of_its_own_on_a_worker_outweighs_the_machine_speed() {
        let activity = Activity::new("A-1", "A", 0)
            .with_duration(ActivityDuration::new(1_000, 60_000, 0))
            .with_requirement(ResourceRequirement::new("Machine"))
            .with_requirement(ResourceRequirement::new("Worker"))
            .with_speed_on("M", 2.0)
            .with_speed_on("ALICE", 1.5)
            .with_duration_on("BOB", ActivityDuration::new(0, 50_000, 0));
        assert_eq!(activity.duration_on(["M", "ALICE"]).process_ms, 20_000); // 60 s ÷ 3
        assert_eq!(activity.duration_on(["M", "ALICE"]).setup_ms, 1_000); // setup unchanged
        assert_eq!(
            activity.duration_on(["M", "BOB"]),
            ActivityDuration::new(0, 50_000, 0)
        );
        assert_eq!(activity.duration_on(["X"]), activity.duration);

        // Scheduled: the machine's speed does not apply under BOB's own time.
        let mut activity = activity;
        activity.resource_timings.remove("ALICE");
        let tasks = vec![Task::new("A").with_activity(activity)];
        let resources = vec![
            Resource::new("M", "Machine"),
            Resource::new("BOB", "Worker"),
        ];
        let schedule = SimpleScheduler::new().schedule(&valid(&tasks, &resources), 0);
        assert!(schedule.violations.is_empty(), "{:?}", schedule.violations);
        for a in &schedule.assignments {
            assert_eq!((a.start_ms, a.end_ms), (0, 50_000), "{}", a.resource_id);
        }
    }

    #[test]
    fn test_feasibility_reads_the_time_on_the_resources_used() {
        let tasks = vec![sped_task("A", &[("FAST", 2.0)])];
        let resources = vec![Resource::new("FAST", "Machine")];
        let input = super::super::feasibility::FeasibilityInput {
            tasks: &tasks,
            resources: &resources,
            constraints: &[],
        };
        let mut schedule = Schedule::new();
        schedule.add_assignment(Assignment::new("A-1", "A", "FAST", 0, 30_000));
        assert!(super::super::feasibility::check_schedule(&schedule, &input).is_empty());
        let mut short = Schedule::new();
        short.add_assignment(Assignment::new("A-1", "A", "FAST", 0, 29_999));
        let found = super::super::feasibility::check_schedule(&short, &input);
        assert!(
            found
                .iter()
                .any(|v| v.violation_type == crate::models::ViolationType::DurationTooShort),
            "{found:?}"
        );
    }

    fn make_resource(id: &str) -> Resource {
        Resource::new(id, "Machine")
    }

    fn make_task_with_resource(
        id: &str,
        duration_ms: i64,
        resource_id: &str,
        priority: i32,
    ) -> Task {
        Task::new(id)
            .with_priority(priority)
            .with_category("default")
            .with_activity(
                Activity::new(format!("{id}_O1"), id, 0)
                    .with_duration(ActivityDuration::fixed(duration_ms))
                    .with_requirement(
                        ResourceRequirement::new("Machine")
                            .with_candidates(vec![resource_id.into()]),
                    ),
            )
    }

    #[test]
    fn test_multi_resource_simultaneous_hold() {
        // probe S1: 설비+금형 — 금형(TOOL1) 공유로 두 작업 직렬화
        let mk = |tid: &str, machine: &str| {
            Task::new(tid)
                .with_priority(1)
                .with_category("default")
                .with_activity(
                    Activity::new(format!("{tid}_O1"), tid, 0)
                        .with_duration(ActivityDuration::fixed(1000))
                        .with_requirement(
                            ResourceRequirement::new("Machine")
                                .with_candidates(vec![machine.into()]),
                        )
                        .with_requirement(
                            ResourceRequirement::new("Mold").with_candidates(vec!["TOOL1".into()]),
                        ),
                )
        };
        let tasks = vec![mk("J1", "M1"), mk("J2", "M2")];
        let resources = vec![
            Resource::new("M1", "Machine"),
            Resource::new("M2", "Machine"),
            Resource::new("TOOL1", "Mold"),
        ];
        let s = SimpleScheduler::new().schedule(&valid(&tasks, &resources), 0);

        // 각 activity가 설비+금형 2개 배정
        assert_eq!(s.assignments_for_task("J1").len(), 2);
        assert_eq!(s.assignments_for_task("J2").len(), 2);
        // 금형 위에서 직렬화 (겹침 없음)
        let tool = s.assignments_for_resource("TOOL1");
        assert_eq!(tool.len(), 2);
        let (a, b) = (tool[0], tool[1]);
        assert!(
            a.end_ms <= b.start_ms || b.end_ms <= a.start_ms,
            "TOOL1 double-booked: {a:?} vs {b:?}"
        );
        // 동시 점유: 같은 activity의 두 배정 구간 동일
        let j1 = s.assignments_for_task("J1");
        assert_eq!(j1[0].start_ms, j1[1].start_ms);
        assert_eq!(j1[0].end_ms, j1[1].end_ms);
    }

    #[test]
    fn test_calendar_enforced() {
        // probe S3: 가용창 [0,5000)∪[10000,20000), 3000ms 작업 2건 → 둘째는 10000 시작
        let cal = crate::models::Calendar::new("shift")
            .with_window(0, 5000)
            .with_window(10_000, 20_000);
        let tasks = vec![
            make_task_with_resource("J1", 3000, "M1", 10),
            make_task_with_resource("J2", 3000, "M1", 5),
        ];
        let resources = vec![Resource::new("M1", "Machine").with_calendar(cal)];
        let s = SimpleScheduler::new().schedule(&valid(&tasks, &resources), 0);
        let j2 = s.assignment_for_activity("J2_O1").unwrap();
        assert_eq!(j2.start_ms, 10_000);
        assert_eq!(j2.end_ms, 13_000);
    }

    /// Machine work, a cure that holds nothing, then machine work again.
    fn cured(cure: Activity) -> Vec<Task> {
        let on_m = |id: &str, seq: i32| {
            Activity::new(id, "J", seq)
                .with_process_time(HOUR)
                .with_requirement(ResourceRequirement::new("Machine"))
        };
        vec![Task::new("J")
            .with_activity(on_m("J-1", 0))
            .with_activity(cure)
            .with_activity(on_m("J-3", 2))]
    }

    #[test]
    fn an_activity_without_resources_takes_its_time_before_the_next() {
        let tasks = cured(Activity::new("J-2", "J", 1).with_process_time(3 * HOUR));
        let resources = vec![Resource::new("M", "Machine")];
        let s = SimpleScheduler::new().schedule(&valid(&tasks, &resources), 0);
        assert!(s.violations.is_empty(), "{:?}", s.violations);
        let cure = s
            .time_only_for_activity("J-2")
            .expect("the cure has a span");
        assert_eq!((cure.start_ms, cure.end_ms), (HOUR, 4 * HOUR));
        assert_eq!(cure.sequence, Some(2));
        assert_eq!(s.assignment_for_activity("J-3").unwrap().start_ms, 4 * HOUR);
        assert_eq!(s.makespan_ms(), 5 * HOUR);
        assert!(s.assignment_for_activity("J-2").is_none());
    }

    #[test]
    fn an_activity_without_resources_runs_in_its_own_calendar() {
        // 1 h of machine work from 0, then a 6 h cure in day shifts: it waits
        // for 08:00 and, resumable, pauses over lunch.
        let cure = Activity::new("J-2", "J", 1)
            .with_process_time(6 * HOUR)
            .with_calendar(day_shifts("plant", 2));
        let resources = vec![Resource::new("M", "Machine")];
        // Not resumable: it needs one 6 h stretch, and the shifts are 4 h —
        // never placed, and said so.
        let s = SimpleScheduler::new().schedule(&valid(&cured(cure.clone()), &resources), 0);
        assert!(s.time_only_for_activity("J-2").is_none());
        assert!(s.violations.iter().any(|v| v.entity_id == "J-2"
            && v.violation_type == crate::models::ViolationType::TimeWindowViolation));
        let s = SimpleScheduler::new().schedule(&valid(&cured(cure.resumable()), &resources), 0);
        assert!(s.violations.is_empty(), "{:?}", s.violations);
        let span = s.time_only_for_activity("J-2").unwrap();
        assert_eq!((span.start_ms, span.end_ms), (8 * HOUR, 15 * HOUR));
    }

    #[test]
    fn an_activity_calendar_holds_resourced_work_to_its_windows() {
        let tasks = vec![Task::new("J").with_activity(
            Activity::new("J-1", "J", 0)
                .with_process_time(HOUR)
                .with_requirement(ResourceRequirement::new("Machine"))
                .with_calendar(crate::models::Calendar::new("w").with_window(5 * HOUR, 9 * HOUR)),
        )];
        let resources = vec![Resource::new("M", "Machine")];
        let s = SimpleScheduler::new().schedule(&valid(&tasks, &resources), 0);
        assert!(s.violations.is_empty(), "{:?}", s.violations);
        assert_eq!(s.assignment_for_activity("J-1").unwrap().start_ms, 5 * HOUR);
    }

    #[test]
    fn test_capacity_two_parallel() {
        // probe S4: capacity=2 → 두 작업 병렬
        let tasks = vec![
            make_task_with_resource("J1", 1000, "M1", 10),
            make_task_with_resource("J2", 1000, "M1", 5),
        ];
        let resources = vec![Resource::new("M1", "Machine").with_capacity(2)];
        let s = SimpleScheduler::new().schedule(&valid(&tasks, &resources), 0);
        assert_eq!(s.assignment_for_activity("J1_O1").unwrap().start_ms, 0);
        assert_eq!(s.assignment_for_activity("J2_O1").unwrap().start_ms, 0);
    }

    #[test]
    fn test_duration_components_enforced() {
        // probe S5: (500,1000,300) → 점유 1800ms, setup 기록 500
        let task = Task::new("J1")
            .with_priority(1)
            .with_category("default")
            .with_activity(
                Activity::new("J1_O1", "J1", 0)
                    .with_duration(ActivityDuration::new(500, 1000, 300))
                    .with_requirement(
                        ResourceRequirement::new("Machine").with_candidates(vec!["M1".into()]),
                    ),
            );
        let s =
            SimpleScheduler::new().schedule(&valid(&[task], &[Resource::new("M1", "Machine")]), 0);
        let a = s.assignment_for_activity("J1_O1").unwrap();
        assert_eq!(a.end_ms - a.start_ms, 1800);
        assert_eq!(a.setup_ms, 500);
    }

    #[test]
    fn test_simple_single_task() {
        let tasks = vec![make_task_with_resource("J1", 1000, "M1", 0)];
        let resources = vec![make_resource("M1")];
        let scheduler = SimpleScheduler::new();

        let schedule = scheduler.schedule(&valid(&tasks, &resources), 0);
        assert_eq!(schedule.assignment_count(), 1);

        let a = schedule.assignment_for_activity("J1_O1").unwrap();
        assert_eq!(a.start_ms, 0);
        assert_eq!(a.end_ms, 1000);
        assert_eq!(a.resource_id, "M1");
    }

    #[test]
    fn test_priority_ordering() {
        let tasks = vec![
            make_task_with_resource("low", 1000, "M1", 1),
            make_task_with_resource("high", 1000, "M1", 10),
        ];
        let resources = vec![make_resource("M1")];
        let scheduler = SimpleScheduler::new();

        let schedule = scheduler.schedule(&valid(&tasks, &resources), 0);

        // High priority scheduled first
        let high_a = schedule.assignment_for_activity("high_O1").unwrap();
        let low_a = schedule.assignment_for_activity("low_O1").unwrap();
        assert!(high_a.start_ms < low_a.start_ms);
    }

    #[test]
    fn test_two_resources() {
        let tasks = vec![
            make_task_with_resource("J1", 2000, "M1", 10),
            make_task_with_resource("J2", 1000, "M1", 5),
        ];
        // Only M1 → J1 first (priority), then J2 at 2000
        let resources = vec![make_resource("M1")];
        let scheduler = SimpleScheduler::new();

        let schedule = scheduler.schedule(&valid(&tasks, &resources), 0);
        let j1 = schedule.assignment_for_activity("J1_O1").unwrap();
        let j2 = schedule.assignment_for_activity("J2_O1").unwrap();
        assert_eq!(j1.start_ms, 0);
        assert_eq!(j1.end_ms, 2000);
        assert_eq!(j2.start_ms, 2000);
        assert_eq!(j2.end_ms, 3000);
    }

    #[test]
    fn test_parallel_resources() {
        // J1→M1, J2→M2 can run in parallel
        let tasks = vec![
            make_task_with_resource("J1", 2000, "M1", 10),
            make_task_with_resource("J2", 1000, "M2", 5),
        ];
        let resources = vec![make_resource("M1"), make_resource("M2")];
        let scheduler = SimpleScheduler::new();

        let schedule = scheduler.schedule(&valid(&tasks, &resources), 0);
        let j1 = schedule.assignment_for_activity("J1_O1").unwrap();
        let j2 = schedule.assignment_for_activity("J2_O1").unwrap();
        // Both start at 0 since they use different resources
        assert_eq!(j1.start_ms, 0);
        assert_eq!(j2.start_ms, 0);
    }

    #[test]
    fn test_multi_activity_task() {
        let task = Task::new("J1")
            .with_priority(1)
            .with_category("TypeA")
            .with_activity(
                Activity::new("O1", "J1", 0)
                    .with_duration(ActivityDuration::fixed(1000))
                    .with_requirement(
                        ResourceRequirement::new("Machine").with_candidates(vec!["M1".into()]),
                    ),
            )
            .with_activity(
                Activity::new("O2", "J1", 1)
                    .with_duration(ActivityDuration::fixed(2000))
                    .with_requirement(
                        ResourceRequirement::new("Machine").with_candidates(vec!["M1".into()]),
                    ),
            );

        let resources = vec![make_resource("M1")];
        let scheduler = SimpleScheduler::new();
        let schedule = scheduler.schedule(&valid(&[task], &resources), 0);

        let o1 = schedule.assignment_for_activity("O1").unwrap();
        let o2 = schedule.assignment_for_activity("O2").unwrap();
        // O2 must start after O1 ends (intra-task precedence)
        assert_eq!(o1.end_ms, 1000);
        assert!(o2.start_ms >= o1.end_ms);
        assert_eq!(o2.end_ms, 3000);
    }

    #[test]
    fn test_transition_matrix_setup() {
        let mut tm = TransitionMatrix::new("changeover", "M1").with_default(500);
        tm.set_transition("TypeA", "TypeB", 1000);

        let matrices = TransitionMatrixCollection::new().with_matrix(tm);

        let tasks = vec![
            Task::new("J1")
                .with_priority(10)
                .with_category("TypeA")
                .with_activity(
                    Activity::new("O1", "J1", 0)
                        .with_duration(ActivityDuration::fixed(1000))
                        .with_requirement(
                            ResourceRequirement::new("Machine").with_candidates(vec!["M1".into()]),
                        ),
                ),
            Task::new("J2")
                .with_priority(5)
                .with_category("TypeB")
                .with_activity(
                    Activity::new("O2", "J2", 0)
                        .with_duration(ActivityDuration::fixed(1000))
                        .with_requirement(
                            ResourceRequirement::new("Machine").with_candidates(vec!["M1".into()]),
                        ),
                ),
        ];

        let resources = vec![make_resource("M1")];
        let scheduler = SimpleScheduler::new().with_transition_matrices(matrices);

        let schedule = scheduler.schedule(&valid(&tasks, &resources), 0);
        let o2 = schedule.assignment_for_activity("O2").unwrap();
        // J1 ends at 1000, setup A→B = 1000, J2 starts at 1000, ends at 1000+1000+1000 = 3000
        assert_eq!(o2.start_ms, 1000);
        assert_eq!(o2.setup_ms, 1000);
        assert_eq!(o2.end_ms, 3000);
    }

    #[test]
    fn test_with_rule_engine() {
        // Use SPT rule → shorter task first regardless of priority
        let tasks = vec![
            make_task_with_resource("long", 5000, "M1", 100), // High priority but long
            make_task_with_resource("short", 1000, "M1", 1),  // Low priority but short
        ];
        let resources = vec![make_resource("M1")];
        let engine = RuleEngine::new().with_rule(rules::Spt);
        let scheduler = SimpleScheduler::new().with_rule_engine(engine);

        let schedule = scheduler.schedule(&valid(&tasks, &resources), 0);
        let short_a = schedule.assignment_for_activity("short_O1").unwrap();
        let long_a = schedule.assignment_for_activity("long_O1").unwrap();
        // SPT orders short first despite lower priority
        assert_eq!(short_a.start_ms, 0);
        assert!(long_a.start_ms >= short_a.end_ms);
    }

    #[test]
    fn test_schedule_request() {
        let tasks = vec![make_task_with_resource("J1", 1000, "M1", 0)];
        let resources = vec![make_resource("M1")];
        let request = ScheduleRequest::new(valid(&tasks, &resources)).with_start_time(5000);

        let scheduler = SimpleScheduler::new();
        let schedule = scheduler.schedule_request(&request);

        let a = schedule.assignment_for_activity("J1_O1").unwrap();
        assert_eq!(a.start_ms, 5000);
        assert_eq!(a.end_ms, 6000);
    }

    #[test]
    fn test_release_time_respected() {
        let mut task = make_task_with_resource("J1", 1000, "M1", 0);
        task.release_time = Some(5000);
        let resources = vec![make_resource("M1")];
        let scheduler = SimpleScheduler::new();

        let schedule = scheduler.schedule(&valid(&[task], &resources), 0);
        let a = schedule.assignment_for_activity("J1_O1").unwrap();
        // Must not start before release_time
        assert_eq!(a.start_ms, 5000);
    }

    #[test]
    fn test_empty_input() {
        let scheduler = SimpleScheduler::new();
        let schedule = scheduler.schedule(&valid(&[], &[]), 0);
        assert_eq!(schedule.assignment_count(), 0);
        assert_eq!(schedule.makespan_ms(), 0);
    }

    #[test]
    fn without_candidates_only_a_resource_of_the_requirement_type_is_picked() {
        // The operator is free at 0 and has the skill, but is not a machine;
        // the machine is busy until 1000 with a pinned job.
        let task = Task::new("J1").with_priority(1).with_activity(
            Activity::new("O1", "J1", 0)
                .with_duration(ActivityDuration::fixed(500))
                .with_requirement(ResourceRequirement::new("Machine").with_skill("milling")),
        );
        let resources = vec![
            Resource::new("W1", "Operator").with_skill("milling"),
            Resource::new("M1", "Machine").with_skill("milling"),
            Resource::new("M2", "Machine"), // right type, no skill
        ];
        let scheduler = SimpleScheduler::new()
            .with_fixed_assignments(vec![Assignment::new("PIN", "PIN", "M1", 0, 1000)]);
        let problem = valid(&[task], &resources);
        let s = scheduler.schedule(&problem, 0);
        let a = s.assignment_for_activity("O1").expect("placed");
        assert_eq!(a.resource_id, "M1");
        assert_eq!(a.start_ms, 1000);

        let input = crate::scheduler::FeasibilityInput {
            tasks: problem.tasks(),
            resources: problem.resources(),
            constraints: &[],
        };
        assert!(crate::scheduler::check_schedule(&s, &input).is_empty());
    }

    #[test]
    fn without_candidates_and_no_resource_of_the_type_the_activity_is_unfilled() {
        let task = Task::new("J1").with_priority(1).with_activity(
            Activity::new("O1", "J1", 0)
                .with_duration(ActivityDuration::fixed(500))
                .with_requirement(ResourceRequirement::new("Machine")),
        );
        let resources = vec![Resource::new("W1", "Operator")];
        let request = ScheduleRequest::new(valid(&[task], &resources));
        let s = SimpleScheduler::new().schedule_request(&request);
        assert_eq!(s.assignment_count(), 0);
        assert!(s
            .violations
            .iter()
            .any(|v| v.violation_type == crate::models::ViolationType::RequirementUnfilled));
    }

    #[test]
    fn test_no_candidate_resources() {
        // Activity with no resource requirement → skipped
        let task = Task::new("J1").with_priority(1).with_activity(
            Activity::new("O1", "J1", 0).with_duration(ActivityDuration::fixed(1000)),
            // No resource requirement
        );
        let resources = vec![make_resource("M1")];
        let scheduler = SimpleScheduler::new();
        let schedule = scheduler.schedule(&valid(&[task], &resources), 0);
        assert_eq!(schedule.assignment_count(), 0);
    }

    // ── decision-time context: each rule reads the state it is about ──

    /// A task of operations `(resource, ms)` run in order.
    fn routed(id: &str, ops: &[(&str, i64)]) -> Task {
        let mut task = Task::new(id).with_category("default");
        for (k, &(resource, ms)) in ops.iter().enumerate() {
            task = task.with_activity(
                Activity::new(format!("{id}_O{}", k + 1), id, k as i32)
                    .with_duration(ActivityDuration::fixed(ms))
                    .with_requirement(
                        ResourceRequirement::new("Machine").with_candidates(vec![resource.into()]),
                    ),
            );
        }
        task
    }

    fn machines(ids: &[&str]) -> Vec<Resource> {
        ids.iter().map(|id| make_resource(id)).collect()
    }

    fn by_rule<R: crate::dispatching::DispatchingRule + 'static>(
        rule: R,
        tasks: &[Task],
        resources: &[Resource],
    ) -> Schedule {
        SimpleScheduler::new()
            .with_rule_engine(RuleEngine::new().with_rule(rule))
            .schedule(&valid(tasks, resources), 0)
    }

    fn start_of(s: &Schedule, activity: &str) -> i64 {
        s.assignment_for_activity(activity)
            .expect(activity)
            .start_ms
    }

    #[test]
    fn critical_ratio_is_judged_at_each_decision_not_once_at_the_start() {
        // t=0: CR J2 = 60/50 = 1.2 < J1 = 10 < J3 = 14 → J2 runs 0..50.
        // t=50: J1 = 50/10 = 5, J3 = 20/5 = 4 → J3 before J1 (a sort at
        // t=0 would have kept J1 ahead of J3).
        let tasks = vec![
            routed("J1", &[("M1", 10)]).with_deadline(100),
            routed("J2", &[("M1", 50)]).with_deadline(60),
            routed("J3", &[("M1", 5)]).with_deadline(70),
        ];
        let s = by_rule(rules::Cr, &tasks, &machines(&["M1"]));
        assert_eq!(start_of(&s, "J2_O1"), 0);
        assert_eq!(start_of(&s, "J3_O1"), 50);
        assert_eq!(start_of(&s, "J1_O1"), 55);
    }

    #[test]
    fn spt_reads_the_imminent_operation_and_lwkr_the_work_left() {
        // Both start on M1. J1's first operation is short but its job is
        // long; J2 is one medium operation.
        let tasks = vec![
            routed("J1", &[("M1", 5), ("M2", 100)]),
            routed("J2", &[("M1", 30)]),
        ];
        let resources = machines(&["M1", "M2"]);
        let spt = by_rule(rules::Spt, &tasks, &resources);
        assert_eq!(start_of(&spt, "J1_O1"), 0, "SPT: 5 < 30");
        let lwkr = by_rule(rules::Lwkr, &tasks, &resources);
        assert_eq!(start_of(&lwkr, "J2_O1"), 0, "LWKR: 30 < 105");
        let mwkr = by_rule(rules::Mwkr, &tasks, &resources);
        assert_eq!(start_of(&mwkr, "J1_O1"), 0, "MWKR: 105 > 30");
    }

    #[test]
    fn winq_counts_who_waits_at_the_next_machine() {
        // J1 goes on to M2, where J3 already waits; J2 goes on to the idle M3.
        let tasks = vec![
            routed("J1", &[("M1", 10), ("M2", 10)]),
            routed("J2", &[("M1", 10), ("M3", 10)]),
            routed("J3", &[("M2", 10)]),
        ];
        let s = by_rule(rules::Winq, &tasks, &machines(&["M1", "M2", "M3"]));
        assert_eq!(start_of(&s, "J2_O1"), 0);
        assert_eq!(start_of(&s, "J1_O1"), 10);
    }

    #[test]
    fn lpul_prefers_the_less_loaded_machine() {
        // M1 (capacity 2) is half booked by a pin; M2 is idle. Both jobs can
        // start at 0, so the decision order is the rule's.
        let pin_task = routed("P", &[("M1", 100)]);
        let tasks = vec![
            pin_task,
            routed("J1", &[("M1", 10)]),
            routed("J2", &[("M2", 10)]),
        ];
        let resources = vec![make_resource("M1").with_capacity(2), make_resource("M2")];
        let s = SimpleScheduler::new()
            .with_rule_engine(RuleEngine::new().with_rule(rules::Lpul))
            .with_fixed_assignments(vec![Assignment::new("P_O1", "P", "M1", 0, 100)])
            .schedule(&valid(&tasks, &resources), 0);
        let order: Vec<&str> = s.assignments.iter().map(|a| a.task_id.as_str()).collect();
        assert_eq!(order, ["P", "J2", "J1"]);
    }

    #[test]
    fn atc_normalizes_slack_by_the_queue_average() {
        // k = 2. Own-p normalization: J1 0.1·e^(−20/20) = 0.037 < J2 0.05.
        // Queue average p̄ = 15: J1 0.1·e^(−20/30) = 0.051 > J2 0.05.
        let tasks = vec![
            routed("J1", &[("M1", 10)]).with_deadline(30),
            routed("J2", &[("M1", 20)]).with_deadline(20),
        ];
        let s = by_rule(rules::Atc::default(), &tasks, &machines(&["M1"]));
        assert_eq!(start_of(&s, "J1_O1"), 0);
        assert_eq!(start_of(&s, "J2_O1"), 10);
    }

    #[test]
    fn a_waiting_task_is_not_held_back_for_a_later_release() {
        // SPT prefers the short J1, but it is not released until after J2
        // would finish: the machine runs J2 instead of idling.
        let tasks = vec![
            routed("J1", &[("M1", 2)]).with_release_time(5),
            routed("J2", &[("M1", 3)]),
        ];
        let s = by_rule(rules::Spt, &tasks, &machines(&["M1"]));
        assert_eq!(start_of(&s, "J2_O1"), 0);
        assert_eq!(start_of(&s, "J1_O1"), 5);
    }

    #[test]
    fn no_slot_is_held_for_a_later_higher_priority_task() {
        // J1 (priority 10) arrives at 100; J2 (priority 1) waits from 0. The
        // machine starts J2 rather than idling for J1.
        let tasks = vec![
            make_task_with_resource("J1", 500, "M1", 10).with_release_time(100),
            make_task_with_resource("J2", 1000, "M1", 1),
            make_task_with_resource("J3", 300, "M1", 5),
        ];
        let s = SimpleScheduler::new().schedule(&valid(&tasks, &machines(&["M1"])), 0);
        assert_eq!(start_of(&s, "J3_O1"), 0, "J3 outranks J2 at 0");
        // At 300 both J1 (arrived) and J2 wait: priority decides.
        assert_eq!(start_of(&s, "J1_O1"), 300);
        assert_eq!(start_of(&s, "J2_O1"), 800);
    }

    #[test]
    fn ties_between_machines_go_to_declaration_order() {
        let tasks = vec![Task::new("J1").with_category("default").with_activity(
            Activity::new("J1_O1", "J1", 0)
                .with_duration(ActivityDuration::fixed(10))
                .with_requirement(ResourceRequirement::new("Machine")),
        )];
        // "M10" sorts before "M2" as a string; declaration order puts M2 first.
        let s = SimpleScheduler::new().schedule(&valid(&tasks, &machines(&["M2", "M10"])), 0);
        assert_eq!(
            s.assignment_for_activity("J1_O1")
                .expect("placed")
                .resource_id,
            "M2"
        );
    }

    /// A task of two 1 s activities: on `first`, then on any of `next`.
    fn moved(first: &str, next: &[&str]) -> Task {
        let on = |id: &str, seq: i32, machines: Vec<String>| {
            Activity::new(id, "J", seq)
                .with_process_time(1_000)
                .with_requirement(ResourceRequirement::new("Machine").with_candidates(machines))
        };
        Task::new("J")
            .with_activity(on("J-1", 0, vec![first.to_string()]))
            .with_activity(on("J-2", 1, next.iter().map(|m| m.to_string()).collect()))
    }

    #[test]
    fn the_next_activity_waits_for_the_work_to_arrive() {
        let resources = vec![
            Resource::new("A", "Machine").with_transport_to("B", 5_000),
            Resource::new("B", "Machine"),
        ];
        let s = SimpleScheduler::new().schedule(&valid(&[moved("A", &["B"])], &resources), 0);
        assert_eq!(start_of(&s, "J-2"), 6_000);
        assert!(s.violations.is_empty(), "{:?}", s.violations);
    }

    #[test]
    fn the_resource_the_work_reaches_first_is_taken() {
        let resources = vec![
            Resource::new("A", "Machine")
                .with_transport_to("B1", 5_000)
                .with_transport_to("B2", 1_000),
            Resource::new("B1", "Machine"),
            Resource::new("B2", "Machine"),
        ];
        let s =
            SimpleScheduler::new().schedule(&valid(&[moved("A", &["B1", "B2"])], &resources), 0);
        let next = s.assignment_for_activity("J-2").expect("placed");
        assert_eq!((next.resource_id.as_str(), next.start_ms), ("B2", 2_000));
    }

    /// A one-activity task on its own machine `M{id}` with operator `W1`,
    /// taking `usage` of the operator.
    fn shared(id: &str, usage: i32) -> Task {
        Task::new(id).with_activity(
            Activity::new(format!("{id}-1"), id, 0)
                .with_process_time(1_000)
                .with_requirement(
                    ResourceRequirement::new("Machine").with_candidates(vec![format!("M{id}")]),
                )
                .with_requirement(
                    ResourceRequirement::new("Operator")
                        .with_candidates(vec!["W1".into()])
                        .with_usage(usage),
                ),
        )
    }

    fn operator_and_machines(ids: &[&str]) -> Vec<Resource> {
        let mut resources: Vec<Resource> = ids
            .iter()
            .map(|id| Resource::new(format!("M{id}"), "Machine"))
            .collect();
        resources.push(Resource::new("W1", "Operator").with_capacity(10));
        resources
    }

    #[test]
    fn usages_within_the_capacity_run_side_by_side() {
        let s = SimpleScheduler::new().schedule(
            &valid(
                &[shared("A", 5), shared("B", 5), shared("C", 5)],
                &operator_and_machines(&["A", "B", "C"]),
            ),
            0,
        );
        assert_eq!(
            [
                start_of(&s, "A-1"),
                start_of(&s, "B-1"),
                start_of(&s, "C-1")
            ],
            [0, 0, 1_000]
        );
        assert!(s.violations.is_empty(), "{:?}", s.violations);
    }

    #[test]
    fn a_usage_above_every_capacity_is_left_unfilled() {
        let s = SimpleScheduler::new().schedule(
            &valid(&[shared("A", 11)], &operator_and_machines(&["A"])),
            0,
        );
        assert!(s.assignment_for_activity("A-1").is_none());
        assert!(!s.violations.is_empty());
    }

    #[test]
    fn a_rule_policy_chooses_the_rules_at_each_decision() {
        // On one machine: the longest first at the start, the shortest after.
        let tasks = [
            on("A", "M1", 1_000),
            on("B", "M1", 3_000),
            on("C", "M1", 2_000),
        ];
        let problem = valid(&tasks, &machines(&["M1"]));
        let s = SimpleScheduler::new()
            .with_rule_policy(|d: &Decision<'_>| {
                let engine = RuleEngine::new();
                if d.context.current_time_ms == 0 {
                    engine.with_rule(rules::Lpt)
                } else {
                    engine.with_rule(rules::Spt)
                }
            })
            .schedule(&problem, 0);
        assert_eq!(
            [
                start_of(&s, "B-1"),
                start_of(&s, "A-1"),
                start_of(&s, "C-1")
            ],
            [0, 3_000, 4_000]
        );

        // The decision names who waits and who competes.
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let log = std::sync::Arc::clone(&seen);
        SimpleScheduler::new()
            .with_rule_policy(move |d: &Decision<'_>| {
                log.lock()
                    .expect("not poisoned")
                    .push((d.waiting.len(), d.competing.len()));
                RuleEngine::new().with_rule(rules::Spt)
            })
            .schedule(&problem, 0);
        assert_eq!(
            *seen.lock().expect("not poisoned"),
            [(3, 3), (2, 2), (1, 1)]
        );
    }

    /// J1: 1 s on A, then 1 s on B released at 5 s.
    fn released_second() -> Task {
        let on = |id: &str, seq: i32, machine: &str| {
            Activity::new(id, "J1", seq)
                .with_process_time(1_000)
                .with_requirement(
                    ResourceRequirement::new("Machine").with_candidates(vec![machine.into()]),
                )
        };
        Task::new("J1")
            .with_activity(on("J1-1", 0, "A"))
            .with_activity(on("J1-2", 1, "B").with_release_time(5_000))
    }

    #[test]
    fn an_activity_starts_no_earlier_than_its_own_release() {
        let resources = machines(&["A", "B"]);
        let s = SimpleScheduler::new().schedule(&valid(&[released_second()], &resources), 0);
        assert_eq!((start_of(&s, "J1-1"), start_of(&s, "J1-2")), (0, 5_000));
        assert!(s.violations.is_empty(), "{:?}", s.violations);
    }
}
