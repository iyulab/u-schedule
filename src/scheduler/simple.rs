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
use crate::dispatching::{rules, RuleEngine, SchedulingContext};
use crate::models::{Activity, Assignment, Resource, Schedule, Task, TransitionMatrixCollection};
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
    rule_engine: Option<RuleEngine>,
    fixed: Vec<Assignment>,
}

impl SimpleScheduler {
    /// Creates a new scheduler.
    pub fn new() -> Self {
        Self {
            transition_matrices: TransitionMatrixCollection::new(),
            rule_engine: None,
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
    pub fn with_rule_engine(mut self, engine: RuleEngine) -> Self {
        self.rule_engine = Some(engine);
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

        // Fixed-assignment seeding: index pins by activity, pre-book their
        // resources, and emit them verbatim. Pins are opaque reservations —
        // feasibility (below) validates them; we never relocate them.
        let mut fixed_by_activity: HashMap<&str, Vec<&Assignment>> = HashMap::new();
        for pin in &self.fixed {
            fixed_by_activity
                .entry(pin.activity_id.as_str())
                .or_default()
                .push(pin);
            if let Some(tl) = timelines.get_mut(&pin.resource_id) {
                tl.book(pin.start_ms, pin.end_ms); // 선점 예약 (자원 미상이면 feasibility가 보고)
            }
            schedule.add_assignment(pin.clone());
        }

        let by_priority;
        let engine = match &self.rule_engine {
            Some(engine) => engine,
            None => {
                by_priority = RuleEngine::new().with_rule(rules::Priority);
                &by_priority
            }
        };

        // Per task: the next activity to place, and the earliest it may start.
        let mut cursor: Vec<usize> = vec![0; tasks.len()];
        let mut ready: Vec<i64> = tasks
            .iter()
            .map(|t| t.release_time.unwrap_or(start_time_ms).max(start_time_ms))
            .collect();

        loop {
            let mut offers: Vec<(usize, Placement)> = Vec::new();
            for (i, task) in tasks.iter().enumerate() {
                while let Some(activity) = task.activities.get(cursor[i]) {
                    if let Some(pins) = fixed_by_activity.get(activity.id.as_str()) {
                        // 고정 activity: SGS 스킵, 후속 커서를 pin end 이후로 전진
                        let pin_end = pins.iter().map(|p| p.end_ms).max().unwrap_or(ready[i]);
                        ready[i] = ready[i].max(pin_end);
                        cursor[i] += 1;
                        continue;
                    }
                    if activity.resource_requirements.is_empty() {
                        cursor[i] += 1; // 자원 무요구 activity는 배정하지 않음 (기존 의미 유지)
                        continue;
                    }
                    match self.place_activity(
                        activity,
                        task,
                        &timelines,
                        &last_category,
                        resources,
                        ready[i],
                    ) {
                        Some(placement) => {
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
                resources,
                &timelines,
                &fixed_by_activity,
                &cursor,
                &ready,
                &offers,
                &conflict,
                start_time_ms,
            );
            let candidates: Vec<usize> = conflict.iter().map(|&k| offers[k].0).collect();
            let chosen = engine
                .best_of(tasks, &candidates, &context)
                .expect("the conflict set is non-empty");
            let k = offers
                .iter()
                .position(|(i, _)| *i == chosen)
                .expect("the chosen task made an offer");
            let (i, placement) = offers.swap_remove(k);
            let (task, activity) = (&tasks[i], &tasks[i].activities[cursor[i]]);

            for (resource_id, setup_ms) in &placement.selections {
                let tl = timelines
                    .get_mut(resource_id)
                    .expect("selection references known resource");
                tl.book(placement.start_ms, placement.end_ms);
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
                if self.transition_matrices.has_matrix(resource_id) {
                    last_category.insert(resource_id.clone(), task.category.clone());
                }
            }
            ready[i] = placement.end_ms;
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

    /// Finds the earliest common start for an activity across all its
    /// resource requirements (serial-SGS fixed point).
    fn place_activity(
        &self,
        activity: &Activity,
        task: &Task,
        timelines: &HashMap<String, ResourceTimeline>,
        last_category: &HashMap<String, String>,
        resources: &[Resource],
        ready: i64,
    ) -> Option<Placement> {
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
        let mut selections: Vec<String> = Vec::new();
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
                        timelines
                            .get(rid)?
                            .earliest_fit(t, span)
                            .map(|fit| (fit, d.process_ms, rid))
                    })
                    .collect();
                // 동시각이면 짧은 처리, 그다음 자원 선언 순서 (stable sort)
                fits.sort_by_key(|&(fit, process, _)| (fit, process));
                if (fits.len() as i32) < req.quantity {
                    return None; // 후보 부족 — 영원히 충족 불가
                }
                for &(fit, _, rid) in fits.iter().take(req.quantity as usize) {
                    t_next = t_next.max(fit);
                    selections.push(rid.to_string());
                }
            }
            if t_next == t {
                break;
            }
            t = t_next;
        }

        // Phase 2: the time on the chosen resources together, with a common
        // span D (the largest setup), re-checked on every resource.
        let d = activity.duration_on(selections.iter().map(String::as_str));
        let selections: Vec<(String, i64)> = selections
            .into_iter()
            .map(|rid| {
                let setup = setup_for(&rid, d.setup_ms);
                (rid, setup)
            })
            .collect();
        let max_setup = selections.iter().map(|&(_, s)| s).max().unwrap_or(0);
        let d = max_setup + d.process_ms + d.teardown_ms;
        for _ in 0..MAX_FIXED_POINT_ITERS {
            let mut t_next = t;
            for (rid, _) in &selections {
                let tl = timelines.get(rid).expect("selected resource has timeline");
                // None: 캘린더가 D를 영원히 못 담음
                let fit = tl.earliest_fit(t, d)?;
                t_next = t_next.max(fit);
            }
            if t_next == t {
                return Some(Placement {
                    start_ms: t,
                    end_ms: t + d,
                    selections,
                });
            }
            t = t_next;
        }
        None // 고정점 미수렴 (방어적 상한 — property test가 실질 커버)
    }

    /// Schedules from a request.
    ///
    /// Re-annotates with the request's constraints so `TimeWindow`,
    /// `Synchronize`, `NoOverlap`, tightened `Capacity`, and cross-task
    /// `Precedence` breaches are reported in `Schedule::violations`.
    pub fn schedule_request(&self, request: &ScheduleRequest) -> Schedule {
        let scheduler = Self {
            transition_matrices: request.transition_matrices.clone(),
            rule_engine: self.rule_engine.clone(),
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

/// The state a rule judges one decision on.
///
/// Every task with an offer is waiting: its imminent activity, the work left
/// on it, when it joined the queue, and how many other waiting tasks want a
/// resource its next activity can use (WINQ). The average imminent operation
/// is over the conflict set (ATC's `p̄`). Utilization is booked time over the
/// planned span so far, per unit of capacity (LPUL).
#[allow(clippy::too_many_arguments)]
fn decision_context(
    tasks: &[Task],
    resources: &[Resource],
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

    let wanted: Vec<(usize, Vec<&str>)> = offers
        .iter()
        .map(|&(i, _)| {
            (
                i,
                tasks[i].activities[cursor[i]].eligible_resources(resources),
            )
        })
        .collect();
    for &(i, _) in offers {
        let (task, c) = (&tasks[i], cursor[i]);
        let remaining: i64 = task.activities[c..]
            .iter()
            .map(|a| a.duration.total_ms())
            .sum();
        let next_queue = task.activities[c + 1..]
            .iter()
            .find(|a| !a.resource_requirements.is_empty() && !fixed.contains_key(a.id.as_str()))
            .map(|next| {
                let usable = next.eligible_resources(resources);
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
struct Placement {
    start_ms: i64,
    end_ms: i64,
    selections: Vec<(String, i64)>, // (resource_id, setup_ms)
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
}
