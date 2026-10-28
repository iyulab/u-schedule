//! Scheduling GA problem definition.
//!
//! Implements `u_metaheur::ga::GaProblem` for scheduling optimization.
//! Bridges domain models (Task, Resource) to the generic GA framework.
//!
//! # Reference
//! Cheng et al. (1996), "A Tutorial Survey of JSSP using GA"

use std::collections::{BTreeMap, HashMap};

use rand::{Rng, RngExt};
use u_metaheur::ga::GaProblem;

use super::chromosome::ScheduleChromosome;
use super::operators::GeneticOperators;
use crate::models::{
    tardiness_ms, ActivityDuration, Assignment, Overlap, Resource, ResourceTiming, Schedule,
    TransitionMatrixCollection,
};
use crate::Problem;

mod local_search;

/// Compact activity descriptor for GA encoding.
///
/// Extracted from `Task`/`Activity` to avoid cloning full domain objects.
#[derive(Debug, Clone)]
pub struct ActivityInfo {
    /// The activity's own ID.
    ///
    /// Dropping it here is what made the decoder put the *task* id in each
    /// assignment's `activity_id`: with nothing else to hand, a plausible
    /// string filled the slot.
    pub id: String,
    /// Parent task ID.
    pub task_id: String,
    /// Activity sequence within task (1-based).
    pub sequence: i32,
    /// Setup, processing and teardown time (ms) -- the activity's own
    /// duration. A resource with a transition matrix replaces the setup with
    /// its changeover time, as [`SimpleScheduler`](crate::scheduler::SimpleScheduler) does.
    pub duration: ActivityDuration,
    /// IDs of the resources the activity may run on: those some requirement
    /// [accepts](crate::models::ResourceRequirement::accepts) (listed
    /// candidates, or resources of the requirement's type when none are
    /// listed, holding the required skills), in resource order.
    pub candidates: Vec<String>,
}

impl ActivityInfo {
    /// Extracts activity info from a checked problem, resolving each
    /// activity's eligible resources against the problem's resources.
    pub fn from_problem(problem: &Problem) -> Vec<Self> {
        let mut infos = Vec::new();
        for task in problem.tasks() {
            for (i, activity) in task.activities.iter().enumerate() {
                infos.push(ActivityInfo {
                    id: activity.id.clone(),
                    task_id: task.id.clone(),
                    sequence: (i + 1) as i32,
                    duration: activity.duration,
                    candidates: activity
                        .eligible_resources(problem.resources())
                        .into_iter()
                        .map(str::to_string)
                        .collect(),
                });
            }
        }
        infos
    }
}

/// GA problem definition for scheduling optimization.
///
/// Decodes chromosomes into schedules and evaluates fitness as makespan.
///
/// # Example
/// ```no_run
/// use u_schedule::ga::{SchedulingGaProblem, ActivityInfo};
/// use u_schedule::models::{Task, Resource};
/// use u_schedule::Problem;
/// use u_metaheur::ga::{GaConfig, GaRunner};
///
/// let tasks: Vec<Task> = vec![/* ... */];
/// let resources: Vec<Resource> = vec![/* ... */];
/// let problem = SchedulingGaProblem::new(&Problem::new(tasks, resources).expect("valid input"));
/// let config = GaConfig::default();
/// let result = GaRunner::run(&problem, &config);
/// ```
pub struct SchedulingGaProblem {
    /// Activity info (extracted from tasks).
    pub activities: Vec<ActivityInfo>,
    /// Available resources.
    pub resources: Vec<Resource>,
    /// Task categories (task_id → category).
    pub task_categories: HashMap<String, String>,
    /// Transition matrices for setup times.
    pub transition_matrices: TransitionMatrixCollection,
    /// Task deadlines (task_id → deadline_ms).
    pub deadlines: HashMap<String, i64>,
    /// Task release times (task_id → release_ms).
    pub release_times: HashMap<String, i64>,
    /// Weight for tardiness in fitness (default: 0.5).
    pub tardiness_weight: f64,
    /// Genetic operators for crossover/mutation strategy selection.
    ///
    /// Default: POX crossover + Swap mutation.
    /// Override with [`with_operators`](SchedulingGaProblem::with_operators).
    pub operators: GeneticOperators,
    /// Swaps the local search may make to improve each child of a crossover and
    /// each mutated chromosome (0 turns it off): each swap exchanges two
    /// neighbours at the edge of a critical block, the best one even when it
    /// lengthens the schedule (taboo search), and the best sequences seen are kept.
    ///
    /// Default: 60. Set with
    /// [`with_local_search_steps`](SchedulingGaProblem::with_local_search_steps).
    pub local_search_steps: usize,
    /// Precomputed index: `(task_id, sequence) → activities index`.
    ///
    /// Built once at construction, enables O(1) activity lookup during decode.
    activity_index: HashMap<(String, i32), usize>,
    /// The activity indices of each task, in sequence order (what
    /// [`decode_active`](Self::decode_active) steps through).
    task_activities: Vec<Vec<usize>>,
    /// The ids of the tasks, in the order of `task_activities`.
    task_ids: Vec<String>,
    /// `task id → index in task_ids`.
    task_index: HashMap<String, usize>,
    /// The task (index in `task_ids`) of each activity.
    task_of: Vec<usize>,
    /// Each activity's [`resource_timings`](crate::models::Activity::resource_timings),
    /// by activity index.
    timings: Vec<BTreeMap<String, ResourceTiming>>,
    /// Each activity's own [`release_time`](crate::models::Activity::release_time)
    /// (0 without one), by activity index.
    activity_release: Vec<i64>,
    /// Each activity's [`overlap`](crate::models::Activity::overlap) of the
    /// previous step of its task, by activity index.
    overlap: Vec<Option<Overlap>>,
    /// Whether any activity's time depends on its resource — when none does,
    /// the decoders read each activity's own duration without a lookup.
    timed: bool,
    /// Each activity's [`predecessors`](crate::models::Activity::predecessors)
    /// in any task, as activity indices: a decoder places an activity only
    /// once they are, and not before they end.
    predecessors: Vec<Vec<usize>>,
    /// `resource id → index in resources`.
    resource_index: HashMap<String, usize>,
    /// The problem itself, which the dispatching seeds are scheduled on.
    problem: Problem,
}

impl SchedulingGaProblem {
    /// Creates a GA problem from a checked scheduling problem.
    pub fn new(problem: &Problem) -> Self {
        let (tasks, resources) = (problem.tasks(), problem.resources());
        let activities = ActivityInfo::from_problem(problem);
        // In the order `from_problem` lists the activities.
        let timings: Vec<BTreeMap<String, ResourceTiming>> = problem
            .tasks()
            .iter()
            .flat_map(|t| &t.activities)
            .map(|a| a.resource_timings.clone())
            .collect();
        let activity_release: Vec<i64> = problem
            .tasks()
            .iter()
            .flat_map(|t| &t.activities)
            .map(|a| a.release_time.unwrap_or(0))
            .collect();
        let overlap: Vec<Option<Overlap>> = problem
            .tasks()
            .iter()
            .flat_map(|t| &t.activities)
            .map(|a| a.overlap)
            .collect();
        let mut task_categories = HashMap::new();
        let mut deadlines = HashMap::new();
        let mut release_times = HashMap::new();

        for task in tasks {
            task_categories.insert(task.id.clone(), task.category.clone());
            if let Some(dl) = task.deadline {
                deadlines.insert(task.id.clone(), dl);
            }
            if let Some(rt) = task.release_time {
                release_times.insert(task.id.clone(), rt);
            }
        }

        // Build (task_id, sequence) → index lookup for O(1) decode
        let activity_index: HashMap<(String, i32), usize> = activities
            .iter()
            .enumerate()
            .map(|(i, a)| ((a.task_id.clone(), a.sequence), i))
            .collect();

        let mut by_task: Vec<(&str, Vec<usize>)> = Vec::new();
        for (i, act) in activities.iter().enumerate() {
            match by_task.iter_mut().find(|(t, _)| *t == act.task_id) {
                Some((_, ops)) => ops.push(i),
                None => by_task.push((act.task_id.as_str(), vec![i])),
            }
        }
        let task_ids: Vec<String> = by_task.iter().map(|(t, _)| t.to_string()).collect();
        let task_activities: Vec<Vec<usize>> = by_task
            .into_iter()
            .map(|(_, mut ops)| {
                ops.sort_by_key(|&i| activities[i].sequence);
                ops
            })
            .collect();
        let task_index: HashMap<String, usize> = task_ids
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i))
            .collect();
        let mut task_of = vec![0usize; activities.len()];
        for (t, ops) in task_activities.iter().enumerate() {
            for &a in ops {
                task_of[a] = t;
            }
        }
        let resource_index: HashMap<String, usize> = resources
            .iter()
            .enumerate()
            .map(|(i, r)| (r.id.clone(), i))
            .collect();
        // `Problem::new` refused unknown predecessors and cycles.
        let by_id: HashMap<&str, usize> = activities
            .iter()
            .enumerate()
            .map(|(i, a)| (a.id.as_str(), i))
            .collect();
        let predecessors: Vec<Vec<usize>> = problem
            .tasks()
            .iter()
            .flat_map(|t| &t.activities)
            .map(|a| {
                a.predecessors
                    .iter()
                    .filter_map(|p| by_id.get(p.as_str()).copied())
                    .collect()
            })
            .collect();

        Self {
            activities,
            resources: resources.to_vec(),
            task_categories,
            transition_matrices: TransitionMatrixCollection::new(),
            deadlines,
            release_times,
            tardiness_weight: 0.5,
            operators: GeneticOperators::default(),
            local_search_steps: 60,
            activity_index,
            task_activities,
            task_ids,
            task_index,
            task_of,
            activity_release,
            overlap,
            timed: timings.iter().any(|t| !t.is_empty()),
            timings,
            predecessors,
            resource_index,
            problem: problem.clone(),
        }
    }

    /// `schedule` as a chromosome: the activities in order of start (ties by
    /// sequence), each on the resource it was given. Decoding it yields a
    /// schedule no longer than `schedule` -- the decoder starts every activity
    /// as early as its resource and its predecessor allow, in this order.
    /// Activities the schedule leaves out go last, in sequence order, on their
    /// first candidate.
    pub fn chromosome_of(&self, schedule: &Schedule) -> ScheduleChromosome {
        let placed: HashMap<&str, (i64, &str)> = schedule
            .assignments
            .iter()
            .map(|a| (a.activity_id.as_str(), (a.start_ms, a.resource_id.as_str())))
            .collect();
        let mut order: Vec<(i64, i32, usize)> = self
            .activities
            .iter()
            .enumerate()
            .map(|(i, act)| {
                let start = placed.get(act.id.as_str()).map_or(i64::MAX, |p| p.0);
                (start, act.sequence, i)
            })
            .collect();
        order.sort();
        let osv = order
            .iter()
            .map(|&(_, _, i)| self.activities[i].task_id.clone())
            .collect();
        let mav = self
            .activities
            .iter()
            .map(|act| match placed.get(act.id.as_str()) {
                Some(&(_, resource)) => resource.to_string(),
                None => act.candidates.first().cloned().unwrap_or_default(),
            })
            .collect();
        ScheduleChromosome {
            osv,
            mav,
            activity_index: self.activity_index.clone(),
            fitness: f64::INFINITY,
        }
    }

    /// The dispatching rules' schedules ([`SimpleScheduler`]) -- MWKR, SPT,
    /// LPT, FIFO, LWKR, and EDD and ATC when a task has a deadline -- as
    /// chromosomes: the population starts from them as well as from random
    /// ones, so the search never ends worse than the best of them.
    ///
    /// [`SimpleScheduler`]: crate::scheduler::SimpleScheduler
    fn dispatching_seeds(&self) -> Vec<ScheduleChromosome> {
        use crate::dispatching::{rules, RuleEngine};
        let mut engines = vec![
            RuleEngine::new().with_rule(rules::Mwkr),
            RuleEngine::new().with_rule(rules::Spt),
            RuleEngine::new().with_rule(rules::Lpt),
            RuleEngine::new().with_rule(rules::Fifo),
            RuleEngine::new().with_rule(rules::Lwkr),
        ];
        if !self.deadlines.is_empty() {
            engines.push(RuleEngine::new().with_rule(rules::Edd));
            engines.push(RuleEngine::new().with_rule(rules::Atc::default()));
        }
        engines
            .into_iter()
            .map(|engine| {
                let schedule = crate::scheduler::SimpleScheduler::new()
                    .with_transition_matrices(self.transition_matrices.clone())
                    .with_rule_engine(engine)
                    .schedule(&self.problem, 0);
                self.chromosome_of(&schedule)
            })
            .collect()
    }

    /// Sets transition matrices.
    pub fn with_transition_matrices(mut self, matrices: TransitionMatrixCollection) -> Self {
        self.transition_matrices = matrices;
        self
    }

    /// Sets tardiness weight (0.0 = pure makespan, 1.0 = pure tardiness).
    ///
    /// # Errors
    /// `TardinessWeightOutOfRange` when `weight` is not a number in `[0, 1]`.
    /// (It used to be clamped, so 1.5 ran as pure tardiness and NaN slipped
    /// through to every fitness.)
    pub fn with_tardiness_weight(
        mut self,
        weight: f64,
    ) -> Result<Self, crate::validation::ValidationError> {
        if !(0.0..=1.0).contains(&weight) {
            return Err(crate::validation::ValidationError::new(
                crate::validation::ValidationErrorKind::TardinessWeightOutOfRange { weight },
            ));
        }
        self.tardiness_weight = weight;
        Ok(self)
    }

    /// Sets how many swaps the local search may make to improve each child and
    /// each mutated chromosome; 0 turns it off.
    pub fn with_local_search_steps(mut self, steps: usize) -> Self {
        self.local_search_steps = steps;
        self
    }

    /// Sets the genetic operators (crossover + mutation strategy).
    ///
    /// # Example
    /// ```no_run
    /// use u_schedule::ga::SchedulingGaProblem;
    /// use u_schedule::ga::operators::{GeneticOperators, CrossoverType, MutationType};
    ///
    /// # let checked = u_schedule::Problem::new(vec![], vec![]).unwrap();
    /// let problem = SchedulingGaProblem::new(&checked)
    ///     .with_operators(GeneticOperators {
    ///         crossover_type: CrossoverType::LOX,
    ///         mutation_type: MutationType::Invert,
    ///     });
    /// ```
    pub fn with_operators(mut self, operators: GeneticOperators) -> Self {
        self.operators = operators;
        self
    }

    /// Decodes a chromosome into a schedule: the better, by the fitness the
    /// search minimizes, of the chromosome's own order
    /// ([`decode_semi_active`](Self::decode_semi_active)) and the active
    /// schedule that order prioritizes ([`decode_active`](Self::decode_active)).
    /// An active schedule lets an activity jump ahead into the idle time
    /// before a longer one when that delays nothing, so the search reaches
    /// schedules the order alone cannot express; taking the better of the two
    /// keeps the guarantee that a chromosome made from a schedule decodes to
    /// one no longer than it.
    pub fn decode(&self, chromosome: &ScheduleChromosome) -> Schedule {
        let view = self.view();
        let ix = self.indexed(chromosome);
        let (placed, _) = self.better_placement(&view, &ix);
        self.schedule_of(&ix, &placed)
    }

    /// Decodes a chromosome into a Schedule, placing the activities in the
    /// order of its OSV, each as early as its resource and its predecessor
    /// allow.
    pub fn decode_semi_active(&self, chromosome: &ScheduleChromosome) -> Schedule {
        let view = self.view();
        let ix = self.indexed(chromosome);
        self.schedule_of(&ix, &self.place_in_order(&view, &ix))
    }

    /// Decodes a chromosome into an *active* schedule (Giffler & Thompson 1960):
    /// repeatedly take, among the activities next in their tasks, the one that
    /// could finish earliest, and of those waiting for its resource before it
    /// finishes, start the one the chromosome's order ranks first. No activity
    /// can then start earlier without delaying another, and an optimal
    /// schedule is among the active ones. Each activity runs on the resource
    /// the chromosome assigns it.
    pub fn decode_active(&self, chromosome: &ScheduleChromosome) -> Schedule {
        let view = self.view();
        let ix = self.indexed(chromosome);
        self.schedule_of(&ix, &self.place_active(&view, &ix))
    }

    /// The tasks' release times, categories and deadlines, by task index --
    /// read from the public maps each time, since they may be changed after
    /// the problem is built.
    fn view(&self) -> TaskView<'_> {
        let ids = &self.task_ids;
        TaskView {
            release: ids
                .iter()
                .map(|t| self.release_times.get(t).copied().unwrap_or(0))
                .collect(),
            category: ids
                .iter()
                .map(|t| self.task_categories.get(t).map_or("", String::as_str))
                .collect(),
            deadline: ids.iter().map(|t| self.deadlines.get(t).copied()).collect(),
            moves: self.resources.iter().any(|r| !r.transport_ms.is_empty()),
        }
    }

    /// A chromosome as indices: its OSV as the activities in order, and the
    /// resource of each activity. An OSV entry with no activity behind it, and
    /// an activity with no resource, are left out, as the decoders always
    /// have.
    fn indexed(&self, chromosome: &ScheduleChromosome) -> Indexed {
        let mut seen = vec![0usize; self.task_activities.len()];
        let order = chromosome
            .osv
            .iter()
            .filter_map(|task_id| {
                let t = *self.task_index.get(task_id.as_str())?;
                let k = seen[t];
                seen[t] += 1;
                self.task_activities[t].get(k).copied()
            })
            .collect();
        let mut extra: Vec<&str> = Vec::new();
        let resource = (0..self.activities.len())
            .map(|a| match chromosome.mav.get(a).map(String::as_str) {
                Some(id) if !id.is_empty() => Some(match self.resource_index.get(id) {
                    Some(&r) => r,
                    // A resource the problem does not list: it is free from the
                    // start, like any other.
                    None => {
                        let at = extra.iter().position(|&e| e == id).unwrap_or_else(|| {
                            extra.push(id);
                            extra.len() - 1
                        });
                        self.resources.len() + at
                    }
                }),
                _ => None,
            })
            .collect::<Vec<_>>();
        let duration = (0..self.activities.len())
            .map(|a| match (self.timed, chromosome.mav.get(a)) {
                (true, Some(id)) if !id.is_empty() => self.duration_on(a, id),
                _ => self.activities[a].duration,
            })
            .collect();
        // The names of the resources the problem does not list, which a
        // schedule needs; `extra` borrowed them from the chromosome, so keep
        // them as owned strings.
        Indexed {
            order,
            resource,
            duration,
            extra: extra.into_iter().map(str::to_string).collect(),
        }
    }

    /// The time of the activity at index `activity` on the resource named
    /// `resource` ([`Activity::duration_on`](crate::models::Activity::duration_on)).
    fn duration_on(&self, activity: usize, resource: &str) -> ActivityDuration {
        crate::models::timed_duration(
            &self.activities[activity].duration,
            &self.timings[activity],
            [resource],
        )
    }

    /// The earliest `activity` may start: its task's release, or its own if
    /// later.
    fn release_of(&self, view: &TaskView<'_>, activity: usize) -> i64 {
        view.release[self.task_of[activity]].max(self.activity_release[activity])
    }

    /// The time work takes to move from the resource at index `from` to the
    /// one at `to` ([`Resource::transport_ms`]); a resource the problem does
    /// not list moves nothing.
    fn transport(&self, from: usize, to: usize) -> i64 {
        match (self.resources.get(from), self.resources.get(to)) {
            (Some(f), Some(t)) if !f.transport_ms.is_empty() => f.transport_to(&t.id),
            _ => 0,
        }
    }

    /// The earliest `activity`, taking `span` on `resource`, may start after
    /// the activity before it in its task (`pred`, if any): that one's end —
    /// or, for a step that overlaps it ([`Overlap`]), its share of that one's
    /// span, and late enough to end no earlier than the overlap allows — plus
    /// the move from the resource that one ran on.
    #[allow(clippy::too_many_arguments)]
    fn arrival(
        &self,
        view: &TaskView<'_>,
        ix: &Indexed,
        placed: &Placed,
        pred: Option<usize>,
        activity: usize,
        resource: usize,
        span: i64,
    ) -> i64 {
        let Some(p) = pred else { return 0 };
        let moved = if view.moves {
            ix.resource[p].map_or(0, |from| self.transport(from, resource))
        } else {
            0
        };
        let after = match &self.overlap[activity] {
            None => placed.end[p],
            Some(overlap) => overlap
                .earliest_start((placed.start[p], placed.end[p]))
                .max(overlap.earliest_end(placed.end[p]) - span),
        };
        after + moved
    }

    /// The setup of `activity` on `resource`, after the task last run there
    /// (`previous`): the activity's own, or the transition matrix's.
    fn setup_of(
        &self,
        view: &TaskView<'_>,
        ix: &Indexed,
        activity: usize,
        resource: usize,
        previous: Option<usize>,
    ) -> i64 {
        let own = ix.duration[activity].setup_ms;
        if self.transition_matrices.is_empty() {
            return own;
        }
        self.transition_matrices.setup_ms(
            ix.resource_name(self, resource),
            previous.map(|t| view.category[t]),
            view.category[self.task_of[activity]],
            own,
        )
    }

    /// Places the activities in the order of the chromosome, each as early as
    /// its resource and its predecessor allow.
    fn place_in_order(&self, view: &TaskView<'_>, ix: &Indexed) -> Placed {
        let n_resources = self.resources.len() + ix.extra.len();
        let mut resource_available = vec![0i64; n_resources];
        let mut last_task: Vec<Option<usize>> = vec![None; n_resources];
        let mut task_last: Vec<Option<usize>> = vec![None; self.task_activities.len()];
        let mut placed = Placed::new(self.activities.len());
        let mut settled = vec![false; self.activities.len()];
        for a in self.ready_order(ix) {
            settled[a] = true;
            let Some(r) = ix.resource[a] else { continue };
            let t = self.task_of[a];
            let setup = self.setup_of(view, ix, a, r, last_task[r]);
            let duration = ix.duration[a];
            let span = setup + duration.process_ms + duration.teardown_ms;
            let start = resource_available[r]
                .max(self.arrival(view, ix, &placed, task_last[t], a, r, span))
                .max(self.release_of(view, a))
                .max(self.predecessors_end(a, &placed.end, &settled));
            let end = start + span;
            placed.put(a, start, end, setup);
            resource_available[r] = end;
            task_last[t] = Some(a);
            last_task[r] = Some(t);
        }
        placed
    }

    /// The chromosome's order with each activity moved after its
    /// [`predecessors`](Self::predecessors) in other tasks (and so after the
    /// rest of its task's earlier steps): repeatedly the first activity of the
    /// order still waiting whose predecessors have all been taken — a
    /// priority list read in a precedence-feasible order, as a serial
    /// schedule generation scheme reads it (Kolisch & Hartmann 1999). The
    /// order itself when no activity has such predecessors.
    fn ready_order(&self, ix: &Indexed) -> Vec<usize> {
        if self.predecessors.iter().all(Vec::is_empty) {
            return ix.order.clone();
        }
        let mut taken = vec![false; self.activities.len()];
        // An activity the order leaves out never comes: its successors do not
        // wait for it.
        let mut in_order = vec![false; self.activities.len()];
        for &a in &ix.order {
            in_order[a] = true;
        }
        let mut waiting = ix.order.clone();
        let mut out = Vec::with_capacity(waiting.len());
        while !waiting.is_empty() {
            let k = waiting
                .iter()
                .position(|&a| {
                    self.predecessors[a]
                        .iter()
                        .all(|&p| taken[p] || !in_order[p])
                        && self.task_activities[self.task_of[a]]
                            .iter()
                            .take_while(|&&b| b != a)
                            .all(|&b| taken[b] || !in_order[b])
                })
                // Only a cycle `Problem::new` refuses leaves none ready.
                .unwrap_or(0);
            let a = waiting.remove(k);
            taken[a] = true;
            out.push(a);
        }
        out
    }

    /// When the last of `activity`'s predecessors in other tasks ends, of
    /// those placed (`settled` and given a resource); 0 when none.
    fn predecessors_end(&self, activity: usize, end: &[i64], settled: &[bool]) -> i64 {
        self.predecessors[activity]
            .iter()
            .filter(|&&p| settled[p])
            .map(|&p| end[p])
            .max()
            .unwrap_or(0)
    }

    /// Places the activities as an active schedule; see
    /// [`decode_active`](Self::decode_active).
    fn place_active(&self, view: &TaskView<'_>, ix: &Indexed) -> Placed {
        // Rank of each activity in the chromosome's order.
        let mut rank = vec![usize::MAX; self.activities.len()];
        for (position, &a) in ix.order.iter().enumerate() {
            rank[a] = position;
        }
        let n_resources = self.resources.len() + ix.extra.len();
        let mut resource_available = vec![0i64; n_resources];
        let mut last_task: Vec<Option<usize>> = vec![None; n_resources];
        let mut task_last: Vec<Option<usize>> = vec![None; self.task_activities.len()];
        let mut next = vec![0usize; self.task_activities.len()];
        let mut placed = Placed::new(self.activities.len());
        // Placed, or passed over for want of a resource.
        let mut settled = vec![false; self.activities.len()];

        // The activity next in a task, with the resource it runs on and when
        // it could start and finish there.
        struct Candidate {
            activity: usize,
            resource: usize,
            setup: i64,
            start: i64,
            finish: i64,
        }
        let mut candidates: Vec<Candidate> = Vec::with_capacity(self.task_activities.len());

        loop {
            candidates.clear();
            for (t, ops) in self.task_activities.iter().enumerate() {
                // An activity with no resource is left out, as the order-based
                // decoder leaves it out.
                while let Some(&a) = ops.get(next[t]) {
                    let Some(r) = ix.resource[a] else {
                        settled[a] = true;
                        next[t] += 1;
                        continue;
                    };
                    // It waits for its predecessors in other tasks.
                    if !self.predecessors[a].iter().all(|&p| settled[p]) {
                        break;
                    }
                    let setup = self.setup_of(view, ix, a, r, last_task[r]);
                    let duration = ix.duration[a];
                    let span = setup + duration.process_ms + duration.teardown_ms;
                    let start = resource_available[r]
                        .max(self.arrival(view, ix, &placed, task_last[t], a, r, span))
                        .max(self.release_of(view, a))
                        .max(self.predecessors_end(a, &placed.end, &settled));
                    candidates.push(Candidate {
                        activity: a,
                        resource: r,
                        setup,
                        start,
                        finish: start + span,
                    });
                    break;
                }
            }

            // The one that could finish first fixes the resource and the
            // moment the conflict is judged at.
            let Some(first) = candidates
                .iter()
                .min_by_key(|c| (c.finish, rank[c.activity]))
            else {
                break;
            };
            let (resource, finish, first_activity) = (first.resource, first.finish, first.activity);
            let chosen = candidates
                .iter()
                .filter(|c| {
                    c.resource == resource && (c.start < finish || c.activity == first_activity)
                })
                .min_by_key(|c| rank[c.activity])
                .expect("the first candidate is in the conflict set");

            let t = self.task_of[chosen.activity];
            placed.put(chosen.activity, chosen.start, chosen.finish, chosen.setup);
            settled[chosen.activity] = true;
            resource_available[resource] = chosen.finish;
            task_last[t] = Some(chosen.activity);
            last_task[resource] = Some(t);
            next[t] += 1;
        }
        placed
    }

    /// The better of the two placements of a chromosome, with its fitness.
    /// The one placed in the chromosome's order wins a tie.
    fn better_placement(&self, view: &TaskView<'_>, ix: &Indexed) -> (Placed, f64) {
        let in_order = self.place_in_order(view, ix);
        let in_order_fitness = self.fitness_of(view, &in_order);
        let active = self.place_active(view, ix);
        let active_fitness = self.fitness_of(view, &active);
        if active_fitness < in_order_fitness {
            (active, active_fitness)
        } else {
            (in_order, in_order_fitness)
        }
    }

    /// A placement as a [`Schedule`], its assignments in the order they were
    /// placed.
    fn schedule_of(&self, ix: &Indexed, placed: &Placed) -> Schedule {
        let mut schedule = Schedule::new();
        for &a in &placed.order {
            let act = &self.activities[a];
            let resource = ix.resource[a].expect("a placed activity has a resource");
            schedule.add_assignment(
                Assignment::new(
                    &act.id,
                    &act.task_id,
                    ix.resource_name(self, resource),
                    placed.start[a],
                    placed.end[a],
                )
                .with_sequence(act.sequence)
                .with_setup(placed.setup[a]),
            );
        }
        schedule
    }

    /// Fitness of a placement: weighted combination of makespan and tardiness.
    fn fitness_of(&self, view: &TaskView<'_>, placed: &Placed) -> f64 {
        let makespan = placed
            .order
            .iter()
            .map(|&a| placed.end[a])
            .max()
            .unwrap_or(0) as f64;
        let mut completion = vec![0i64; self.task_activities.len()];
        for &a in &placed.order {
            let t = self.task_of[a];
            completion[t] = completion[t].max(placed.end[a]);
        }
        let total_tardiness: f64 = completion
            .iter()
            .zip(&view.deadline)
            .map(|(&c, &d)| tardiness_ms(c, d) as f64)
            .sum();

        // Weighted combination (both terms in ms, comparable scale)
        let makespan_weight = 1.0 - self.tardiness_weight;
        makespan_weight * makespan + self.tardiness_weight * total_tardiness
    }
}

/// The tasks' release times, categories and deadlines, by task index.
struct TaskView<'a> {
    release: Vec<i64>,
    category: Vec<&'a str>,
    deadline: Vec<Option<i64>>,
    /// Whether any resource gives a transport time.
    moves: bool,
}

/// A chromosome as indices into the problem's activities and resources.
struct Indexed {
    /// The activities in the order of the OSV.
    order: Vec<usize>,
    /// The resource each activity runs on: an index into the problem's
    /// resources, or past them for one the problem does not list. `None` when
    /// the chromosome gives it none.
    resource: Vec<Option<usize>>,
    /// Each activity's time on the resource it runs on
    /// ([`Activity::duration_on`](crate::models::Activity::duration_on)) —
    /// resolved once per chromosome, read by every placement.
    duration: Vec<ActivityDuration>,
    /// The ids of the unlisted resources, in the order of their indices past
    /// the problem's.
    extra: Vec<String>,
}

impl Indexed {
    fn resource_name<'s>(&'s self, problem: &'s SchedulingGaProblem, resource: usize) -> &'s str {
        match problem.resources.get(resource) {
            Some(r) => r.id.as_str(),
            None => self.extra[resource - problem.resources.len()].as_str(),
        }
    }
}

/// Where and when each activity was placed.
struct Placed {
    /// The activities in the order they were placed.
    order: Vec<usize>,
    start: Vec<i64>,
    end: Vec<i64>,
    setup: Vec<i64>,
}

impl Placed {
    fn new(n: usize) -> Self {
        Self {
            order: Vec::with_capacity(n),
            start: vec![0; n],
            end: vec![0; n],
            setup: vec![0; n],
        }
    }

    fn put(&mut self, activity: usize, start: i64, end: i64, setup: i64) {
        self.order.push(activity);
        self.start[activity] = start;
        self.end[activity] = end;
        self.setup[activity] = setup;
    }
}

impl GaProblem for SchedulingGaProblem {
    type Individual = ScheduleChromosome;

    /// The dispatching rules' schedules seed the population.
    fn initial_individuals(&self) -> Vec<ScheduleChromosome> {
        self.dispatching_seeds()
    }

    fn create_individual<R: Rng>(&self, rng: &mut R) -> ScheduleChromosome {
        // 50% random, 25% load-balanced, 25% SPT (or load-balanced when no
        // activity's time depends on its resource)
        let roll: f64 = rng.random_range(0.0..1.0);
        if roll < 0.5 {
            ScheduleChromosome::random(&self.activities, rng)
        } else if roll < 0.75 || !self.timed {
            ScheduleChromosome::with_load_balancing(
                &self.activities,
                |a, r| self.duration_on(a, r).total_ms(),
                rng,
            )
        } else {
            ScheduleChromosome::with_shortest_time(
                &self.activities,
                |a, r| self.duration_on(a, r).process_ms,
                rng,
            )
        }
    }

    fn evaluate(&self, individual: &ScheduleChromosome) -> f64 {
        let view = self.view();
        let ix = self.indexed(individual);
        self.better_placement(&view, &ix).1
    }

    fn crossover<R: Rng>(
        &self,
        parent1: &ScheduleChromosome,
        parent2: &ScheduleChromosome,
        rng: &mut R,
    ) -> Vec<ScheduleChromosome> {
        let (c1, c2) = self
            .operators
            .crossover(parent1, parent2, &self.activities, rng);
        let mut children = vec![c1, c2];
        for child in &mut children {
            self.improve(child);
        }
        children
    }

    fn mutate<R: Rng>(&self, individual: &mut ScheduleChromosome, rng: &mut R) {
        self.operators.mutate(individual, &self.activities, rng);
        self.improve(individual);
    }
}

// Make SchedulingGaProblem Send + Sync (all fields are owned data)
unsafe impl Send for SchedulingGaProblem {}
unsafe impl Sync for SchedulingGaProblem {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ga::operators::{CrossoverType, MutationType};
    use crate::models::{Activity, ActivityDuration, ResourceRequirement, Task};
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use u_metaheur::ga::{GaConfig, GaRunner};

    /// A checked problem, for tests whose input is valid by construction.
    fn valid(tasks: &[Task], resources: &[Resource]) -> crate::Problem {
        crate::Problem::new(tasks.to_vec(), resources.to_vec()).expect("a valid problem")
    }

    fn make_test_problem() -> (Vec<Task>, Vec<Resource>) {
        let tasks = vec![
            Task::new("T1")
                .with_category("TypeA")
                .with_priority(5)
                .with_deadline(10_000)
                .with_activity(
                    Activity::new("T1_O1", "T1", 0)
                        .with_duration(ActivityDuration::fixed(1000))
                        .with_requirement(
                            ResourceRequirement::new("Machine")
                                .with_candidates(vec!["M1".into(), "M2".into()]),
                        ),
                )
                .with_activity(
                    Activity::new("T1_O2", "T1", 1)
                        .with_duration(ActivityDuration::fixed(2000))
                        .with_requirement(
                            ResourceRequirement::new("Machine").with_candidates(vec!["M2".into()]),
                        ),
                ),
            Task::new("T2")
                .with_category("TypeB")
                .with_priority(3)
                .with_activity(
                    Activity::new("T2_O1", "T2", 0)
                        .with_duration(ActivityDuration::fixed(1500))
                        .with_requirement(
                            ResourceRequirement::new("Machine")
                                .with_candidates(vec!["M1".into(), "M3".into()]),
                        ),
                ),
        ];

        let resources = vec![
            Resource::new("M1", "Machine"),
            Resource::new("M2", "Machine"),
            Resource::new("M3", "Machine"),
        ];

        (tasks, resources)
    }

    #[test]
    fn test_activity_info_from_problem() {
        let (tasks, resources) = make_test_problem();
        let infos = ActivityInfo::from_problem(&valid(&tasks, &resources));
        assert_eq!(infos.len(), 3);
        assert_eq!(infos[0].id, "T1_O1");
        assert_eq!(infos[0].task_id, "T1");
        assert_eq!(infos[0].sequence, 1);
        assert_eq!(infos[0].duration.process_ms, 1000);
        assert_eq!(infos[1].id, "T1_O2");
        assert_eq!(infos[2].id, "T2_O1");
        assert_eq!(infos[2].task_id, "T2");
        assert_eq!(infos[0].candidates, vec!["M1", "M2"]);
        assert_eq!(infos[2].candidates, vec!["M1", "M3"]);
    }

    #[test]
    fn without_candidates_the_ga_draws_only_resources_of_the_requirement_type() {
        let tasks = vec![Task::new("T1").with_activity(
            Activity::new("T1_O1", "T1", 0)
                .with_duration(ActivityDuration::fixed(1000))
                .with_requirement(ResourceRequirement::new("Machine").with_skill("milling")),
        )];
        let resources = vec![
            Resource::new("W1", "Operator").with_skill("milling"),
            Resource::new("M1", "Machine").with_skill("milling"),
            Resource::new("M2", "Machine"),
        ];
        let infos = ActivityInfo::from_problem(&valid(&tasks, &resources));
        assert_eq!(infos[0].candidates, vec!["M1"]);
    }

    /// An assignment names the activity it scheduled, and says which step of
    /// its task that activity is.
    ///
    /// Both used to be wrong together, and one hid the other. `ActivityInfo`
    /// dropped the activity's own id, so the decoder had nothing to put in the
    /// `activity_id` slot and put the *task* id there -- the same string for
    /// every step of a job. The WebAssembly binding then recovered the step by
    /// parsing a trailing number out of that id, falling back to 1 when it
    /// could not, so every row of every job-shop schedule read "operation 1"
    /// while machines, times and makespan were all correct.
    #[test]
    fn an_assignment_names_its_activity_and_its_step() {
        let (tasks, resources) = make_test_problem();
        let problem = SchedulingGaProblem::new(&valid(&tasks, &resources));
        let mut rng = SmallRng::seed_from_u64(42);
        let ch = problem.create_individual(&mut rng);
        let schedule = problem.decode(&ch);
        assert_eq!(schedule.assignment_count(), 3);

        for a in &schedule.assignments {
            assert_ne!(
                a.activity_id, a.task_id,
                "the activity id must not be the task id: {a:?}"
            );
            assert!(
                a.activity_id.starts_with(&a.task_id),
                "activity ids in this fixture are '<task>_O<n>': {a:?}"
            );
        }

        // T1 has two steps and they are distinct, in the order the task lists
        // them; T2 has one.
        let steps = |task: &str| {
            let mut v: Vec<_> = schedule
                .assignments
                .iter()
                .filter(|a| a.task_id == task)
                .map(|a| {
                    (
                        a.sequence.expect("the decoder says the step"),
                        a.activity_id.clone(),
                    )
                })
                .collect();
            v.sort();
            v
        };
        assert_eq!(
            steps("T1"),
            vec![(1, "T1_O1".to_string()), (2, "T1_O2".to_string())]
        );
        assert_eq!(steps("T2"), vec![(1, "T2_O1".to_string())]);
    }

    #[test]
    fn test_decode_chromosome() {
        let (tasks, resources) = make_test_problem();
        let problem = SchedulingGaProblem::new(&valid(&tasks, &resources));
        let mut rng = SmallRng::seed_from_u64(42);
        let ch = problem.create_individual(&mut rng);

        let schedule = problem.decode(&ch);
        // Should have assignments for all 3 activities
        assert!(schedule.assignment_count() > 0);
        assert!(schedule.makespan_ms() > 0);
    }

    /// A seed decodes to a schedule no longer than the dispatching schedule it
    /// came from, so the GA (which keeps its best) never ends worse.
    #[test]
    fn a_dispatching_seed_decodes_no_longer_than_its_schedule() {
        use crate::dispatching::{rules, RuleEngine};
        use crate::scheduler::SimpleScheduler;
        let mut tasks = Vec::new();
        for j in 0..6 {
            let mut task = Task::new(format!("J{j}")).with_category("c");
            for k in 0..3 {
                let m = (j + k) % 3;
                task = task.with_activity(
                    Activity::new(format!("J{j}_O{k}"), format!("J{j}"), k)
                        .with_duration(ActivityDuration::fixed(10 + ((j * 7 + k * 3) % 11) as i64))
                        .with_requirement(
                            ResourceRequirement::new("M").with_candidates(vec![format!("M{m}")]),
                        ),
                );
            }
            tasks.push(task);
        }
        let resources: Vec<Resource> = (0..3)
            .map(|m| Resource::new(format!("M{m}"), "M"))
            .collect();
        let checked = Problem::new(tasks, resources).expect("valid");
        let ga = SchedulingGaProblem::new(&checked);
        let dispatched = SimpleScheduler::new()
            .with_rule_engine(RuleEngine::new().with_rule(rules::Mwkr))
            .schedule(&checked, 0);
        let seed = ga.chromosome_of(&dispatched);
        assert!(seed.is_valid(&ga.activities));
        let decoded = ga.decode(&seed);
        assert!(decoded.makespan_ms() <= dispatched.makespan_ms());
        assert_eq!(
            ga.initial_individuals().len(),
            5,
            "no deadlines: five rules"
        );
    }

    /// Job A is one short activity on M1; job B runs on M2 and then on M1.
    /// With the order B, B, A the semi-active decoder leaves M1 idle until B
    /// is ready and runs A after it; the active one lets A use the idle time.
    fn idle_gap_problem() -> (SchedulingGaProblem, ScheduleChromosome) {
        let op = |id: &str, task: &str, k: i32, machine: &str, ms: i64| {
            Activity::new(id, task, k)
                .with_duration(ActivityDuration::fixed(ms))
                .with_requirement(
                    ResourceRequirement::new("M").with_candidates(vec![machine.to_string()]),
                )
        };
        let tasks = vec![
            Task::new("A").with_activity(op("A1", "A", 0, "M1", 3)),
            Task::new("B")
                .with_activity(op("B1", "B", 0, "M2", 5))
                .with_activity(op("B2", "B", 1, "M1", 3)),
        ];
        let resources = vec![Resource::new("M1", "M"), Resource::new("M2", "M")];
        let ga = SchedulingGaProblem::new(&valid(&tasks, &resources))
            .with_tardiness_weight(0.0)
            .expect("a weight in [0, 1]");
        let chromosome = ScheduleChromosome {
            osv: vec!["B".into(), "B".into(), "A".into()],
            mav: vec!["M1".into(), "M2".into(), "M1".into()],
            activity_index: ga.activity_index.clone(),
            fitness: f64::INFINITY,
        };
        (ga, chromosome)
    }

    #[test]
    fn the_active_decoder_uses_idle_time_the_order_leaves() {
        let (ga, chromosome) = idle_gap_problem();
        assert_eq!(ga.decode_semi_active(&chromosome).makespan_ms(), 11);
        assert_eq!(ga.decode_active(&chromosome).makespan_ms(), 8);
        // `decode` takes the better of the two; the search sees 8.
        assert_eq!(ga.decode(&chromosome).makespan_ms(), 8);
        assert_eq!(ga.evaluate(&chromosome), 8.0);
    }

    /// Whatever the chromosome, an active schedule is a schedule: every
    /// activity once, each resource and each task one thing at a time, a task's
    /// activities in sequence -- and `decode` is never longer than the order
    /// alone gives.
    #[test]
    fn decoded_schedules_are_valid_and_never_longer_than_the_order_gives() {
        let mut rng = SmallRng::seed_from_u64(7);
        let machines = 4;
        for instance in 0..20 {
            let tasks: Vec<Task> = (0..5)
                .map(|j| {
                    let mut order: Vec<usize> = (0..machines).collect();
                    for i in (1..order.len()).rev() {
                        order.swap(i, rng.random_range(0..=i));
                    }
                    let mut task = Task::new(format!("J{j}"));
                    for (k, m) in order.into_iter().enumerate() {
                        let ms = rng.random_range(1..20) as i64;
                        task = task.with_activity(
                            Activity::new(format!("J{j}_{k}"), format!("J{j}"), k as i32)
                                .with_duration(ActivityDuration::fixed(ms))
                                .with_requirement(
                                    ResourceRequirement::new("M")
                                        .with_candidates(vec![format!("M{m}")]),
                                ),
                        );
                    }
                    task
                })
                .collect();
            let resources: Vec<Resource> = (0..machines)
                .map(|m| Resource::new(format!("M{m}"), "M"))
                .collect();
            let ga = SchedulingGaProblem::new(&valid(&tasks, &resources));
            for _ in 0..10 {
                let chromosome = ga.create_individual(&mut rng);
                let semi = ga.decode_semi_active(&chromosome);
                let active = ga.decode_active(&chromosome);
                let best = ga.decode(&chromosome);
                assert!(
                    best.makespan_ms() <= semi.makespan_ms(),
                    "instance {instance}"
                );
                for schedule in [&semi, &active, &best] {
                    assert_eq!(schedule.assignments.len(), 5 * machines);
                    let mut by_resource: HashMap<&str, Vec<(i64, i64)>> = HashMap::new();
                    let mut by_task: HashMap<&str, Vec<(i32, i64, i64)>> = HashMap::new();
                    for a in &schedule.assignments {
                        by_resource
                            .entry(a.resource_id.as_str())
                            .or_default()
                            .push((a.start_ms, a.end_ms));
                        by_task.entry(a.task_id.as_str()).or_default().push((
                            a.sequence.unwrap_or(0),
                            a.start_ms,
                            a.end_ms,
                        ));
                    }
                    for spans in by_resource.values_mut() {
                        spans.sort();
                        assert!(spans.windows(2).all(|w| w[0].1 <= w[1].0), "overlap");
                    }
                    for spans in by_task.values_mut() {
                        spans.sort();
                        assert!(spans.windows(2).all(|w| w[0].2 <= w[1].1), "precedence");
                    }
                }
            }
        }
    }

    /// A random job shop: every job visits every machine once, in a random order.
    fn random_job_shop(
        rng: &mut SmallRng,
        jobs: usize,
        machines: usize,
    ) -> (Vec<Task>, Vec<Resource>) {
        let tasks: Vec<Task> = (0..jobs)
            .map(|j| {
                let mut order: Vec<usize> = (0..machines).collect();
                for i in (1..order.len()).rev() {
                    order.swap(i, rng.random_range(0..=i));
                }
                let mut task = Task::new(format!("J{j}"));
                for (k, m) in order.into_iter().enumerate() {
                    let ms = rng.random_range(1..20) as i64;
                    task = task.with_activity(
                        Activity::new(format!("J{j}_{k}"), format!("J{j}"), k as i32)
                            .with_duration(ActivityDuration::fixed(ms))
                            .with_requirement(
                                ResourceRequirement::new("M")
                                    .with_candidates(vec![format!("M{m}")]),
                            ),
                    );
                }
                task
            })
            .collect();
        let resources: Vec<Resource> = (0..machines)
            .map(|m| Resource::new(format!("M{m}"), "M"))
            .collect();
        (tasks, resources)
    }

    /// The taboo search continues the descent: the same chromosome improved with
    /// more steps is never longer than with fewer, since the first steps are
    /// the same and the best sequences seen are kept.
    #[test]
    fn the_local_search_never_lengthens_a_schedule_with_predecessors_in_other_tasks() {
        // Its timing used to leave them out, so a swap that looked shorter
        // decoded longer: 28 of these 200 improvements were.
        let mut rng = SmallRng::seed_from_u64(7);
        for _ in 0..40 {
            let (mut tasks, resources) = random_job_shop(&mut rng, 6, 5);
            for (j, task) in tasks.iter_mut().enumerate().skip(1) {
                let (k, p) = (rng.random_range(0..5), rng.random_range(0..5));
                task.activities[k]
                    .predecessors
                    .push(format!("J{}_{p}", j - 1));
            }
            let ga = SchedulingGaProblem::new(&valid(&tasks, &resources))
                .with_tardiness_weight(0.0)
                .expect("a weight in [0, 1]");
            for _ in 0..5 {
                let start = ga.create_individual(&mut rng);
                let mut improved = start.clone();
                ga.improve(&mut improved);
                assert!(
                    ga.decode_semi_active(&improved).makespan_ms()
                        <= ga.decode_semi_active(&start).makespan_ms()
                );
            }
        }
    }

    #[test]
    fn more_local_search_steps_never_give_a_longer_schedule() {
        let mut rng = SmallRng::seed_from_u64(23);
        for _ in 0..15 {
            let (tasks, resources) = random_job_shop(&mut rng, 6, 5);
            let few = SchedulingGaProblem::new(&valid(&tasks, &resources))
                .with_tardiness_weight(0.0)
                .expect("a weight in [0, 1]")
                .with_local_search_steps(1);
            let many = SchedulingGaProblem::new(&valid(&tasks, &resources))
                .with_tardiness_weight(0.0)
                .expect("a weight in [0, 1]")
                .with_local_search_steps(60);
            for _ in 0..5 {
                let start = few.create_individual(&mut rng);
                let (mut a, mut b) = (start.clone(), start);
                few.improve(&mut a);
                many.improve(&mut b);
                assert!(
                    many.decode_semi_active(&b).makespan_ms()
                        <= few.decode_semi_active(&a).makespan_ms()
                );
            }
        }
    }

    /// The local search never lengthens the schedule the order gives, leaves a
    /// valid schedule, and on random chromosomes of random 5 x 4 job shops
    /// shortens a good many of them -- a search that never found a swap, or one
    /// that corrupted the order, would fail one of the three.
    #[test]
    fn the_local_search_shortens_and_keeps_the_schedule_valid() {
        let mut rng = SmallRng::seed_from_u64(11);
        let machines = 4;
        let (mut shortened, mut total) = (0, 0);
        for _ in 0..20 {
            let (tasks, resources) = random_job_shop(&mut rng, 5, machines);
            let ga = SchedulingGaProblem::new(&valid(&tasks, &resources))
                .with_tardiness_weight(0.0)
                .expect("a weight in [0, 1]");
            for _ in 0..10 {
                let mut chromosome = ga.create_individual(&mut rng);
                let before = ga.decode_semi_active(&chromosome).makespan_ms();
                ga.improve(&mut chromosome);
                let schedule = ga.decode_semi_active(&chromosome);
                assert!(schedule.makespan_ms() <= before);
                assert_eq!(schedule.assignments.len(), 5 * machines);
                let mut by_resource: HashMap<&str, Vec<(i64, i64)>> = HashMap::new();
                let mut by_task: HashMap<&str, Vec<(i32, i64, i64)>> = HashMap::new();
                for a in &schedule.assignments {
                    by_resource
                        .entry(a.resource_id.as_str())
                        .or_default()
                        .push((a.start_ms, a.end_ms));
                    by_task.entry(a.task_id.as_str()).or_default().push((
                        a.sequence.unwrap_or(0),
                        a.start_ms,
                        a.end_ms,
                    ));
                }
                for spans in by_resource.values_mut() {
                    spans.sort();
                    assert!(spans.windows(2).all(|w| w[0].1 <= w[1].0), "overlap");
                }
                for spans in by_task.values_mut() {
                    spans.sort();
                    assert!(spans.windows(2).all(|w| w[0].2 <= w[1].1), "precedence");
                }
                total += 1;
                shortened += usize::from(schedule.makespan_ms() < before);
            }
        }
        assert!(
            shortened * 4 >= total,
            "only {shortened} of {total} shortened"
        );
    }

    /// With the steps at 0 the local search leaves the chromosome as it is.
    #[test]
    fn no_local_search_steps_leave_the_chromosome_alone() {
        let (tasks, resources) = make_test_problem();
        let ga = SchedulingGaProblem::new(&valid(&tasks, &resources)).with_local_search_steps(0);
        let mut rng = SmallRng::seed_from_u64(3);
        let mut chromosome = ga.create_individual(&mut rng);
        let before = chromosome.osv.clone();
        ga.improve(&mut chromosome);
        assert_eq!(chromosome.osv, before);
    }

    /// A seed reproduces the run: two GAs with the same seed on the same
    /// problem end with the same schedule.
    #[test]
    fn a_seeded_run_repeats() {
        let op = |id: String, task: &str, k: i32, machine: usize, ms: i64| {
            Activity::new(id, task, k)
                .with_duration(ActivityDuration::fixed(ms))
                .with_requirement(
                    ResourceRequirement::new("M").with_candidates(vec![format!("M{machine}")]),
                )
        };
        let tasks: Vec<Task> = (0..6)
            .map(|j| {
                (0..3).fold(Task::new(format!("J{j}")), |task, k| {
                    let machine = (j + k) % 3;
                    let ms = 1 + ((j * 7 + k * 3) % 9) as i64;
                    task.with_activity(op(
                        format!("J{j}_{k}"),
                        &format!("J{j}"),
                        k as i32,
                        machine,
                        ms,
                    ))
                })
            })
            .collect();
        let resources: Vec<Resource> = (0..3)
            .map(|m| Resource::new(format!("M{m}"), "M"))
            .collect();
        let problem = valid(&tasks, &resources);
        let run = || {
            let ga = SchedulingGaProblem::new(&problem)
                .with_tardiness_weight(0.0)
                .expect("a weight in [0, 1]");
            let config = GaConfig::default()
                .with_population_size(20)
                .with_max_generations(30)
                .with_seed(1)
                .with_parallel(false);
            let result = GaRunner::run(&ga, &config).expect("valid settings");
            let means: Vec<f64> = result
                .generation_stats
                .iter()
                .map(|g| g.mean_fitness)
                .collect();
            (ga.decode(&result.best), result.best.osv.clone(), means)
        };
        let (first, first_osv, first_means) = run();
        for _ in 0..5 {
            let (again, osv, means) = run();
            assert_eq!(again.makespan_ms(), first.makespan_ms());
            assert_eq!(osv, first_osv);
            // The whole run, not only where it ended.
            assert_eq!(means, first_means);
        }
    }

    #[test]
    fn test_fitness_computation() {
        let (tasks, resources) = make_test_problem();
        let problem = SchedulingGaProblem::new(&valid(&tasks, &resources));
        let mut rng = SmallRng::seed_from_u64(42);
        let ch = problem.create_individual(&mut rng);

        let fitness = problem.evaluate(&ch);
        assert!(fitness.is_finite());
        assert!(fitness > 0.0);
    }

    #[test]
    fn test_ga_runner_integration() {
        let (tasks, resources) = make_test_problem();
        let problem = SchedulingGaProblem::new(&valid(&tasks, &resources));
        let config = GaConfig::default()
            .with_population_size(20)
            .with_max_generations(10)
            .with_seed(42)
            .with_parallel(false);

        let result = GaRunner::run(&problem, &config).expect("GA should succeed");
        assert!(result.best_fitness.is_finite());
        assert!(result.best_fitness < f64::INFINITY);
        assert!(result.generations > 0);
    }

    #[test]
    fn test_crossover_and_mutation() {
        let (tasks, resources) = make_test_problem();
        let problem = SchedulingGaProblem::new(&valid(&tasks, &resources));
        let mut rng = SmallRng::seed_from_u64(42);

        let p1 = problem.create_individual(&mut rng);
        let p2 = problem.create_individual(&mut rng);

        let children = problem.crossover(&p1, &p2, &mut rng);
        assert_eq!(children.len(), 2);

        let mut child = children[0].clone();
        problem.mutate(&mut child, &mut rng);
        assert_eq!(child.osv.len(), p1.osv.len());
    }

    #[test]
    fn a_tardiness_weight_outside_0_1_is_refused_not_clamped() {
        let (tasks, resources) = make_test_problem();
        for weight in [1.5, -0.1, f64::NAN] {
            let err = SchedulingGaProblem::new(&valid(&tasks, &resources))
                .with_tardiness_weight(weight)
                .err()
                .expect("out of range");
            assert_eq!(err.kind.code(), "parameter_out_of_range");
        }
    }

    #[test]
    fn test_tardiness_weight() {
        let (tasks, resources) = make_test_problem();
        let problem_makespan = SchedulingGaProblem::new(&valid(&tasks, &resources))
            .with_tardiness_weight(0.0)
            .expect("weight in [0, 1]");
        let problem_tardy = SchedulingGaProblem::new(&valid(&tasks, &resources))
            .with_tardiness_weight(1.0)
            .expect("weight in [0, 1]");

        let mut rng = SmallRng::seed_from_u64(42);
        let ch = problem_makespan.create_individual(&mut rng);

        let f1 = problem_makespan.evaluate(&ch);
        let f2 = problem_tardy.evaluate(&ch);
        // Different weights should give different fitness
        assert!(f1 != f2 || (f1 == 0.0 && f2 == 0.0));
    }

    #[test]
    fn test_spt_initialization() {
        let (mut tasks, resources) = make_test_problem();
        tasks[0].activities[0] = tasks[0].activities[0].clone().with_speed_on("M1", 2.0);
        tasks[1].activities[0] = tasks[1].activities[0]
            .clone()
            .with_duration_on("M3", ActivityDuration::fixed(800));

        let problem = SchedulingGaProblem::new(&valid(&tasks, &resources));

        // Generate many individuals to exercise all 3 init strategies
        let mut rng = SmallRng::seed_from_u64(42);
        for _ in 0..100 {
            let ch = problem.create_individual(&mut rng);
            assert_eq!(ch.osv.len(), 3);
            assert_eq!(ch.mav.len(), 3);
        }
    }

    #[test]
    fn test_with_operators_lox_invert() {
        let (tasks, resources) = make_test_problem();
        let ops = GeneticOperators {
            crossover_type: CrossoverType::LOX,
            mutation_type: MutationType::Invert,
        };
        let problem = SchedulingGaProblem::new(&valid(&tasks, &resources)).with_operators(ops);
        let config = GaConfig::default()
            .with_population_size(20)
            .with_max_generations(10)
            .with_seed(42)
            .with_parallel(false);

        let result = GaRunner::run(&problem, &config).expect("GA should succeed");
        assert!(result.best_fitness.is_finite());
        assert!(result.best_fitness < f64::INFINITY);
    }

    #[test]
    fn test_with_operators_jox_insert() {
        let (tasks, resources) = make_test_problem();
        let ops = GeneticOperators {
            crossover_type: CrossoverType::JOX,
            mutation_type: MutationType::Insert,
        };
        let problem = SchedulingGaProblem::new(&valid(&tasks, &resources)).with_operators(ops);
        let config = GaConfig::default()
            .with_population_size(20)
            .with_max_generations(10)
            .with_seed(99)
            .with_parallel(false);

        let result = GaRunner::run(&problem, &config).expect("GA should succeed");
        assert!(result.best_fitness.is_finite());
        assert!(result.best_fitness < f64::INFINITY);
    }

    #[test]
    fn test_default_operators_backward_compatible() {
        // Default operators should be POX + Swap (same as original hardcoded behavior)
        let (tasks, resources) = make_test_problem();
        let problem = SchedulingGaProblem::new(&valid(&tasks, &resources));
        assert_eq!(problem.operators.crossover_type, CrossoverType::POX);
        assert_eq!(problem.operators.mutation_type, MutationType::Swap);
    }

    #[test]
    fn test_ga_schedules_with_the_time_on_each_resource() {
        let (mut tasks, resources) = make_test_problem();
        // T1's first activity runs twice as fast on M1, T2's takes 800 ms on M3.
        tasks[0].activities[0] = tasks[0].activities[0].clone().with_speed_on("M1", 2.0);
        tasks[1].activities[0] = tasks[1].activities[0]
            .clone()
            .with_duration_on("M3", ActivityDuration::fixed(800));
        let problem = SchedulingGaProblem::new(&valid(&tasks, &resources));
        let config = GaConfig::default()
            .with_population_size(20)
            .with_max_generations(10)
            .with_seed(42)
            .with_parallel(false);

        let result = GaRunner::run(&problem, &config).expect("GA should succeed");
        let schedule = problem.decode(&result.best);

        assert!(schedule.violations.is_empty(), "{:?}", schedule.violations);
        for a in &schedule.assignments {
            let activity = tasks
                .iter()
                .flat_map(|t| &t.activities)
                .find(|x| x.id == a.activity_id)
                .expect("assigned activity exists");
            let want = activity.duration_on([a.resource_id.as_str()]);
            assert_eq!(
                a.end_ms - a.start_ms,
                a.setup_ms + want.process_ms + want.teardown_ms,
                "{} on {}",
                a.activity_id,
                a.resource_id
            );
        }
    }

    #[test]
    fn both_decoders_and_the_local_search_keep_an_overlap_s_two_bounds() {
        use crate::models::Overlap;
        // J1 cuts on A (1 s), then bends on B (`bend` ms) overlapping the cut;
        // moving A -> B takes 100 ms.
        let step = |id: &str, seq: i32, ms: i64, machine: &str| {
            Activity::new(id, "J1", seq)
                .with_process_time(ms)
                .with_requirement(
                    ResourceRequirement::new("Machine").with_candidates(vec![machine.into()]),
                )
        };
        let resources = vec![
            Resource::new("A", "Machine").with_transport_to("B", 100),
            Resource::new("B", "Machine"),
        ];
        for (bend, overlap, expected) in [
            // Starts at 0.3 of the cut, plus the move.
            (1_000, Overlap::new(0.3), (400, 1_400)),
            // Short: held back to end after the cut (and the move).
            (200, Overlap::new(0.3), (900, 1_100)),
            (200, Overlap::new(0.0).with_finish_after(50), (950, 1_150)),
        ] {
            let tasks = vec![Task::new("J1")
                .with_activity(step("cut", 0, 1_000, "A"))
                .with_activity(step("bend", 1, bend, "B").with_overlap(overlap))];
            let problem = valid(&tasks, &resources);
            let ga = SchedulingGaProblem::new(&problem);
            let mut chromosome = ScheduleChromosome {
                osv: vec!["J1".into(), "J1".into()],
                mav: vec!["A".into(), "B".into()],
                activity_index: ga.activity_index.clone(),
                fitness: f64::INFINITY,
            };
            ga.improve(&mut chromosome);
            for schedule in [
                ga.decode_semi_active(&chromosome),
                ga.decode_active(&chromosome),
            ] {
                let a = schedule.assignment_for_activity("bend").expect("placed");
                assert_eq!((a.start_ms, a.end_ms), expected, "bend {bend}");
                let input = crate::scheduler::FeasibilityInput {
                    tasks: problem.tasks(),
                    resources: problem.resources(),
                    constraints: &[],
                };
                let violations = crate::scheduler::check_schedule(&schedule, &input);
                assert!(violations.is_empty(), "{violations:?}");
            }
        }
    }

    #[test]
    fn both_decoders_wait_for_the_work_to_arrive() {
        // J1 runs A then B; moving A -> B takes 5 s.
        let on = |id: &str, task: &str, seq: i32, machine: &str| {
            Activity::new(id, task, seq)
                .with_process_time(1_000)
                .with_requirement(
                    ResourceRequirement::new("Machine").with_candidates(vec![machine.into()]),
                )
        };
        let tasks = vec![
            Task::new("J1")
                .with_activity(on("J1-1", "J1", 0, "A"))
                .with_activity(on("J1-2", "J1", 1, "B")),
            Task::new("J2").with_activity(on("J2-1", "J2", 0, "B")),
        ];
        let resources = vec![
            Resource::new("A", "Machine").with_transport_to("B", 5_000),
            Resource::new("B", "Machine"),
        ];
        let problem = valid(&tasks, &resources);
        let ga = SchedulingGaProblem::new(&problem);
        let chromosome = ScheduleChromosome {
            osv: vec!["J1".into(), "J1".into(), "J2".into()],
            mav: vec!["A".into(), "B".into(), "B".into()],
            activity_index: ga.activity_index.clone(),
            fitness: f64::INFINITY,
        };
        for schedule in [
            ga.decode_semi_active(&chromosome),
            ga.decode_active(&chromosome),
        ] {
            let start = schedule
                .assignment_for_activity("J1-2")
                .expect("placed")
                .start_ms;
            assert!(start >= 6_000, "J1-2 starts at {start}");
            let input = crate::scheduler::FeasibilityInput {
                tasks: problem.tasks(),
                resources: problem.resources(),
                constraints: &[],
            };
            let violations = crate::scheduler::check_schedule(&schedule, &input);
            assert!(violations.is_empty(), "{violations:?}");
        }
    }

    #[test]
    fn both_decoders_wait_for_predecessors_in_other_tasks() {
        // J2-1 waits for J1-2, though the order puts J2 first and its machine
        // is free from the start.
        let on = |id: &str, task: &str, seq: i32, machine: &str| {
            Activity::new(id, task, seq)
                .with_process_time(1_000)
                .with_requirement(
                    ResourceRequirement::new("Machine").with_candidates(vec![machine.into()]),
                )
        };
        let tasks = vec![
            Task::new("J1")
                .with_activity(on("J1-1", "J1", 0, "A"))
                .with_activity(on("J1-2", "J1", 1, "A")),
            Task::new("J2").with_activity(on("J2-1", "J2", 0, "B").with_predecessor("J1-2")),
        ];
        let resources = vec![Resource::new("A", "Machine"), Resource::new("B", "Machine")];
        let problem = valid(&tasks, &resources);
        let ga = SchedulingGaProblem::new(&problem);
        let chromosome = ScheduleChromosome {
            osv: vec!["J2".into(), "J1".into(), "J1".into()],
            mav: vec!["A".into(), "A".into(), "B".into()],
            activity_index: ga.activity_index.clone(),
            fitness: f64::INFINITY,
        };
        for schedule in [
            ga.decode_semi_active(&chromosome),
            ga.decode_active(&chromosome),
            ga.decode(&chromosome),
        ] {
            let start = schedule
                .assignment_for_activity("J2-1")
                .expect("placed")
                .start_ms;
            assert_eq!(start, 2_000);
            let input = crate::scheduler::FeasibilityInput {
                tasks: problem.tasks(),
                resources: problem.resources(),
                constraints: &[],
            };
            let violations = crate::scheduler::check_schedule(&schedule, &input);
            assert!(violations.is_empty(), "{violations:?}");
        }
    }

    #[test]
    fn the_local_search_keeps_the_transport_times() {
        let mut rng = SmallRng::seed_from_u64(7);
        let (tasks, mut resources) = random_job_shop(&mut rng, 6, 4);
        let ids: Vec<String> = resources.iter().map(|r| r.id.clone()).collect();
        for r in &mut resources {
            for to in ids.iter().filter(|t| **t != r.id) {
                r.transport_ms.insert(to.clone(), 700);
            }
        }
        let problem = valid(&tasks, &resources);
        let ga = SchedulingGaProblem::new(&problem).with_local_search_steps(40);
        let config = GaConfig::default()
            .with_population_size(20)
            .with_max_generations(10)
            .with_seed(3);
        let result = GaRunner::run(&ga, &config).expect("valid settings");
        let schedule = ga.decode(&result.best);
        let input = crate::scheduler::FeasibilityInput {
            tasks: problem.tasks(),
            resources: problem.resources(),
            constraints: &[],
        };
        let violations = crate::scheduler::check_schedule(&schedule, &input);
        assert!(violations.is_empty(), "{violations:?}");
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
    fn both_decoders_keep_an_activity_release() {
        let problem = valid(
            &[released_second()],
            &[Resource::new("A", "Machine"), Resource::new("B", "Machine")],
        );
        let ga = SchedulingGaProblem::new(&problem);
        let chromosome = ScheduleChromosome {
            osv: vec!["J1".into(), "J1".into()],
            mav: vec!["A".into(), "B".into()],
            activity_index: ga.activity_index.clone(),
            fitness: f64::INFINITY,
        };
        for schedule in [
            ga.decode_semi_active(&chromosome),
            ga.decode_active(&chromosome),
        ] {
            let start = schedule
                .assignment_for_activity("J1-2")
                .expect("placed")
                .start_ms;
            assert_eq!(start, 5_000);
        }
    }
}
