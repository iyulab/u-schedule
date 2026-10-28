//! CP-based scheduling formulation.
//!
//! Bridges scheduling domain models to `u-metaheur`'s CP framework.
//! Builds a `CpModel` from tasks, resources, and constraints, then
//! solves it using a `CpSolver`.
//!
//! # Reference
//! - Laborie et al. (2018), "IBM ILOG CP Optimizer for Scheduling"
//! - Baptiste et al. (2001), "Constraint-Based Scheduling"

use std::collections::HashMap;

use u_metaheur::cp::{
    Constraint as CpConstraint, CpModel, CpSolution, CpSolver, IntervalVar, Objective, SolverConfig,
};

use crate::models::{
    transport_between, Assignment, Constraint, Schedule, TransitionMatrixCollection,
};
use crate::Problem;

/// Builds a CP model from scheduling domain objects.
///
/// Translates tasks, resources, and constraints into a CpModel
/// suitable for solving with any `CpSolver` implementation.
///
/// `u_metaheur::cp::BranchAndBoundSolver` solves the model exactly when it is
/// a disjunctive one: no step [overlaps](crate::models::Overlap) the one
/// before (a negative delay), and no user constraint adds a second
/// `NoOverlap` over an activity or a constraint of another kind. Otherwise
/// that solver reports the model not searched (`SolverStatus::Unknown`).
///
/// # Example
/// ```
/// use u_metaheur::cp::{BranchAndBoundSolver, SolverConfig, SolverStatus};
/// use u_schedule::cp::ScheduleCpBuilder;
/// use u_schedule::models::{Activity, Resource, ResourceRequirement, Task};
/// use u_schedule::Problem;
///
/// let step = |id: &str, task: &str, seq: i32, ms: i64, machine: &str| {
///     Activity::new(id, task, seq)
///         .with_process_time(ms)
///         .with_requirement(
///             ResourceRequirement::new("Machine").with_candidates(vec![machine.into()]),
///         )
/// };
/// let tasks = vec![
///     Task::new("A")
///         .with_activity(step("A1", "A", 0, 3, "M1"))
///         .with_activity(step("A2", "A", 1, 2, "M2")),
///     Task::new("B")
///         .with_activity(step("B1", "B", 0, 2, "M2"))
///         .with_activity(step("B2", "B", 1, 4, "M1")),
/// ];
/// let resources = vec![Resource::new("M1", "Machine"), Resource::new("M2", "Machine")];
/// let problem = Problem::new(tasks, resources).expect("valid input");
///
/// let (schedule, solution) = ScheduleCpBuilder::new(&problem).solve(
///     &BranchAndBoundSolver::new(),
///     &SolverConfig::default(),
///     1_000,
/// );
/// assert_eq!(solution.status, SolverStatus::Optimal);
/// assert_eq!(schedule.makespan_ms(), 7);
/// ```
pub struct ScheduleCpBuilder<'a> {
    problem: &'a Problem,
    constraints: Vec<Constraint>,
    transition_matrices: TransitionMatrixCollection,
}

impl<'a> ScheduleCpBuilder<'a> {
    /// Creates a new CP builder.
    pub fn new(problem: &'a Problem) -> Self {
        Self {
            problem,
            constraints: Vec::new(),
            transition_matrices: TransitionMatrixCollection::new(),
        }
    }

    /// Adds scheduling constraints.
    pub fn with_constraints(mut self, constraints: Vec<Constraint>) -> Self {
        self.constraints = constraints;
        self
    }

    /// Sets transition matrices.
    pub fn with_transition_matrices(mut self, matrices: TransitionMatrixCollection) -> Self {
        self.transition_matrices = matrices;
        self
    }

    /// Builds a CP model with the given planning horizon.
    ///
    /// Creates:
    /// - An `IntervalVar` per activity
    /// - For an activity that [chooses its resource](Self::choices), one
    ///   optional interval per candidate, for the activity's time on it, and
    ///   an `Alternative` over them: the solution performs it on one
    /// - `NoOverlap` constraints per resource, over the activities on it --
    ///   the optional intervals of those choosing, and each other activity on
    ///   its first eligible resource
    /// - `Precedence` constraints for intra-task activity ordering and for
    ///   each activity's `predecessors` in other tasks — a step that
    ///   [overlaps](crate::models::Overlap) the one before by a delay that may
    ///   be negative (both bounds, exact since every span here is fixed)
    /// - User-defined constraints
    /// - `MinimizeMaxEnd` objective (makespan minimization)
    pub fn build(&self, horizon_ms: i64) -> CpModel {
        let mut model = CpModel::new("scheduling", horizon_ms);
        let choices = self.choices();

        // Create interval variables for each activity
        for task in self.problem.tasks() {
            let release = task.release_time.unwrap_or(0);

            for activity in &task.activities {
                let release = activity.release_time.map_or(release, |r| release.max(r));
                // Setup, processing and teardown: the activity's whole span,
                // on each resource it may be performed on.
                let interval = |name: &str, duration: i64| {
                    IntervalVar::new(name, release, horizon_ms - duration, duration, horizon_ms)
                };
                match choices.get(activity.id.as_str()) {
                    None => model.add_interval(interval(
                        &activity.id,
                        self.duration_of(activity).total_ms(),
                    )),
                    Some(candidates) => {
                        let spans: Vec<i64> = candidates
                            .iter()
                            .map(|r| activity.duration_on([*r]).total_ms())
                            .collect();
                        let (shortest, longest) = (
                            *spans.iter().min().expect("a choice has candidates"),
                            *spans.iter().max().expect("a choice has candidates"),
                        );
                        model.add_interval(
                            interval(&activity.id, shortest)
                                .with_variable_duration(shortest, longest),
                        );
                        let mut alternatives = Vec::with_capacity(candidates.len());
                        for (r, span) in candidates.iter().zip(spans) {
                            let name = Self::alternative(&activity.id, r);
                            model.add_interval(
                                interval(&name, span).as_optional(format!("{name}?")),
                            );
                            alternatives.push(name);
                        }
                        model.add_constraint(CpConstraint::Alternative {
                            main: activity.id.clone(),
                            alternatives,
                        });
                    }
                }
            }

            // Intra-task precedence: activity[i] before activity[i+1], with
            // the time the work moves between the resources they are
            // reported on.
            for pair in task.activities.windows(2) {
                // The same for every choice: an activity chooses only then.
                let transport = transport_between(
                    |id| self.problem.resources().iter().find(|r| r.id == id),
                    self.performed_on(&pair[0], &choices),
                    &self.performed_on(&pair[1], &choices),
                );
                let delay = match &pair[1].overlap {
                    None => transport,
                    // start(b) >= start(a) + share of a's span, and end(b) >=
                    // end(a) + lag: both as end(a) + delay <= start(b).
                    Some(overlap) => {
                        let (before, after) = (
                            self.duration_of(&pair[0]).total_ms(),
                            self.duration_of(&pair[1]).total_ms(),
                        );
                        (overlap.earliest_start((0, before)) - before)
                            .max(overlap.finish_after_ms - after)
                            + transport
                    }
                };
                model.add_precedence(pair[0].id.clone(), pair[1].id.clone(), delay);
            }
        }

        // Predecessors in any task: each ends before the activity starts.
        for task in self.problem.tasks() {
            for activity in &task.activities {
                for predecessor in &activity.predecessors {
                    model.add_precedence(predecessor.clone(), activity.id.clone(), 0);
                }
            }
        }

        // No-overlap constraints per resource
        let resource_activities = self.collect_resource_activities(&choices);
        for activity_ids in resource_activities.values() {
            if activity_ids.len() > 1 {
                model.add_no_overlap(activity_ids.clone());
            }
        }

        // User-defined constraints
        for constraint in &self.constraints {
            match constraint {
                Constraint::Precedence {
                    before,
                    after,
                    min_delay_ms,
                } => {
                    model.add_precedence(before.clone(), after.clone(), *min_delay_ms);
                }
                Constraint::NoOverlap {
                    resource_id: _,
                    activity_ids,
                } => {
                    model.add_no_overlap(activity_ids.clone());
                }
                Constraint::Capacity {
                    resource_id: _,
                    max_capacity,
                } => {
                    // Cumulative constraint — would need interval→demand mapping
                    // Simplified: skip (handled by no-overlap for capacity=1)
                    let _ = max_capacity;
                }
                _ => {
                    // TimeWindow, TransitionCost, Synchronize — advanced constraints
                    // Not yet supported by the simple CP formulation
                }
            }
        }

        // Objective: minimize makespan
        model.set_objective(Objective::MinimizeMaxEnd);

        model
    }

    /// Solves the scheduling problem and returns a Schedule.
    pub fn solve<S: CpSolver>(
        &self,
        solver: &S,
        config: &SolverConfig,
        horizon_ms: i64,
    ) -> (Schedule, CpSolution) {
        let model = self.build(horizon_ms);
        let solution = solver.solve(&model, config);

        let schedule = self.decode_solution(&solution);
        (schedule, solution)
    }

    /// A schedule of the problem as a solution of the model [`build`](Self::build)
    /// makes -- to start a solver's search from (a heuristic's schedule as
    /// `u_metaheur::cp::BranchAndBoundSolver::with_incumbent`). Each activity
    /// is taken from its assignment; one that [chooses](Self::choices) its
    /// resource, from its assignment on a candidate, the alternative for that
    /// candidate present and the others absent. An activity without an
    /// assignment is left out, and the solver does not use a solution that
    /// leaves one out or breaks the model.
    pub fn incumbent(&self, schedule: &Schedule) -> CpSolution {
        use u_metaheur::cp::{IntervalSolution, SolverStatus};
        let choices = self.choices();
        let mut solution = CpSolution::empty(SolverStatus::Feasible);
        for task in self.problem.tasks() {
            for activity in &task.activities {
                let candidates = choices.get(activity.id.as_str());
                let Some(assignment) = schedule.assignments.iter().find(|a| {
                    a.activity_id == activity.id
                        && candidates.is_none_or(|c| c.contains(&a.resource_id.as_str()))
                }) else {
                    continue;
                };
                let interval = IntervalSolution {
                    start: assignment.start_ms,
                    end: assignment.end_ms,
                    duration: assignment.end_ms - assignment.start_ms,
                    is_present: true,
                };
                for r in candidates.into_iter().flatten() {
                    solution.intervals.insert(
                        Self::alternative(&activity.id, r),
                        IntervalSolution {
                            is_present: *r == assignment.resource_id,
                            ..interval.clone()
                        },
                    );
                }
                solution.intervals.insert(activity.id.clone(), interval);
            }
        }
        solution
    }

    /// Decodes a CP solution into a Schedule.
    fn decode_solution(&self, solution: &CpSolution) -> Schedule {
        let mut schedule = Schedule::new();

        if !solution.is_solution_found() {
            return schedule;
        }
        let choices = self.choices();

        for task in self.problem.tasks() {
            for (step, activity) in task.activities.iter().enumerate() {
                if let Some(interval_sol) = solution.intervals.get(&activity.id) {
                    if interval_sol.is_present {
                        // The candidate whose interval is present, or the
                        // resource it is reported on.
                        let resource_id = match choices.get(activity.id.as_str()) {
                            Some(candidates) => candidates
                                .iter()
                                .find(|r| {
                                    solution
                                        .intervals
                                        .get(&Self::alternative(&activity.id, r))
                                        .is_some_and(|s| s.is_present)
                                })
                                .copied(),
                            None => self.reported_on(activity),
                        }
                        .unwrap_or_default()
                        .to_string();

                        schedule.add_assignment(
                            Assignment::new(
                                &activity.id,
                                &task.id,
                                &resource_id,
                                interval_sol.start,
                                interval_sol.end,
                            )
                            .with_sequence(step as i32 + 1)
                            .with_setup(activity.duration_on([resource_id.as_str()]).setup_ms),
                        );
                    }
                }
            }
        }

        schedule
    }

    /// The resources each activity chooses among, by activity id: an
    /// activity with one requirement that several resources fill, unless
    /// what lies between it and its neighbours would depend on the choice --
    /// an [overlap](crate::models::Overlap) with the step before or after
    /// (its delay follows the spans) or a transport time that differs between
    /// candidates -- or a user `NoOverlap` names it. Every other activity is
    /// performed on its first eligible resource.
    pub fn choices(&self) -> HashMap<&'a str, Vec<&'a str>> {
        let resources = self.problem.resources();
        let named: std::collections::HashSet<&str> = self
            .constraints
            .iter()
            .flat_map(|c| match c {
                Constraint::NoOverlap { activity_ids, .. } => activity_ids.as_slice(),
                _ => &[],
            })
            .map(String::as_str)
            .collect();
        let mut choices: HashMap<&'a str, Vec<&'a str>> = HashMap::new();
        for task in self.problem.tasks() {
            for activity in &task.activities {
                let eligible = activity.eligible_resources(resources);
                if activity.resource_requirements.len() == 1
                    && eligible.len() > 1
                    && !named.contains(activity.id.as_str())
                {
                    choices.insert(activity.id.as_str(), eligible);
                }
            }
            for pair in task.activities.windows(2) {
                let (from, to) = (
                    pair[0].eligible_resources(resources),
                    pair[1].eligible_resources(resources),
                );
                let lookup = |id: &str| resources.iter().find(|r| r.id == id);
                let mut moves = from
                    .iter()
                    .flat_map(|f| to.iter().map(move |t| (*f, *t)))
                    .map(|(f, t)| transport_between(lookup, [f], &[t]));
                let first = moves.next();
                if pair[1].overlap.is_some() || moves.any(|m| Some(m) != first) {
                    choices.remove(pair[0].id.as_str());
                    choices.remove(pair[1].id.as_str());
                }
            }
        }
        choices
    }

    /// The name of the optional interval for performing `activity` on
    /// `resource`.
    fn alternative(activity: &str, resource: &str) -> String {
        format!("{activity}@{resource}")
    }

    /// The resources `activity` may be performed on in the model.
    fn performed_on(
        &self,
        activity: &crate::models::Activity,
        choices: &HashMap<&'a str, Vec<&'a str>>,
    ) -> Vec<&'a str> {
        match choices.get(activity.id.as_str()) {
            Some(candidates) => candidates.clone(),
            None => self.reported_on(activity).into_iter().collect(),
        }
    }

    /// The resource a solution reports an activity that does not
    /// [choose](Self::choices) on -- its first eligible one.
    fn reported_on(&self, activity: &crate::models::Activity) -> Option<&'a str> {
        activity
            .eligible_resources(self.problem.resources())
            .first()
            .copied()
    }

    /// The activity's time on the resource a solution reports it on
    /// ([`reported_on`](Self::reported_on)).
    fn duration_of(&self, activity: &crate::models::Activity) -> crate::models::ActivityDuration {
        activity.duration_on(self.reported_on(activity))
    }

    /// Collects interval names per resource: each candidate's optional
    /// interval of an activity that [chooses](Self::choices), and every other
    /// activity on the resource it is reported on
    /// ([`reported_on`](Self::reported_on)) -- the one its duration is taken
    /// on, and the only one it occupies.
    fn collect_resource_activities(
        &self,
        choices: &HashMap<&'a str, Vec<&'a str>>,
    ) -> HashMap<String, Vec<String>> {
        let mut map: HashMap<String, Vec<String>> = HashMap::new();

        for task in self.problem.tasks() {
            for activity in &task.activities {
                match choices.get(activity.id.as_str()) {
                    Some(candidates) => {
                        for r in candidates {
                            map.entry(r.to_string())
                                .or_default()
                                .push(Self::alternative(&activity.id, r));
                        }
                    }
                    None => {
                        if let Some(resource) = self.reported_on(activity) {
                            map.entry(resource.to_string())
                                .or_default()
                                .push(activity.id.clone());
                        }
                    }
                }
            }
        }

        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Activity, ActivityDuration, Resource, ResourceRequirement, Task};
    use u_metaheur::cp::SimpleCpSolver;

    fn make_test_data() -> Problem {
        let tasks = vec![
            Task::new("T1")
                .with_activity(
                    Activity::new("T1_O1", "T1", 0)
                        .with_duration(ActivityDuration::fixed(1000))
                        .with_requirement(
                            ResourceRequirement::new("Machine").with_candidates(vec!["M1".into()]),
                        ),
                )
                .with_activity(
                    Activity::new("T1_O2", "T1", 1)
                        .with_duration(ActivityDuration::fixed(2000))
                        .with_requirement(
                            ResourceRequirement::new("Machine").with_candidates(vec!["M1".into()]),
                        ),
                ),
            Task::new("T2").with_activity(
                Activity::new("T2_O1", "T2", 0)
                    .with_duration(ActivityDuration::fixed(1500))
                    .with_requirement(
                        ResourceRequirement::new("Machine").with_candidates(vec!["M1".into()]),
                    ),
            ),
        ];

        let resources = vec![Resource::new("M1", "Machine")];
        Problem::new(tasks, resources).expect("a valid problem")
    }

    #[test]
    fn test_build_model() {
        let problem = make_test_data();
        let builder = ScheduleCpBuilder::new(&problem);
        let model = builder.build(100_000);

        // 3 intervals (T1_O1, T1_O2, T2_O1)
        assert_eq!(model.interval_count(), 3);
        // Constraints: 1 precedence (T1_O1→T1_O2) + 1 no-overlap (M1)
        assert!(model.constraint_count() >= 2);
    }

    #[test]
    fn test_build_with_constraints() {
        let problem = make_test_data();
        let constraints = vec![Constraint::precedence("T1_O2", "T2_O1")];
        let builder = ScheduleCpBuilder::new(&problem).with_constraints(constraints);
        let model = builder.build(100_000);

        // Additional precedence constraint
        assert!(model.constraint_count() >= 3);
    }

    #[test]
    fn test_solve_basic() {
        let problem = make_test_data();
        let builder = ScheduleCpBuilder::new(&problem);
        let solver = SimpleCpSolver::new();
        let config = SolverConfig::default();

        let (schedule, solution) = builder.solve(&solver, &config, 100_000);
        assert!(solution.is_solution_found());
        assert!(schedule.assignment_count() > 0);
        assert!(schedule.makespan_ms() > 0);
    }

    #[test]
    fn test_intra_task_precedence() {
        let problem = make_test_data();
        let builder = ScheduleCpBuilder::new(&problem);
        let solver = SimpleCpSolver::new();
        let config = SolverConfig::default();

        let (schedule, _) = builder.solve(&solver, &config, 100_000);

        // T1_O1 must finish before T1_O2 starts
        if let (Some(o1), Some(o2)) = (
            schedule.assignment_for_activity("T1_O1"),
            schedule.assignment_for_activity("T1_O2"),
        ) {
            assert!(o1.end_ms <= o2.start_ms);
        }
    }

    #[test]
    fn test_no_overlap() {
        let problem = make_test_data();
        let builder = ScheduleCpBuilder::new(&problem);
        let solver = SimpleCpSolver::new();
        let config = SolverConfig::default();

        let (schedule, _) = builder.solve(&solver, &config, 100_000);

        // All activities on M1 should not overlap
        let m1_assignments = schedule.assignments_for_resource("M1");
        for i in 0..m1_assignments.len() {
            for j in (i + 1)..m1_assignments.len() {
                let a = m1_assignments[i];
                let b = m1_assignments[j];
                // No overlap: a ends before b starts OR b ends before a starts
                assert!(
                    a.end_ms <= b.start_ms || b.end_ms <= a.start_ms,
                    "Overlap detected: {} [{}, {}] and {} [{}, {}]",
                    a.activity_id,
                    a.start_ms,
                    a.end_ms,
                    b.activity_id,
                    b.start_ms,
                    b.end_ms,
                );
            }
        }
    }

    #[test]
    fn the_next_activity_waits_for_the_work_to_arrive() {
        let on = |id: &str, seq: i32, machine: &str| {
            Activity::new(id, "J", seq)
                .with_process_time(1_000)
                .with_requirement(
                    ResourceRequirement::new("Machine").with_candidates(vec![machine.into()]),
                )
        };
        let problem = crate::Problem::new(
            vec![Task::new("J")
                .with_activity(on("J-1", 0, "A"))
                .with_activity(on("J-2", 1, "B"))],
            vec![
                Resource::new("A", "Machine").with_transport_to("B", 5_000),
                Resource::new("B", "Machine"),
            ],
        )
        .expect("valid");
        let (schedule, _) = ScheduleCpBuilder::new(&problem).solve(
            &SimpleCpSolver::new(),
            &SolverConfig::default(),
            100_000,
        );
        let first = schedule.assignment_for_activity("J-1").expect("placed");
        let next = schedule.assignment_for_activity("J-2").expect("placed");
        assert!(next.start_ms >= first.end_ms + 5_000);
    }

    #[test]
    fn an_overlapping_step_keeps_both_bounds() {
        use crate::models::Overlap;
        let on = |id: &str, seq: i32, ms: i64, machine: &str| {
            Activity::new(id, "J", seq)
                .with_process_time(ms)
                .with_requirement(
                    ResourceRequirement::new("Machine").with_candidates(vec![machine.into()]),
                )
        };
        for (bend, overlap, expected) in [
            (1_000, Overlap::new(0.3), (400, 1_400)),
            (200, Overlap::new(0.3), (900, 1_100)),
            (200, Overlap::new(0.0).with_finish_after(50), (950, 1_150)),
        ] {
            let problem = crate::Problem::new(
                vec![Task::new("J")
                    .with_activity(on("cut", 0, 1_000, "A"))
                    .with_activity(on("bend", 1, bend, "B").with_overlap(overlap))],
                vec![
                    Resource::new("A", "Machine").with_transport_to("B", 100),
                    Resource::new("B", "Machine"),
                ],
            )
            .expect("valid");
            let (schedule, _) = ScheduleCpBuilder::new(&problem).solve(
                &SimpleCpSolver::new(),
                &SolverConfig::default(),
                100_000,
            );
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

    #[test]
    fn a_predecessor_in_another_task_ends_before_the_activity_starts() {
        let on = |id: &str, task: &str, machine: &str| {
            Activity::new(id, task, 0)
                .with_process_time(1_000)
                .with_requirement(
                    ResourceRequirement::new("Machine").with_candidates(vec![machine.into()]),
                )
        };
        let problem = crate::Problem::new(
            vec![
                Task::new("J1").with_activity(on("J1-1", "J1", "A")),
                Task::new("J2").with_activity(on("J2-1", "J2", "B").with_predecessor("J1-1")),
            ],
            vec![Resource::new("A", "Machine"), Resource::new("B", "Machine")],
        )
        .expect("valid");
        let (schedule, _) = ScheduleCpBuilder::new(&problem).solve(
            &SimpleCpSolver::new(),
            &SolverConfig::default(),
            100_000,
        );
        let at = |id: &str| schedule.assignment_for_activity(id).expect("placed");
        assert!(at("J2-1").start_ms >= at("J1-1").end_ms);
    }

    #[test]
    fn an_activity_release_bounds_its_start() {
        let problem = crate::Problem::new(
            vec![Task::new("J").with_activity(
                Activity::new("J-1", "J", 0)
                    .with_process_time(1_000)
                    .with_release_time(4_000)
                    .with_requirement(
                        ResourceRequirement::new("Machine").with_candidates(vec!["A".into()]),
                    ),
            )],
            vec![Resource::new("A", "Machine")],
        )
        .expect("valid");
        let (schedule, _) = ScheduleCpBuilder::new(&problem).solve(
            &SimpleCpSolver::new(),
            &SolverConfig::default(),
            100_000,
        );
        assert!(
            schedule
                .assignment_for_activity("J-1")
                .expect("placed")
                .start_ms
                >= 4_000
        );
    }

    #[test]
    fn an_activity_chooses_the_resource_that_finishes_first() {
        // "flex" may run on M1 or M2, "other" needs M1 alone: on M2 the two
        // run together. "slow" runs 4 ms on M3 or 12 ms on M4 (2 / 6 per ms
        // of its 2 ms base): it takes M3.
        let one_step = |task: &str, id: &str, candidates: &[&str]| {
            Task::new(task).with_activity(
                Activity::new(id, task, 0)
                    .with_process_time(10)
                    .with_requirement(
                        ResourceRequirement::new("Machine")
                            .with_candidates(candidates.iter().map(|c| c.to_string()).collect()),
                    ),
            )
        };
        let tasks = vec![
            one_step("F", "flex", &["M1", "M2"]),
            one_step("O", "other", &["M1"]),
            Task::new("S").with_activity(
                Activity::new("slow", "S", 0)
                    .with_process_time(10)
                    .with_duration_on("M3", ActivityDuration::new(0, 4, 0))
                    .with_duration_on("M4", ActivityDuration::new(0, 12, 0))
                    .with_requirement(
                        ResourceRequirement::new("Machine")
                            .with_candidates(vec!["M3".into(), "M4".into()]),
                    ),
            ),
        ];
        let resources: Vec<Resource> = ["M1", "M2", "M3", "M4"]
            .iter()
            .map(|m| Resource::new(*m, "Machine"))
            .collect();
        let problem = Problem::new(tasks, resources).expect("valid");
        let builder = ScheduleCpBuilder::new(&problem);
        assert_eq!(builder.choices()["flex"], vec!["M1", "M2"]);
        assert!(!builder.choices().contains_key("other"));
        let (schedule, solution) = builder.solve(
            &u_metaheur::cp::BranchAndBoundSolver::new(),
            &SolverConfig::default(),
            1_000,
        );
        assert_eq!(solution.status, u_metaheur::cp::SolverStatus::Optimal);
        assert_eq!(schedule.makespan_ms(), 10);
        let on = |id: &str| {
            let a = schedule.assignment_for_activity(id).expect("placed");
            (a.resource_id.clone(), a.end_ms - a.start_ms)
        };
        assert_eq!(on("flex"), ("M2".to_string(), 10));
        assert_eq!(on("other"), ("M1".to_string(), 10));
        assert_eq!(on("slow"), ("M3".to_string(), 4));
        assert!(solution.intervals["flex@M2"].is_present);
        assert!(!solution.intervals["flex@M1"].is_present);
    }

    #[test]
    fn a_schedule_is_read_back_as_the_solvers_incumbent() {
        // "a" and "b" each choose M1 or M2; the schedule has them together,
        // "a" on M2. A search told to stop at its first schedule returns the
        // incumbent as it is.
        let step = |task: &str, id: &str| {
            Task::new(task).with_activity(
                Activity::new(id, task, 0)
                    .with_process_time(10)
                    .with_requirement(
                        ResourceRequirement::new("Machine")
                            .with_candidates(vec!["M1".into(), "M2".into()]),
                    ),
            )
        };
        let resources: Vec<Resource> = ["M1", "M2"]
            .iter()
            .map(|m| Resource::new(*m, "Machine"))
            .collect();
        let problem = Problem::new(vec![step("A", "a"), step("B", "b")], resources).expect("valid");
        let builder = ScheduleCpBuilder::new(&problem);
        let mut schedule = Schedule::new();
        schedule.add_assignment(Assignment::new("a", "A", "M2", 0, 10));
        schedule.add_assignment(Assignment::new("b", "B", "M1", 0, 10));
        let incumbent = builder.incumbent(&schedule);
        assert!(incumbent.intervals["a@M2"].is_present);
        assert!(!incumbent.intervals["a@M1"].is_present);
        assert_eq!(incumbent.intervals["b"].end, 10);
        let config = SolverConfig {
            stop_after_first: true,
            ..SolverConfig::default()
        };
        let (found, solution) = builder.solve(
            &u_metaheur::cp::BranchAndBoundSolver::new().with_incumbent(incumbent),
            &config,
            1_000,
        );
        assert!(solution.is_solution_found());
        assert_eq!(found.makespan_ms(), 10);
        assert_eq!(
            found
                .assignment_for_activity("a")
                .expect("placed")
                .resource_id,
            "M2"
        );
    }

    #[test]
    fn an_activity_keeps_its_first_resource_when_the_choice_would_move_a_delay() {
        let two_steps = |task: &str, overlap: bool| {
            let second = Activity::new(format!("{task}2"), task, 1)
                .with_process_time(10)
                .with_requirement(
                    ResourceRequirement::new("Machine")
                        .with_candidates(vec!["M1".into(), "M2".into()]),
                );
            Task::new(task)
                .with_activity(
                    Activity::new(format!("{task}1"), task, 0)
                        .with_process_time(10)
                        .with_requirement(
                            ResourceRequirement::new("Machine")
                                .with_candidates(vec!["M1".into(), "M2".into()]),
                        ),
                )
                .with_activity(if overlap {
                    second.with_overlap(crate::models::Overlap::new(0.5))
                } else {
                    second
                })
        };
        // An overlap: its delay follows the spans on the chosen resources.
        let resources = vec![
            Resource::new("M1", "Machine"),
            Resource::new("M2", "Machine"),
        ];
        let problem = Problem::new(vec![two_steps("A", true)], resources.clone()).expect("valid");
        assert!(ScheduleCpBuilder::new(&problem).choices().is_empty());
        // A transport time from M1 but not from M2.
        let moving = vec![
            Resource::new("M1", "Machine").with_transport_to("M2", 5),
            Resource::new("M2", "Machine"),
        ];
        let problem = Problem::new(vec![two_steps("B", false)], moving).expect("valid");
        assert!(ScheduleCpBuilder::new(&problem).choices().is_empty());
        // Neither: both steps choose.
        let problem = Problem::new(vec![two_steps("C", false)], resources).expect("valid");
        assert_eq!(ScheduleCpBuilder::new(&problem).choices().len(), 2);
        // A user no-overlap names the first step: it keeps its resource.
        let builder =
            ScheduleCpBuilder::new(&problem).with_constraints(vec![Constraint::NoOverlap {
                resource_id: "M1".into(),
                activity_ids: vec!["C1".into()],
            }]);
        assert_eq!(
            builder.choices().keys().copied().collect::<Vec<_>>(),
            vec!["C2"]
        );
    }

    #[test]
    fn the_exact_solver_proves_ft06() {
        // Fisher & Thompson's 6 x 6 job shop, optimum 55.
        let jobs: [[(usize, i64); 6]; 6] = [
            [(2, 1), (0, 3), (1, 6), (3, 7), (5, 3), (4, 6)],
            [(1, 8), (2, 5), (4, 10), (5, 10), (0, 10), (3, 4)],
            [(2, 5), (3, 4), (5, 8), (0, 9), (1, 1), (4, 7)],
            [(1, 5), (0, 5), (2, 5), (3, 3), (4, 8), (5, 9)],
            [(2, 9), (1, 3), (4, 5), (5, 4), (0, 3), (3, 1)],
            [(1, 3), (3, 3), (5, 9), (0, 10), (4, 4), (2, 1)],
        ];
        let tasks = jobs
            .iter()
            .enumerate()
            .map(|(j, ops)| {
                let id = format!("J{j}");
                ops.iter()
                    .enumerate()
                    .fold(Task::new(&id), |task, (k, &(m, d))| {
                        task.with_activity(
                            Activity::new(format!("{id}_{k}"), &id, k as i32)
                                .with_process_time(d)
                                .with_requirement(
                                    ResourceRequirement::new("Machine")
                                        .with_candidates(vec![format!("M{m}")]),
                                ),
                        )
                    })
            })
            .collect();
        let resources = (0..6)
            .map(|m| Resource::new(format!("M{m}"), "Machine"))
            .collect();
        let problem = Problem::new(tasks, resources).expect("valid");
        let (schedule, solution) = ScheduleCpBuilder::new(&problem).solve(
            &u_metaheur::cp::BranchAndBoundSolver::new(),
            &SolverConfig::default(),
            1_000,
        );
        assert_eq!(solution.status, u_metaheur::cp::SolverStatus::Optimal);
        assert_eq!(schedule.makespan_ms(), 55);
        let input = crate::scheduler::FeasibilityInput {
            tasks: problem.tasks(),
            resources: problem.resources(),
            constraints: &[],
        };
        let violations = crate::scheduler::check_schedule(&schedule, &input);
        assert!(violations.is_empty(), "{violations:?}");
    }
}
