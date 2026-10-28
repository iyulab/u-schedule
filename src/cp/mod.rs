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

use u_metaheur::cp::{CpModel, CpSolution, CpSolver, IntervalVar, Objective, SolverConfig};

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
    /// - `NoOverlap` constraints per resource, over the activities reported
    ///   on it (each activity on its first eligible resource -- this
    ///   formulation does not choose among alternatives)
    /// - `Precedence` constraints for intra-task activity ordering and for
    ///   each activity's `predecessors` in other tasks — a step that
    ///   [overlaps](crate::models::Overlap) the one before by a delay that may
    ///   be negative (both bounds, exact since every span here is fixed)
    /// - User-defined constraints
    /// - `MinimizeMaxEnd` objective (makespan minimization)
    pub fn build(&self, horizon_ms: i64) -> CpModel {
        let mut model = CpModel::new("scheduling", horizon_ms);

        // Create interval variables for each activity
        for task in self.problem.tasks() {
            let release = task.release_time.unwrap_or(0);

            for activity in &task.activities {
                // Setup, processing and teardown: the activity's whole span,
                // on the resource the solution is reported on.
                let duration = self.duration_of(activity).total_ms();
                let release = activity.release_time.map_or(release, |r| release.max(r));
                let interval = IntervalVar::new(
                    &activity.id,
                    release,               // start_min
                    horizon_ms - duration, // start_max
                    duration,              // fixed duration
                    horizon_ms,            // end_max
                );
                model.add_interval(interval);
            }

            // Intra-task precedence: activity[i] before activity[i+1], with
            // the time the work moves between the resources they are
            // reported on.
            for pair in task.activities.windows(2) {
                let transport = match (self.reported_on(&pair[0]), self.reported_on(&pair[1])) {
                    (Some(from), Some(to)) => transport_between(
                        |id| self.problem.resources().iter().find(|r| r.id == id),
                        [from],
                        &[to],
                    ),
                    _ => 0,
                };
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
        let resource_activities = self.collect_resource_activities();
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

    /// Decodes a CP solution into a Schedule.
    fn decode_solution(&self, solution: &CpSolution) -> Schedule {
        let mut schedule = Schedule::new();

        if !solution.is_solution_found() {
            return schedule;
        }

        for task in self.problem.tasks() {
            for (step, activity) in task.activities.iter().enumerate() {
                if let Some(interval_sol) = solution.intervals.get(&activity.id) {
                    if interval_sol.is_present {
                        let resource_id =
                            self.reported_on(activity).unwrap_or_default().to_string();

                        schedule.add_assignment(
                            Assignment::new(
                                &activity.id,
                                &task.id,
                                &resource_id,
                                interval_sol.start,
                                interval_sol.end,
                            )
                            .with_sequence(step as i32 + 1)
                            .with_setup(self.duration_of(activity).setup_ms),
                        );
                    }
                }
            }
        }

        schedule
    }

    /// The resource a solution reports `activity` on — the first eligible
    /// one (this formulation does not choose among alternatives).
    fn reported_on<'p>(&'p self, activity: &crate::models::Activity) -> Option<&'p str> {
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

    /// Collects activity IDs per resource: each activity on the resource it
    /// is reported on ([`reported_on`](Self::reported_on)) -- the one its
    /// duration is taken on, and the only one it occupies.
    fn collect_resource_activities(&self) -> HashMap<String, Vec<String>> {
        let mut map: HashMap<String, Vec<String>> = HashMap::new();

        for task in self.problem.tasks() {
            for activity in &task.activities {
                if let Some(resource) = self.reported_on(activity) {
                    map.entry(resource.to_string())
                        .or_default()
                        .push(activity.id.clone());
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
    fn an_activity_occupies_only_the_resource_it_is_reported_on() {
        // "flex" may run on M1 or M2 and is reported on M1; "other" needs M2
        // alone. The two can run together.
        let tasks = vec![
            Task::new("F").with_activity(
                Activity::new("flex", "F", 0)
                    .with_process_time(10)
                    .with_requirement(
                        ResourceRequirement::new("Machine")
                            .with_candidates(vec!["M1".into(), "M2".into()]),
                    ),
            ),
            Task::new("O").with_activity(
                Activity::new("other", "O", 0)
                    .with_process_time(10)
                    .with_requirement(
                        ResourceRequirement::new("Machine").with_candidates(vec!["M2".into()]),
                    ),
            ),
        ];
        let resources = vec![
            Resource::new("M1", "Machine"),
            Resource::new("M2", "Machine"),
        ];
        let problem = Problem::new(tasks, resources).expect("valid");
        let (schedule, solution) = ScheduleCpBuilder::new(&problem).solve(
            &u_metaheur::cp::BranchAndBoundSolver::new(),
            &SolverConfig::default(),
            1_000,
        );
        assert_eq!(solution.status, u_metaheur::cp::SolverStatus::Optimal);
        assert_eq!(schedule.makespan_ms(), 10);
        let flex = schedule.assignment_for_activity("flex").expect("placed");
        assert_eq!(flex.resource_id, "M1");
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
