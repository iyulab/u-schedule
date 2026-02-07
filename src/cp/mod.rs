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
    CpModel, CpSolver, CpSolution, IntervalVar,
    Objective, SolverConfig,
};

use crate::models::{
    Assignment, Constraint, Resource, Schedule, Task, TransitionMatrixCollection,
};

/// Builds a CP model from scheduling domain objects.
///
/// Translates tasks, resources, and constraints into a CpModel
/// suitable for solving with any `CpSolver` implementation.
///
/// # Example
/// ```no_run
/// use u_schedule::cp::ScheduleCpBuilder;
/// use u_schedule::models::{Task, Resource};
/// use u_metaheur::cp::{SimpleCpSolver, SolverConfig};
///
/// let tasks = vec![/* ... */];
/// let resources = vec![/* ... */];
/// let builder = ScheduleCpBuilder::new(&tasks, &resources);
/// let model = builder.build(100_000);
/// ```
pub struct ScheduleCpBuilder<'a> {
    tasks: &'a [Task],
    #[allow(dead_code)]
    resources: &'a [Resource],
    constraints: Vec<Constraint>,
    transition_matrices: TransitionMatrixCollection,
}

impl<'a> ScheduleCpBuilder<'a> {
    /// Creates a new CP builder.
    pub fn new(tasks: &'a [Task], resources: &'a [Resource]) -> Self {
        Self {
            tasks,
            resources,
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
    /// - `NoOverlap` constraints per resource (from candidate assignments)
    /// - `Precedence` constraints for intra-task activity ordering
    /// - User-defined constraints
    /// - `MinimizeMaxEnd` objective (makespan minimization)
    pub fn build(&self, horizon_ms: i64) -> CpModel {
        let mut model = CpModel::new("scheduling", horizon_ms);

        // Create interval variables for each activity
        for task in self.tasks {
            let release = task.release_time.unwrap_or(0);

            for activity in &task.activities {
                let duration = activity.duration.process_ms;
                let interval = IntervalVar::new(
                    &activity.id,
                    release,         // start_min
                    horizon_ms - duration, // start_max
                    duration,        // fixed duration
                    horizon_ms,      // end_max
                );
                model.add_interval(interval);
            }

            // Intra-task precedence: activity[i] before activity[i+1]
            for i in 0..task.activities.len().saturating_sub(1) {
                model.add_precedence(
                    task.activities[i].id.clone(),
                    task.activities[i + 1].id.clone(),
                    0,
                );
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

        for task in self.tasks {
            for activity in &task.activities {
                if let Some(interval_sol) = solution.intervals.get(&activity.id) {
                    if interval_sol.is_present {
                        // Determine resource (from candidates, pick first for now)
                        let resource_id = activity
                            .candidate_resources()
                            .first()
                            .map(|s| s.to_string())
                            .unwrap_or_default();

                        schedule.add_assignment(Assignment::new(
                            &activity.id,
                            &task.id,
                            &resource_id,
                            interval_sol.start,
                            interval_sol.end,
                        ));
                    }
                }
            }
        }

        schedule
    }

    /// Collects activity IDs per resource (from candidate lists).
    fn collect_resource_activities(&self) -> HashMap<String, Vec<String>> {
        let mut map: HashMap<String, Vec<String>> = HashMap::new();

        for task in self.tasks {
            for activity in &task.activities {
                for candidate in activity.candidate_resources() {
                    map.entry(candidate.to_string())
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
    use crate::models::{Activity, ActivityDuration, ResourceRequirement, ResourceType};
    use u_metaheur::cp::SimpleCpSolver;

    fn make_test_data() -> (Vec<Task>, Vec<Resource>) {
        let tasks = vec![
            Task::new("T1")
                .with_activity(
                    Activity::new("T1_O1", "T1", 0)
                        .with_duration(ActivityDuration::fixed(1000))
                        .with_requirement(
                            ResourceRequirement::new("Machine")
                                .with_candidates(vec!["M1".into()]),
                        ),
                )
                .with_activity(
                    Activity::new("T1_O2", "T1", 1)
                        .with_duration(ActivityDuration::fixed(2000))
                        .with_requirement(
                            ResourceRequirement::new("Machine")
                                .with_candidates(vec!["M1".into()]),
                        ),
                ),
            Task::new("T2").with_activity(
                Activity::new("T2_O1", "T2", 0)
                    .with_duration(ActivityDuration::fixed(1500))
                    .with_requirement(
                        ResourceRequirement::new("Machine")
                            .with_candidates(vec!["M1".into()]),
                    ),
            ),
        ];

        let resources = vec![Resource::new("M1", ResourceType::Primary)];
        (tasks, resources)
    }

    #[test]
    fn test_build_model() {
        let (tasks, resources) = make_test_data();
        let builder = ScheduleCpBuilder::new(&tasks, &resources);
        let model = builder.build(100_000);

        // 3 intervals (T1_O1, T1_O2, T2_O1)
        assert_eq!(model.interval_count(), 3);
        // Constraints: 1 precedence (T1_O1→T1_O2) + 1 no-overlap (M1)
        assert!(model.constraint_count() >= 2);
    }

    #[test]
    fn test_build_with_constraints() {
        let (tasks, resources) = make_test_data();
        let constraints = vec![Constraint::precedence("T1_O2", "T2_O1")];
        let builder = ScheduleCpBuilder::new(&tasks, &resources)
            .with_constraints(constraints);
        let model = builder.build(100_000);

        // Additional precedence constraint
        assert!(model.constraint_count() >= 3);
    }

    #[test]
    fn test_solve_basic() {
        let (tasks, resources) = make_test_data();
        let builder = ScheduleCpBuilder::new(&tasks, &resources);
        let solver = SimpleCpSolver::new();
        let config = SolverConfig::default();

        let (schedule, solution) = builder.solve(&solver, &config, 100_000);
        assert!(solution.is_solution_found());
        assert!(schedule.assignment_count() > 0);
        assert!(schedule.makespan_ms() > 0);
    }

    #[test]
    fn test_intra_task_precedence() {
        let (tasks, resources) = make_test_data();
        let builder = ScheduleCpBuilder::new(&tasks, &resources);
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
        let (tasks, resources) = make_test_data();
        let builder = ScheduleCpBuilder::new(&tasks, &resources);
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
                    a.activity_id, a.start_ms, a.end_ms,
                    b.activity_id, b.start_ms, b.end_ms,
                );
            }
        }
    }
}
