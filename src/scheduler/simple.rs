//! Simple priority-driven greedy scheduler.
//!
//! # Algorithm
//!
//! 1. Sort tasks by dispatching rule (or priority if no rule engine).
//! 2. For each task, process activities sequentially.
//! 3. For each activity, select the earliest-available candidate resource.
//! 4. Apply sequence-dependent setup times from transition matrices.
//!
//! # Complexity
//! O(n * m * c) where n=tasks, m=activities/task, c=candidate resources.
//!
//! # Reference
//! Pinedo (2016), "Scheduling", Ch. 4: Priority Dispatching

use std::collections::HashMap;

use crate::dispatching::{RuleEngine, SchedulingContext};
use crate::models::{Assignment, Resource, Schedule, Task, TransitionMatrixCollection};

/// Input container for scheduling.
#[derive(Debug, Clone)]
pub struct ScheduleRequest {
    /// Tasks to schedule.
    pub tasks: Vec<Task>,
    /// Available resources.
    pub resources: Vec<Resource>,
    /// Schedule start time (ms).
    pub start_time_ms: i64,
    /// Sequence-dependent setup time matrices.
    pub transition_matrices: TransitionMatrixCollection,
}

impl ScheduleRequest {
    /// Creates a new schedule request.
    pub fn new(tasks: Vec<Task>, resources: Vec<Resource>) -> Self {
        Self {
            tasks,
            resources,
            start_time_ms: 0,
            transition_matrices: TransitionMatrixCollection::new(),
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
}

/// Simple priority-driven greedy scheduler.
///
/// Schedules tasks by priority (or dispatching rule), assigning each
/// activity to the earliest-available candidate resource. Supports
/// sequence-dependent setup times via transition matrices.
///
/// # Example
///
/// ```
/// use u_schedule::scheduler::{SimpleScheduler, ScheduleRequest};
/// use u_schedule::models::{Task, Resource, ResourceType, Activity, ActivityDuration, ResourceRequirement};
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
/// let resources = vec![Resource::new("M1", ResourceType::Primary)];
/// let request = ScheduleRequest::new(tasks, resources);
///
/// let scheduler = SimpleScheduler::new();
/// let schedule = scheduler.schedule_request(&request);
/// assert_eq!(schedule.assignment_count(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct SimpleScheduler {
    transition_matrices: TransitionMatrixCollection,
    rule_engine: Option<RuleEngine>,
}

impl SimpleScheduler {
    /// Creates a new scheduler.
    pub fn new() -> Self {
        Self {
            transition_matrices: TransitionMatrixCollection::new(),
            rule_engine: None,
        }
    }

    /// Sets transition matrices.
    pub fn with_transition_matrices(mut self, matrices: TransitionMatrixCollection) -> Self {
        self.transition_matrices = matrices;
        self
    }

    /// Sets a rule engine for task ordering.
    ///
    /// When set, tasks are sorted by the rule engine instead of by priority.
    pub fn with_rule_engine(mut self, engine: RuleEngine) -> Self {
        self.rule_engine = Some(engine);
        self
    }

    /// Schedules tasks on resources.
    ///
    /// # Algorithm
    /// 1. Sort tasks by rule engine or priority (descending).
    /// 2. For each task, schedule activities in sequence order.
    /// 3. For each activity, find the earliest-available candidate resource.
    /// 4. Apply setup time from transition matrices.
    pub fn schedule(&self, tasks: &[Task], resources: &[Resource], start_time_ms: i64) -> Schedule {
        let mut schedule = Schedule::new();
        let mut resource_available: HashMap<String, i64> = HashMap::new();
        let mut last_category: HashMap<String, String> = HashMap::new();

        // Initialize resource availability
        for resource in resources {
            resource_available.insert(resource.id.clone(), start_time_ms);
        }

        // Determine task order
        let task_order = self.sort_tasks(tasks, start_time_ms);

        // Schedule each task
        for &task_idx in &task_order {
            let task = &tasks[task_idx];
            let mut task_start = task
                .release_time
                .unwrap_or(start_time_ms)
                .max(start_time_ms);

            for activity in &task.activities {
                let candidates = activity.candidate_resources();
                if candidates.is_empty() {
                    continue;
                }

                // Select resource with earliest availability
                let mut best_resource: Option<&str> = None;
                let mut best_start = i64::MAX;

                for candidate in &candidates {
                    if let Some(&available) = resource_available.get(*candidate) {
                        let actual_start = available.max(task_start);
                        if actual_start < best_start {
                            best_start = actual_start;
                            best_resource = Some(candidate);
                        }
                    }
                }

                if let Some(resource_id) = best_resource {
                    // Calculate setup time from transition matrices
                    let setup_time = if let Some(prev_cat) = last_category.get(resource_id) {
                        self.transition_matrices.get_transition_time(
                            resource_id,
                            prev_cat,
                            &task.category,
                        )
                    } else {
                        0
                    };

                    let start = best_start;
                    let end = start + setup_time + activity.duration.process_ms;

                    let assignment =
                        Assignment::new(&activity.id, &task.id, resource_id, start, end)
                            .with_setup(setup_time);

                    schedule.add_assignment(assignment);

                    // Update state
                    resource_available.insert(resource_id.to_string(), end);
                    last_category.insert(resource_id.to_string(), task.category.clone());
                    task_start = end; // Enforce intra-task precedence
                }
            }
        }

        schedule
    }

    /// Schedules from a request.
    pub fn schedule_request(&self, request: &ScheduleRequest) -> Schedule {
        let scheduler = Self {
            transition_matrices: request.transition_matrices.clone(),
            rule_engine: self.rule_engine.clone(),
        };
        scheduler.schedule(&request.tasks, &request.resources, request.start_time_ms)
    }

    /// Returns task indices sorted by rule engine or priority.
    fn sort_tasks(&self, tasks: &[Task], start_time_ms: i64) -> Vec<usize> {
        if let Some(ref engine) = self.rule_engine {
            let ctx = SchedulingContext::at_time(start_time_ms);
            engine.sort_indices(tasks, &ctx)
        } else {
            // Default: sort by priority descending
            let mut indices: Vec<usize> = (0..tasks.len()).collect();
            indices.sort_by(|&a, &b| tasks[b].priority.cmp(&tasks[a].priority));
            indices
        }
    }
}

impl Default for SimpleScheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatching::rules;
    use crate::models::{
        Activity, ActivityDuration, Resource, ResourceRequirement, ResourceType, TransitionMatrix,
    };

    fn make_resource(id: &str) -> Resource {
        Resource::new(id, ResourceType::Primary)
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
    fn test_simple_single_task() {
        let tasks = vec![make_task_with_resource("J1", 1000, "M1", 0)];
        let resources = vec![make_resource("M1")];
        let scheduler = SimpleScheduler::new();

        let schedule = scheduler.schedule(&tasks, &resources, 0);
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

        let schedule = scheduler.schedule(&tasks, &resources, 0);

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

        let schedule = scheduler.schedule(&tasks, &resources, 0);
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

        let schedule = scheduler.schedule(&tasks, &resources, 0);
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
        let schedule = scheduler.schedule(&[task], &resources, 0);

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

        let schedule = scheduler.schedule(&tasks, &resources, 0);
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

        let schedule = scheduler.schedule(&tasks, &resources, 0);
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
        let request = ScheduleRequest::new(tasks, resources).with_start_time(5000);

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

        let schedule = scheduler.schedule(&[task], &resources, 0);
        let a = schedule.assignment_for_activity("J1_O1").unwrap();
        // Must not start before release_time
        assert_eq!(a.start_ms, 5000);
    }

    #[test]
    fn test_empty_input() {
        let scheduler = SimpleScheduler::new();
        let schedule = scheduler.schedule(&[], &[], 0);
        assert_eq!(schedule.assignment_count(), 0);
        assert_eq!(schedule.makespan_ms(), 0);
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
        let schedule = scheduler.schedule(&[task], &resources, 0);
        assert_eq!(schedule.assignment_count(), 0);
    }
}
