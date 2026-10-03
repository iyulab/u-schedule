//! Input validation for scheduling problems.
//!
//! Checks structural integrity of tasks, activities, and resources
//! before scheduling. [`Problem::new`](crate::Problem::new) runs it, and the
//! solvers take only a [`Problem`](crate::Problem). Detects:
//! - Duplicate IDs
//! - Missing resource references
//! - Circular precedence dependencies (DAG validation)
//! - Empty tasks
//! - Weights that are not finite and positive
//! - Activity durations below 0 and resource capacities below 1
//!
//! # Reference
//! Cormen et al. (2009), "Introduction to Algorithms", Ch. 22.4 (Topological Sort)

use crate::models::{Resource, Task};
use std::collections::{HashMap, HashSet};

/// Validation result.
pub type ValidationResult = Result<(), Vec<ValidationError>>;

/// A validation error.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationError {
    /// Error category.
    pub kind: ValidationErrorKind,
    /// Human-readable description.
    pub message: String,
}

/// What a validation error is about, with the ids that locate it.
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationErrorKind {
    /// Two entities of the same kind share an id.
    DuplicateId {
        /// Which kind of entity the id is repeated among.
        entity: Entity,
        /// The repeated id.
        id: String,
    },
    /// An activity names a candidate resource that doesn't exist.
    InvalidResourceReference {
        /// The activity naming the resource.
        activity: String,
        /// The resource id that matches no resource.
        resource: String,
    },
    /// The precedence graph contains a cycle.
    CyclicDependency {
        /// An activity on the cycle.
        activity: String,
    },
    /// A task has no activities.
    EmptyTask {
        /// The task.
        task: String,
    },
    /// An activity names a predecessor that doesn't exist.
    InvalidPredecessor {
        /// The activity naming the predecessor.
        activity: String,
        /// The predecessor id that matches no activity.
        predecessor: String,
    },
    /// A task's [`weight`](crate::models::Task::weight) is not finite and
    /// `> 0`. WSPT and ATC divide by processing time and multiply by the
    /// weight, so zero, negative, or non-finite weights would rank the task
    /// silently wrong rather than fail.
    WeightOutOfRange {
        /// The task.
        task: String,
        /// The weight it carries.
        weight: f64,
    },
    /// An activity's setup, process or teardown time is below 0. A negative
    /// part would end the activity before it starts.
    NegativeDuration {
        /// The activity.
        activity: String,
        /// Its total duration (ms).
        duration_ms: i64,
    },
    /// A resource's capacity is below 1: it could never hold an activity.
    CapacityOutOfRange {
        /// The resource.
        resource: String,
        /// The capacity it declares.
        capacity: i32,
    },
}

/// The kinds of entity a scheduling problem identifies by id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Entity {
    /// A [`Task`].
    Task,
    /// An [`Activity`](crate::models::Activity).
    Activity,
    /// A [`Resource`].
    Resource,
}

impl Entity {
    /// The entity's name in lower case, as messages and wire fields spell it.
    pub fn name(self) -> &'static str {
        match self {
            Entity::Task => "task",
            Entity::Activity => "activity",
            Entity::Resource => "resource",
        }
    }
}

impl ValidationErrorKind {
    /// A stable snake_case name for the reason, for callers that branch on it
    /// across a wire (`duplicate_id`, `invalid_resource_reference`, ...).
    pub fn code(&self) -> &'static str {
        match self {
            ValidationErrorKind::DuplicateId { .. } => "duplicate_id",
            ValidationErrorKind::InvalidResourceReference { .. } => "invalid_resource_reference",
            ValidationErrorKind::CyclicDependency { .. } => "cyclic_dependency",
            ValidationErrorKind::EmptyTask { .. } => "empty_task",
            ValidationErrorKind::InvalidPredecessor { .. } => "invalid_predecessor",
            ValidationErrorKind::WeightOutOfRange { .. }
            | ValidationErrorKind::NegativeDuration { .. }
            | ValidationErrorKind::CapacityOutOfRange { .. } => "parameter_out_of_range",
        }
    }
}

impl ValidationError {
    fn new(kind: ValidationErrorKind) -> Self {
        let message = match &kind {
            ValidationErrorKind::DuplicateId { entity, id } => {
                format!("Duplicate {} ID: {id}", entity.name())
            }
            ValidationErrorKind::InvalidResourceReference { activity, resource } => {
                format!("Activity '{activity}' references unknown resource '{resource}'")
            }
            ValidationErrorKind::CyclicDependency { activity } => {
                format!("Circular dependency detected involving activity '{activity}'")
            }
            ValidationErrorKind::EmptyTask { task } => format!("Task '{task}' has no activities"),
            ValidationErrorKind::InvalidPredecessor {
                activity,
                predecessor,
            } => format!("Activity '{activity}' references unknown predecessor '{predecessor}'"),
            ValidationErrorKind::WeightOutOfRange { task, weight } => format!(
                "Task '{task}' has weight {weight}; a weight must be finite and greater than 0"
            ),
            ValidationErrorKind::NegativeDuration {
                activity,
                duration_ms,
            } => format!(
                "Activity '{activity}' has a negative duration part (total {duration_ms} ms); \
                 setup, process and teardown must each be >= 0"
            ),
            ValidationErrorKind::CapacityOutOfRange { resource, capacity } => format!(
                "Resource '{resource}' has capacity {capacity}; a capacity must be at least 1"
            ),
        };
        Self { kind, message }
    }
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ValidationError {}

/// Validates the input data for a scheduling problem.
///
/// Checks:
/// 1. No duplicate task IDs
/// 2. No duplicate activity IDs (across all tasks)
/// 3. No duplicate resource IDs
/// 4. All tasks have at least one activity
/// 5. All resource references in activities point to existing resources
/// 6. All predecessor references point to existing activities
/// 7. No circular precedence dependencies
/// 8. Every task's weight is finite and `> 0`
/// 9. Every activity's setup, process and teardown times are `>= 0`
/// 10. Every resource's capacity is at least 1
///
/// # Returns
/// `Ok(())` if all checks pass, `Err(errors)` with all detected issues.
pub fn validate_input(tasks: &[Task], resources: &[Resource]) -> ValidationResult {
    let mut errors = Vec::new();

    // Collect resource IDs
    let mut resource_ids = HashSet::new();
    for r in resources {
        if r.capacity < 1 {
            errors.push(ValidationError::new(
                ValidationErrorKind::CapacityOutOfRange {
                    resource: r.id.clone(),
                    capacity: r.capacity,
                },
            ));
        }
        if !resource_ids.insert(r.id.as_str()) {
            errors.push(ValidationError::new(ValidationErrorKind::DuplicateId {
                entity: Entity::Resource,
                id: r.id.clone(),
            }));
        }
    }

    // Collect task and activity IDs
    let mut task_ids = HashSet::new();
    let mut activity_ids = HashSet::new();

    for task in tasks {
        if !task_ids.insert(task.id.as_str()) {
            errors.push(ValidationError::new(ValidationErrorKind::DuplicateId {
                entity: Entity::Task,
                id: task.id.clone(),
            }));
        }

        if !(task.weight.is_finite() && task.weight > 0.0) {
            errors.push(ValidationError::new(
                ValidationErrorKind::WeightOutOfRange {
                    task: task.id.clone(),
                    weight: task.weight,
                },
            ));
        }

        if task.activities.is_empty() {
            errors.push(ValidationError::new(ValidationErrorKind::EmptyTask {
                task: task.id.clone(),
            }));
        }

        for act in &task.activities {
            let d = &act.duration;
            if d.setup_ms < 0 || d.process_ms < 0 || d.teardown_ms < 0 {
                errors.push(ValidationError::new(
                    ValidationErrorKind::NegativeDuration {
                        activity: act.id.clone(),
                        duration_ms: d.total_ms(),
                    },
                ));
            }
            if !activity_ids.insert(act.id.as_str()) {
                errors.push(ValidationError::new(ValidationErrorKind::DuplicateId {
                    entity: Entity::Activity,
                    id: act.id.clone(),
                }));
            }
        }
    }

    // Check resource references
    for task in tasks {
        for act in &task.activities {
            for req in &act.resource_requirements {
                for cand in &req.candidates {
                    if !resource_ids.contains(cand.as_str()) {
                        errors.push(ValidationError::new(
                            ValidationErrorKind::InvalidResourceReference {
                                activity: act.id.clone(),
                                resource: cand.clone(),
                            },
                        ));
                    }
                }
            }
        }
    }

    // Check predecessor references
    for task in tasks {
        for act in &task.activities {
            for pred in &act.predecessors {
                if !activity_ids.contains(pred.as_str()) {
                    errors.push(ValidationError::new(
                        ValidationErrorKind::InvalidPredecessor {
                            activity: act.id.clone(),
                            predecessor: pred.clone(),
                        },
                    ));
                }
            }
        }
    }

    // Check for cycles in precedence graph (DFS-based)
    if let Some(cycle_err) = detect_cycles(tasks) {
        errors.push(cycle_err);
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Detects cycles in the precedence graph using DFS.
///
/// # Algorithm
/// Topological sort via DFS. If a back-edge is found (visiting a node
/// currently in the recursion stack), a cycle exists.
///
/// # Reference
/// Cormen et al. (2009), "Introduction to Algorithms", Ch. 22.4
fn detect_cycles(tasks: &[Task]) -> Option<ValidationError> {
    // Build adjacency list: activity_id → successors
    let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut all_ids: HashSet<&str> = HashSet::new();

    for task in tasks {
        for act in &task.activities {
            all_ids.insert(&act.id);
            for pred in &act.predecessors {
                adj.entry(pred.as_str()).or_default().push(act.id.as_str());
            }
        }
    }

    // DFS cycle detection
    let mut visited = HashSet::new();
    let mut in_stack = HashSet::new();

    for &node in &all_ids {
        if !visited.contains(node) && has_cycle_dfs(node, &adj, &mut visited, &mut in_stack) {
            return Some(ValidationError::new(
                ValidationErrorKind::CyclicDependency {
                    activity: node.to_string(),
                },
            ));
        }
    }

    None
}

fn has_cycle_dfs<'a>(
    node: &'a str,
    adj: &HashMap<&'a str, Vec<&'a str>>,
    visited: &mut HashSet<&'a str>,
    in_stack: &mut HashSet<&'a str>,
) -> bool {
    visited.insert(node);
    in_stack.insert(node);

    if let Some(neighbors) = adj.get(node) {
        for &next in neighbors {
            if in_stack.contains(next) {
                return true; // Back edge → cycle
            }
            if !visited.contains(next) && has_cycle_dfs(next, adj, visited, in_stack) {
                return true;
            }
        }
    }

    in_stack.remove(node);
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Activity, ActivityDuration, Resource, ResourceRequirement, Task};

    fn sample_resources() -> Vec<Resource> {
        vec![
            Resource::primary("M1").with_name("Machine 1"),
            Resource::primary("M2").with_name("Machine 2"),
            Resource::human("W1").with_name("Worker 1"),
        ]
    }

    fn sample_tasks() -> Vec<Task> {
        vec![
            Task::new("J1")
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
                        .with_predecessor("O1")
                        .with_requirement(
                            ResourceRequirement::new("Machine").with_candidates(vec!["M2".into()]),
                        ),
                ),
            Task::new("J2").with_activity(
                Activity::new("O3", "J2", 0)
                    .with_duration(ActivityDuration::fixed(1500))
                    .with_requirement(
                        ResourceRequirement::new("Machine").with_candidates(vec!["M1".into()]),
                    ),
            ),
        ]
    }

    #[test]
    fn test_valid_input() {
        let tasks = sample_tasks();
        let resources = sample_resources();
        assert!(validate_input(&tasks, &resources).is_ok());
    }

    #[test]
    fn test_duplicate_task_id() {
        let tasks = vec![
            Task::new("J1").with_activity(Activity::new("O1", "J1", 0).with_process_time(100)),
            Task::new("J1").with_activity(Activity::new("O2", "J1", 0).with_process_time(100)),
        ];
        let resources = sample_resources();

        let errors = validate_input(&tasks, &resources).unwrap_err();
        assert!(errors.iter().any(|e| e.kind
            == ValidationErrorKind::DuplicateId {
                entity: Entity::Task,
                id: "J1".into()
            }));
    }

    #[test]
    fn test_duplicate_resource_id() {
        let tasks = sample_tasks();
        let resources = vec![Resource::primary("M1"), Resource::primary("M1")];

        let errors = validate_input(&tasks, &resources).unwrap_err();
        assert!(errors.iter().any(|e| e.kind
            == ValidationErrorKind::DuplicateId {
                entity: Entity::Resource,
                id: "M1".into()
            }));
    }

    #[test]
    fn test_empty_task() {
        let tasks = vec![Task::new("empty")]; // No activities
        let resources = sample_resources();

        let errors = validate_input(&tasks, &resources).unwrap_err();
        assert!(errors.iter().any(|e| e.kind
            == ValidationErrorKind::EmptyTask {
                task: "empty".into()
            }));
    }

    #[test]
    fn test_invalid_resource_reference() {
        let tasks = vec![Task::new("J1").with_activity(
            Activity::new("O1", "J1", 0)
                .with_process_time(100)
                .with_requirement(
                    ResourceRequirement::new("Machine").with_candidates(vec!["NONEXISTENT".into()]),
                ),
        )];
        let resources = sample_resources();

        let errors = validate_input(&tasks, &resources).unwrap_err();
        assert!(errors.iter().any(|e| e.kind
            == ValidationErrorKind::InvalidResourceReference {
                activity: "O1".into(),
                resource: "NONEXISTENT".into()
            }));
    }

    #[test]
    fn test_invalid_predecessor() {
        let tasks = vec![Task::new("J1").with_activity(
            Activity::new("O1", "J1", 0)
                .with_process_time(100)
                .with_predecessor("NONEXISTENT"),
        )];
        let resources = sample_resources();

        let errors = validate_input(&tasks, &resources).unwrap_err();
        assert!(errors.iter().any(|e| e.kind
            == ValidationErrorKind::InvalidPredecessor {
                activity: "O1".into(),
                predecessor: "NONEXISTENT".into()
            }));
    }

    #[test]
    fn test_cyclic_dependency() {
        // O1 → O2 → O3 → O1 (cycle)
        let tasks = vec![Task::new("J1")
            .with_activity(
                Activity::new("O1", "J1", 0)
                    .with_process_time(100)
                    .with_predecessor("O3"),
            )
            .with_activity(
                Activity::new("O2", "J1", 1)
                    .with_process_time(100)
                    .with_predecessor("O1"),
            )
            .with_activity(
                Activity::new("O3", "J1", 2)
                    .with_process_time(100)
                    .with_predecessor("O2"),
            )];
        let resources = sample_resources();

        let errors = validate_input(&tasks, &resources).unwrap_err();
        assert!(errors
            .iter()
            .any(|e| matches!(e.kind, ValidationErrorKind::CyclicDependency { .. })));
    }

    #[test]
    fn test_no_cycle_in_chain() {
        // O1 → O2 → O3 (linear chain, no cycle)
        let tasks = vec![Task::new("J1")
            .with_activity(Activity::new("O1", "J1", 0).with_process_time(100))
            .with_activity(
                Activity::new("O2", "J1", 1)
                    .with_process_time(100)
                    .with_predecessor("O1"),
            )
            .with_activity(
                Activity::new("O3", "J1", 2)
                    .with_process_time(100)
                    .with_predecessor("O2"),
            )];
        let resources = sample_resources();

        assert!(validate_input(&tasks, &resources).is_ok());
    }

    #[test]
    fn test_multiple_errors() {
        // Empty task + invalid resource reference
        let tasks = vec![
            Task::new("empty"), // Empty task
            Task::new("J1").with_activity(
                Activity::new("O1", "J1", 0)
                    .with_process_time(100)
                    .with_requirement(
                        ResourceRequirement::new("M").with_candidates(vec!["UNKNOWN".into()]),
                    ),
            ),
        ];
        let resources = vec![];

        let errors = validate_input(&tasks, &resources).unwrap_err();
        assert!(errors.len() >= 2);
    }

    #[test]
    fn a_weight_that_is_not_finite_and_positive_is_refused() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut tasks = sample_tasks();
            tasks[0].weight = bad;
            let errors = validate_input(&tasks, &sample_resources()).unwrap_err();
            assert_eq!(errors.len(), 1, "weight {bad}");
            assert_eq!(errors[0].kind.code(), "parameter_out_of_range");
            match &errors[0].kind {
                ValidationErrorKind::WeightOutOfRange { task, weight } => {
                    assert_eq!(task, "J1");
                    assert!(weight.to_bits() == bad.to_bits());
                }
                other => panic!("unexpected {other:?}"),
            }
        }
        let mut tasks = sample_tasks();
        tasks[0].weight = 1e-6;
        assert!(validate_input(&tasks, &sample_resources()).is_ok());
    }

    #[test]
    fn a_negative_duration_and_a_capacity_below_one_are_refused() {
        let mut tasks = sample_tasks();
        tasks[0].activities[0].duration = ActivityDuration::fixed(-5);
        let mut resources = sample_resources();
        resources[0].capacity = 0;
        let errors = validate_input(&tasks, &resources).unwrap_err();
        let kinds: Vec<_> = errors.iter().map(|e| e.kind.clone()).collect();
        assert!(kinds.contains(&ValidationErrorKind::CapacityOutOfRange {
            resource: "M1".into(),
            capacity: 0,
        }));
        assert!(kinds.contains(&ValidationErrorKind::NegativeDuration {
            activity: "O1".into(),
            duration_ms: -5,
        }));
        assert!(errors
            .iter()
            .all(|e| e.kind.code() == "parameter_out_of_range"));
    }
}
