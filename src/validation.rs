//! Input validation for scheduling problems.
//!
//! Checks structural integrity of tasks, activities, and resources
//! before scheduling. Detects:
//! - Duplicate IDs
//! - Missing resource references
//! - Circular precedence dependencies (DAG validation)
//! - Empty tasks
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

/// Categories of validation errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationErrorKind {
    /// Two entities share the same ID.
    DuplicateId,
    /// An activity references a resource that doesn't exist.
    InvalidResourceReference,
    /// Precedence graph contains a cycle.
    CyclicDependency,
    /// A task has no activities.
    EmptyTask,
    /// An activity references a predecessor that doesn't exist.
    InvalidPredecessor,
}

impl ValidationError {
    fn new(kind: ValidationErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

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
///
/// # Returns
/// `Ok(())` if all checks pass, `Err(errors)` with all detected issues.
pub fn validate_input(tasks: &[Task], resources: &[Resource]) -> ValidationResult {
    let mut errors = Vec::new();

    // Collect resource IDs
    let mut resource_ids = HashSet::new();
    for r in resources {
        if !resource_ids.insert(r.id.as_str()) {
            errors.push(ValidationError::new(
                ValidationErrorKind::DuplicateId,
                format!("Duplicate resource ID: {}", r.id),
            ));
        }
    }

    // Collect task and activity IDs
    let mut task_ids = HashSet::new();
    let mut activity_ids = HashSet::new();

    for task in tasks {
        if !task_ids.insert(task.id.as_str()) {
            errors.push(ValidationError::new(
                ValidationErrorKind::DuplicateId,
                format!("Duplicate task ID: {}", task.id),
            ));
        }

        if task.activities.is_empty() {
            errors.push(ValidationError::new(
                ValidationErrorKind::EmptyTask,
                format!("Task '{}' has no activities", task.id),
            ));
        }

        for act in &task.activities {
            if !activity_ids.insert(act.id.as_str()) {
                errors.push(ValidationError::new(
                    ValidationErrorKind::DuplicateId,
                    format!("Duplicate activity ID: {}", act.id),
                ));
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
                            ValidationErrorKind::InvalidResourceReference,
                            format!(
                                "Activity '{}' references unknown resource '{}'",
                                act.id, cand
                            ),
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
                        ValidationErrorKind::InvalidPredecessor,
                        format!(
                            "Activity '{}' references unknown predecessor '{}'",
                            act.id, pred
                        ),
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
                ValidationErrorKind::CyclicDependency,
                format!("Circular dependency detected involving activity '{node}'"),
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
        assert!(errors
            .iter()
            .any(|e| e.kind == ValidationErrorKind::DuplicateId));
    }

    #[test]
    fn test_duplicate_resource_id() {
        let tasks = sample_tasks();
        let resources = vec![Resource::primary("M1"), Resource::primary("M1")];

        let errors = validate_input(&tasks, &resources).unwrap_err();
        assert!(errors
            .iter()
            .any(|e| e.kind == ValidationErrorKind::DuplicateId && e.message.contains("resource")));
    }

    #[test]
    fn test_empty_task() {
        let tasks = vec![Task::new("empty")]; // No activities
        let resources = sample_resources();

        let errors = validate_input(&tasks, &resources).unwrap_err();
        assert!(errors
            .iter()
            .any(|e| e.kind == ValidationErrorKind::EmptyTask));
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
        assert!(errors
            .iter()
            .any(|e| e.kind == ValidationErrorKind::InvalidResourceReference));
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
        assert!(errors
            .iter()
            .any(|e| e.kind == ValidationErrorKind::InvalidPredecessor));
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
            .any(|e| e.kind == ValidationErrorKind::CyclicDependency));
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
}
