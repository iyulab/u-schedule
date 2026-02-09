//! Activity (operation) model.
//!
//! An activity is the smallest schedulable unit of work. It belongs to a task,
//! requires resources, has a duration, and may have precedence constraints.
//!
//! # Duration Model
//!
//! Each activity has three time components:
//! - **Setup**: Preparation time (may depend on previous activity via TransitionMatrix)
//! - **Process**: Core work time
//! - **Teardown**: Cleanup/cooldown time
//!
//! # Reference
//! Pinedo (2016), "Scheduling: Theory, Algorithms, and Systems", Ch. 2

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// An activity (operation) to be scheduled.
///
/// Represents a single processing step that requires one or more resources
/// for a specified duration. Activities within a task are linked by
/// precedence constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Activity {
    /// Unique activity identifier.
    pub id: String,
    /// Parent task identifier.
    pub task_id: String,
    /// Position within the task (0-indexed).
    pub sequence: i32,
    /// Time required to complete this activity.
    pub duration: ActivityDuration,
    /// Resources needed (type + quantity + candidates).
    pub resource_requirements: Vec<ResourceRequirement>,
    /// IDs of activities that must complete before this one starts.
    pub predecessors: Vec<String>,
    /// Whether this activity can be preempted and resumed later.
    pub splittable: bool,
    /// Minimum duration (ms) of each split segment.
    pub min_split_ms: i64,
    /// Domain-specific metadata.
    pub attributes: HashMap<String, String>,
}

impl Activity {
    /// Creates a new activity.
    pub fn new(id: impl Into<String>, task_id: impl Into<String>, sequence: i32) -> Self {
        Self {
            id: id.into(),
            task_id: task_id.into(),
            sequence,
            duration: ActivityDuration::default(),
            resource_requirements: Vec::new(),
            predecessors: Vec::new(),
            splittable: false,
            min_split_ms: 0,
            attributes: HashMap::new(),
        }
    }

    /// Sets the duration.
    pub fn with_duration(mut self, duration: ActivityDuration) -> Self {
        self.duration = duration;
        self
    }

    /// Sets the processing time (setup=0, teardown=0).
    pub fn with_process_time(mut self, process_ms: i64) -> Self {
        self.duration = ActivityDuration::fixed(process_ms);
        self
    }

    /// Adds a resource requirement.
    pub fn with_requirement(mut self, req: ResourceRequirement) -> Self {
        self.resource_requirements.push(req);
        self
    }

    /// Adds a predecessor activity ID.
    pub fn with_predecessor(mut self, predecessor_id: impl Into<String>) -> Self {
        self.predecessors.push(predecessor_id.into());
        self
    }

    /// Enables preemption with a minimum split size.
    pub fn with_splitting(mut self, min_split_ms: i64) -> Self {
        self.splittable = true;
        self.min_split_ms = min_split_ms;
        self
    }

    /// Returns all candidate resource IDs across all requirements.
    pub fn candidate_resources(&self) -> Vec<&str> {
        self.resource_requirements
            .iter()
            .flat_map(|r| r.candidates.iter().map(|s| s.as_str()))
            .collect()
    }
}

/// Time components of an activity.
///
/// # Components
/// - **Setup**: Preparation before processing (e.g., machine changeover).
///   May be overridden by `TransitionMatrix` for sequence-dependent setups.
/// - **Process**: Core work time (the actual operation).
/// - **Teardown**: Cleanup after processing (e.g., cooling, inspection).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityDuration {
    /// Setup/preparation time (ms).
    pub setup_ms: i64,
    /// Core processing time (ms).
    pub process_ms: i64,
    /// Teardown/cleanup time (ms).
    pub teardown_ms: i64,
}

impl ActivityDuration {
    /// Creates a duration with all three components.
    pub fn new(setup_ms: i64, process_ms: i64, teardown_ms: i64) -> Self {
        Self {
            setup_ms,
            process_ms,
            teardown_ms,
        }
    }

    /// Creates a fixed-duration activity (setup=0, teardown=0).
    pub fn fixed(process_ms: i64) -> Self {
        Self::new(0, process_ms, 0)
    }

    /// Total duration (setup + process + teardown).
    pub fn total_ms(&self) -> i64 {
        self.setup_ms + self.process_ms + self.teardown_ms
    }
}

impl Default for ActivityDuration {
    fn default() -> Self {
        Self::fixed(0)
    }
}

/// A resource requirement for an activity.
///
/// Specifies what type and quantity of resources are needed,
/// with optional candidate filtering and skill requirements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirement {
    /// Required resource type (e.g., "Machine", "Operator").
    pub resource_type: String,
    /// Number of resource units needed simultaneously.
    pub quantity: i32,
    /// Specific resource IDs that can fulfill this requirement.
    /// Empty = any resource of the correct type.
    pub candidates: Vec<String>,
    /// Required skills (matched against `Resource.skills`).
    pub required_skills: Vec<String>,
}

impl ResourceRequirement {
    /// Creates a new requirement for one unit of a resource type.
    pub fn new(resource_type: impl Into<String>) -> Self {
        Self {
            resource_type: resource_type.into(),
            quantity: 1,
            candidates: Vec::new(),
            required_skills: Vec::new(),
        }
    }

    /// Sets the required quantity.
    pub fn with_quantity(mut self, quantity: i32) -> Self {
        self.quantity = quantity;
        self
    }

    /// Adds candidate resource IDs.
    pub fn with_candidates(mut self, candidates: Vec<String>) -> Self {
        self.candidates = candidates;
        self
    }

    /// Adds a required skill.
    pub fn with_skill(mut self, skill: impl Into<String>) -> Self {
        self.required_skills.push(skill.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activity_builder() {
        let act = Activity::new("O1", "J1", 0)
            .with_duration(ActivityDuration::new(100, 500, 50))
            .with_requirement(ResourceRequirement::new("Machine").with_quantity(1))
            .with_predecessor("O0")
            .with_splitting(200);

        assert_eq!(act.id, "O1");
        assert_eq!(act.task_id, "J1");
        assert_eq!(act.sequence, 0);
        assert_eq!(act.duration.total_ms(), 650);
        assert_eq!(act.resource_requirements.len(), 1);
        assert_eq!(act.predecessors, vec!["O0"]);
        assert!(act.splittable);
        assert_eq!(act.min_split_ms, 200);
    }

    #[test]
    fn test_activity_duration_fixed() {
        let d = ActivityDuration::fixed(1000);
        assert_eq!(d.setup_ms, 0);
        assert_eq!(d.process_ms, 1000);
        assert_eq!(d.teardown_ms, 0);
        assert_eq!(d.total_ms(), 1000);
    }

    #[test]
    fn test_activity_duration_components() {
        let d = ActivityDuration::new(100, 500, 50);
        assert_eq!(d.total_ms(), 650);
    }

    #[test]
    fn test_resource_requirement() {
        let req = ResourceRequirement::new("CNC")
            .with_quantity(2)
            .with_candidates(vec!["M1".into(), "M2".into(), "M3".into()])
            .with_skill("milling");

        assert_eq!(req.resource_type, "CNC");
        assert_eq!(req.quantity, 2);
        assert_eq!(req.candidates.len(), 3);
        assert_eq!(req.required_skills, vec!["milling"]);
    }

    #[test]
    fn test_candidate_resources() {
        let act = Activity::new("O1", "J1", 0)
            .with_requirement(
                ResourceRequirement::new("Machine").with_candidates(vec!["M1".into(), "M2".into()]),
            )
            .with_requirement(
                ResourceRequirement::new("Operator").with_candidates(vec!["W1".into()]),
            );

        let candidates = act.candidate_resources();
        assert_eq!(candidates.len(), 3);
        assert!(candidates.contains(&"M1"));
        assert!(candidates.contains(&"W1"));
    }
}
