//! Task (job) model.
//!
//! A task represents a unit of work to be scheduled, consisting of
//! one or more activities (operations) with precedence constraints.
//!
//! # Reference
//! Pinedo (2016), "Scheduling: Theory, Algorithms, and Systems", Ch. 1

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::Activity;

/// A task (job) to be scheduled.
///
/// Contains one or more activities and scheduling metadata (priority, deadlines).
/// Activities within a task may have precedence constraints forming a DAG.
///
/// # Time Representation
/// All times are in milliseconds relative to a scheduling epoch (t=0).
/// The consumer defines what t=0 means (e.g., shift start, midnight UTC).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Unique task identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Task category (for transition matrix lookups and grouping).
    pub category: String,
    /// Scheduling priority (higher = more important).
    pub priority: i32,
    /// Latest completion time (ms). `None` = no deadline.
    pub deadline: Option<i64>,
    /// Earliest start time (ms). `None` = available immediately.
    pub release_time: Option<i64>,
    /// Activities (operations) that compose this task.
    pub activities: Vec<Activity>,
    /// Domain-specific key-value metadata.
    pub attributes: HashMap<String, String>,
}

impl Task {
    /// Creates a new task with the given ID.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: String::new(),
            category: String::new(),
            priority: 0,
            deadline: None,
            release_time: None,
            activities: Vec::new(),
            attributes: HashMap::new(),
        }
    }

    /// Sets the task name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Sets the task category.
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = category.into();
        self
    }

    /// Sets the scheduling priority.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Sets the deadline (latest completion time in ms).
    pub fn with_deadline(mut self, deadline_ms: i64) -> Self {
        self.deadline = Some(deadline_ms);
        self
    }

    /// Sets the release time (earliest start time in ms).
    pub fn with_release_time(mut self, release_ms: i64) -> Self {
        self.release_time = Some(release_ms);
        self
    }

    /// Adds an activity to this task.
    pub fn with_activity(mut self, activity: Activity) -> Self {
        self.activities.push(activity);
        self
    }

    /// Adds a domain-specific attribute.
    pub fn with_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    /// Total processing duration across all activities (ms).
    pub fn total_duration_ms(&self) -> i64 {
        self.activities.iter().map(|a| a.duration.total_ms()).sum()
    }

    /// Whether this task has any activities.
    pub fn has_activities(&self) -> bool {
        !self.activities.is_empty()
    }

    /// Number of activities.
    pub fn activity_count(&self) -> usize {
        self.activities.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ActivityDuration;

    #[test]
    fn test_task_builder() {
        let task = Task::new("J1")
            .with_name("Job 1")
            .with_category("TypeA")
            .with_priority(10)
            .with_deadline(100_000)
            .with_release_time(0)
            .with_attribute("customer", "ACME");

        assert_eq!(task.id, "J1");
        assert_eq!(task.name, "Job 1");
        assert_eq!(task.category, "TypeA");
        assert_eq!(task.priority, 10);
        assert_eq!(task.deadline, Some(100_000));
        assert_eq!(task.release_time, Some(0));
        assert_eq!(task.attributes.get("customer"), Some(&"ACME".to_string()));
    }

    #[test]
    fn test_task_total_duration() {
        let task = Task::new("J1")
            .with_activity(
                Activity::new("O1", "J1", 0).with_duration(ActivityDuration::fixed(1000)),
            )
            .with_activity(
                Activity::new("O2", "J1", 1).with_duration(ActivityDuration::fixed(2000)),
            );

        assert_eq!(task.total_duration_ms(), 3000);
        assert_eq!(task.activity_count(), 2);
        assert!(task.has_activities());
    }

    #[test]
    fn test_task_empty() {
        let task = Task::new("empty");
        assert_eq!(task.total_duration_ms(), 0);
        assert!(!task.has_activities());
    }
}
