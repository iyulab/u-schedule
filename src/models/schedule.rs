//! Schedule (solution) model.
//!
//! A schedule is a complete assignment of activities to resources and
//! time slots. It may include constraint violations for infeasible
//! or suboptimal solutions.
//!
//! # Reference
//! Pinedo (2016), "Scheduling: Theory, Algorithms, and Systems", Ch. 3

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A complete schedule (solution to a scheduling problem).
///
/// Contains activity-resource-time assignments and any constraint violations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Schedule {
    /// Activity assignments (activity → resource × time).
    pub assignments: Vec<Assignment>,
    /// Constraint violations detected in this schedule.
    pub violations: Vec<Violation>,
}

/// An activity-resource-time assignment.
///
/// Records that a specific activity is scheduled on a specific resource
/// during a specific time interval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assignment {
    /// Assigned activity ID.
    pub activity_id: String,
    /// Parent task ID (denormalized for query convenience).
    pub task_id: String,
    /// Assigned resource ID.
    pub resource_id: String,
    /// Start time (ms).
    pub start_ms: i64,
    /// End time (ms).
    pub end_ms: i64,
    /// Setup time portion (ms). Included in [start_ms, start_ms + setup_ms).
    pub setup_ms: i64,
}

/// A constraint violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Violation {
    /// Type of violation.
    pub violation_type: ViolationType,
    /// Related entity ID (task, resource, or activity).
    pub entity_id: String,
    /// Human-readable description.
    pub message: String,
    /// Severity (0-100, higher = worse).
    pub severity: i32,
}

/// Classification of constraint violations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationType {
    /// Task completed after its deadline.
    DeadlineMiss,
    /// Resource allocated beyond its capacity.
    CapacityExceeded,
    /// Activity started before its predecessor finished.
    PrecedenceViolation,
    /// Activity scheduled when resource is unavailable.
    ResourceUnavailable,
    /// Resource lacks a required skill.
    SkillMismatch,
    /// Domain-specific violation.
    Custom(String),
}

impl Assignment {
    /// Creates a new assignment.
    pub fn new(
        activity_id: impl Into<String>,
        task_id: impl Into<String>,
        resource_id: impl Into<String>,
        start_ms: i64,
        end_ms: i64,
    ) -> Self {
        Self {
            activity_id: activity_id.into(),
            task_id: task_id.into(),
            resource_id: resource_id.into(),
            start_ms,
            end_ms,
            setup_ms: 0,
        }
    }

    /// Sets the setup time.
    pub fn with_setup(mut self, setup_ms: i64) -> Self {
        self.setup_ms = setup_ms;
        self
    }

    /// Total duration (end - start) in ms.
    #[inline]
    pub fn duration_ms(&self) -> i64 {
        self.end_ms - self.start_ms
    }

    /// Processing duration excluding setup (ms).
    #[inline]
    pub fn process_ms(&self) -> i64 {
        self.duration_ms() - self.setup_ms
    }
}

impl Violation {
    /// Creates a deadline miss violation.
    pub fn deadline_miss(task_id: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            violation_type: ViolationType::DeadlineMiss,
            entity_id: task_id.into(),
            message: message.into(),
            severity: 80,
        }
    }

    /// Creates a capacity exceeded violation.
    pub fn capacity_exceeded(resource_id: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            violation_type: ViolationType::CapacityExceeded,
            entity_id: resource_id.into(),
            message: message.into(),
            severity: 90,
        }
    }

    /// Creates a precedence violation.
    pub fn precedence_violation(
        activity_id: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            violation_type: ViolationType::PrecedenceViolation,
            entity_id: activity_id.into(),
            message: message.into(),
            severity: 95,
        }
    }
}

impl Schedule {
    /// Creates an empty schedule.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an assignment.
    pub fn add_assignment(&mut self, assignment: Assignment) {
        self.assignments.push(assignment);
    }

    /// Adds a violation.
    pub fn add_violation(&mut self, violation: Violation) {
        self.violations.push(violation);
    }

    /// Whether the schedule has no violations.
    pub fn is_valid(&self) -> bool {
        self.violations.is_empty()
    }

    /// Makespan: latest end time across all assignments (ms).
    pub fn makespan_ms(&self) -> i64 {
        self.assignments.iter().map(|a| a.end_ms).max().unwrap_or(0)
    }

    /// Finds the assignment for a given activity.
    pub fn assignment_for_activity(&self, activity_id: &str) -> Option<&Assignment> {
        self.assignments
            .iter()
            .find(|a| a.activity_id == activity_id)
    }

    /// Returns all assignments for a given task.
    pub fn assignments_for_task(&self, task_id: &str) -> Vec<&Assignment> {
        self.assignments
            .iter()
            .filter(|a| a.task_id == task_id)
            .collect()
    }

    /// Returns all assignments for a given resource.
    pub fn assignments_for_resource(&self, resource_id: &str) -> Vec<&Assignment> {
        self.assignments
            .iter()
            .filter(|a| a.resource_id == resource_id)
            .collect()
    }

    /// Computes resource utilization: busy_time / horizon.
    ///
    /// Returns `None` if `horizon_ms` is zero.
    pub fn resource_utilization(&self, resource_id: &str, horizon_ms: i64) -> Option<f64> {
        if horizon_ms <= 0 {
            return None;
        }
        let busy: i64 = self
            .assignments_for_resource(resource_id)
            .iter()
            .map(|a| a.duration_ms())
            .sum();
        Some(busy as f64 / horizon_ms as f64)
    }

    /// Computes utilization for all resources that have assignments.
    ///
    /// Uses makespan as the horizon.
    pub fn all_utilizations(&self) -> HashMap<String, f64> {
        let horizon = self.makespan_ms();
        if horizon <= 0 {
            return HashMap::new();
        }

        let mut resource_busy: HashMap<String, i64> = HashMap::new();
        for a in &self.assignments {
            *resource_busy.entry(a.resource_id.clone()).or_insert(0) += a.duration_ms();
        }

        resource_busy
            .into_iter()
            .map(|(id, busy)| (id, busy as f64 / horizon as f64))
            .collect()
    }

    /// Completion time for a task (latest end of its assignments).
    pub fn task_completion_time(&self, task_id: &str) -> Option<i64> {
        self.assignments_for_task(task_id)
            .iter()
            .map(|a| a.end_ms)
            .max()
    }

    /// Number of assignments.
    pub fn assignment_count(&self) -> usize {
        self.assignments.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_schedule() -> Schedule {
        let mut s = Schedule::new();
        s.add_assignment(Assignment::new("O1", "J1", "M1", 0, 5000).with_setup(500));
        s.add_assignment(Assignment::new("O2", "J1", "M2", 1000, 4000));
        s.add_assignment(Assignment::new("O3", "J2", "M1", 5000, 8000));
        s
    }

    #[test]
    fn test_schedule_makespan() {
        let s = sample_schedule();
        assert_eq!(s.makespan_ms(), 8000);
    }

    #[test]
    fn test_schedule_is_valid() {
        let s = sample_schedule();
        assert!(s.is_valid());

        let mut s2 = sample_schedule();
        s2.add_violation(Violation::deadline_miss("J1", "Late by 1000ms"));
        assert!(!s2.is_valid());
    }

    #[test]
    fn test_assignment_duration() {
        let a = Assignment::new("O1", "J1", "M1", 0, 5000).with_setup(500);
        assert_eq!(a.duration_ms(), 5000);
        assert_eq!(a.process_ms(), 4500);
        assert_eq!(a.setup_ms, 500);
    }

    #[test]
    fn test_assignment_for_activity() {
        let s = sample_schedule();
        let a = s.assignment_for_activity("O1").unwrap();
        assert_eq!(a.resource_id, "M1");
        assert!(s.assignment_for_activity("O99").is_none());
    }

    #[test]
    fn test_assignments_for_task() {
        let s = sample_schedule();
        let j1 = s.assignments_for_task("J1");
        assert_eq!(j1.len(), 2);
        let j2 = s.assignments_for_task("J2");
        assert_eq!(j2.len(), 1);
    }

    #[test]
    fn test_assignments_for_resource() {
        let s = sample_schedule();
        let m1 = s.assignments_for_resource("M1");
        assert_eq!(m1.len(), 2); // O1 and O3
    }

    #[test]
    fn test_resource_utilization() {
        let s = sample_schedule();
        // M1: busy 5000 + 3000 = 8000 over horizon 8000 → 1.0
        let util = s.resource_utilization("M1", 8000).unwrap();
        assert!((util - 1.0).abs() < 1e-10);

        // M2: busy 3000 over horizon 8000 → 0.375
        let util2 = s.resource_utilization("M2", 8000).unwrap();
        assert!((util2 - 0.375).abs() < 1e-10);
    }

    #[test]
    fn test_task_completion_time() {
        let s = sample_schedule();
        assert_eq!(s.task_completion_time("J1"), Some(5000)); // max(5000, 4000)
        assert_eq!(s.task_completion_time("J2"), Some(8000));
        assert_eq!(s.task_completion_time("J99"), None);
    }

    #[test]
    fn test_all_utilizations() {
        let s = sample_schedule();
        let utils = s.all_utilizations();
        assert!((utils["M1"] - 1.0).abs() < 1e-10);
        assert!((utils["M2"] - 0.375).abs() < 1e-10);
    }

    #[test]
    fn test_empty_schedule() {
        let s = Schedule::new();
        assert_eq!(s.makespan_ms(), 0);
        assert!(s.is_valid());
        assert_eq!(s.assignment_count(), 0);
    }

    #[test]
    fn test_violation_factories() {
        let v1 = Violation::deadline_miss("J1", "Late");
        assert_eq!(v1.violation_type, ViolationType::DeadlineMiss);
        assert_eq!(v1.entity_id, "J1");

        let v2 = Violation::capacity_exceeded("M1", "Over capacity");
        assert_eq!(v2.violation_type, ViolationType::CapacityExceeded);

        let v3 = Violation::precedence_violation("O2", "Started before O1");
        assert_eq!(v3.violation_type, ViolationType::PrecedenceViolation);
    }
}
