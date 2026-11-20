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
    /// The activities that hold no resource, each with the time it takes
    /// (a cure, a wait — its task's next activity starts after it).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub time_only: Vec<TimeOnlySpan>,
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
    /// Which step of its task this activity is, counting from 1 in the order
    /// the task lists them.
    ///
    /// `None` when the solver that produced the assignment did not say. That
    /// is deliberately not a number: a schedule that reports every entry as
    /// step 1 is wrong in a way nothing downstream can detect, which is
    /// exactly how this field came to exist.
    pub sequence: Option<i32>,
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
    /// Which part of a [split](crate::models::Split) activity this is, and
    /// how much of its processing it does. `None`: the activity runs whole.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part: Option<crate::models::Part>,
}

/// When an activity that holds no resource runs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeOnlySpan {
    /// The activity.
    pub activity_id: String,
    /// Which step of its task it is, counting from 1 (as
    /// [`Assignment::sequence`]).
    pub sequence: Option<i32>,
    /// Its task.
    pub task_id: String,
    /// Start time (ms).
    pub start_ms: i64,
    /// End time (ms).
    pub end_ms: i64,
}

impl TimeOnlySpan {
    /// The span of `activity_id` of `task_id` over `[start_ms, end_ms)`.
    pub fn new(
        activity_id: impl Into<String>,
        task_id: impl Into<String>,
        start_ms: i64,
        end_ms: i64,
    ) -> Self {
        Self {
            activity_id: activity_id.into(),
            sequence: None,
            task_id: task_id.into(),
            start_ms,
            end_ms,
        }
    }

    /// Records which step of its task the activity is, counting from 1.
    pub fn with_sequence(mut self, sequence: i32) -> Self {
        self.sequence = Some(sequence);
        self
    }
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
    /// An activity's resource requirement has no or insufficient assignments.
    RequirementUnfilled,
    /// Activity scheduled outside its required time window.
    TimeWindowViolation,
    /// Synchronized activities do not share the same start time.
    SynchronizeViolation,
    /// An assignment names an activity the model does not have, under
    /// another task, or one more time than its requirements take units.
    UnknownActivity,
    /// An assignment is shorter than its setup, processing and teardown.
    DurationTooShort,
    /// An activity starts before its task's release time.
    EarlyStart,
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
            sequence: None,
            task_id: task_id.into(),
            resource_id: resource_id.into(),
            start_ms,
            end_ms,
            setup_ms: 0,
            part: None,
        }
    }

    /// Records which part of a split activity this is
    /// ([`part`](Self::part)).
    pub fn with_part(mut self, part: crate::models::Part) -> Self {
        self.part = Some(part);
        self
    }

    /// Records which step of its task this activity is, counting from 1.
    pub fn with_sequence(mut self, sequence: i32) -> Self {
        self.sequence = Some(sequence);
        self
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

    /// Creates an unfilled-requirement violation.
    pub fn requirement_unfilled(
        activity_id: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            violation_type: ViolationType::RequirementUnfilled,
            entity_id: activity_id.into(),
            message: message.into(),
            severity: 95,
        }
    }

    /// Creates a resource-unavailable (outside calendar) violation.
    pub fn resource_unavailable(
        resource_id: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            violation_type: ViolationType::ResourceUnavailable,
            entity_id: resource_id.into(),
            message: message.into(),
            severity: 85,
        }
    }

    /// Creates a skill-mismatch violation.
    pub fn skill_mismatch(resource_id: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            violation_type: ViolationType::SkillMismatch,
            entity_id: resource_id.into(),
            message: message.into(),
            severity: 70,
        }
    }

    /// Creates a time-window violation.
    pub fn time_window(activity_id: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            violation_type: ViolationType::TimeWindowViolation,
            entity_id: activity_id.into(),
            message: message.into(),
            severity: 75,
        }
    }

    /// Creates an unknown-activity violation.
    pub fn unknown_activity(activity_id: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            violation_type: ViolationType::UnknownActivity,
            entity_id: activity_id.into(),
            message: message.into(),
            severity: 95,
        }
    }

    /// Creates a too-short-assignment violation.
    pub fn duration_too_short(activity_id: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            violation_type: ViolationType::DurationTooShort,
            entity_id: activity_id.into(),
            message: message.into(),
            severity: 90,
        }
    }

    /// Creates an early-start (before release) violation.
    pub fn early_start(activity_id: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            violation_type: ViolationType::EarlyStart,
            entity_id: activity_id.into(),
            message: message.into(),
            severity: 90,
        }
    }

    /// Creates a synchronization violation.
    pub fn synchronize(entity_id: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            violation_type: ViolationType::SynchronizeViolation,
            entity_id: entity_id.into(),
            message: message.into(),
            severity: 75,
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

    /// Makespan: latest end across all assignments and time-only spans (ms).
    pub fn makespan_ms(&self) -> i64 {
        self.assignments
            .iter()
            .map(|a| a.end_ms)
            .chain(self.time_only.iter().map(|s| s.end_ms))
            .max()
            .unwrap_or(0)
    }

    /// The time-only span of an activity ([`time_only`](Self::time_only)).
    ///
    /// Looks through every span; to ask about many activities, build an
    /// [`index`](Self::index) once.
    pub fn time_only_for_activity(&self, activity_id: &str) -> Option<&TimeOnlySpan> {
        self.time_only.iter().find(|s| s.activity_id == activity_id)
    }

    /// Finds the first assignment for a given activity.
    ///
    /// A multi-resource activity has one assignment per held resource;
    /// this returns the first in insertion order. Use
    /// [`assignments_for_activity_all`](Self::assignments_for_activity_all)
    /// for the full set.
    pub fn assignment_for_activity(&self, activity_id: &str) -> Option<&Assignment> {
        self.assignments
            .iter()
            .find(|a| a.activity_id == activity_id)
    }

    /// Returns all assignments for a given activity (one per held resource).
    ///
    /// Looks through every assignment; to ask about many activities, build an
    /// [`index`](Self::index) once.
    pub fn assignments_for_activity_all(&self, activity_id: &str) -> Vec<&Assignment> {
        self.assignments
            .iter()
            .filter(|a| a.activity_id == activity_id)
            .collect()
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

    /// Completion time for a task (latest end of its assignments and
    /// time-only spans).
    ///
    /// Looks through every assignment; to ask about many tasks, build an
    /// [`index`](Self::index) once.
    pub fn task_completion_time(&self, task_id: &str) -> Option<i64> {
        self.assignments_for_task(task_id)
            .iter()
            .map(|a| a.end_ms)
            .chain(
                self.time_only
                    .iter()
                    .filter(|s| s.task_id == task_id)
                    .map(|s| s.end_ms),
            )
            .max()
    }

    /// Looks the schedule's assignments and time-only spans up by activity
    /// and by task: built once in one pass over them, then each lookup takes
    /// constant time, where the lookups of the schedule itself look through
    /// everything (asking about every task or activity of `n` of them is
    /// then linear, not quadratic).
    ///
    /// It borrows the schedule, so the schedule cannot change while it is
    /// held.
    pub fn index(&self) -> ScheduleIndex<'_> {
        let mut by_activity: HashMap<&str, Vec<&Assignment>> = HashMap::new();
        let mut completion: HashMap<&str, i64> = HashMap::new();
        for a in &self.assignments {
            by_activity.entry(&a.activity_id).or_default().push(a);
            completion
                .entry(&a.task_id)
                .and_modify(|end| *end = (*end).max(a.end_ms))
                .or_insert(a.end_ms);
        }
        let mut time_only: HashMap<&str, &TimeOnlySpan> = HashMap::new();
        for span in &self.time_only {
            // The first of an activity's spans, as `time_only_for_activity` finds it.
            time_only.entry(&span.activity_id).or_insert(span);
            completion
                .entry(&span.task_id)
                .and_modify(|end| *end = (*end).max(span.end_ms))
                .or_insert(span.end_ms);
        }
        ScheduleIndex {
            by_activity,
            time_only,
            completion,
        }
    }

    /// Number of assignments.
    pub fn assignment_count(&self) -> usize {
        self.assignments.len()
    }
}

/// A schedule's assignments and time-only spans looked up by activity and by
/// task, in constant time ([`Schedule::index`]).
///
/// The answers are those of the same-named lookups of [`Schedule`].
#[derive(Debug, Clone)]
pub struct ScheduleIndex<'a> {
    by_activity: HashMap<&'a str, Vec<&'a Assignment>>,
    time_only: HashMap<&'a str, &'a TimeOnlySpan>,
    completion: HashMap<&'a str, i64>,
}

impl<'a> ScheduleIndex<'a> {
    /// All assignments of an activity (one per held resource, and one per
    /// part), in the order they were added.
    pub fn assignments_for_activity_all(&self, activity_id: &str) -> &[&'a Assignment] {
        self.by_activity.get(activity_id).map_or(&[], Vec::as_slice)
    }

    /// The time-only span of an activity.
    pub fn time_only_for_activity(&self, activity_id: &str) -> Option<&'a TimeOnlySpan> {
        self.time_only.get(activity_id).copied()
    }

    /// Completion time of a task: the latest end of its assignments and
    /// time-only spans.
    pub fn task_completion_time(&self, task_id: &str) -> Option<i64> {
        self.completion.get(task_id).copied()
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
    fn test_new_violation_ctors() {
        let v = Violation::requirement_unfilled("O1", "Mold requirement unfilled");
        assert_eq!(v.violation_type, ViolationType::RequirementUnfilled);
        let v = Violation::resource_unavailable("M1", "outside calendar");
        assert_eq!(v.violation_type, ViolationType::ResourceUnavailable);
        let v = Violation::skill_mismatch("M1", "missing welding");
        assert_eq!(v.violation_type, ViolationType::SkillMismatch);
        let v = Violation::time_window("O1", "outside [5000,8000)");
        assert_eq!(v.violation_type, ViolationType::TimeWindowViolation);
        let v = Violation::synchronize("O1", "start differs");
        assert_eq!(v.violation_type, ViolationType::SynchronizeViolation);
    }

    #[test]
    fn test_assignments_for_activity_all() {
        let mut s = sample_schedule();
        s.add_assignment(Assignment::new("O1", "J1", "T1", 0, 5000));
        assert_eq!(s.assignments_for_activity_all("O1").len(), 2);
        assert_eq!(s.assignments_for_activity_all("O2").len(), 1);
        assert!(s.assignments_for_activity_all("O99").is_empty());
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

    /// The index answers as the schedule's own lookups do, wherever they differ in how they
    /// could: an activity held on several resources or placed in parts, two time-only spans
    /// of one activity (the first one counts), a task that has only time-only spans, one that
    /// has only assignments, and an id the schedule has never heard of.
    #[test]
    fn the_index_answers_as_the_schedule_does() {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move |n: u64| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) % n
        };
        for _ in 0..400 {
            let mut s = Schedule::new();
            for _ in 0..next(60) {
                let (task, activity) = (next(7), next(12));
                let start = next(5_000) as i64;
                let mut a = Assignment::new(
                    format!("A{activity}"),
                    format!("T{task}"),
                    format!("M{}", next(4)),
                    start,
                    start + next(900) as i64,
                );
                if next(5) == 0 {
                    a = a.with_part(crate::models::Part {
                        index: next(3) as i32,
                        work_ms: 100,
                    });
                }
                s.add_assignment(a);
            }
            for _ in 0..next(10) {
                let start = next(5_000) as i64;
                s.time_only.push(TimeOnlySpan::new(
                    format!("A{}", 20 + next(6)),
                    format!("T{}", next(9)),
                    start,
                    start + next(900) as i64,
                ));
            }
            let index = s.index();
            for activity in (0..32).map(|k| format!("A{k}")) {
                let want: Vec<*const Assignment> = s
                    .assignments_for_activity_all(&activity)
                    .into_iter()
                    .map(|a| a as *const Assignment)
                    .collect();
                let got: Vec<*const Assignment> = index
                    .assignments_for_activity_all(&activity)
                    .iter()
                    .map(|&a| a as *const Assignment)
                    .collect();
                assert_eq!(got, want, "assignments of {activity}, in order");
                assert_eq!(
                    index.time_only_for_activity(&activity),
                    s.time_only_for_activity(&activity),
                    "time-only span of {activity}"
                );
            }
            for task in (0..12).map(|k| format!("T{k}")) {
                assert_eq!(
                    index.task_completion_time(&task),
                    s.task_completion_time(&task),
                    "completion of {task}"
                );
            }
            assert!(index.assignments_for_activity_all("nothing").is_empty());
            assert_eq!(index.time_only_for_activity("nothing"), None);
            assert_eq!(index.task_completion_time("nothing"), None);
        }
    }
}
