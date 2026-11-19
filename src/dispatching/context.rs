//! Scheduling context for dispatching rule evaluation.

use std::collections::HashMap;

use crate::models::{Activity, Task};

/// Runtime scheduling state passed to dispatching rules.
///
/// Contains the current simulation clock, remaining work estimates,
/// resource utilization, and arrival times needed by context-aware rules.
///
/// All times are in milliseconds relative to the scheduling epoch (t=0).
///
/// [`SimpleScheduler`](crate::scheduler::SimpleScheduler) fills, at each
/// decision point, the fields the rules deciding say they read
/// ([`DispatchingRule::reads`](super::DispatchingRule::reads)) — every field
/// when a rule does not say, or when a policy chooses the rules. A context
/// built by hand may leave fields empty; each rule then falls back to the
/// task's static data (see the rule).
#[derive(Debug, Clone, Default)]
pub struct SchedulingContext {
    /// Current simulation time (ms).
    pub current_time_ms: i64,
    /// Remaining processing work per task (task_id → ms).
    pub remaining_work: HashMap<String, i64>,
    /// Queue length at next resource per task.
    pub next_queue_length: HashMap<String, usize>,
    /// Current resource utilization (resource_id → 0.0..1.0).
    pub resource_utilization: HashMap<String, f64>,
    /// Task arrival times (task_id → ms).
    pub arrival_times: HashMap<String, i64>,
    /// Average imminent processing time of the tasks competing at this
    /// decision (for ATC normalization).
    pub average_processing_time: Option<f64>,
    /// Index into `task.activities` of the activity each task would start
    /// next (task_id → index). Absent: the task is treated as a whole.
    pub imminent_activity: HashMap<String, usize>,
}

/// The parts of a [`SchedulingContext`] a rule reads (the clock, `current_time_ms`, is
/// always there).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ContextFields(u8);

impl ContextFields {
    /// Nothing but the clock.
    pub const NONE: Self = Self(0);
    /// [`SchedulingContext::remaining_work`].
    pub const REMAINING_WORK: Self = Self(1);
    /// [`SchedulingContext::next_queue_length`].
    pub const NEXT_QUEUE: Self = Self(1 << 1);
    /// [`SchedulingContext::resource_utilization`].
    pub const UTILIZATION: Self = Self(1 << 2);
    /// [`SchedulingContext::arrival_times`].
    pub const ARRIVALS: Self = Self(1 << 3);
    /// [`SchedulingContext::average_processing_time`].
    pub const AVERAGE_PROCESSING_TIME: Self = Self(1 << 4);
    /// [`SchedulingContext::imminent_activity`] (and what is read through it:
    /// [`SchedulingContext::imminent`], [`processing_time_ms`](SchedulingContext::processing_time_ms),
    /// [`remaining_operations`](SchedulingContext::remaining_operations)).
    pub const IMMINENT_ACTIVITY: Self = Self(1 << 5);
    /// Every field.
    pub const ALL: Self = Self(0b11_1111);

    /// These fields and those.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether every field of `other` is among these.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Whether any field of `other` is among these.
    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }
}

impl SchedulingContext {
    /// The context with only `fields` kept (the rest empty, as in a context built by hand).
    #[cfg(test)]
    pub(crate) fn restricted_to(&self, fields: ContextFields) -> Self {
        let mut kept = Self::at_time(self.current_time_ms);
        if fields.contains(ContextFields::REMAINING_WORK) {
            kept.remaining_work = self.remaining_work.clone();
        }
        if fields.contains(ContextFields::NEXT_QUEUE) {
            kept.next_queue_length = self.next_queue_length.clone();
        }
        if fields.contains(ContextFields::UTILIZATION) {
            kept.resource_utilization = self.resource_utilization.clone();
        }
        if fields.contains(ContextFields::ARRIVALS) {
            kept.arrival_times = self.arrival_times.clone();
        }
        if fields.contains(ContextFields::AVERAGE_PROCESSING_TIME) {
            kept.average_processing_time = self.average_processing_time;
        }
        if fields.contains(ContextFields::IMMINENT_ACTIVITY) {
            kept.imminent_activity = self.imminent_activity.clone();
        }
        kept
    }

    /// Creates a context at the given time.
    pub fn at_time(current_time_ms: i64) -> Self {
        Self {
            current_time_ms,
            ..Default::default()
        }
    }

    /// Sets remaining work for a task.
    pub fn with_remaining_work(mut self, task_id: impl Into<String>, ms: i64) -> Self {
        self.remaining_work.insert(task_id.into(), ms);
        self
    }

    /// Sets queue length for a task.
    pub fn with_next_queue(mut self, task_id: impl Into<String>, length: usize) -> Self {
        self.next_queue_length.insert(task_id.into(), length);
        self
    }

    /// Sets resource utilization.
    pub fn with_utilization(mut self, resource_id: impl Into<String>, load: f64) -> Self {
        self.resource_utilization.insert(resource_id.into(), load);
        self
    }

    /// Sets arrival time for a task.
    pub fn with_arrival_time(mut self, task_id: impl Into<String>, time_ms: i64) -> Self {
        self.arrival_times.insert(task_id.into(), time_ms);
        self
    }

    /// Sets the average processing time.
    pub fn with_average_processing_time(mut self, avg_ms: f64) -> Self {
        self.average_processing_time = Some(avg_ms);
        self
    }

    /// Sets the activity a task would start next.
    pub fn with_imminent_activity(mut self, task_id: impl Into<String>, index: usize) -> Self {
        self.imminent_activity.insert(task_id.into(), index);
        self
    }

    /// The activity `task` would start next, when the context names one.
    pub fn imminent<'a>(&self, task: &'a Task) -> Option<&'a Activity> {
        self.imminent_activity
            .get(&task.id)
            .and_then(|&i| task.activities.get(i))
    }

    /// Processing time of the decision at hand: the imminent activity's
    /// duration (setup + process + teardown), or the whole task's when the
    /// context names no imminent activity.
    pub fn processing_time_ms(&self, task: &Task) -> i64 {
        self.imminent(task)
            .map(|a| a.duration.total_ms())
            .unwrap_or_else(|| task.total_duration_ms())
    }

    /// Work left on `task` (this activity and the ones after it), or the
    /// whole task's duration when the context does not say.
    pub fn remaining_work_ms(&self, task: &Task) -> i64 {
        self.remaining_work
            .get(&task.id)
            .copied()
            .unwrap_or_else(|| task.total_duration_ms())
    }

    /// Activities left on `task`, the imminent one included.
    pub fn remaining_operations(&self, task: &Task) -> usize {
        let done = self.imminent_activity.get(&task.id).copied().unwrap_or(0);
        task.activity_count().saturating_sub(done)
    }
}
