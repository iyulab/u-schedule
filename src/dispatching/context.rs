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
/// [`SimpleScheduler`](crate::scheduler::SimpleScheduler) fills every field
/// at each decision point. A context built by hand may leave fields empty;
/// each rule then falls back to the task's static data (see the rule).
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

impl SchedulingContext {
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
