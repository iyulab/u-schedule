//! Scheduling context for dispatching rule evaluation.

use std::collections::HashMap;

/// Runtime scheduling state passed to dispatching rules.
///
/// Contains the current simulation clock, remaining work estimates,
/// resource utilization, and arrival times needed by context-aware rules.
///
/// All times are in milliseconds relative to the scheduling epoch (t=0).
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
    /// Average processing time across all tasks (for ATC normalization).
    pub average_processing_time: Option<f64>,
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
}
