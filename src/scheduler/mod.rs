//! Greedy schedulers and KPI evaluation.
//!
//! Provides a simple priority-driven scheduler and schedule quality metrics.
//!
//! # Algorithm
//!
//! `SimpleScheduler` uses a greedy, priority-driven, earliest-available-resource
//! heuristic. It is not optimal, but provides fast baseline solutions.
//!
//! # KPI
//!
//! `ScheduleKpi` computes standard scheduling metrics: makespan, tardiness,
//! on-time rate, utilization, and flow time.
//!
//! # References
//!
//! - Pinedo (2016), "Scheduling: Theory, Algorithms, and Systems", Ch. 3-4
//! - Baker & Trietsch (2019), "Principles of Sequencing and Scheduling"

mod kpi;
mod simple;

pub use kpi::ScheduleKpi;
pub use simple::{ScheduleRequest, SimpleScheduler};
