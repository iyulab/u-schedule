//! Scheduling domain models.
//!
//! Provides the core data types for representing scheduling problems
//! and solutions. Domain-agnostic within scheduling — applicable to
//! job-shop, flow-shop, project scheduling, and resource-constrained problems.
//!
//! # Domain Mappings
//!
//! | u-schedule | Manufacturing | Healthcare | Logistics |
//! |------------|--------------|------------|-----------|
//! | Task | Job/Order | Patient Case | Shipment |
//! | Activity | Operation | Procedure | Transport Leg |
//! | Resource | Machine/Worker | Room/Doctor | Truck/Driver |
//! | Schedule | Production Plan | OR Schedule | Route Plan |

mod activity;
mod calendar;
mod constraint;
mod resource;
mod schedule;
mod task;
pub mod time_constraints;

pub(crate) use activity::timed_duration;
pub use activity::{Activity, ActivityDuration, ResourceRequirement, ResourceTiming};
pub use calendar::{joint_available_ms, joint_work_end, Calendar, TimeWindow};
pub use constraint::{Constraint, TransitionMatrix, TransitionMatrixCollection};
pub use resource::{transport_between, Resource};
pub use schedule::{Assignment, Schedule, TimeOnlySpan, Violation, ViolationType};
pub use task::{tardiness_ms, Task};
pub use time_constraints::{DurationDistribution, PertEstimate};
