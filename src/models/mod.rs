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

pub use activity::{Activity, ActivityDuration, ResourceRequirement};
pub use calendar::{Calendar, TimeWindow};
pub use constraint::{Constraint, TransitionMatrix, TransitionMatrixCollection};
pub use resource::{Resource, ResourceType, Skill};
pub use schedule::{Assignment, Schedule, Violation, ViolationType};
pub use task::Task;
