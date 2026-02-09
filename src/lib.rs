//! Scheduling framework for the U-Engine ecosystem.
//!
//! Provides domain models, constraints, validation, dispatching rules,
//! and a greedy scheduler for scheduling problems. This crate defines
//! the scheduling domain language — metaheuristic algorithms (GA, SA, CP)
//! are provided by `u-metaheur` at a lower layer.
//!
//! # Modules
//!
//! - **`models`**: Domain types — `Task`, `Activity`, `Resource`, `Schedule`,
//!   `Assignment`, `Calendar`, `Constraint`, `TransitionMatrix`
//! - **`validation`**: Input integrity checks (duplicate IDs, DAG cycles, resource refs)
//! - **`dispatching`**: Priority dispatching rules (SPT, EDD, ATC, etc.) and rule engine
//! - **`scheduler`**: Greedy scheduler and KPI evaluation
//! - **`ga`**: GA-based scheduling with OSV/MAV encoding
//! - **`cp`**: CP-based scheduling formulation
//!
//! # Architecture
//!
//! This crate sits at Layer 3 (Frameworks) in the U-Engine ecosystem.
//! It depends on `u-metaheur` and `u-numerics` but contains only scheduling
//! domain logic — no nesting, packing, or manufacturing concepts.
//!
//! # References
//!
//! - Pinedo (2016), "Scheduling: Theory, Algorithms, and Systems"
//! - Brucker (2007), "Scheduling Algorithms"
//! - Blazewicz et al. (2019), "Handbook on Scheduling"
//! - Haupt (1989), "A Survey of Priority Rule-Based Scheduling"

pub mod cp;
pub mod dispatching;
pub mod ga;
pub mod models;
pub mod scheduler;
pub mod validation;
