//! Scheduling framework for the U-Engine ecosystem.
//!
//! Provides domain models, constraints, and validation for scheduling problems.
//! This crate defines the scheduling domain language — algorithms (GA, SA, CP)
//! are provided by `u-metaheur` at a lower layer.
//!
//! # Modules
//!
//! - **`models`**: Domain types — `Task`, `Activity`, `Resource`, `Schedule`,
//!   `Assignment`, `Calendar`, `Constraint`, `TransitionMatrix`
//! - **`validation`**: Input integrity checks (duplicate IDs, DAG cycles, resource refs)
//!
//! # Architecture
//!
//! This crate sits at Layer 3 (Frameworks) in the U-Engine ecosystem.
//! It depends on `u-metaheur` and `u-optim` but contains only scheduling
//! domain logic — no nesting, packing, or manufacturing concepts.
//!
//! # References
//!
//! - Pinedo (2016), "Scheduling: Theory, Algorithms, and Systems"
//! - Brucker (2007), "Scheduling Algorithms"
//! - Blazewicz et al. (2019), "Handbook on Scheduling"

pub mod models;
pub mod validation;
