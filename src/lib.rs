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
//! It depends on `u-metaheur` and `u-numflow` but contains only scheduling
//! domain logic — no nesting, packing, or manufacturing concepts.
//!
//! # Solver enforcement matrix (0.4.0)
//!
//! Model expressiveness and solver enforcement are distinct. What each
//! execution path actually enforces:
//!
//! | Feature | `SimpleScheduler` | GA decode | CP builder |
//! |---|---|---|---|
//! | Multi-requirement simultaneous hold | ✅ | ❌ (single resource per activity) | ❌ (first candidate) |
//! | Resource calendar | ✅ | ❌ | ❌ |
//! | Capacity > 1 | ✅ | ❌ | ❌ |
//! | Setup/teardown duration components | ✅ | ❌ | ❌ |
//! | `TransitionMatrix` sequence-dependent setup | ✅ | ✅ | ❌ |
//! | `Constraint::{TimeWindow, Synchronize}` | validated only | validated only | ❌ (skipped) |
//! | `Constraint::TransitionCost` | ❌ (unsupported) | ❌ | ❌ |
//!
//! Every [`scheduler::SimpleScheduler`] result self-annotates via
//! [`scheduler::check_schedule`], so `Schedule::is_valid()` is
//! meaningful on that path. Run the checker manually on GA/CP output to
//! obtain honest violation reports.
//!
//! # References
//!
//! - Pinedo (2016), "Scheduling: Theory, Algorithms, and Systems"
//! - Brucker (2007), "Scheduling Algorithms"
//! - Blazewicz et al. (2019), "Handbook on Scheduling"
//! - Haupt (1989), "A Survey of Priority Rule-Based Scheduling"
//! - Kolisch & Hartmann (1999), "Heuristic algorithms for the RCPSP" (serial SGS)

pub mod cp;
pub mod dispatching;
pub mod ga;
pub mod models;
pub mod scheduler;
pub mod validation;

#[cfg(feature = "wasm")]
pub mod wasm;
