//! GA-based scheduling optimization.
//!
//! Implements scheduling-specific GA encodings on top of `u-metaheur`'s
//! generic GA framework. Uses the OSV/MAV dual-vector encoding.
//!
//! # Encoding
//!
//! - **OSV** (Operation Sequence Vector): Permutation of task IDs encoding
//!   activity execution order. The k-th occurrence of task T = T's k-th activity.
//! - **MAV** (Machine Assignment Vector): Resource assignment for each activity.
//!
//! # Submodules
//!
//! - [`operators`]: Runtime-selectable crossover and mutation strategies
//!
//! # Reference
//! - Cheng et al. (1996), "A Tutorial Survey of JSSP using GA"
//! - Bierwirth (1995), "A generalized permutation approach to JSSP"

mod chromosome;
pub mod operators;
mod problem;

pub use chromosome::{
    ScheduleChromosome, invert_mutation, insert_mutation, jox_crossover, lox_crossover,
    mav_mutation, pox_crossover, swap_mutation,
};
pub use problem::{ActivityInfo, SchedulingGaProblem};
