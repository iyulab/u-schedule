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
//! # Quick Start
//!
//! ```no_run
//! use u_schedule::ga::{SchedulingGaProblem, ActivityInfo};
//! use u_schedule::ga::operators::{GeneticOperators, CrossoverType, MutationType};
//! use u_schedule::models::{Task, Resource};
//! use u_metaheur::ga::{GaConfig, GaRunner};
//!
//! // 1. Define tasks and resources
//! let tasks: Vec<Task> = vec![/* ... */];
//! let resources: Vec<Resource> = vec![/* ... */];
//!
//! // 2. Create problem (optionally configure operators)
//! let problem = SchedulingGaProblem::new(&tasks, &resources)
//!     .with_operators(GeneticOperators {
//!         crossover_type: CrossoverType::LOX,
//!         mutation_type: MutationType::Invert,
//!     });
//!
//! // 3. Configure and run GA
//! let config = GaConfig::auto_select(tasks.len())
//!     .with_seed(42);
//! let result = GaRunner::run(&problem, &config);
//!
//! // 4. Decode best solution
//! let schedule = problem.decode(&result.best);
//! println!("Makespan: {} ms", schedule.makespan_ms());
//! ```
//!
//! # Initialization Strategies
//!
//! Initial population uses a mixed strategy:
//! - **50%** random (full diversity)
//! - **25%** load-balanced (even resource utilization)
//! - **25%** SPT (shortest processing time, if `process_times` provided)
//!
//! # Crossover Operators
//!
//! | Operator | Description | Reference |
//! |----------|-------------|-----------|
//! | [`CrossoverType::POX`](operators::CrossoverType::POX) | Precedence Operation Crossover | Bierwirth et al. (1996) |
//! | [`CrossoverType::LOX`](operators::CrossoverType::LOX) | Linear Order Crossover | Falkenauer & Bouffouix (1991) |
//! | [`CrossoverType::JOX`](operators::CrossoverType::JOX) | Job-based Order Crossover | Yamada & Nakano (1997) |
//!
//! # Mutation Operators
//!
//! | Operator | Description |
//! |----------|-------------|
//! | [`MutationType::Swap`](operators::MutationType::Swap) | Exchange two random positions |
//! | [`MutationType::Insert`](operators::MutationType::Insert) | Remove and reinsert at random |
//! | [`MutationType::Invert`](operators::MutationType::Invert) | Reverse a random segment |
//!
//! All mutations also apply MAV mutation (random resource reassignment).
//!
//! # Submodules
//!
//! - [`operators`]: Runtime-selectable crossover and mutation strategies
//!
//! # References
//!
//! - Cheng et al. (1996), "A Tutorial Survey of JSSP using GA"
//! - Bierwirth (1995), "A generalized permutation approach to JSSP"
//! - Conway et al. (1967), "Theory of Scheduling" (SPT heuristic)

mod chromosome;
pub mod operators;
mod problem;

pub use chromosome::{
    insert_mutation, invert_mutation, jox_crossover, lox_crossover, mav_mutation, pox_crossover,
    swap_mutation, ScheduleChromosome,
};
pub use problem::{ActivityInfo, SchedulingGaProblem};
