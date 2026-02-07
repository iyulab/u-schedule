//! Dispatching rules and rule engine for scheduling.
//!
//! Provides priority-based dispatching rules (SPT, EDD, ATC, etc.)
//! and a composable rule engine for multi-criteria task prioritization.
//!
//! # Usage
//!
//! ```
//! use u_schedule::dispatching::{RuleEngine, SchedulingContext};
//! use u_schedule::dispatching::rules;
//!
//! let engine = RuleEngine::new()
//!     .with_rule(rules::Edd)
//!     .with_tie_breaker(rules::Spt);
//!
//! let context = SchedulingContext::at_time(0);
//! // let sorted = engine.sort(&tasks, &context);
//! ```
//!
//! # References
//!
//! - Pinedo (2016), "Scheduling: Theory, Algorithms, and Systems", Ch. 4
//! - Haupt (1989), "A Survey of Priority Rule-Based Scheduling"

mod context;
mod engine;
pub mod rules;

pub use context::SchedulingContext;
pub use engine::{EvaluationMode, RuleEngine, TieBreaker};

use crate::models::Task;
use std::fmt::Debug;

/// Score returned by a dispatching rule.
///
/// Lower scores = higher priority (scheduled first).
/// This follows the academic convention where SPT = shortest processing time first.
pub type RuleScore = f64;

/// A dispatching rule that evaluates task priority.
///
/// # Score Convention
/// **Lower score = higher priority.** Rules should return smaller values
/// for tasks that should be scheduled first.
///
/// # Reference
/// Pinedo (2016), "Scheduling", Ch. 4: Priority Dispatching
pub trait DispatchingRule: Send + Sync + Debug {
    /// Rule name (e.g., "SPT", "EDD").
    fn name(&self) -> &'static str;

    /// Evaluates the priority of a task given the current scheduling context.
    ///
    /// Returns a score where lower = higher priority.
    fn evaluate(&self, task: &Task, context: &SchedulingContext) -> RuleScore;

    /// Rule description.
    fn description(&self) -> &'static str {
        self.name()
    }
}
