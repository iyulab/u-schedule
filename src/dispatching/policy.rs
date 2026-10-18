//! Rules chosen decision by decision.
//!
//! A [`RuleEngine`] ranks the competing tasks the same way at every
//! decision. A [`RulePolicy`] chooses the engine for each decision from that
//! decision's state — a rule for the morning and another once the shop is
//! late, a rule while a bottleneck is busy — so the conditions stay the
//! caller's while the scheduler stays one.
//!
//! # Reference
//! Priore, Gómez, Pino & Rosillo (2014), "Dynamic scheduling of
//! manufacturing systems using machine learning: An updated review", *AI
//! EDAM* 28(1) — choosing the dispatching rule from the system state at each
//! decision.

use std::fmt;
use std::sync::Arc;

use super::{RuleEngine, SchedulingContext};
use crate::models::Task;

/// One decision of a dispatching scheduler.
#[derive(Debug, Clone, Copy)]
pub struct Decision<'a> {
    /// Every task of the problem.
    pub tasks: &'a [Task],
    /// The tasks waiting with an activity to place (indices into `tasks`).
    pub waiting: &'a [usize],
    /// Among them, the ones that compete at this decision — they can start
    /// now.
    pub competing: &'a [usize],
    /// The state the rules judge the decision on.
    pub context: &'a SchedulingContext,
}

/// Chooses the rules each decision is made by.
///
/// Any `Fn(&Decision) -> RuleEngine` is one.
pub trait RulePolicy: Send + Sync {
    /// The rules for `decision`.
    fn rules_for(&self, decision: &Decision<'_>) -> RuleEngine;
}

impl<F> RulePolicy for F
where
    F: Fn(&Decision<'_>) -> RuleEngine + Send + Sync,
{
    fn rules_for(&self, decision: &Decision<'_>) -> RuleEngine {
        self(decision)
    }
}

/// A shared [`RulePolicy`], for the schedulers that hold one.
#[derive(Clone)]
pub(crate) struct SharedPolicy(pub(crate) Arc<dyn RulePolicy>);

impl fmt::Debug for SharedPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SharedPolicy(..)")
    }
}
