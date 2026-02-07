//! Rule engine for multi-criteria dispatching.
//!
//! Composes multiple dispatching rules with configurable evaluation modes
//! and tie-breaking strategies.
//!
//! # Reference
//! Haupt (1989), "A Survey of Priority Rule-Based Scheduling"

use std::sync::Arc;

use super::{DispatchingRule, RuleScore, SchedulingContext};
use crate::models::Task;

/// How multiple rules are combined.
#[derive(Debug, Clone, Default)]
pub enum EvaluationMode {
    /// Apply rules in sequence; use next rule only on ties.
    #[default]
    Sequential,
    /// Compute weighted sum of all rule scores.
    Weighted,
}

/// How ties are broken after all rules are exhausted.
#[derive(Debug, Clone, Default)]
pub enum TieBreaker {
    /// Use the next rule in the chain (default).
    #[default]
    NextRule,
    /// Deterministic by task ID (lexicographic).
    ById,
}

#[derive(Clone)]
struct WeightedRule {
    rule: Arc<dyn DispatchingRule>,
    weight: f64,
}

/// A composable rule engine for task prioritization.
///
/// Supports sequential multi-layer evaluation (primary rule → tie-breaker)
/// and weighted combination modes.
///
/// # Example
/// ```
/// use u_schedule::dispatching::{RuleEngine, SchedulingContext};
/// use u_schedule::dispatching::rules;
///
/// let engine = RuleEngine::new()
///     .with_rule(rules::Edd)
///     .with_tie_breaker(rules::Spt);
/// ```
#[derive(Clone)]
pub struct RuleEngine {
    rules: Vec<WeightedRule>,
    mode: EvaluationMode,
    tie_breaker: TieBreaker,
    epsilon: f64,
}

impl RuleEngine {
    /// Creates an empty rule engine.
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            mode: EvaluationMode::Sequential,
            tie_breaker: TieBreaker::NextRule,
            epsilon: 1e-9,
        }
    }

    /// Adds a primary rule (weight 1.0).
    pub fn with_rule<R: DispatchingRule + 'static>(mut self, rule: R) -> Self {
        self.rules.push(WeightedRule {
            rule: Arc::new(rule),
            weight: 1.0,
        });
        self
    }

    /// Adds a weighted rule.
    pub fn with_weighted_rule<R: DispatchingRule + 'static>(
        mut self,
        rule: R,
        weight: f64,
    ) -> Self {
        self.rules.push(WeightedRule {
            rule: Arc::new(rule),
            weight,
        });
        self
    }

    /// Adds a tie-breaking rule (weight 0.0, used only in Sequential mode).
    pub fn with_tie_breaker<R: DispatchingRule + 'static>(mut self, rule: R) -> Self {
        self.rules.push(WeightedRule {
            rule: Arc::new(rule),
            weight: 0.0,
        });
        self
    }

    /// Sets the evaluation mode.
    pub fn with_mode(mut self, mode: EvaluationMode) -> Self {
        self.mode = mode;
        self
    }

    /// Sets the final tie-breaking strategy.
    pub fn with_final_tie_breaker(mut self, tie_breaker: TieBreaker) -> Self {
        self.tie_breaker = tie_breaker;
        self
    }

    /// Sorts tasks by priority (highest priority first).
    ///
    /// Returns indices into the original task slice, sorted by rule evaluation.
    pub fn sort_indices(&self, tasks: &[Task], context: &SchedulingContext) -> Vec<usize> {
        if tasks.is_empty() {
            return Vec::new();
        }

        let mut indices: Vec<usize> = (0..tasks.len()).collect();

        match &self.mode {
            EvaluationMode::Sequential => {
                indices.sort_by(|&a, &b| {
                    self.compare_sequential(&tasks[a], &tasks[b], context)
                });
            }
            EvaluationMode::Weighted => {
                let scores: Vec<f64> = tasks
                    .iter()
                    .map(|t| self.weighted_score(t, context))
                    .collect();
                indices.sort_by(|&a, &b| {
                    scores[a]
                        .partial_cmp(&scores[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        indices
    }

    /// Returns the index of the highest-priority task.
    pub fn select_best(&self, tasks: &[Task], context: &SchedulingContext) -> Option<usize> {
        self.sort_indices(tasks, context).first().copied()
    }

    /// Evaluates a single task and returns scores from each rule.
    pub fn evaluate(&self, task: &Task, context: &SchedulingContext) -> Vec<RuleScore> {
        self.rules
            .iter()
            .map(|wr| wr.rule.evaluate(task, context) * wr.weight)
            .collect()
    }

    fn compare_sequential(
        &self,
        a: &Task,
        b: &Task,
        context: &SchedulingContext,
    ) -> std::cmp::Ordering {
        for wr in &self.rules {
            let score_a = wr.rule.evaluate(a, context);
            let score_b = wr.rule.evaluate(b, context);

            if (score_a - score_b).abs() > self.epsilon {
                return score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal);
            }
        }

        // All rules tied → use final tie-breaker
        match &self.tie_breaker {
            TieBreaker::NextRule => std::cmp::Ordering::Equal,
            TieBreaker::ById => a.id.cmp(&b.id),
        }
    }

    fn weighted_score(&self, task: &Task, context: &SchedulingContext) -> f64 {
        self.rules
            .iter()
            .map(|wr| wr.rule.evaluate(task, context) * wr.weight)
            .sum()
    }
}

impl Default for RuleEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for RuleEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuleEngine")
            .field(
                "rules",
                &self
                    .rules
                    .iter()
                    .map(|r| format!("{}(w={})", r.rule.name(), r.weight))
                    .collect::<Vec<_>>(),
            )
            .field("mode", &self.mode)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatching::rules;
    use crate::models::{Activity, ActivityDuration, Task};

    fn make_task(id: &str, duration_ms: i64, deadline: Option<i64>, priority: i32) -> Task {
        Task::new(id)
            .with_priority(priority)
            .with_activity(
                Activity::new(format!("{id}_O1"), id, 0)
                    .with_duration(ActivityDuration::fixed(duration_ms)),
            )
            .with_deadline_opt(deadline)
    }

    // Helper: Task with optional deadline
    trait TaskExt {
        fn with_deadline_opt(self, deadline: Option<i64>) -> Self;
    }
    impl TaskExt for Task {
        fn with_deadline_opt(mut self, deadline: Option<i64>) -> Self {
            self.deadline = deadline;
            self
        }
    }

    #[test]
    fn test_spt_ordering() {
        let tasks = vec![
            make_task("long", 5000, None, 0),
            make_task("short", 1000, None, 0),
            make_task("medium", 3000, None, 0),
        ];
        let ctx = SchedulingContext::at_time(0);
        let engine = RuleEngine::new().with_rule(rules::Spt);

        let indices = engine.sort_indices(&tasks, &ctx);
        assert_eq!(tasks[indices[0]].id, "short");
        assert_eq!(tasks[indices[1]].id, "medium");
        assert_eq!(tasks[indices[2]].id, "long");
    }

    #[test]
    fn test_edd_ordering() {
        let tasks = vec![
            make_task("late", 1000, Some(50_000), 0),
            make_task("early", 1000, Some(10_000), 0),
            make_task("no_deadline", 1000, None, 0),
        ];
        let ctx = SchedulingContext::at_time(0);
        let engine = RuleEngine::new().with_rule(rules::Edd);

        let indices = engine.sort_indices(&tasks, &ctx);
        assert_eq!(tasks[indices[0]].id, "early");
        assert_eq!(tasks[indices[1]].id, "late");
        assert_eq!(tasks[indices[2]].id, "no_deadline");
    }

    #[test]
    fn test_sequential_with_tie_breaker() {
        let tasks = vec![
            make_task("A", 1000, Some(10_000), 0),
            make_task("B", 2000, Some(10_000), 0), // Same deadline as A
        ];
        let ctx = SchedulingContext::at_time(0);
        let engine = RuleEngine::new()
            .with_rule(rules::Edd)
            .with_tie_breaker(rules::Spt);

        let indices = engine.sort_indices(&tasks, &ctx);
        // EDD ties → SPT breaks it → A (shorter) first
        assert_eq!(tasks[indices[0]].id, "A");
    }

    #[test]
    fn test_weighted_mode() {
        let tasks = vec![
            make_task("A", 1000, Some(50_000), 0),
            make_task("B", 5000, Some(10_000), 0),
        ];
        let ctx = SchedulingContext::at_time(0);
        let engine = RuleEngine::new()
            .with_mode(EvaluationMode::Weighted)
            .with_weighted_rule(rules::Edd, 0.5)
            .with_weighted_rule(rules::Spt, 0.5);

        let indices = engine.sort_indices(&tasks, &ctx);
        // A: 0.5*50000 + 0.5*1000 = 25500
        // B: 0.5*10000 + 0.5*5000 = 7500
        // B wins (lower weighted score)
        assert_eq!(tasks[indices[0]].id, "B");
    }

    #[test]
    fn test_by_id_tie_breaker() {
        let tasks = vec![
            make_task("B", 1000, None, 0),
            make_task("A", 1000, None, 0),
        ];
        let ctx = SchedulingContext::at_time(0);
        let engine = RuleEngine::new()
            .with_rule(rules::Spt)
            .with_final_tie_breaker(TieBreaker::ById);

        let indices = engine.sort_indices(&tasks, &ctx);
        // SPT ties → ById → A before B
        assert_eq!(tasks[indices[0]].id, "A");
    }

    #[test]
    fn test_empty_tasks() {
        let ctx = SchedulingContext::at_time(0);
        let engine = RuleEngine::new().with_rule(rules::Spt);
        assert!(engine.sort_indices(&[], &ctx).is_empty());
        assert!(engine.select_best(&[], &ctx).is_none());
    }

    #[test]
    fn test_select_best() {
        let tasks = vec![
            make_task("long", 5000, None, 0),
            make_task("short", 1000, None, 0),
        ];
        let ctx = SchedulingContext::at_time(0);
        let engine = RuleEngine::new().with_rule(rules::Spt);

        assert_eq!(engine.select_best(&tasks, &ctx), Some(1));
    }

    #[test]
    fn test_evaluate_scores() {
        let task = make_task("T1", 3000, Some(20_000), 0);
        let ctx = SchedulingContext::at_time(0);
        let engine = RuleEngine::new()
            .with_rule(rules::Spt)
            .with_rule(rules::Edd);

        let scores = engine.evaluate(&task, &ctx);
        assert_eq!(scores.len(), 2);
        assert!((scores[0] - 3000.0).abs() < 1e-10); // SPT score
        assert!((scores[1] - 20_000.0).abs() < 1e-10); // EDD score
    }
}
