//! Built-in dispatching rules.
//!
//! # Categories
//!
//! - **Time-based**: SPT, LPT, LWKR, MWKR, WSPT
//! - **Due-date**: EDD, MST, CR, SRO, ATC
//! - **Queue/Load**: FIFO, WINQ, LPUL
//! - **Priority**: PRIORITY
//!
//! # Score Convention
//! All rules return lower scores for higher priority tasks.
//!
//! # References
//! - Pinedo (2016), "Scheduling: Theory, Algorithms, and Systems", Ch. 4
//! - Haupt (1989), "A Survey of Priority Rule-Based Scheduling"

use super::{DispatchingRule, RuleScore, SchedulingContext};
use crate::models::Task;

// ======================== Time-based rules ========================

/// Shortest Processing Time.
///
/// Prioritizes tasks with shorter total processing time.
/// Minimizes average flow time and WIP (Work-In-Process).
///
/// # Reference
/// Smith (1956), optimal for minimizing mean flow time on single machine.
#[derive(Debug, Clone, Copy)]
pub struct Spt;

impl DispatchingRule for Spt {
    fn name(&self) -> &'static str {
        "SPT"
    }

    fn evaluate(&self, task: &Task, _context: &SchedulingContext) -> RuleScore {
        task.total_duration_ms() as f64
    }

    fn description(&self) -> &'static str {
        "Shortest Processing Time"
    }
}

/// Longest Processing Time.
///
/// Prioritizes tasks with longer total processing time.
/// Useful for load balancing in parallel machine environments.
#[derive(Debug, Clone, Copy)]
pub struct Lpt;

impl DispatchingRule for Lpt {
    fn name(&self) -> &'static str {
        "LPT"
    }

    fn evaluate(&self, task: &Task, _context: &SchedulingContext) -> RuleScore {
        -(task.total_duration_ms() as f64)
    }

    fn description(&self) -> &'static str {
        "Longest Processing Time"
    }
}

/// Least Work Remaining.
///
/// Prioritizes tasks closer to completion. Uses `context.remaining_work`
/// if available, falls back to total duration.
#[derive(Debug, Clone, Copy)]
pub struct Lwkr;

impl DispatchingRule for Lwkr {
    fn name(&self) -> &'static str {
        "LWKR"
    }

    fn evaluate(&self, task: &Task, context: &SchedulingContext) -> RuleScore {
        context
            .remaining_work
            .get(&task.id)
            .copied()
            .unwrap_or_else(|| task.total_duration_ms()) as f64
    }

    fn description(&self) -> &'static str {
        "Least Work Remaining"
    }
}

/// Most Work Remaining.
///
/// Prioritizes tasks with the most remaining work.
/// Prevents starvation of long tasks.
#[derive(Debug, Clone, Copy)]
pub struct Mwkr;

impl DispatchingRule for Mwkr {
    fn name(&self) -> &'static str {
        "MWKR"
    }

    fn evaluate(&self, task: &Task, context: &SchedulingContext) -> RuleScore {
        let remaining = context
            .remaining_work
            .get(&task.id)
            .copied()
            .unwrap_or_else(|| task.total_duration_ms());
        -(remaining as f64)
    }

    fn description(&self) -> &'static str {
        "Most Work Remaining"
    }
}

/// Weighted Shortest Processing Time.
///
/// Prioritizes by the ratio of importance to processing time.
/// Weight is derived from priority: `weight = 1000 / (priority + 1)`.
///
/// # Reference
/// Smith (1956), optimal for minimizing weighted mean flow time.
#[derive(Debug, Clone, Copy)]
pub struct Wspt;

impl DispatchingRule for Wspt {
    fn name(&self) -> &'static str {
        "WSPT"
    }

    fn evaluate(&self, task: &Task, _context: &SchedulingContext) -> RuleScore {
        let processing_time = task.total_duration_ms() as f64;
        if processing_time <= 0.0 {
            return f64::MAX;
        }
        let weight = 1000.0 / (task.priority as f64 + 1.0);
        -(weight / processing_time) // Higher ratio = higher priority → negate
    }

    fn description(&self) -> &'static str {
        "Weighted Shortest Processing Time"
    }
}

// ======================== Due-date rules ========================

/// Earliest Due Date.
///
/// Prioritizes tasks with earlier deadlines. Tasks without deadlines
/// are assigned lowest priority.
///
/// # Reference
/// Jackson (1955), optimal for minimizing maximum lateness on single machine.
#[derive(Debug, Clone, Copy)]
pub struct Edd;

impl DispatchingRule for Edd {
    fn name(&self) -> &'static str {
        "EDD"
    }

    fn evaluate(&self, task: &Task, _context: &SchedulingContext) -> RuleScore {
        task.deadline.map(|d| d as f64).unwrap_or(f64::MAX)
    }

    fn description(&self) -> &'static str {
        "Earliest Due Date"
    }
}

/// Minimum Slack Time.
///
/// Slack = (deadline - current_time) - remaining_work.
/// Prioritizes tasks with least slack (most urgent).
///
/// Tasks without deadlines get maximum slack (lowest priority).
#[derive(Debug, Clone, Copy)]
pub struct Mst;

impl DispatchingRule for Mst {
    fn name(&self) -> &'static str {
        "MST"
    }

    fn evaluate(&self, task: &Task, context: &SchedulingContext) -> RuleScore {
        let deadline = match task.deadline {
            Some(d) => d,
            None => return f64::MAX,
        };

        let remaining = context
            .remaining_work
            .get(&task.id)
            .copied()
            .unwrap_or_else(|| task.total_duration_ms());

        let time_until_deadline = deadline - context.current_time_ms;
        (time_until_deadline - remaining) as f64
    }

    fn description(&self) -> &'static str {
        "Minimum Slack Time"
    }
}

/// Critical Ratio.
///
/// CR = (deadline - current_time) / remaining_work.
/// - CR < 1.0: behind schedule
/// - CR = 1.0: on track
/// - CR > 1.0: ahead of schedule
///
/// Prioritizes tasks with lowest CR (most behind).
#[derive(Debug, Clone, Copy)]
pub struct Cr;

impl DispatchingRule for Cr {
    fn name(&self) -> &'static str {
        "CR"
    }

    fn evaluate(&self, task: &Task, context: &SchedulingContext) -> RuleScore {
        let deadline = match task.deadline {
            Some(d) => d,
            None => return f64::MAX,
        };

        let remaining = context
            .remaining_work
            .get(&task.id)
            .copied()
            .unwrap_or_else(|| task.total_duration_ms());

        if remaining <= 0 {
            return f64::MAX; // Already done
        }

        let time_until_deadline = (deadline - context.current_time_ms) as f64;
        time_until_deadline / remaining as f64
    }

    fn description(&self) -> &'static str {
        "Critical Ratio"
    }
}

/// Slack per Remaining Operations.
///
/// S/RO = slack / remaining_operation_count.
/// Accounts for the number of remaining steps, not just total work.
#[derive(Debug, Clone, Copy)]
pub struct Sro;

impl DispatchingRule for Sro {
    fn name(&self) -> &'static str {
        "S/RO"
    }

    fn evaluate(&self, task: &Task, context: &SchedulingContext) -> RuleScore {
        let deadline = match task.deadline {
            Some(d) => d,
            None => return f64::MAX,
        };

        let remaining_work = context
            .remaining_work
            .get(&task.id)
            .copied()
            .unwrap_or_else(|| task.total_duration_ms());

        let op_count = task.activity_count().max(1);
        let slack = (deadline - context.current_time_ms - remaining_work) as f64;
        slack / op_count as f64
    }

    fn description(&self) -> &'static str {
        "Slack per Remaining Operations"
    }
}

/// Apparent Tardiness Cost.
///
/// Combines WSPT with deadline urgency using an exponential function.
/// The parameter `k` controls the balance:
/// - k > 2: more SPT-like (processing time dominates)
/// - k < 2: more EDD-like (deadline dominates)
///
/// # Reference
/// Vepsalainen & Morton (1987), "Priority Rules for Job Shops with
/// Weighted Tardiness Costs"
#[derive(Debug, Clone, Copy)]
pub struct Atc {
    /// Lookahead parameter (default: 2.0).
    pub k: f64,
}

impl Default for Atc {
    fn default() -> Self {
        Self { k: 2.0 }
    }
}

impl Atc {
    /// Creates an ATC rule with custom k parameter.
    pub fn with_k(k: f64) -> Self {
        Self { k }
    }
}

impl DispatchingRule for Atc {
    fn name(&self) -> &'static str {
        "ATC"
    }

    fn evaluate(&self, task: &Task, context: &SchedulingContext) -> RuleScore {
        let processing_time = task.total_duration_ms() as f64;
        if processing_time <= 0.0 {
            return f64::MAX;
        }

        let weight = 1000.0 / (task.priority as f64 + 1.0);

        let deadline = match task.deadline {
            Some(d) => d as f64,
            None => return -(weight / processing_time), // Fallback to WSPT
        };

        let slack = deadline - processing_time - context.current_time_ms as f64;
        let p_avg = context.average_processing_time.unwrap_or(processing_time).max(1.0);

        let urgency = if slack <= 0.0 {
            1.0
        } else {
            (-slack / (self.k * p_avg)).exp()
        };

        -(weight / processing_time * urgency) // Higher ATC = higher priority → negate
    }

    fn description(&self) -> &'static str {
        "Apparent Tardiness Cost"
    }
}

// ======================== Queue/Load rules ========================

/// First In First Out.
///
/// Prioritizes tasks by arrival time. Uses `context.arrival_times`
/// or falls back to `task.release_time`.
#[derive(Debug, Clone, Copy)]
pub struct Fifo;

impl DispatchingRule for Fifo {
    fn name(&self) -> &'static str {
        "FIFO"
    }

    fn evaluate(&self, task: &Task, context: &SchedulingContext) -> RuleScore {
        context
            .arrival_times
            .get(&task.id)
            .copied()
            .unwrap_or_else(|| task.release_time.unwrap_or(0)) as f64
    }

    fn description(&self) -> &'static str {
        "First In First Out"
    }
}

/// Work In Next Queue.
///
/// Prioritizes tasks whose next resource has the shortest queue.
/// Uses `context.next_queue_length`.
#[derive(Debug, Clone, Copy)]
pub struct Winq;

impl DispatchingRule for Winq {
    fn name(&self) -> &'static str {
        "WINQ"
    }

    fn evaluate(&self, task: &Task, context: &SchedulingContext) -> RuleScore {
        context
            .next_queue_length
            .get(&task.id)
            .copied()
            .unwrap_or(0) as f64
    }

    fn description(&self) -> &'static str {
        "Work In Next Queue"
    }
}

/// Least Planned Utilization Level.
///
/// Prioritizes tasks whose candidate resources have the lowest utilization.
/// Uses `context.resource_utilization` and `task.activities[0].resource_requirements`.
#[derive(Debug, Clone, Copy)]
pub struct Lpul;

impl DispatchingRule for Lpul {
    fn name(&self) -> &'static str {
        "LPUL"
    }

    fn evaluate(&self, task: &Task, context: &SchedulingContext) -> RuleScore {
        if let Some(activity) = task.activities.first() {
            let min_util = activity
                .resource_requirements
                .iter()
                .flat_map(|req| req.candidates.iter())
                .filter_map(|res_id| context.resource_utilization.get(res_id))
                .copied()
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            min_util.unwrap_or(0.0)
        } else {
            0.0
        }
    }

    fn description(&self) -> &'static str {
        "Least Planned Utilization Level"
    }
}

// ======================== Priority-based rule ========================

/// Simple priority rule.
///
/// Prioritizes tasks with higher `task.priority` values.
/// (Negated because lower score = higher priority in convention.)
#[derive(Debug, Clone, Copy)]
pub struct Priority;

impl DispatchingRule for Priority {
    fn name(&self) -> &'static str {
        "PRIORITY"
    }

    fn evaluate(&self, task: &Task, _context: &SchedulingContext) -> RuleScore {
        -(task.priority as f64)
    }

    fn description(&self) -> &'static str {
        "Task Priority"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Activity, ActivityDuration, ResourceRequirement};

    fn make_task(id: &str, duration_ms: i64, deadline: Option<i64>, priority: i32) -> Task {
        let mut task = Task::new(id).with_priority(priority).with_activity(
            Activity::new(format!("{id}_O1"), id, 0)
                .with_duration(ActivityDuration::fixed(duration_ms)),
        );
        task.deadline = deadline;
        task
    }

    #[test]
    fn test_spt() {
        let ctx = SchedulingContext::at_time(0);
        let short = make_task("short", 1000, None, 0);
        let long = make_task("long", 5000, None, 0);
        assert!(Spt.evaluate(&short, &ctx) < Spt.evaluate(&long, &ctx));
    }

    #[test]
    fn test_lpt() {
        let ctx = SchedulingContext::at_time(0);
        let short = make_task("short", 1000, None, 0);
        let long = make_task("long", 5000, None, 0);
        assert!(Lpt.evaluate(&long, &ctx) < Lpt.evaluate(&short, &ctx));
    }

    #[test]
    fn test_lwkr_with_context() {
        let ctx = SchedulingContext::at_time(0)
            .with_remaining_work("almost_done", 100)
            .with_remaining_work("lots_left", 5000);

        let t1 = make_task("almost_done", 10000, None, 0);
        let t2 = make_task("lots_left", 10000, None, 0);
        assert!(Lwkr.evaluate(&t1, &ctx) < Lwkr.evaluate(&t2, &ctx));
    }

    #[test]
    fn test_lwkr_fallback() {
        let ctx = SchedulingContext::at_time(0); // No remaining_work data
        let t1 = make_task("short", 1000, None, 0);
        let t2 = make_task("long", 5000, None, 0);
        assert!(Lwkr.evaluate(&t1, &ctx) < Lwkr.evaluate(&t2, &ctx));
    }

    #[test]
    fn test_mwkr() {
        let ctx = SchedulingContext::at_time(0)
            .with_remaining_work("a", 100)
            .with_remaining_work("b", 5000);
        let t1 = make_task("a", 10000, None, 0);
        let t2 = make_task("b", 10000, None, 0);
        assert!(Mwkr.evaluate(&t2, &ctx) < Mwkr.evaluate(&t1, &ctx));
    }

    #[test]
    fn test_wspt() {
        let ctx = SchedulingContext::at_time(0);
        // High priority + short duration → highest WSPT
        let important_short = make_task("is", 1000, None, 1);
        // Low priority + long duration → lowest WSPT
        let unimportant_long = make_task("ul", 5000, None, 10);
        assert!(Wspt.evaluate(&important_short, &ctx) < Wspt.evaluate(&unimportant_long, &ctx));
    }

    #[test]
    fn test_edd() {
        let ctx = SchedulingContext::at_time(0);
        let early = make_task("early", 1000, Some(10_000), 0);
        let late = make_task("late", 1000, Some(50_000), 0);
        let none = make_task("none", 1000, None, 0);
        assert!(Edd.evaluate(&early, &ctx) < Edd.evaluate(&late, &ctx));
        assert!(Edd.evaluate(&late, &ctx) < Edd.evaluate(&none, &ctx));
    }

    #[test]
    fn test_mst() {
        let ctx = SchedulingContext::at_time(1000);
        // Deadline 5000, remaining 3000 → slack = (5000-1000)-3000 = 1000
        let urgent = make_task("urgent", 3000, Some(5000), 0);
        // Deadline 50000, remaining 3000 → slack = (50000-1000)-3000 = 46000
        let relaxed = make_task("relaxed", 3000, Some(50000), 0);
        assert!(Mst.evaluate(&urgent, &ctx) < Mst.evaluate(&relaxed, &ctx));
    }

    #[test]
    fn test_cr() {
        let ctx = SchedulingContext::at_time(1000);
        // Deadline 4000, remaining 3000 → CR = (4000-1000)/3000 = 1.0
        let on_track = make_task("on_track", 3000, Some(4000), 0);
        // Deadline 10000, remaining 3000 → CR = (10000-1000)/3000 = 3.0
        let ahead = make_task("ahead", 3000, Some(10000), 0);
        assert!(Cr.evaluate(&on_track, &ctx) < Cr.evaluate(&ahead, &ctx));
    }

    #[test]
    fn test_cr_behind_schedule() {
        let ctx = SchedulingContext::at_time(5000);
        // Deadline 4000 at t=5000 → CR = (4000-5000)/3000 = -0.33 (negative = behind)
        let behind = make_task("behind", 3000, Some(4000), 0);
        let normal = make_task("normal", 3000, Some(20000), 0);
        assert!(Cr.evaluate(&behind, &ctx) < Cr.evaluate(&normal, &ctx));
    }

    #[test]
    fn test_sro() {
        let ctx = SchedulingContext::at_time(0);
        let few_ops = make_task("few", 1000, Some(5000), 0); // 1 op, slack=4000, S/RO=4000
        // Task with 3 activities
        let mut many_ops = Task::new("many").with_priority(0);
        many_ops.deadline = Some(5000);
        for i in 0..3 {
            many_ops.activities.push(
                Activity::new(format!("many_O{i}"), "many", i)
                    .with_duration(ActivityDuration::fixed(333)),
            );
        }
        // slack = 5000-0-999 = 4001, S/RO = 4001/3 ≈ 1333
        assert!(Sro.evaluate(&many_ops, &ctx) < Sro.evaluate(&few_ops, &ctx));
    }

    #[test]
    fn test_atc() {
        let ctx = SchedulingContext::at_time(0).with_average_processing_time(2000.0);
        let atc = Atc::default();
        let urgent = make_task("urgent", 1000, Some(2000), 0); // Tight deadline
        let relaxed = make_task("relaxed", 1000, Some(100_000), 0); // Loose deadline
        // Urgent has higher ATC score → lower (more negative) return
        assert!(atc.evaluate(&urgent, &ctx) < atc.evaluate(&relaxed, &ctx));
    }

    #[test]
    fn test_atc_no_deadline() {
        let ctx = SchedulingContext::at_time(0);
        let atc = Atc::default();
        let no_dl = make_task("no_dl", 1000, None, 0);
        // Falls back to WSPT
        assert!(atc.evaluate(&no_dl, &ctx).is_finite());
    }

    #[test]
    fn test_fifo() {
        let ctx = SchedulingContext::at_time(5000)
            .with_arrival_time("first", 1000)
            .with_arrival_time("second", 3000);
        let t1 = make_task("first", 2000, None, 0);
        let t2 = make_task("second", 2000, None, 0);
        assert!(Fifo.evaluate(&t1, &ctx) < Fifo.evaluate(&t2, &ctx));
    }

    #[test]
    fn test_fifo_fallback() {
        let ctx = SchedulingContext::at_time(0);
        let mut t1 = make_task("t1", 1000, None, 0);
        t1.release_time = Some(500);
        let mut t2 = make_task("t2", 1000, None, 0);
        t2.release_time = Some(1000);
        assert!(Fifo.evaluate(&t1, &ctx) < Fifo.evaluate(&t2, &ctx));
    }

    #[test]
    fn test_winq() {
        let ctx = SchedulingContext::at_time(0)
            .with_next_queue("short_q", 2)
            .with_next_queue("long_q", 10);
        let t1 = make_task("short_q", 1000, None, 0);
        let t2 = make_task("long_q", 1000, None, 0);
        assert!(Winq.evaluate(&t1, &ctx) < Winq.evaluate(&t2, &ctx));
    }

    #[test]
    fn test_lpul() {
        let ctx = SchedulingContext::at_time(0)
            .with_utilization("M1", 0.3)
            .with_utilization("M2", 0.9);

        let t1 = Task::new("t1").with_activity(
            Activity::new("t1_O1", "t1", 0)
                .with_process_time(1000)
                .with_requirement(
                    ResourceRequirement::new("Machine").with_candidates(vec!["M1".into()]),
                ),
        );
        let t2 = Task::new("t2").with_activity(
            Activity::new("t2_O1", "t2", 0)
                .with_process_time(1000)
                .with_requirement(
                    ResourceRequirement::new("Machine").with_candidates(vec!["M2".into()]),
                ),
        );

        assert!(Lpul.evaluate(&t1, &ctx) < Lpul.evaluate(&t2, &ctx));
    }

    #[test]
    fn test_priority() {
        let ctx = SchedulingContext::at_time(0);
        let high = make_task("high", 1000, None, 100);
        let low = make_task("low", 1000, None, 1);
        assert!(Priority.evaluate(&high, &ctx) < Priority.evaluate(&low, &ctx));
    }
}
