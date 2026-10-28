//! Searching the order dispatching takes the tasks in.
//!
//! A biased random-key genetic algorithm gives each activity a key; a
//! chromosome is decoded by [`SimpleScheduler`] with the keys as its rule (the
//! task whose next activity has the lower key wins each decision), so every decoded schedule keeps everything the
//! dispatcher keeps — calendars, several requirements at once, usages,
//! companions, transport times, releases. The search returns the best of the
//! keyed schedules and the scheduler's own (its configured rules), so it is
//! never worse than dispatching alone.
//!
//! # Reference
//! Hartmann (1998), "A competitive genetic algorithm for resource-constrained
//! project scheduling", *Naval Research Logistics* 45 — priority-value
//! representation decoded by a schedule generation scheme; Gonçalves,
//! Resende & Mendes (2011), "A biased random-key genetic algorithm with
//! forward-backward improvement for the RCPSP", *J. Heuristics* 17.

use std::collections::HashMap;

use u_metaheur::brkga::{BrkgaConfig, BrkgaDecoder, BrkgaRunner};
use u_metaheur::{ConfigError, StopReason};

use super::SimpleScheduler;
use crate::dispatching::{DispatchingRule, RuleEngine, RuleScore, SchedulingContext};
use crate::models::{Schedule, Task};
use crate::Problem;

/// The best schedule a [`DispatchSearch`] found.
#[derive(Debug, Clone)]
pub struct DispatchSearchResult {
    /// The schedule.
    pub schedule: Schedule,
    /// Its cost: `(1 − w)` × makespan + `w` × total tardiness (ms), `w` the
    /// search's tardiness weight.
    pub cost: f64,
    /// The cost of the scheduler's own schedule (its configured rules).
    pub dispatching_cost: f64,
    /// Generations the search ran.
    pub generations: usize,
    /// Why it stopped.
    pub stop_reason: StopReason,
}

/// Searches the order a [`SimpleScheduler`] takes the tasks in (see the
/// module documentation).
#[derive(Debug, Clone)]
pub struct DispatchSearch {
    scheduler: SimpleScheduler,
    config: BrkgaConfig,
    tardiness_weight: f64,
}

impl DispatchSearch {
    /// A search over `scheduler`'s decisions, with `config`'s population,
    /// generations, seed and time limit (its chromosome length is set from
    /// the problem: one key per activity).
    pub fn new(scheduler: SimpleScheduler, config: BrkgaConfig) -> Self {
        Self {
            scheduler,
            config,
            tardiness_weight: 0.0,
        }
    }

    /// Weighs total tardiness against makespan in the cost: 0 = makespan
    /// only (the default), 1 = tardiness only.
    pub fn with_tardiness_weight(mut self, weight: f64) -> Self {
        self.tardiness_weight = weight;
        self
    }

    /// Searches `problem` from `start_time_ms`.
    ///
    /// # Errors
    /// The configuration is invalid (a population too small, fractions out
    /// of range, …), or the tardiness weight is outside `[0, 1]`.
    pub fn search(
        &self,
        problem: &Problem,
        start_time_ms: i64,
    ) -> Result<DispatchSearchResult, ConfigError> {
        if !(0.0..=1.0).contains(&self.tardiness_weight) {
            return Err(ConfigError::OutOfRange {
                parameter: "tardiness_weight",
                min: Some(0.0),
                max: Some(1.0),
                exclusive_min: None,
                exclusive_max: None,
                got: self.tardiness_weight,
                range: "in [0, 1]",
            });
        }
        let own = self.scheduler.schedule(problem, start_time_ms);
        let dispatching_cost = cost(&own, problem.tasks(), self.tardiness_weight);
        let tasks = problem.tasks();
        if tasks.len() < 2 {
            return Ok(DispatchSearchResult {
                schedule: own,
                cost: dispatching_cost,
                dispatching_cost,
                generations: 0,
                stop_reason: StopReason::Completed,
            });
        }
        let decoder = Decoder {
            search: self,
            problem,
            start_time_ms,
        };
        let mut config = self.config.clone();
        config.chromosome_length = tasks.iter().map(|t| t.activities.len()).sum();
        let found = BrkgaRunner::run(&decoder, &config)?;
        let keyed = decoder.schedule(&found.best_keys);
        let keyed_cost = cost(&keyed, tasks, self.tardiness_weight);
        let (schedule, cost) = if keyed_cost < dispatching_cost {
            (keyed, keyed_cost)
        } else {
            (own, dispatching_cost)
        };
        Ok(DispatchSearchResult {
            schedule,
            cost,
            dispatching_cost,
            generations: found.generations,
            stop_reason: found.stop_reason,
        })
    }
}

/// `(1 − w)` × makespan + `w` × total tardiness, both in ms from the
/// schedule's clock.
fn cost(schedule: &Schedule, tasks: &[Task], tardiness_weight: f64) -> f64 {
    let makespan = schedule.makespan_ms() as f64;
    let tardiness: i64 = tasks
        .iter()
        .filter_map(|t| Some(t.tardiness_ms(schedule.task_completion_time(&t.id)?)))
        .sum();
    (1.0 - tardiness_weight) * makespan + tardiness_weight * tardiness as f64
}

struct Decoder<'a> {
    search: &'a DispatchSearch,
    problem: &'a Problem,
    start_time_ms: i64,
}

impl Decoder<'_> {
    fn schedule(&self, keys: &[f64]) -> Schedule {
        let mut keys = keys.iter().copied();
        let rule = KeyRule(
            self.problem
                .tasks()
                .iter()
                .map(|t| {
                    (
                        t.id.clone(),
                        keys.by_ref().take(t.activities.len()).collect(),
                    )
                })
                .collect(),
        );
        self.search
            .scheduler
            .clone()
            .with_rule_engine(RuleEngine::new().with_rule(rule))
            .schedule(self.problem, self.start_time_ms)
    }
}

impl BrkgaDecoder for Decoder<'_> {
    fn decode(&self, keys: &[f64]) -> f64 {
        cost(
            &self.schedule(keys),
            self.problem.tasks(),
            self.search.tardiness_weight,
        )
    }

    /// The tasks by priority — the order the scheduler takes them in without
    /// a rule: every activity of a task keyed by its task's rank.
    fn seed_chromosome<R: rand::Rng>(&self, _rng: &mut R) -> Option<Vec<f64>> {
        let tasks = self.problem.tasks();
        let mut order: Vec<usize> = (0..tasks.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(tasks[i].priority));
        let mut rank = vec![0.0; tasks.len()];
        for (r, i) in order.into_iter().enumerate() {
            rank[i] = r as f64 / tasks.len() as f64;
        }
        Some(
            tasks
                .iter()
                .zip(rank)
                .flat_map(|(t, r)| std::iter::repeat_n(r, t.activities.len()))
                .collect(),
        )
    }
}

/// The lower key first: each task's key for the activity it would start
/// next (its first, when the decision does not say).
#[derive(Debug)]
struct KeyRule(HashMap<String, Vec<f64>>);

impl DispatchingRule for KeyRule {
    fn name(&self) -> &'static str {
        "KEY"
    }

    fn evaluate(&self, task: &Task, context: &SchedulingContext) -> RuleScore {
        let next = context
            .imminent_activity
            .get(&task.id)
            .copied()
            .unwrap_or(0);
        self.0
            .get(&task.id)
            .and_then(|keys| keys.get(next))
            .copied()
            .unwrap_or(f64::INFINITY)
    }

    fn description(&self) -> &'static str {
        "the search key of the task's next activity, lower first"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Activity, Resource, ResourceRequirement};

    /// Two tasks on one machine, the later-listed one due early: dispatching
    /// by priority (input order) makes it late; ordering by key need not.
    fn problem() -> Problem {
        let on = |id: &str, task: &str, ms: i64| {
            Activity::new(id, task, 0)
                .with_process_time(ms)
                .with_requirement(ResourceRequirement::new("Machine"))
        };
        Problem::new(
            vec![
                Task::new("LONG").with_activity(on("L", "LONG", 10_000)),
                Task::new("URGENT")
                    .with_activity(on("U", "URGENT", 1_000))
                    .with_deadline(1_000),
            ],
            vec![Resource::new("M", "Machine")],
        )
        .expect("a valid problem")
    }

    fn config() -> BrkgaConfig {
        BrkgaConfig::new(1)
            .with_population_size(20)
            .with_max_generations(20)
            .with_seed(7)
    }

    #[test]
    fn the_search_finds_an_order_dispatching_misses() {
        let search =
            DispatchSearch::new(SimpleScheduler::new(), config()).with_tardiness_weight(1.0);
        let found = search.search(&problem(), 0).expect("a valid search");
        assert_eq!(found.dispatching_cost, 10_000.0); // URGENT ends at 11 000
        assert_eq!(found.cost, 0.0);
        assert_eq!(
            found
                .schedule
                .assignment_for_activity("U")
                .unwrap()
                .start_ms,
            0
        );
        assert!(found.schedule.violations.is_empty());
    }

    #[test]
    fn the_search_is_never_worse_than_dispatching() {
        // Makespan only: every order gives 11 000.
        let found = DispatchSearch::new(SimpleScheduler::new(), config())
            .search(&problem(), 0)
            .expect("a valid search");
        assert_eq!(found.cost, found.dispatching_cost);
    }

    #[test]
    fn a_tardiness_weight_outside_zero_to_one_is_refused() {
        let refused = DispatchSearch::new(SimpleScheduler::new(), config())
            .with_tardiness_weight(1.5)
            .search(&problem(), 0);
        assert!(refused.is_err());
    }
}
