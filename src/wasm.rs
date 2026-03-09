//! WASM bindings for u-schedule.
//!
//! # Functions
//!
//! - [`run_schedule`]: Priority dispatching on a flat job list (single or
//!   parallel machines).
//! - [`solve_jobshop`]: GA-based job-shop scheduling with multi-machine
//!   routing and precedence constraints.
//!
//! # Supported Dispatching Rules
//!
//! | Rule | Description |
//! |------|-------------|
//! | `"SPT"` | Shortest Processing Time |
//! | `"LPT"` | Longest Processing Time |
//! | `"EDD"` | Earliest Due Date |
//! | `"FCFS"` | First Come First Served |
//! | `"CR"` | Critical Ratio |
//! | `"WSPT"` | Weighted Shortest Processing Time |
//! | `"MST"` | Minimum Slack Time |
//! | `"S/RO"` | Slack per Remaining Operations |
//! | `"ATC"` | Apparent Tardiness Cost |
//! | `"LWKR"` | Least Work Remaining |
//! | `"MWKR"` | Most Work Remaining |
//! | `"PRIORITY"` | Task Priority |
//!
//! # Time Convention
//! JSON uses seconds (f64). Internally, the scheduling model uses
//! milliseconds (i64), converted by multiplying/dividing by 1000.

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use crate::dispatching::rules;
use crate::dispatching::{RuleEngine, SchedulingContext};
use crate::ga::operators::{CrossoverType, GeneticOperators, MutationType};
use crate::ga::SchedulingGaProblem;
use crate::models::{
    Activity, ActivityDuration, Resource, ResourceRequirement, ResourceType, Task,
};
use u_metaheur::ga::{GaConfig, GaRunner};

// ── helpers ──────────────────────────────────────────────────────────────────

fn js_err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

// ══════════════════════════════════════════════════════════════════════════════
// run_schedule — dispatching rules (single / parallel machines)
// ══════════════════════════════════════════════════════════════════════════════

// ── input schema ─────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct InputJob {
    id: String,
    processing_time: f64,
    #[serde(default)]
    due_date: Option<f64>,
    #[serde(default)]
    release_time: Option<f64>,
    #[serde(default = "default_weight")]
    weight: f64,
}

fn default_weight() -> f64 {
    1.0
}

fn default_rule() -> String {
    "SPT".to_string()
}

fn default_num_machines() -> usize {
    1
}

fn default_atc_k() -> f64 {
    2.0
}

#[derive(Deserialize, Default)]
struct ScheduleConfig {
    #[serde(default = "default_rule")]
    rule: String,
    /// Number of identical parallel machines (default: 1 = single machine).
    #[serde(default = "default_num_machines")]
    num_machines: usize,
    /// Lookahead parameter for ATC rule (default: 2.0).
    #[serde(default = "default_atc_k")]
    atc_k: f64,
}

#[derive(Deserialize)]
struct ScheduleInput {
    jobs: Vec<InputJob>,
    #[serde(default)]
    config: ScheduleConfig,
}

// ── output schema ────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct OutputJob {
    id: String,
    start: f64,
    end: f64,
    tardiness: f64,
    /// Machine index (0-based). Always 0 for single-machine mode.
    machine: usize,
}

#[derive(Serialize)]
struct MachineUtilization {
    machine: usize,
    busy_time: f64,
    utilization: f64,
}

#[derive(Serialize)]
struct ScheduleOutput {
    schedule: Vec<OutputJob>,
    makespan: f64,
    total_tardiness: f64,
    /// Per-machine utilization. Present only when `num_machines > 1`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    machine_utilization: Vec<MachineUtilization>,
}

// ── conversion helpers ──────────────────────────────────────────────────────

/// Seconds -> milliseconds (i64).
fn sec_to_ms(secs: f64) -> i64 {
    (secs * 1_000.0).round() as i64
}

/// Milliseconds -> seconds (f64).
fn ms_to_sec(ms: i64) -> f64 {
    ms as f64 / 1_000.0
}

/// Build a `Task` with a single activity from an `InputJob`.
///
/// - `processing_time` (seconds) -> `ActivityDuration::fixed` (ms)
/// - `due_date` (seconds) -> `Task::deadline` (ms)
/// - `release_time` (seconds) -> `Task::release_time` (ms)
/// - `weight` is stored in `Task::priority` as `(weight * 1000.0) as i32`
fn build_task(job: &InputJob) -> Task {
    let duration_ms = sec_to_ms(job.processing_time);
    let activity = Activity::new(format!("{}_O1", job.id), &job.id, 0)
        .with_duration(ActivityDuration::fixed(duration_ms));

    let priority = (job.weight * 1_000.0).round() as i32;

    let mut task = Task::new(&job.id)
        .with_priority(priority)
        .with_activity(activity);

    if let Some(dd) = job.due_date {
        task.deadline = Some(sec_to_ms(dd));
    }
    if let Some(rt) = job.release_time {
        task.release_time = Some(sec_to_ms(rt));
    }

    task
}

// ── rule selection ──────────────────────────────────────────────────────────

fn build_engine(rule: &str, config: &ScheduleConfig) -> Result<RuleEngine, String> {
    match rule {
        "SPT" => Ok(RuleEngine::new().with_rule(rules::Spt)),
        "LPT" => Ok(RuleEngine::new().with_rule(rules::Lpt)),
        "EDD" => Ok(RuleEngine::new().with_rule(rules::Edd)),
        "FCFS" => Ok(RuleEngine::new().with_rule(rules::Fifo)),
        "CR" => Ok(RuleEngine::new().with_rule(rules::Cr)),
        "WSPT" => Ok(RuleEngine::new().with_rule(rules::Wspt)),
        // ── newly exposed rules ──
        "MST" => Ok(RuleEngine::new().with_rule(rules::Mst)),
        "S/RO" | "SRO" => Ok(RuleEngine::new().with_rule(rules::Sro)),
        "ATC" => Ok(RuleEngine::new().with_rule(rules::Atc::with_k(config.atc_k))),
        "LWKR" => Ok(RuleEngine::new().with_rule(rules::Lwkr)),
        "MWKR" => Ok(RuleEngine::new().with_rule(rules::Mwkr)),
        "PRIORITY" => Ok(RuleEngine::new().with_rule(rules::Priority)),
        other => Err(format!(
            "Unknown rule '{}'. Supported: SPT, LPT, EDD, FCFS, CR, WSPT, \
             MST, S/RO, ATC, LWKR, MWKR, PRIORITY",
            other
        )),
    }
}

// ── simulation ──────────────────────────────────────────────────────────────

/// Simulate a non-preemptive single-machine schedule.
///
/// 1. Sort tasks by the chosen dispatching rule (at t=0, static priority).
/// 2. Process tasks in that order, respecting `release_time`.
fn simulate_single(tasks: &[Task], engine: &RuleEngine) -> Vec<OutputJob> {
    let context = SchedulingContext::at_time(0);
    let order = engine.sort_indices(tasks, &context);

    let mut current_time_ms: i64 = 0;
    let mut result = Vec::with_capacity(tasks.len());

    for idx in order {
        let task = &tasks[idx];
        let release_ms = task.release_time.unwrap_or(0);

        let start_ms = current_time_ms.max(release_ms);
        let duration_ms = task.total_duration_ms();
        let end_ms = start_ms + duration_ms;

        let tardiness_ms = if let Some(deadline) = task.deadline {
            (end_ms - deadline).max(0)
        } else {
            0
        };

        result.push(OutputJob {
            id: task.id.clone(),
            start: ms_to_sec(start_ms),
            end: ms_to_sec(end_ms),
            tardiness: ms_to_sec(tardiness_ms),
            machine: 0,
        });

        current_time_ms = end_ms;
    }

    result
}

/// Simulate identical parallel machines using LPT-style list scheduling.
///
/// Tasks are sorted by the dispatching rule, then greedily assigned to
/// the machine with the earliest available time (ties broken by lowest
/// index). This is a standard list-scheduling heuristic.
///
/// # Reference
/// Graham (1969), "Bounds on multiprocessing timing anomalies"
fn simulate_parallel(tasks: &[Task], engine: &RuleEngine, num_machines: usize) -> Vec<OutputJob> {
    let context = SchedulingContext::at_time(0);
    let order = engine.sort_indices(tasks, &context);

    // Per-machine earliest available time.
    let mut machine_available = vec![0_i64; num_machines];
    let mut result = Vec::with_capacity(tasks.len());

    for idx in order {
        let task = &tasks[idx];
        let release_ms = task.release_time.unwrap_or(0);
        let duration_ms = task.total_duration_ms();

        // Pick machine with earliest availability.
        let (m, &earliest) = machine_available
            .iter()
            .enumerate()
            .min_by_key(|&(_, &t)| t)
            .expect("num_machines >= 1");

        let start_ms = earliest.max(release_ms);
        let end_ms = start_ms + duration_ms;

        let tardiness_ms = if let Some(deadline) = task.deadline {
            (end_ms - deadline).max(0)
        } else {
            0
        };

        result.push(OutputJob {
            id: task.id.clone(),
            start: ms_to_sec(start_ms),
            end: ms_to_sec(end_ms),
            tardiness: ms_to_sec(tardiness_ms),
            machine: m,
        });

        machine_available[m] = end_ms;
    }

    result
}

/// Compute per-machine utilization from a schedule and makespan.
fn compute_utilization(
    schedule: &[OutputJob],
    num_machines: usize,
    makespan: f64,
) -> Vec<MachineUtilization> {
    if makespan <= 0.0 {
        return Vec::new();
    }
    let mut busy = vec![0.0_f64; num_machines];
    for job in schedule {
        busy[job.machine] += job.end - job.start;
    }
    busy.iter()
        .enumerate()
        .map(|(i, &b)| MachineUtilization {
            machine: i,
            busy_time: b,
            utilization: b / makespan,
        })
        .collect()
}

// ── public API: run_schedule ────────────────────────────────────────────────

/// Run a dispatching schedule on single or parallel identical machines.
///
/// # Arguments
/// `jobs_json` -- A JS object matching `ScheduleInput`.
///
/// # Returns
/// A JS object matching `ScheduleOutput` on success, or a JS string error.
///
/// # Backward Compatibility
/// When `config.num_machines` is omitted (defaults to 1), behavior is
/// identical to the original single-machine implementation.
#[wasm_bindgen]
pub fn run_schedule(jobs_json: JsValue) -> Result<JsValue, JsValue> {
    let input: ScheduleInput = serde_wasm_bindgen::from_value(jobs_json).map_err(js_err)?;

    if input.jobs.is_empty() {
        let output = ScheduleOutput {
            schedule: vec![],
            makespan: 0.0,
            total_tardiness: 0.0,
            machine_utilization: vec![],
        };
        return serde_wasm_bindgen::to_value(&output).map_err(js_err);
    }

    let num_machines = input.config.num_machines.max(1);
    let engine = build_engine(&input.config.rule, &input.config).map_err(js_err)?;
    let tasks: Vec<Task> = input.jobs.iter().map(build_task).collect();

    let schedule = if num_machines == 1 {
        simulate_single(&tasks, &engine)
    } else {
        simulate_parallel(&tasks, &engine, num_machines)
    };

    let makespan = schedule.iter().map(|j| j.end).fold(0.0_f64, f64::max);
    let total_tardiness: f64 = schedule.iter().map(|j| j.tardiness).sum();

    let machine_utilization = if num_machines > 1 {
        compute_utilization(&schedule, num_machines, makespan)
    } else {
        vec![]
    };

    let output = ScheduleOutput {
        schedule,
        makespan,
        total_tardiness,
        machine_utilization,
    };

    serde_wasm_bindgen::to_value(&output).map_err(js_err)
}

// ══════════════════════════════════════════════════════════════════════════════
// solve_jobshop — GA-based job-shop scheduling
// ══════════════════════════════════════════════════════════════════════════════

// ── input schema ────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct JobShopOperation {
    /// Machine ID (e.g., "M1", "M2"). If multiple candidates, use an array.
    #[serde(default)]
    machine: Option<String>,
    /// Candidate machine IDs. Takes precedence over `machine`.
    #[serde(default)]
    machines: Vec<String>,
    /// Processing time in seconds.
    processing_time: f64,
}

impl JobShopOperation {
    /// Resolved candidate machine IDs.
    fn candidates(&self) -> Vec<String> {
        if !self.machines.is_empty() {
            self.machines.clone()
        } else if let Some(ref m) = self.machine {
            vec![m.clone()]
        } else {
            vec![]
        }
    }
}

#[derive(Deserialize)]
struct JobShopJob {
    id: String,
    operations: Vec<JobShopOperation>,
    #[serde(default)]
    due_date: Option<f64>,
    #[serde(default)]
    release_time: Option<f64>,
}

fn default_population_size() -> usize {
    100
}

fn default_max_generations() -> usize {
    200
}

fn default_mutation_rate() -> f64 {
    0.1
}

fn default_tardiness_weight() -> f64 {
    0.5
}

fn default_crossover_type() -> String {
    "POX".to_string()
}

fn default_mutation_type() -> String {
    "Swap".to_string()
}

#[derive(Deserialize)]
struct JobShopGaConfig {
    #[serde(default = "default_population_size")]
    population_size: usize,
    #[serde(default = "default_max_generations")]
    max_generations: usize,
    #[serde(default = "default_mutation_rate")]
    mutation_rate: f64,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default = "default_tardiness_weight")]
    tardiness_weight: f64,
    #[serde(default = "default_crossover_type")]
    crossover: String,
    #[serde(default = "default_mutation_type")]
    mutation: String,
}

impl Default for JobShopGaConfig {
    fn default() -> Self {
        Self {
            population_size: default_population_size(),
            max_generations: default_max_generations(),
            mutation_rate: default_mutation_rate(),
            seed: None,
            tardiness_weight: default_tardiness_weight(),
            crossover: default_crossover_type(),
            mutation: default_mutation_type(),
        }
    }
}

#[derive(Deserialize)]
struct JobShopInput {
    jobs: Vec<JobShopJob>,
    /// Number of machines. If omitted, inferred from operation machine IDs.
    #[serde(default)]
    num_machines: Option<usize>,
    #[serde(default)]
    ga_config: JobShopGaConfig,
}

// ── output schema ───────────────────────────────────────────────────────────

#[derive(Serialize)]
struct JobShopAssignment {
    job_id: String,
    operation: usize,
    machine: String,
    start: f64,
    end: f64,
}

#[derive(Serialize)]
struct JobShopOutput {
    schedule: Vec<JobShopAssignment>,
    makespan: f64,
    fitness: f64,
    generations: usize,
    fitness_history: Vec<f64>,
}

// ── conversion helpers ──────────────────────────────────────────────────────

fn parse_crossover(s: &str) -> Result<CrossoverType, String> {
    match s {
        "POX" => Ok(CrossoverType::POX),
        "LOX" => Ok(CrossoverType::LOX),
        "JOX" => Ok(CrossoverType::JOX),
        other => Err(format!(
            "Unknown crossover '{}'. Supported: POX, LOX, JOX",
            other
        )),
    }
}

fn parse_mutation(s: &str) -> Result<MutationType, String> {
    match s {
        "Swap" => Ok(MutationType::Swap),
        "Insert" => Ok(MutationType::Insert),
        "Invert" => Ok(MutationType::Invert),
        other => Err(format!(
            "Unknown mutation '{}'. Supported: Swap, Insert, Invert",
            other
        )),
    }
}

// ── GA config validation ────────────────────────────────────────────────────

/// Validates `JobShopGaConfig` fields before they are passed to `GaConfig`.
///
/// This catches invalid values at the WASM boundary with clear error messages,
/// preventing panics deeper in the GA runner.
fn validate_ga_config(cfg: &JobShopGaConfig) -> Result<(), String> {
    if cfg.population_size < 2 {
        return Err(format!(
            "ga_config.population_size must be >= 2, got {}",
            cfg.population_size
        ));
    }
    if cfg.max_generations < 1 {
        return Err(format!(
            "ga_config.max_generations must be >= 1, got {}",
            cfg.max_generations
        ));
    }
    if !(0.0..=1.0).contains(&cfg.mutation_rate) {
        return Err(format!(
            "ga_config.mutation_rate must be in 0.0..=1.0, got {}",
            cfg.mutation_rate
        ));
    }
    if !(0.0..=1.0).contains(&cfg.tardiness_weight) {
        return Err(format!(
            "ga_config.tardiness_weight must be in 0.0..=1.0, got {}",
            cfg.tardiness_weight
        ));
    }
    Ok(())
}

// ── public API: solve_jobshop ───────────────────────────────────────────────

/// Solve a job-shop scheduling problem using Genetic Algorithm.
///
/// # Arguments
/// `problem_json` -- A JS object matching `JobShopInput`.
///
/// # Returns
/// A JS object matching `JobShopOutput` on success, or a JS string error.
///
/// # Input Format
/// ```json
/// {
///   "jobs": [
///     {
///       "id": "J1",
///       "operations": [
///         { "machine": "M1", "processing_time": 3.0 },
///         { "machines": ["M2", "M3"], "processing_time": 2.0 }
///       ],
///       "due_date": 15.0,
///       "release_time": 0.0
///     }
///   ],
///   "num_machines": 3,
///   "ga_config": {
///     "population_size": 100,
///     "max_generations": 200,
///     "mutation_rate": 0.1,
///     "seed": 42,
///     "tardiness_weight": 0.5,
///     "crossover": "POX",
///     "mutation": "Swap"
///   }
/// }
/// ```
#[wasm_bindgen]
pub fn solve_jobshop(problem_json: JsValue) -> Result<JsValue, JsValue> {
    let input: JobShopInput = serde_wasm_bindgen::from_value(problem_json).map_err(js_err)?;

    if input.jobs.is_empty() {
        let output = JobShopOutput {
            schedule: vec![],
            makespan: 0.0,
            fitness: 0.0,
            generations: 0,
            fitness_history: vec![],
        };
        return serde_wasm_bindgen::to_value(&output).map_err(js_err);
    }

    // ── Collect all machine IDs ──
    let mut machine_ids: Vec<String> = Vec::new();
    for job in &input.jobs {
        for op in &job.operations {
            for m in op.candidates() {
                if !machine_ids.contains(&m) {
                    machine_ids.push(m);
                }
            }
        }
    }
    machine_ids.sort();

    // Validate: if num_machines given, ensure enough machines exist.
    if let Some(n) = input.num_machines {
        // Pad machine IDs to match num_machines if needed.
        while machine_ids.len() < n {
            machine_ids.push(format!("M{}", machine_ids.len() + 1));
        }
    }

    if machine_ids.is_empty() {
        return Err(js_err("No machines specified in operations"));
    }

    // ── Build domain Tasks ──
    let mut tasks = Vec::with_capacity(input.jobs.len());
    for job in &input.jobs {
        let mut task = Task::new(&job.id);

        if let Some(dd) = job.due_date {
            task.deadline = Some(sec_to_ms(dd));
        }
        if let Some(rt) = job.release_time {
            task.release_time = Some(sec_to_ms(rt));
        }

        for (i, op) in job.operations.iter().enumerate() {
            let candidates = op.candidates();
            if candidates.is_empty() {
                return Err(js_err(format!(
                    "Job '{}' operation {} has no machine specified",
                    job.id, i
                )));
            }

            let activity = Activity::new(format!("{}_{}", job.id, i + 1), &job.id, i as i32)
                .with_duration(ActivityDuration::fixed(sec_to_ms(op.processing_time)))
                .with_requirement(ResourceRequirement::new("Machine").with_candidates(candidates));

            task = task.with_activity(activity);
        }

        tasks.push(task);
    }

    // ── Build Resources ──
    let resources: Vec<Resource> = machine_ids
        .iter()
        .map(|id| Resource::new(id, ResourceType::Primary))
        .collect();

    // ── Configure operators ──
    let crossover_type = parse_crossover(&input.ga_config.crossover).map_err(js_err)?;
    let mutation_type = parse_mutation(&input.ga_config.mutation).map_err(js_err)?;

    // ── Build GA problem ──
    let problem = SchedulingGaProblem::new(&tasks, &resources)
        .with_tardiness_weight(input.ga_config.tardiness_weight)
        .with_operators(GeneticOperators {
            crossover_type,
            mutation_type,
        });

    // ── Validate & configure GA ──
    validate_ga_config(&input.ga_config).map_err(js_err)?;

    // Compute a safe elite_ratio: ensure at least 1 elite for any population_size.
    // Default 0.1 gives elite_count=0 when population_size < 10.
    let elite_ratio = {
        let pop = input.ga_config.population_size;
        let default_ratio = 0.1_f64;
        let min_ratio = 1.0 / pop as f64; // guarantees at least 1 elite
        default_ratio.max(min_ratio)
    };

    let mut config = GaConfig::default()
        .with_population_size(input.ga_config.population_size)
        .with_max_generations(input.ga_config.max_generations)
        .with_mutation_rate(input.ga_config.mutation_rate)
        .with_elite_ratio(elite_ratio)
        .with_parallel(false); // WASM is single-threaded

    if let Some(seed) = input.ga_config.seed {
        config = config.with_seed(seed);
    }

    // Defence-in-depth: call GaConfig's own validation as well.
    config.validate().map_err(js_err)?;

    // ── Run GA ──
    let result = GaRunner::run(&problem, &config).map_err(js_err)?;

    // ── Decode best solution ──
    let best_schedule = problem.decode(&result.best);

    let schedule: Vec<JobShopAssignment> = best_schedule
        .assignments
        .iter()
        .map(|a| {
            // Parse operation index from activity_id pattern "JOB_N"
            let operation = a
                .activity_id
                .rsplit('_')
                .next()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(1);

            JobShopAssignment {
                job_id: a.task_id.clone(),
                operation,
                machine: a.resource_id.clone(),
                start: ms_to_sec(a.start_ms),
                end: ms_to_sec(a.end_ms),
            }
        })
        .collect();

    let makespan = ms_to_sec(best_schedule.makespan_ms());

    let output = JobShopOutput {
        schedule,
        makespan,
        fitness: result.best_fitness,
        generations: result.generations,
        fitness_history: result.fitness_history,
    };

    serde_wasm_bindgen::to_value(&output).map_err(js_err)
}

// ── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::useless_vec)]
mod tests {
    use super::*;

    fn make_input_job(
        id: &str,
        pt: f64,
        due: Option<f64>,
        release: Option<f64>,
        weight: f64,
    ) -> InputJob {
        InputJob {
            id: id.to_string(),
            processing_time: pt,
            due_date: due,
            release_time: release,
            weight,
        }
    }

    fn run(jobs: Vec<InputJob>, rule: &str) -> (Vec<OutputJob>, f64, f64) {
        let config = ScheduleConfig {
            rule: rule.to_string(),
            num_machines: 1,
            atc_k: 2.0,
        };
        let input = ScheduleInput { jobs, config };
        let tasks: Vec<Task> = input.jobs.iter().map(build_task).collect();
        let engine = build_engine(&input.config.rule, &input.config).expect("valid rule");
        let schedule = simulate_single(&tasks, &engine);
        let makespan = schedule.iter().map(|j| j.end).fold(0.0_f64, f64::max);
        let total_tardiness: f64 = schedule.iter().map(|j| j.tardiness).sum();
        (schedule, makespan, total_tardiness)
    }

    // ── existing tests (backward compatibility) ──

    #[test]
    fn test_spt_order() {
        let jobs = vec![
            make_input_job("A", 5.0, None, None, 1.0),
            make_input_job("B", 2.0, None, None, 1.0),
            make_input_job("C", 8.0, None, None, 1.0),
        ];
        let (schedule, makespan, _) = run(jobs, "SPT");
        // B(2) -> A(5) -> C(8)
        assert_eq!(schedule[0].id, "B");
        assert_eq!(schedule[1].id, "A");
        assert_eq!(schedule[2].id, "C");
        assert!((makespan - 15.0).abs() < 1e-9);
    }

    #[test]
    fn test_edd_order() {
        let jobs = vec![
            make_input_job("A", 3.0, Some(10.0), None, 1.0),
            make_input_job("B", 3.0, Some(5.0), None, 1.0),
            make_input_job("C", 3.0, None, None, 1.0),
        ];
        let (schedule, _, _) = run(jobs, "EDD");
        assert_eq!(schedule[0].id, "B");
        assert_eq!(schedule[1].id, "A");
        assert_eq!(schedule[2].id, "C");
    }

    #[test]
    fn test_lpt_order() {
        let jobs = vec![
            make_input_job("S", 1.0, None, None, 1.0),
            make_input_job("L", 9.0, None, None, 1.0),
        ];
        let (schedule, _, _) = run(jobs, "LPT");
        assert_eq!(schedule[0].id, "L");
    }

    #[test]
    fn test_tardiness_computed() {
        let jobs = vec![
            make_input_job("A", 3.0, Some(2.0), None, 1.0),
            make_input_job("B", 3.0, Some(10.0), None, 1.0),
        ];
        let (schedule, _, total_tardiness) = run(jobs, "SPT");
        let a = schedule.iter().find(|j| j.id == "A").expect("A");
        assert!((a.tardiness - 1.0).abs() < 1e-9);
        let b = schedule.iter().find(|j| j.id == "B").expect("B");
        assert!((b.tardiness - 0.0).abs() < 1e-9);
        assert!((total_tardiness - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_release_time_respected() {
        let jobs = vec![
            make_input_job("A", 2.0, None, Some(5.0), 1.0),
            make_input_job("B", 3.0, None, Some(0.0), 1.0),
        ];
        let (schedule, makespan, _) = run(jobs, "SPT");
        let a = schedule.iter().find(|j| j.id == "A").expect("A");
        assert!(
            (a.start - 5.0).abs() < 1e-9,
            "A start should be 5.0 (release respected)"
        );
        assert!((a.end - 7.0).abs() < 1e-9);
        assert!((makespan - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_jobs() {
        let (schedule, makespan, tardiness) = run(vec![], "SPT");
        assert!(schedule.is_empty());
        assert!((makespan - 0.0).abs() < 1e-9);
        assert!((tardiness - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_unknown_rule_error() {
        let config = ScheduleConfig {
            rule: "UNKNOWN".to_string(),
            num_machines: 1,
            atc_k: 2.0,
        };
        assert!(build_engine("UNKNOWN", &config).is_err());
    }

    #[test]
    fn test_wspt_weight() {
        let jobs = vec![
            make_input_job("A", 5.0, None, None, 1.0),
            make_input_job("B", 2.0, None, None, 1.0),
        ];
        let (schedule, _, _) = run(jobs, "WSPT");
        assert_eq!(
            schedule[0].id, "B",
            "equal weight: shorter job goes first (SPT-like)"
        );
    }

    // ── new rule tests ──

    #[test]
    fn test_mst_rule() {
        let jobs = vec![
            make_input_job("A", 3.0, Some(10.0), None, 1.0),
            make_input_job("B", 3.0, Some(5.0), None, 1.0),
        ];
        let (schedule, _, _) = run(jobs, "MST");
        // B has tighter deadline → less slack → higher priority
        assert_eq!(schedule[0].id, "B");
    }

    #[test]
    fn test_atc_rule() {
        let jobs = vec![
            make_input_job("A", 3.0, Some(20.0), None, 1.0),
            make_input_job("B", 3.0, Some(4.0), None, 1.0),
        ];
        let (schedule, _, _) = run(jobs, "ATC");
        // B has very tight deadline → urgency boost → scheduled first
        assert_eq!(schedule[0].id, "B");
    }

    #[test]
    fn test_lwkr_rule() {
        let jobs = vec![
            make_input_job("A", 5.0, None, None, 1.0),
            make_input_job("B", 2.0, None, None, 1.0),
        ];
        let (schedule, _, _) = run(jobs, "LWKR");
        // LWKR prefers least remaining work → B first (shorter)
        assert_eq!(schedule[0].id, "B");
    }

    #[test]
    fn test_mwkr_rule() {
        let jobs = vec![
            make_input_job("A", 5.0, None, None, 1.0),
            make_input_job("B", 2.0, None, None, 1.0),
        ];
        let (schedule, _, _) = run(jobs, "MWKR");
        // MWKR prefers most remaining work → A first (longer)
        assert_eq!(schedule[0].id, "A");
    }

    #[test]
    fn test_priority_rule() {
        // weight maps to priority via (weight * 1000) as i32
        // PRIORITY rule: score = -(priority) → higher priority first
        let jobs = vec![
            make_input_job("A", 3.0, None, None, 1.0),  // priority=1000
            make_input_job("B", 3.0, None, None, 10.0), // priority=10000
        ];
        let (schedule, _, _) = run(jobs, "PRIORITY");
        // B has higher priority → scheduled first
        assert_eq!(schedule[0].id, "B");
    }

    #[test]
    fn test_sro_alias() {
        // "SRO" alias for "S/RO" should work
        let config = ScheduleConfig {
            rule: "SRO".to_string(),
            num_machines: 1,
            atc_k: 2.0,
        };
        assert!(build_engine("SRO", &config).is_ok());
    }

    // ── parallel machine tests ──

    #[test]
    fn test_parallel_two_machines() {
        // Two identical jobs on 2 machines should run in parallel.
        let jobs = vec![
            make_input_job("A", 5.0, None, None, 1.0),
            make_input_job("B", 5.0, None, None, 1.0),
        ];
        let config = ScheduleConfig {
            rule: "SPT".to_string(),
            num_machines: 2,
            atc_k: 2.0,
        };
        let tasks: Vec<Task> = jobs.iter().map(build_task).collect();
        let engine = build_engine(&config.rule, &config).expect("valid rule");
        let schedule = simulate_parallel(&tasks, &engine, 2);

        let makespan = schedule.iter().map(|j| j.end).fold(0.0_f64, f64::max);
        // Both on separate machines → makespan = 5, not 10
        assert!((makespan - 5.0).abs() < 1e-9);
        // Jobs should be on different machines
        assert_ne!(schedule[0].machine, schedule[1].machine);
    }

    #[test]
    fn test_parallel_utilization() {
        let jobs = vec![
            make_input_job("A", 4.0, None, None, 1.0),
            make_input_job("B", 2.0, None, None, 1.0),
            make_input_job("C", 6.0, None, None, 1.0),
        ];
        let config = ScheduleConfig {
            rule: "LPT".to_string(),
            num_machines: 2,
            atc_k: 2.0,
        };
        let tasks: Vec<Task> = jobs.iter().map(build_task).collect();
        let engine = build_engine(&config.rule, &config).expect("valid rule");
        let schedule = simulate_parallel(&tasks, &engine, 2);

        let makespan = schedule.iter().map(|j| j.end).fold(0.0_f64, f64::max);
        let util = compute_utilization(&schedule, 2, makespan);

        assert_eq!(util.len(), 2);
        // Total busy time should equal sum of processing times (12s)
        let total_busy: f64 = util.iter().map(|u| u.busy_time).sum();
        assert!((total_busy - 12.0).abs() < 1e-9);
    }

    #[test]
    fn test_single_machine_no_utilization() {
        // num_machines=1 → machine_utilization should be empty
        let jobs = vec![make_input_job("A", 3.0, None, None, 1.0)];
        let config = ScheduleConfig {
            rule: "SPT".to_string(),
            num_machines: 1,
            atc_k: 2.0,
        };
        let tasks: Vec<Task> = jobs.iter().map(build_task).collect();
        let engine = build_engine(&config.rule, &config).expect("valid rule");
        let schedule = simulate_single(&tasks, &engine);

        let makespan = schedule.iter().map(|j| j.end).fold(0.0_f64, f64::max);
        // Single machine → no utilization reported
        let util = compute_utilization(&schedule, 1, makespan);
        // (utilization is computed but for 1 machine it would just be 1.0)
        assert_eq!(util.len(), 1);
        assert!((util[0].utilization - 1.0).abs() < 1e-9);
    }

    // ── jobshop tests ──

    #[test]
    fn test_jobshop_basic() {
        // Classic 2-job, 2-machine job-shop instance
        let input = JobShopInput {
            jobs: vec![
                JobShopJob {
                    id: "J1".to_string(),
                    operations: vec![
                        JobShopOperation {
                            machine: Some("M1".to_string()),
                            machines: vec![],
                            processing_time: 3.0,
                        },
                        JobShopOperation {
                            machine: Some("M2".to_string()),
                            machines: vec![],
                            processing_time: 2.0,
                        },
                    ],
                    due_date: None,
                    release_time: None,
                },
                JobShopJob {
                    id: "J2".to_string(),
                    operations: vec![
                        JobShopOperation {
                            machine: Some("M2".to_string()),
                            machines: vec![],
                            processing_time: 4.0,
                        },
                        JobShopOperation {
                            machine: Some("M1".to_string()),
                            machines: vec![],
                            processing_time: 1.0,
                        },
                    ],
                    due_date: None,
                    release_time: None,
                },
            ],
            num_machines: Some(2),
            ga_config: JobShopGaConfig {
                population_size: 20,
                max_generations: 30,
                mutation_rate: 0.1,
                seed: Some(42),
                tardiness_weight: 0.0,
                crossover: "POX".to_string(),
                mutation: "Swap".to_string(),
            },
        };

        let tasks = build_jobshop_tasks(&input).expect("valid input");
        let machine_ids = collect_machine_ids(&input);
        let resources: Vec<Resource> = machine_ids
            .iter()
            .map(|id| Resource::new(id, ResourceType::Primary))
            .collect();

        let problem = SchedulingGaProblem::new(&tasks, &resources)
            .with_tardiness_weight(input.ga_config.tardiness_weight);

        let config = GaConfig::default()
            .with_population_size(20)
            .with_max_generations(30)
            .with_seed(42)
            .with_parallel(false);

        let result = GaRunner::run(&problem, &config).expect("GA should succeed");
        let schedule = problem.decode(&result.best);

        assert!(schedule.makespan_ms() > 0);
        assert_eq!(schedule.assignments.len(), 4); // 2 jobs * 2 ops
        assert!(result.best_fitness.is_finite());
    }

    #[test]
    fn test_jobshop_flexible() {
        // Flexible job-shop: operations can go on multiple machines
        let input = JobShopInput {
            jobs: vec![JobShopJob {
                id: "J1".to_string(),
                operations: vec![JobShopOperation {
                    machine: None,
                    machines: vec!["M1".to_string(), "M2".to_string()],
                    processing_time: 5.0,
                }],
                due_date: None,
                release_time: None,
            }],
            num_machines: Some(2),
            ga_config: JobShopGaConfig::default(),
        };

        let tasks = build_jobshop_tasks(&input).expect("valid input");
        assert_eq!(tasks[0].activities.len(), 1);
        // Activity should have 2 candidate machines
        let candidates = tasks[0].activities[0].candidate_resources();
        assert_eq!(candidates.len(), 2);
    }

    // ── test helpers for jobshop (used by tests only) ──

    fn collect_machine_ids(input: &JobShopInput) -> Vec<String> {
        let mut ids: Vec<String> = Vec::new();
        for job in &input.jobs {
            for op in &job.operations {
                for m in op.candidates() {
                    if !ids.contains(&m) {
                        ids.push(m);
                    }
                }
            }
        }
        ids.sort();
        if let Some(n) = input.num_machines {
            while ids.len() < n {
                ids.push(format!("M{}", ids.len() + 1));
            }
        }
        ids
    }

    fn build_jobshop_tasks(input: &JobShopInput) -> Result<Vec<Task>, String> {
        let mut tasks = Vec::new();
        for job in &input.jobs {
            let mut task = Task::new(&job.id);
            if let Some(dd) = job.due_date {
                task.deadline = Some(sec_to_ms(dd));
            }
            if let Some(rt) = job.release_time {
                task.release_time = Some(sec_to_ms(rt));
            }
            for (i, op) in job.operations.iter().enumerate() {
                let candidates = op.candidates();
                if candidates.is_empty() {
                    return Err(format!("Job '{}' operation {} has no machine", job.id, i));
                }
                let activity = Activity::new(format!("{}_{}", job.id, i + 1), &job.id, i as i32)
                    .with_duration(ActivityDuration::fixed(sec_to_ms(op.processing_time)))
                    .with_requirement(
                        ResourceRequirement::new("Machine").with_candidates(candidates),
                    );
                task = task.with_activity(activity);
            }
            tasks.push(task);
        }
        Ok(tasks)
    }

    // ── GA config validation tests ──

    #[test]
    fn test_validate_ga_config_defaults_ok() {
        let cfg = JobShopGaConfig::default();
        assert!(validate_ga_config(&cfg).is_ok());
    }

    #[test]
    fn test_validate_ga_config_population_size_zero() {
        let cfg = JobShopGaConfig {
            population_size: 0,
            ..Default::default()
        };
        let err = validate_ga_config(&cfg).unwrap_err();
        assert!(err.contains("population_size"), "got: {}", err);
    }

    #[test]
    fn test_validate_ga_config_population_size_one() {
        let cfg = JobShopGaConfig {
            population_size: 1,
            ..Default::default()
        };
        let err = validate_ga_config(&cfg).unwrap_err();
        assert!(err.contains("population_size"), "got: {}", err);
    }

    #[test]
    fn test_validate_ga_config_population_size_two_ok() {
        let cfg = JobShopGaConfig {
            population_size: 2,
            ..Default::default()
        };
        assert!(validate_ga_config(&cfg).is_ok());
    }

    #[test]
    fn test_validate_ga_config_max_generations_zero() {
        let cfg = JobShopGaConfig {
            max_generations: 0,
            ..Default::default()
        };
        let err = validate_ga_config(&cfg).unwrap_err();
        assert!(err.contains("max_generations"), "got: {}", err);
    }

    #[test]
    fn test_validate_ga_config_mutation_rate_negative() {
        let cfg = JobShopGaConfig {
            mutation_rate: -0.1,
            ..Default::default()
        };
        let err = validate_ga_config(&cfg).unwrap_err();
        assert!(err.contains("mutation_rate"), "got: {}", err);
    }

    #[test]
    fn test_validate_ga_config_mutation_rate_above_one() {
        let cfg = JobShopGaConfig {
            mutation_rate: 1.5,
            ..Default::default()
        };
        let err = validate_ga_config(&cfg).unwrap_err();
        assert!(err.contains("mutation_rate"), "got: {}", err);
    }

    #[test]
    fn test_validate_ga_config_mutation_rate_boundary_ok() {
        // 0.0 and 1.0 are both valid
        let cfg0 = JobShopGaConfig {
            mutation_rate: 0.0,
            ..Default::default()
        };
        assert!(validate_ga_config(&cfg0).is_ok());

        let cfg1 = JobShopGaConfig {
            mutation_rate: 1.0,
            ..Default::default()
        };
        assert!(validate_ga_config(&cfg1).is_ok());
    }

    #[test]
    fn test_validate_ga_config_tardiness_weight_invalid() {
        let cfg = JobShopGaConfig {
            tardiness_weight: -0.5,
            ..Default::default()
        };
        let err = validate_ga_config(&cfg).unwrap_err();
        assert!(err.contains("tardiness_weight"), "got: {}", err);

        let cfg2 = JobShopGaConfig {
            tardiness_weight: 2.0,
            ..Default::default()
        };
        let err2 = validate_ga_config(&cfg2).unwrap_err();
        assert!(err2.contains("tardiness_weight"), "got: {}", err2);
    }

    #[test]
    fn test_jobshop_invalid_population_returns_error() {
        // Verify that invalid GA config is caught as an error, not a panic.
        // We test via validate_ga_config since solve_jobshop requires JsValue (WASM).
        let cfg = JobShopGaConfig {
            population_size: 0,
            max_generations: 10,
            mutation_rate: 0.1,
            seed: Some(42),
            tardiness_weight: 0.0,
            crossover: "POX".to_string(),
            mutation: "Swap".to_string(),
        };
        assert!(validate_ga_config(&cfg).is_err());
    }

    #[test]
    fn test_jobshop_edge_case_single_job_single_op() {
        // Minimal valid jobshop: 1 job, 1 operation, 1 machine
        let input = JobShopInput {
            jobs: vec![JobShopJob {
                id: "J1".to_string(),
                operations: vec![JobShopOperation {
                    machine: Some("M1".to_string()),
                    machines: vec![],
                    processing_time: 5.0,
                }],
                due_date: None,
                release_time: None,
            }],
            num_machines: Some(1),
            ga_config: JobShopGaConfig {
                population_size: 4,
                max_generations: 5,
                mutation_rate: 0.1,
                seed: Some(1),
                tardiness_weight: 0.0,
                crossover: "POX".to_string(),
                mutation: "Swap".to_string(),
            },
        };

        let tasks = build_jobshop_tasks(&input).expect("valid input");
        let machine_ids = collect_machine_ids(&input);
        let resources: Vec<Resource> = machine_ids
            .iter()
            .map(|id| Resource::new(id, ResourceType::Primary))
            .collect();

        let problem = SchedulingGaProblem::new(&tasks, &resources);
        let config = GaConfig::default()
            .with_population_size(4)
            .with_max_generations(5)
            .with_seed(1)
            .with_parallel(false);

        config.validate().expect("config should be valid");
        let result = GaRunner::run(&problem, &config).expect("GA should succeed");
        let schedule = problem.decode(&result.best);

        assert_eq!(schedule.assignments.len(), 1);
        assert!(schedule.makespan_ms() > 0);
    }
}
