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
use serde_json::json;
use wasm_bindgen::prelude::*;

use crate::dispatching::rules;
use crate::dispatching::{RuleEngine, SchedulingContext};
use crate::ga::operators::{CrossoverType, GeneticOperators, MutationType};
use crate::ga::SchedulingGaProblem;
use crate::models::{
    Activity, ActivityDuration, Resource, ResourceRequirement, ResourceType, Task,
};
use crate::validation::{ValidationError, ValidationErrorKind};
use crate::Problem;
use u_metaheur::ga::{GaConfig, GaRunner};

// ── helpers ──────────────────────────────────────────────────────────────────

/// A refusal on its way to JavaScript: the text for `Error.message`, and the
/// fields -- `code` first among them -- copied onto the `Error`.
#[derive(Debug)]
struct WireError {
    message: String,
    fields: serde_json::Value,
}

impl WireError {
    fn new(code: &str, message: String, mut extra: serde_json::Value) -> Self {
        let mut fields = serde_json::Map::new();
        fields.insert("code".into(), json!(code));
        if let Some(extra) = extra.as_object_mut() {
            fields.append(extra);
        }
        WireError {
            message,
            fields: serde_json::Value::Object(fields),
        }
    }

    /// An argument that is not the shape the function takes: a JSON string
    /// instead of a value, a wrong type, a missing or unknown key.
    fn malformed_input(parameter: &str, message: String) -> Self {
        Self::new(
            "malformed_input",
            message,
            json!({ "parameter": parameter }),
        )
    }

    /// A string option that names none of the values the function knows.
    fn unknown_option(parameter: &str, got: &str, expected: &[&str]) -> Self {
        Self::new(
            "unknown_option",
            format!(
                "Unknown {parameter} '{got}'. Supported: {}",
                expected.join(", ")
            ),
            json!({ "parameter": parameter, "got": got, "expected": expected }),
        )
    }

    /// A numeric option outside the range the solver accepts.
    fn out_of_range(parameter: &str, min: f64, max: Option<f64>, got: f64) -> Self {
        let range = match max {
            Some(max) => format!("in {min}..={max}"),
            None => format!(">= {min}"),
        };
        Self::new(
            "parameter_out_of_range",
            format!("{parameter} must be {range}, got {got}"),
            json!({ "parameter": parameter, "min": min, "max": max, "got": got }),
        )
    }

    /// The stable reason, as the `code` field carries it.
    #[cfg(test)]
    fn code(&self) -> &str {
        self.fields["code"]
            .as_str()
            .expect("every refusal carries a code")
    }
}

impl std::fmt::Display for WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// The checked problem's first finding, with every finding's text in the
/// message. The code and fields are the first one's: they name one thing to
/// fix, which is what a program branching on them can act on.
impl From<Vec<ValidationError>> for WireError {
    fn from(errors: Vec<ValidationError>) -> Self {
        let message = errors
            .iter()
            .map(|e| e.message.as_str())
            .collect::<Vec<_>>()
            .join("; ");
        let Some(first) = errors.first() else {
            return WireError::new("invalid_input", message, json!({}));
        };
        let fields = match &first.kind {
            ValidationErrorKind::DuplicateId { entity, id } => {
                json!({ "entity": entity.name(), "id": id })
            }
            ValidationErrorKind::InvalidResourceReference { activity, resource } => {
                json!({ "activity": activity, "resource": resource })
            }
            ValidationErrorKind::CyclicDependency { activity } => json!({ "activity": activity }),
            ValidationErrorKind::EmptyTask { task } => json!({ "task": task }),
            ValidationErrorKind::InvalidPredecessor {
                activity,
                predecessor,
            } => json!({ "activity": activity, "predecessor": predecessor }),
        };
        WireError::new(first.kind.code(), message, fields)
    }
}

/// Every refusal crosses into JavaScript as an `Error` whose `message` is the
/// readable text and which carries `code` -- a stable reason -- and the values
/// behind it as further properties (`id`, `parameter`, `got`, ...). A program
/// branches on `err.code` and reads the fields; `err.message` reads as it
/// always did.
fn js_err(error: impl Into<WireError>) -> JsValue {
    let error = error.into();
    let js = js_sys::Error::new(&error.message);
    // `json_compatible` turns the map into a plain object; the default would
    // produce a JavaScript `Map`, which `Object.assign` does not read.
    if let Ok(fields) = error
        .fields
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
    {
        js_sys::Object::assign(&js, &fields.into());
    }
    js.into()
}

/// Serializes a response; a failure is reported rather than unwrapped.
fn to_js<T: Serialize>(value: &T) -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(value)
        .map_err(|e| js_err(WireError::malformed_input("result", e.to_string())))
}

/// Deserialize a native JS value, rejecting JSON strings with an actionable
/// message and prefixing the offending parameter name to any serde error.
fn from_js<T: serde::de::DeserializeOwned>(value: JsValue, param: &str) -> Result<T, JsValue> {
    let refuse = |message: String| js_err(WireError::malformed_input(param, message));
    if value.as_string().is_some() {
        return Err(refuse(format!(
            "{param}: expected a native JS object/array, got a string — \
             pass the value directly, not JSON.stringify(...)"
        )));
    }
    // serde-wasm-bindgen reads only a struct's declared fields from a JS
    // object, so `deny_unknown_fields` never sees extra keys. Round-trip
    // through serde_json::Value so the strict wire schema is enforced.
    let json: serde_json::Value =
        serde_wasm_bindgen::from_value(value).map_err(|e| refuse(format!("{param}: {e}")))?;
    serde_json::from_value(json).map_err(|e| refuse(format!("{param}: {e}")))
}

// ══════════════════════════════════════════════════════════════════════════════
// run_schedule — dispatching rules (single / parallel machines)
// ══════════════════════════════════════════════════════════════════════════════

// ── input schema ─────────────────────────────────────────────────────────────

#[derive(Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct InputJob {
    id: String,
    processing_time: f64,
    #[serde(default)]
    #[tsify(optional)]
    #[tsify(type = "number | null")]
    due_date: Option<f64>,
    #[serde(default)]
    #[tsify(optional)]
    #[tsify(type = "number | null")]
    release_time: Option<f64>,
    #[serde(default = "default_weight")]
    #[tsify(optional)]
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

#[derive(Deserialize, Default, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct ScheduleConfig {
    #[serde(default = "default_rule")]
    #[tsify(optional)]
    #[tsify(
        type = "\"SPT\" | \"LPT\" | \"EDD\" | \"FCFS\" | \"CR\" | \"WSPT\" | \"MST\" | \"S/RO\" | \"SRO\" | \"ATC\" | \"LWKR\" | \"MWKR\" | \"PRIORITY\""
    )]
    rule: String,
    /// Number of identical parallel machines (default: 1 = single machine).
    #[serde(default = "default_num_machines")]
    #[tsify(optional)]
    num_machines: usize,
    /// Lookahead parameter for ATC rule (default: 2.0).
    #[serde(default = "default_atc_k")]
    #[tsify(optional)]
    atc_k: f64,
}

#[derive(Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct ScheduleInput {
    jobs: Vec<InputJob>,
    #[serde(default)]
    #[tsify(optional)]
    config: ScheduleConfig,
}

// ── output schema ────────────────────────────────────────────────────────────

#[derive(Serialize, tsify::Tsify)]
struct OutputJob {
    id: String,
    start: f64,
    end: f64,
    tardiness: f64,
    /// Machine index (0-based). Always 0 for single-machine mode.
    machine: usize,
}

#[derive(Serialize, tsify::Tsify)]
struct MachineUtilization {
    machine: usize,
    busy_time: f64,
    utilization: f64,
}

#[derive(Serialize, tsify::Tsify)]
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

/// Refuses a job `id` given to two jobs.
///
/// The output names jobs by `id` alone. A dispatching schedule would list two
/// entries under one name that the caller cannot tell apart, and the job-shop
/// solver keys its problem by task id, so one of the two jobs silently left
/// the schedule.
fn refuse_repeated_ids<'a>(ids: impl IntoIterator<Item = &'a str>) -> Result<(), WireError> {
    let mut first_at: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for (position, id) in ids.into_iter().enumerate() {
        if let Some(first) = first_at.insert(id, position) {
            return Err(WireError::new(
                "duplicate_id",
                format!(
                    "job {id:?}: the id is given twice, at positions {first} and {position} \
                     of jobs (counting from 0); the schedule names jobs by id, so every \
                     job needs its own"
                ),
                json!({ "entity": "job", "id": id, "first": first, "second": position }),
            ));
        }
    }
    Ok(())
}

// ── rule selection ──────────────────────────────────────────────────────────

/// The rule names `build_engine` accepts, in the order the README lists them.
const RULES: [&str; 13] = [
    "SPT", "LPT", "EDD", "FCFS", "CR", "WSPT", "MST", "S/RO", "SRO", "ATC", "LWKR", "MWKR",
    "PRIORITY",
];

fn build_engine(rule: &str, config: &ScheduleConfig) -> Result<RuleEngine, WireError> {
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
        other => Err(WireError::unknown_option("rule", other, &RULES)),
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
/// `jobs` -- A native JS object matching `ScheduleInput`.
///
/// # Returns
/// A JS object matching `ScheduleOutput` on success. A refusal throws an
/// `Error` carrying `code` (see the README's *Errors*).
///
/// # Backward Compatibility
/// When `config.num_machines` is omitted (defaults to 1), behavior is
/// identical to the original single-machine implementation.
#[wasm_bindgen(unchecked_return_type = "ScheduleOutput")]
pub fn run_schedule(
    #[wasm_bindgen(unchecked_param_type = "ScheduleInput")] jobs: JsValue,
) -> Result<JsValue, JsValue> {
    let input: ScheduleInput = from_js(jobs, "jobs")?;

    if input.jobs.is_empty() {
        let output = ScheduleOutput {
            schedule: vec![],
            makespan: 0.0,
            total_tardiness: 0.0,
            machine_utilization: vec![],
        };
        return to_js(&output);
    }

    refuse_repeated_ids(input.jobs.iter().map(|j| j.id.as_str())).map_err(js_err)?;
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

    to_js(&output)
}

// ══════════════════════════════════════════════════════════════════════════════
// solve_jobshop — GA-based job-shop scheduling
// ══════════════════════════════════════════════════════════════════════════════

// ── input schema ────────────────────────────────────────────────────────────

#[derive(Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct JobShopOperation {
    /// Machine ID (e.g., "M1", "M2"). If multiple candidates, use an array.
    #[serde(default)]
    #[tsify(optional)]
    #[tsify(type = "string | null")]
    machine: Option<String>,
    /// Candidate machine IDs. Takes precedence over `machine`.
    #[serde(default)]
    #[tsify(optional)]
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

#[derive(Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct JobShopJob {
    id: String,
    operations: Vec<JobShopOperation>,
    #[serde(default)]
    #[tsify(optional)]
    #[tsify(type = "number | null")]
    due_date: Option<f64>,
    #[serde(default)]
    #[tsify(optional)]
    #[tsify(type = "number | null")]
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

#[derive(Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct JobShopGaConfig {
    #[serde(default = "default_population_size")]
    #[tsify(optional)]
    population_size: usize,
    #[serde(default = "default_max_generations")]
    #[tsify(optional)]
    max_generations: usize,
    #[serde(default = "default_mutation_rate")]
    #[tsify(optional)]
    mutation_rate: f64,
    #[serde(default)]
    #[tsify(optional)]
    #[tsify(type = "number | null")]
    seed: Option<u64>,
    #[serde(default = "default_tardiness_weight")]
    #[tsify(optional)]
    tardiness_weight: f64,
    #[serde(default = "default_crossover_type")]
    #[tsify(optional)]
    #[tsify(type = "\"POX\" | \"LOX\" | \"JOX\"")]
    crossover: String,
    #[serde(default = "default_mutation_type")]
    #[tsify(optional)]
    #[tsify(type = "\"Swap\" | \"Insert\" | \"Invert\"")]
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

#[derive(Deserialize, tsify::Tsify)]
#[serde(deny_unknown_fields)]
struct JobShopInput {
    jobs: Vec<JobShopJob>,
    /// Number of machines. If omitted, inferred from operation machine IDs.
    #[serde(default)]
    #[tsify(optional)]
    #[tsify(type = "number | null")]
    num_machines: Option<usize>,
    #[serde(default)]
    #[tsify(optional)]
    ga_config: JobShopGaConfig,
}

// ── output schema ───────────────────────────────────────────────────────────

#[derive(Serialize, tsify::Tsify)]
struct JobShopAssignment {
    job_id: String,
    /// Which step of its job this is, counting from 1 in the order the job's
    /// `operations` array lists them -- not the order they run on the timeline.
    operation: usize,
    machine: String,
    start: f64,
    end: f64,
}

#[derive(Serialize, tsify::Tsify)]
struct JobShopOutput {
    schedule: Vec<JobShopAssignment>,
    makespan: f64,
    fitness: f64,
    generations: usize,
    fitness_history: Vec<f64>,
}

// ── conversion helpers ──────────────────────────────────────────────────────

fn parse_crossover(s: &str) -> Result<CrossoverType, WireError> {
    match s {
        "POX" => Ok(CrossoverType::POX),
        "LOX" => Ok(CrossoverType::LOX),
        "JOX" => Ok(CrossoverType::JOX),
        other => Err(WireError::unknown_option(
            "ga_config.crossover",
            other,
            &["POX", "LOX", "JOX"],
        )),
    }
}

fn parse_mutation(s: &str) -> Result<MutationType, WireError> {
    match s {
        "Swap" => Ok(MutationType::Swap),
        "Insert" => Ok(MutationType::Insert),
        "Invert" => Ok(MutationType::Invert),
        other => Err(WireError::unknown_option(
            "ga_config.mutation",
            other,
            &["Swap", "Insert", "Invert"],
        )),
    }
}

// ── GA config validation ────────────────────────────────────────────────────

/// Validates `JobShopGaConfig` fields before they are passed to `GaConfig`.
///
/// This catches invalid values at the WASM boundary with clear error messages,
/// preventing panics deeper in the GA runner.
fn validate_ga_config(cfg: &JobShopGaConfig) -> Result<(), WireError> {
    if cfg.population_size < 2 {
        return Err(WireError::out_of_range(
            "ga_config.population_size",
            2.0,
            None,
            cfg.population_size as f64,
        ));
    }
    if cfg.max_generations < 1 {
        return Err(WireError::out_of_range(
            "ga_config.max_generations",
            1.0,
            None,
            cfg.max_generations as f64,
        ));
    }
    if !(0.0..=1.0).contains(&cfg.mutation_rate) {
        return Err(WireError::out_of_range(
            "ga_config.mutation_rate",
            0.0,
            Some(1.0),
            cfg.mutation_rate,
        ));
    }
    if !(0.0..=1.0).contains(&cfg.tardiness_weight) {
        return Err(WireError::out_of_range(
            "ga_config.tardiness_weight",
            0.0,
            Some(1.0),
            cfg.tardiness_weight,
        ));
    }
    Ok(())
}

/// The checked scheduling problem a job-shop request describes: one task per
/// job, one activity per operation, one resource per machine id (named by the
/// operations, padded to `num_machines`).
fn jobshop_problem(input: &JobShopInput) -> Result<Problem, WireError> {
    refuse_repeated_ids(input.jobs.iter().map(|j| j.id.as_str()))?;

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

    // Pad machine IDs to match num_machines if needed.
    if let Some(n) = input.num_machines {
        while machine_ids.len() < n {
            machine_ids.push(format!("M{}", machine_ids.len() + 1));
        }
    }

    if machine_ids.is_empty() {
        return Err(WireError::new(
            "no_machines",
            "No machines specified in operations".to_string(),
            json!({}),
        ));
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
                return Err(WireError::new(
                    "missing_machine",
                    format!("Job '{}' operation {} has no machine specified", job.id, i),
                    json!({ "job": job.id, "operation": i }),
                ));
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

    Ok(Problem::new(tasks, resources)?)
}

// ── public API: solve_jobshop ───────────────────────────────────────────────

/// Solve a job-shop scheduling problem using Genetic Algorithm.
///
/// # Arguments
/// `problem_json` -- A JS object matching `JobShopInput`.
///
/// # Returns
/// A JS object matching `JobShopOutput` on success. A refusal throws an
/// `Error` carrying `code` (see the README's *Errors*).
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
#[wasm_bindgen(unchecked_return_type = "JobShopOutput")]
pub fn solve_jobshop(
    #[wasm_bindgen(unchecked_param_type = "JobShopInput")] problem: JsValue,
) -> Result<JsValue, JsValue> {
    let input: JobShopInput = from_js(problem, "problem")?;

    if input.jobs.is_empty() {
        let output = JobShopOutput {
            schedule: vec![],
            makespan: 0.0,
            fitness: 0.0,
            generations: 0,
            fitness_history: vec![],
        };
        return to_js(&output);
    }

    let problem = jobshop_problem(&input).map_err(js_err)?;

    // ── Configure operators ──
    let crossover_type = parse_crossover(&input.ga_config.crossover).map_err(js_err)?;
    let mutation_type = parse_mutation(&input.ga_config.mutation).map_err(js_err)?;

    // ── Build GA problem ──
    let ga_problem = SchedulingGaProblem::new(&problem)
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
    let settings_refused = |e: &dyn std::fmt::Display| {
        js_err(WireError::new(
            "invalid_option",
            format!("ga_config refused: {e}"),
            json!({ "parameter": "ga_config" }),
        ))
    };
    config.validate().map_err(|e| settings_refused(&e))?;

    // ── Run GA ──
    let result = GaRunner::run(&ga_problem, &config).map_err(|e| settings_refused(&e))?;

    // ── Decode best solution ──
    let best_schedule = ga_problem.decode(&result.best);

    let schedule: Vec<JobShopAssignment> = best_schedule
        .assignments
        .iter()
        .map(|a| {
            // The schedule says which step this is; it used to be recovered by
            // parsing the trailing number out of `activity_id`, with a fallback
            // of 1 -- so when the decoder stopped putting an activity id there,
            // every row silently read "operation 1".
            let operation = a.sequence.ok_or_else(|| {
                js_err(WireError::new(
                    "internal",
                    format!(
                        "internal: assignment for job '{}' on '{}' carries no operation number",
                        a.task_id, a.resource_id
                    ),
                    json!({}),
                ))
            })? as usize;

            Ok(JobShopAssignment {
                job_id: a.task_id.clone(),
                operation,
                machine: a.resource_id.clone(),
                start: ms_to_sec(a.start_ms),
                end: ms_to_sec(a.end_ms),
            })
        })
        .collect::<Result<_, JsValue>>()?;

    let makespan = ms_to_sec(best_schedule.makespan_ms());

    let output = JobShopOutput {
        schedule,
        makespan,
        fitness: result.best_fitness,
        generations: result.generations,
        fitness_history: result.fitness_history,
    };

    to_js(&output)
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
    fn a_repeated_job_id_is_refused_naming_both_positions() {
        let err = refuse_repeated_ids(["A", "B", "A"]).expect_err("A is given twice");
        assert!(err.message.contains("job \"A\""), "{err}");
        assert!(err.message.contains("positions 0 and 2"), "{err}");
        assert!(
            !err.message.contains("  "),
            "a wrapped literal left a run of spaces: {err}"
        );
        assert_eq!(err.code(), "duplicate_id");
        assert_eq!(
            err.fields,
            json!({ "code": "duplicate_id", "entity": "job", "id": "A", "first": 0, "second": 2 })
        );
        assert!(refuse_repeated_ids(["A", "B", "C"]).is_ok());
    }

    /// Every refusal on the job-shop path names its reason as a `code` and
    /// carries the values behind it -- through the function `solve_jobshop`
    /// itself calls, not a copy of it.
    #[test]
    fn jobshop_refusals_carry_their_code_and_values() {
        let op = |machine: Option<&str>| JobShopOperation {
            machine: machine.map(str::to_string),
            machines: vec![],
            processing_time: 1.0,
        };
        let job = |id: &str, ops: Vec<JobShopOperation>| JobShopJob {
            id: id.to_string(),
            operations: ops,
            due_date: None,
            release_time: None,
        };
        let input = |jobs: Vec<JobShopJob>| JobShopInput {
            jobs,
            num_machines: None,
            ga_config: JobShopGaConfig::default(),
        };

        let missing = jobshop_problem(&input(vec![job("J1", vec![op(Some("M1")), op(None)])]))
            .expect_err("an operation without a machine");
        assert_eq!(
            missing.fields,
            json!({ "code": "missing_machine", "job": "J1", "operation": 1 })
        );

        let none = jobshop_problem(&input(vec![job("J1", vec![])])).expect_err("no machines");
        assert_eq!(none.code(), "no_machines");

        let empty = jobshop_problem(&input(vec![
            job("J1", vec![op(Some("M1"))]),
            job("J2", vec![]),
        ]))
        .expect_err("a job with no operations");
        assert_eq!(empty.fields, json!({ "code": "empty_task", "task": "J2" }));

        let rule = build_engine(
            "FIFO",
            &ScheduleConfig {
                rule: "FIFO".to_string(),
                num_machines: 1,
                atc_k: 2.0,
            },
        )
        .expect_err("FIFO is spelled FCFS here");
        assert_eq!(rule.code(), "unknown_option");
        assert_eq!(rule.fields["parameter"], "rule");
        assert_eq!(rule.fields["got"], "FIFO");

        let crossover = parse_crossover("OX").expect_err("unknown crossover");
        assert_eq!(crossover.fields["parameter"], "ga_config.crossover");

        let population = validate_ga_config(&JobShopGaConfig {
            population_size: 1,
            ..JobShopGaConfig::default()
        })
        .expect_err("population of one");
        assert_eq!(
            population.fields,
            json!({
                "code": "parameter_out_of_range",
                "parameter": "ga_config.population_size",
                "min": 2.0,
                "max": null,
                "got": 1.0,
            })
        );
    }

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

        let checked = jobshop_problem(&input).expect("valid input");
        let problem = SchedulingGaProblem::new(&checked)
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

    /// The tasks of a job-shop request, through the same path `solve_jobshop` takes.
    fn build_jobshop_tasks(input: &JobShopInput) -> Result<Vec<Task>, WireError> {
        jobshop_problem(input).map(|p| p.into_parts().0)
    }

    /// The reporter's job shop: two jobs of three operations each. Every row
    /// used to read `operation: 1`, while machines, times and makespan were
    /// all correct -- a silent wrong column.
    #[test]
    fn jobshop_numbers_each_operation_within_its_job() {
        use u_metaheur::ga::{GaConfig, GaRunner};

        let input = JobShopInput {
            jobs: vec![
                JobShopJob {
                    id: "J1".into(),
                    operations: vec![
                        JobShopOperation {
                            machine: Some("M1".into()),
                            machines: Vec::new(),
                            processing_time: 3.0,
                        },
                        JobShopOperation {
                            machine: Some("M2".into()),
                            machines: Vec::new(),
                            processing_time: 2.0,
                        },
                        JobShopOperation {
                            machine: Some("M3".into()),
                            machines: Vec::new(),
                            processing_time: 4.0,
                        },
                    ],
                    due_date: Some(20.0),
                    release_time: Some(0.0),
                },
                JobShopJob {
                    id: "J2".into(),
                    operations: vec![
                        JobShopOperation {
                            machine: Some("M2".into()),
                            machines: Vec::new(),
                            processing_time: 3.0,
                        },
                        JobShopOperation {
                            machine: Some("M3".into()),
                            machines: Vec::new(),
                            processing_time: 2.0,
                        },
                        JobShopOperation {
                            machine: Some("M1".into()),
                            machines: Vec::new(),
                            processing_time: 5.0,
                        },
                    ],
                    due_date: Some(18.0),
                    release_time: Some(0.0),
                },
            ],
            num_machines: Some(3),
            ga_config: JobShopGaConfig {
                population_size: 30,
                max_generations: 20,
                mutation_rate: 0.1,
                seed: Some(42),
                ..JobShopGaConfig::default()
            },
        };

        let checked = jobshop_problem(&input).expect("every operation names a machine");
        let problem = SchedulingGaProblem::new(&checked);
        let config = GaConfig::default()
            .with_population_size(30)
            .with_max_generations(20)
            .with_seed(42)
            .with_parallel(false);
        let result = GaRunner::run(&problem, &config).expect("a valid GA problem");
        let schedule = problem.decode(&result.best);

        assert_eq!(schedule.assignment_count(), 6, "two jobs of three steps");
        for job in ["J1", "J2"] {
            let mut steps: Vec<i32> = schedule
                .assignments
                .iter()
                .filter(|a| a.task_id == job)
                .map(|a| a.sequence.expect("the schedule says the step"))
                .collect();
            steps.sort_unstable();
            assert_eq!(steps, vec![1, 2, 3], "{job} must number its own steps");
        }

        // The step order is the order the job lists its machines, whatever
        // order the solver puts them in on the timeline.
        for (job, machines) in [("J1", ["M1", "M2", "M3"]), ("J2", ["M2", "M3", "M1"])] {
            for (idx, machine) in machines.iter().enumerate() {
                let a = schedule
                    .assignments
                    .iter()
                    .find(|a| a.task_id == job && a.sequence == Some(idx as i32 + 1))
                    .unwrap_or_else(|| panic!("{job} step {}", idx + 1));
                assert_eq!(&a.resource_id, machine, "{job} step {}", idx + 1);
            }
        }
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
        assert!(err.message.contains("population_size"), "got: {}", err);
    }

    #[test]
    fn test_validate_ga_config_population_size_one() {
        let cfg = JobShopGaConfig {
            population_size: 1,
            ..Default::default()
        };
        let err = validate_ga_config(&cfg).unwrap_err();
        assert!(err.message.contains("population_size"), "got: {}", err);
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
        assert!(err.message.contains("max_generations"), "got: {}", err);
    }

    #[test]
    fn test_validate_ga_config_mutation_rate_negative() {
        let cfg = JobShopGaConfig {
            mutation_rate: -0.1,
            ..Default::default()
        };
        let err = validate_ga_config(&cfg).unwrap_err();
        assert!(err.message.contains("mutation_rate"), "got: {}", err);
    }

    #[test]
    fn test_validate_ga_config_mutation_rate_above_one() {
        let cfg = JobShopGaConfig {
            mutation_rate: 1.5,
            ..Default::default()
        };
        let err = validate_ga_config(&cfg).unwrap_err();
        assert!(err.message.contains("mutation_rate"), "got: {}", err);
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
        assert!(err.message.contains("tardiness_weight"), "got: {}", err);

        let cfg2 = JobShopGaConfig {
            tardiness_weight: 2.0,
            ..Default::default()
        };
        let err2 = validate_ga_config(&cfg2).unwrap_err();
        assert!(err2.message.contains("tardiness_weight"), "got: {}", err2);
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

        let checked = jobshop_problem(&input).expect("valid input");
        let problem = SchedulingGaProblem::new(&checked);
        let config = GaConfig::default()
            .with_population_size(4)
            .with_max_generations(5)
            // population 4 × default elite_ratio 0.1 floors to 0 elites, which
            // GaConfig::validate rejects; 0.25 keeps exactly 1 elite.
            .with_elite_ratio(0.25)
            .with_seed(1)
            .with_parallel(false);

        config.validate().expect("config should be valid");
        let result = GaRunner::run(&problem, &config).expect("GA should succeed");
        let schedule = problem.decode(&result.best);

        assert_eq!(schedule.assignments.len(), 1);
        assert!(schedule.makespan_ms() > 0);
    }
}

// ── Wire-schema strictness tests ─────────────────────────────────────

#[cfg(test)]
mod dto_strictness_tests {
    use serde_json::json;

    fn assert_rejects_unknown<T: serde::de::DeserializeOwned>(v: serde_json::Value) {
        match serde_json::from_value::<T>(v) {
            Ok(_) => panic!("unknown key must be rejected"),
            Err(e) => assert!(e.to_string().contains("unknown field"), "{e}"),
        }
    }

    #[test]
    fn schedule_input_rejects_unknown_keys() {
        assert_rejects_unknown::<super::ScheduleInput>(json!({
            "jobs": [{ "id": "j1", "processing_time": 1.0 }],
            "rule": "spt"
        }));
    }

    #[test]
    fn schedule_nested_job_and_config_reject_unknown_keys() {
        assert_rejects_unknown::<super::ScheduleInput>(json!({
            "jobs": [{ "id": "j1", "processing_time": 1.0, "priority": 2 }]
        }));
        assert_rejects_unknown::<super::ScheduleInput>(json!({
            "jobs": [{ "id": "j1", "processing_time": 1.0 }],
            "config": { "rule": "spt", "machines": 2 }
        }));
    }

    #[test]
    fn jobshop_input_rejects_unknown_keys() {
        assert_rejects_unknown::<super::JobShopInput>(json!({
            "jobs": [{ "id": "j1", "operations": [{ "machine": "M1", "processing_time": 1.0 }] }],
            "time_limit": 5
        }));
    }

    #[test]
    fn jobshop_nested_structs_reject_unknown_keys() {
        assert_rejects_unknown::<super::JobShopJob>(json!({
            "id": "j1", "operations": [], "priority": 1
        }));
        assert_rejects_unknown::<super::JobShopOperation>(json!({
            "machine": "M1", "processing_time": 1.0, "setup_time": 0.5
        }));
        assert_rejects_unknown::<super::JobShopGaConfig>(json!({
            "population_size": 10, "elitism": true
        }));
    }
}
