//! Measures the dispatching rules on Taillard's job-shop instances, through
//! the scheduler a consumer calls (`SimpleScheduler` with a `RuleEngine`),
//! checking every schedule for violations.
//!
//! The instances are not in this repository. Download them in the standard
//! job-shop format (first line `jobs machines`, then one line per job of
//! `machine time` pairs, machines counted from 0 — e.g. the `ta01` …, `ft06`,
//! `la01` … files of the JSPLIB collection) into one directory; every file
//! there without an extension is read. A file ending in `.fjs` is a flexible
//! job shop (Brandimarte's `mk01` …): first line `jobs machines`, then per job
//! its operation count and, per operation, how many machines may run it and
//! that many `machine time` pairs, machines counted from 0:
//!
//! ```text
//! cargo run --release --example taillard -- <dir> [bks.csv] [rules]
//! ```
//!
//! `bks.csv` (optional) has lines `instance,makespan` -- the optimum or best
//! known upper bound; the gap column is against it. `rules` is a comma list of
//! `SPT,LPT,LWKR,MWKR,FIFO,WINQ,LPUL` (default all) and `GA` -- the genetic
//! algorithm (`SchedulingGaProblem` with `GaRunner`: makespan only, population
//! 50, 200 generations, seed 1). `GA@<ms>` gives it a time limit instead of the
//! generation budget, e.g. `GA@10000`; `GA:<n>` sets the local search's swaps
//! per improved chromosome (`GA:0` turns it off), e.g. `GA:5@10000`.
//! `CP@<ms>` solves the instance exactly (`ScheduleCpBuilder` with
//! u-metaheur's `BranchAndBoundSolver`) within that time limit (default
//! 10 000 ms); the `status` column says whether the search proved its
//! makespan optimal (`Optimal`) or was cut off with the best found
//! (`Feasible`). `GA+CP@<ms>` runs the GA (200 generations) first and starts
//! the exact search from its schedule; the time is the two together.

use std::collections::HashMap;
use std::time::Instant;

use u_metaheur::cp::{BranchAndBoundSolver, SolverConfig};
use u_metaheur::ga::{GaConfig, GaRunner};
use u_schedule::cp::ScheduleCpBuilder;
use u_schedule::dispatching::{rules, RuleEngine};
use u_schedule::ga::SchedulingGaProblem;
use u_schedule::models::{Activity, ActivityDuration, Resource, ResourceRequirement, Task};
use u_schedule::scheduler::SimpleScheduler;
use u_schedule::Problem;

const RULES: [&str; 7] = ["SPT", "LPT", "LWKR", "MWKR", "FIFO", "WINQ", "LPUL"];

fn engine(rule: &str) -> RuleEngine {
    let engine = RuleEngine::new();
    match rule {
        "SPT" => engine.with_rule(rules::Spt),
        "LPT" => engine.with_rule(rules::Lpt),
        "LWKR" => engine.with_rule(rules::Lwkr),
        "MWKR" => engine.with_rule(rules::Mwkr),
        "FIFO" => engine.with_rule(rules::Fifo),
        "WINQ" => engine.with_rule(rules::Winq),
        "LPUL" => engine.with_rule(rules::Lpul),
        other => panic!("unknown rule {other:?} -- one of {RULES:?}"),
    }
}

/// Reads one instance: jobs as tasks whose operations each need one machine
/// -- in a flexible one, any of several, each for its own time.
fn read(path: &std::path::Path) -> Problem {
    let flexible = path.extension().is_some_and(|e| e == "fjs");
    let text = std::fs::read_to_string(path).expect("instance file is readable");
    let mut numbers = text
        .split_whitespace()
        .map(|t| t.parse::<i64>().expect("instance holds integers"));
    let mut next = || numbers.next().expect("instance is complete");
    let (jobs, machines) = (next() as usize, next() as usize);
    let tasks = (0..jobs)
        .map(|j| {
            let id = format!("J{j}");
            let mut task = Task::new(&id).with_category("job");
            let operations = if flexible { next() as usize } else { machines };
            for k in 0..operations {
                let choices = if flexible { next() } else { 1 };
                let options: Vec<(String, i64)> = (0..choices)
                    .map(|_| (format!("M{}", next()), next()))
                    .collect();
                // The base time is the longest: a horizon of every base time
                // runs everything in turn on any choice.
                let longest = options.iter().map(|o| o.1).max().expect("a machine");
                let mut step = Activity::new(format!("{id}_O{k}"), &id, k as i32)
                    .with_duration(ActivityDuration::fixed(longest))
                    .with_requirement(
                        ResourceRequirement::new("Machine")
                            .with_candidates(options.iter().map(|o| o.0.clone()).collect()),
                    );
                if options.len() > 1 {
                    for (machine, time) in &options {
                        step = step.with_duration_on(machine, ActivityDuration::fixed(*time));
                    }
                }
                task = task.with_activity(step);
            }
            task
        })
        .collect();
    let resources = (0..machines)
        .map(|m| Resource::new(format!("M{m}"), "Machine"))
        .collect();
    Problem::new(tasks, resources).expect("a valid instance")
}

/// The GA on makespan alone (no due dates here), population 50, seed 1;
/// 200 generations, or as many as fit in `time_limit_ms`.
fn genetic(
    problem: &Problem,
    local_search_steps: Option<usize>,
    time_limit_ms: Option<u64>,
) -> u_schedule::models::Schedule {
    let mut ga = SchedulingGaProblem::new(problem)
        .with_tardiness_weight(0.0)
        .expect("a weight in [0, 1]");
    if let Some(steps) = local_search_steps {
        ga = ga.with_local_search_steps(steps);
    }
    let mut config = GaConfig::default()
        .with_population_size(50)
        .with_max_generations(200)
        .with_seed(1);
    if let Some(ms) = time_limit_ms {
        config = config
            .with_max_generations(usize::MAX)
            .with_time_limit_ms(ms);
    }
    let result = GaRunner::run(&ga, &config).expect("valid GA settings");
    ga.decode(&result.best)
}

/// The exact solver within `time_limit_ms`, on a horizon of the instance's
/// total work (a schedule that runs everything in turn): the schedule and
/// the solver's status.
fn exact(
    problem: &Problem,
    time_limit_ms: i64,
    start_from: Option<&u_schedule::models::Schedule>,
) -> (u_schedule::models::Schedule, String) {
    let horizon: i64 = problem
        .tasks()
        .iter()
        .flat_map(|t| &t.activities)
        .map(|a| a.duration.total_ms())
        .sum();
    let config = SolverConfig {
        time_limit_ms,
        ..SolverConfig::default()
    };
    let builder = ScheduleCpBuilder::new(problem);
    let mut solver = BranchAndBoundSolver::new();
    if let Some(schedule) = start_from {
        solver = solver.with_incumbent(builder.incumbent(schedule));
    }
    let (schedule, solution) = builder.solve(&solver, &config, horizon);
    (schedule, format!("{:?}", solution.status))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = std::path::Path::new(
        args.first()
            .expect("usage: taillard <dir> [bks.csv] [rules]"),
    );
    let best: HashMap<String, f64> = args
        .get(1)
        .filter(|p| !p.is_empty())
        .map(|p| {
            std::fs::read_to_string(p)
                .expect("bks.csv is readable")
                .lines()
                .filter_map(|l| {
                    let (name, value) = l.split_once(',')?;
                    Some((name.trim().to_string(), value.trim().parse().ok()?))
                })
                .collect()
        })
        .unwrap_or_default();
    let chosen: Vec<String> = args.get(2).map_or_else(
        || RULES.iter().map(|r| r.to_string()).collect(),
        |list| list.split(',').map(str::to_string).collect(),
    );

    let mut names: Vec<_> = std::fs::read_dir(dir)
        .expect("instance directory is readable")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| !n.contains('.') || n.ends_with(".fjs"))
        })
        .collect();
    names.sort();

    println!("instance,size,rule,makespan,best,gap_pct,violations,ms,status");
    let mut gaps: HashMap<&str, Vec<f64>> = HashMap::new();
    for path in &names {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .expect("a file name");
        let problem = read(path);
        let size = format!("{}x{}", problem.tasks().len(), problem.resources().len());
        for rule in &chosen {
            let started = Instant::now();
            let mut status = String::new();
            let schedule = match rule.strip_prefix("GA") {
                None if rule.starts_with("CP") => {
                    let limit = rule
                        .strip_prefix("CP@")
                        .map_or(10_000, |ms| ms.parse().expect("CP@<milliseconds>"));
                    let (schedule, s) = exact(&problem, limit, None);
                    status = s;
                    schedule
                }
                Some(cp) if cp.starts_with("+CP") => {
                    let limit = cp
                        .strip_prefix("+CP@")
                        .map_or(10_000, |ms| ms.parse().expect("GA+CP@<milliseconds>"));
                    let seed = genetic(&problem, None, None);
                    let (schedule, s) = exact(&problem, limit, Some(&seed));
                    status = s;
                    schedule
                }
                Some(options) => {
                    let (steps, limit) = options
                        .split_once('@')
                        .map_or((options, ""), |(s, ms)| (s, ms));
                    genetic(
                        &problem,
                        steps
                            .strip_prefix(':')
                            .map(|n| n.parse().expect("GA:<swaps>")),
                        (!limit.is_empty()).then(|| limit.parse().expect("GA@<milliseconds>")),
                    )
                }
                None => SimpleScheduler::new()
                    .with_rule_engine(engine(rule))
                    .schedule(&problem, 0),
            };
            let ms = started.elapsed().as_millis();
            let makespan = schedule.makespan_ms() as f64;
            // The CP path reports no violations of its own: check it.
            let violations = if status.is_empty() {
                schedule.violations.len()
            } else {
                u_schedule::scheduler::check_schedule(
                    &schedule,
                    &u_schedule::scheduler::FeasibilityInput {
                        tasks: problem.tasks(),
                        resources: problem.resources(),
                        constraints: &[],
                    },
                )
                .len()
            };
            let reference = best.get(name).copied();
            let gap = reference.map(|b| 100.0 * (makespan - b) / b);
            if let Some(g) = gap {
                gaps.entry(rule.as_str()).or_default().push(g);
            }
            println!(
                "{name},{size},{rule},{makespan},{},{},{violations},{ms},{status}",
                reference.map_or(String::new(), |b| b.to_string()),
                gap.map_or(String::new(), |g| format!("{g:.1}")),
            );
        }
    }
    for rule in &chosen {
        if let Some(g) = gaps.get(rule.as_str()) {
            let mean = g.iter().sum::<f64>() / g.len() as f64;
            eprintln!("{rule}: mean gap {mean:.1} % over {} instances", g.len());
        }
    }
}
