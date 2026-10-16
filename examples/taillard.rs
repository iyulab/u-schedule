//! Measures the dispatching rules on Taillard's job-shop instances, through
//! the scheduler a consumer calls (`SimpleScheduler` with a `RuleEngine`),
//! checking every schedule for violations.
//!
//! The instances are not in this repository. Download them in the standard
//! job-shop format (first line `jobs machines`, then one line per job of
//! `machine time` pairs, machines counted from 0 — e.g. the `ta01` … files of
//! the JSPLIB collection) into one directory:
//!
//! ```text
//! cargo run --release --example taillard -- <dir> [bks.csv] [rules]
//! ```
//!
//! `bks.csv` (optional) has lines `instance,makespan` -- the optimum or best
//! known upper bound; the gap column is against it. `rules` is a comma list of
//! `SPT,LPT,LWKR,MWKR,FIFO,WINQ,LPUL` (default all).

use std::collections::HashMap;
use std::time::Instant;

use u_schedule::dispatching::{rules, RuleEngine};
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

/// Reads one instance: jobs as tasks whose operations each need one machine.
fn read(path: &std::path::Path) -> Problem {
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
            for k in 0..machines {
                let (machine, time) = (next(), next());
                task = task.with_activity(
                    Activity::new(format!("{id}_O{k}"), &id, k as i32)
                        .with_duration(ActivityDuration::fixed(time))
                        .with_requirement(
                            ResourceRequirement::new("Machine")
                                .with_candidates(vec![format!("M{machine}")]),
                        ),
                );
            }
            task
        })
        .collect();
    let resources = (0..machines)
        .map(|m| Resource::new(format!("M{m}"), "Machine"))
        .collect();
    Problem::new(tasks, resources).expect("a valid instance")
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = std::path::Path::new(
        args.first()
            .expect("usage: taillard <dir> [bks.csv] [rules]"),
    );
    let best: HashMap<String, f64> = args
        .get(1)
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
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("ta") && !n.contains('.'))
        })
        .collect();
    names.sort();

    println!("instance,size,rule,makespan,best,gap_pct,violations,ms");
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
            let schedule = SimpleScheduler::new()
                .with_rule_engine(engine(rule))
                .schedule(&problem, 0);
            let ms = started.elapsed().as_millis();
            let makespan = schedule.makespan_ms() as f64;
            let reference = best.get(name).copied();
            let gap = reference.map(|b| 100.0 * (makespan - b) / b);
            if let Some(g) = gap {
                gaps.entry(rule.as_str()).or_default().push(g);
            }
            println!(
                "{name},{size},{rule},{makespan},{},{},{},{ms}",
                reference.map_or(String::new(), |b| b.to_string()),
                gap.map_or(String::new(), |g| format!("{g:.1}")),
                schedule.violations.len(),
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
