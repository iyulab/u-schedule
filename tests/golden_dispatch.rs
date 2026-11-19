//! The dispatcher's schedules on fixed random problems, pinned as hashes: a change to how
//! `SimpleScheduler::schedule` is implemented (not what it decides) must leave every line of
//! `tests/data/golden_dispatch.txt` as it is. Regenerate only when a decision is meant to change:
//! `UPDATE_GOLDEN=1 cargo test --test golden_dispatch`.

use u_schedule::dispatching::{rules, RuleEngine};
use u_schedule::models::{Activity, Resource, ResourceRequirement, Task};
use u_schedule::scheduler::SimpleScheduler;
use u_schedule::Problem;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Tasks of one to three activities on a pool of identical machines, some also needing one of
/// two operators, with releases, deadlines and priorities.
fn problem(seed: u64) -> Problem {
    let mut rng = Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let machines = 1 + rng.below(4) as usize;
    let tasks_n = 20 + rng.below(50) as usize;
    let mut resources: Vec<Resource> = (0..machines)
        .map(|m| Resource::new(format!("M{m}"), "Machine"))
        .collect();
    resources.push(Resource::new("O0", "Operator"));
    resources.push(Resource::new("O1", "Operator"));
    let tasks: Vec<Task> = (0..tasks_n)
        .map(|i| {
            let id = format!("T{i}");
            let mut task = Task::new(id.clone())
                .with_priority(rng.below(5) as i32)
                .with_release_time(rng.below(300) as i64 * 100)
                .with_deadline(5_000 + rng.below(80_000) as i64 * 100);
            for s in 0..=rng.below(3) {
                let mut activity = Activity::new(format!("{id}-{s}"), id.clone(), s as i32)
                    .with_process_time(500 + rng.below(30) as i64 * 100)
                    .with_requirement(ResourceRequirement::new("Machine"));
                if rng.below(3) == 0 {
                    activity = activity.with_requirement(ResourceRequirement::new("Operator"));
                }
                task = task.with_activity(activity);
            }
            task
        })
        .collect();
    Problem::new(tasks, resources).expect("a valid problem")
}

fn fingerprint(problem: &Problem, engine: Option<RuleEngine>) -> String {
    let scheduler = match engine {
        Some(engine) => SimpleScheduler::new().with_rule_engine(engine),
        None => SimpleScheduler::new(),
    };
    let schedule = scheduler.schedule(problem, 0);
    let mut lines: Vec<String> = schedule
        .assignments
        .iter()
        .map(|a| {
            format!(
                "{}|{}|{}|{}|{}",
                a.activity_id, a.resource_id, a.start_ms, a.end_ms, a.setup_ms
            )
        })
        .collect();
    lines.sort();
    // FNV-1a over the sorted lines.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in lines.join("\n").bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}:{}", lines.len())
}

fn rule_set(name: &str) -> Option<RuleEngine> {
    Some(match name {
        "priority" => return None,
        "spt" => RuleEngine::new().with_rule(rules::Spt),
        "edd" => RuleEngine::new().with_rule(rules::Edd),
        "fifo" => RuleEngine::new().with_rule(rules::Fifo),
        "winq" => RuleEngine::new()
            .with_rule(rules::Winq)
            .with_rule(rules::Spt),
        other => panic!("unknown rule {other}"),
    })
}

#[test]
fn the_dispatcher_decides_what_it_decided() {
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/golden_dispatch.txt");
    let mut now = String::new();
    for seed in 0..24u64 {
        let problem = problem(seed);
        for name in ["priority", "spt", "edd", "fifo", "winq"] {
            now.push_str(&format!(
                "{seed} {name} {}\n",
                fingerprint(&problem, rule_set(name))
            ));
        }
    }
    if std::env::var("UPDATE_GOLDEN").is_ok() {
        std::fs::create_dir_all(path.parent().expect("a directory")).expect("create tests/data");
        std::fs::write(&path, &now).expect("write the golden file");
        return;
    }
    let want = std::fs::read_to_string(&path).expect("tests/data/golden_dispatch.txt");
    let want = want.replace("\r\n", "\n");
    assert_eq!(
        now, want,
        "a schedule changed -- regenerate only if a decision was meant to"
    );
}
