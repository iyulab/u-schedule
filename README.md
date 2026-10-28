# u-schedule

**Scheduling framework in Rust**

[![Crates.io](https://img.shields.io/crates/v/u-schedule.svg)](https://crates.io/crates/u-schedule)
[![docs.rs](https://docs.rs/u-schedule/badge.svg)](https://docs.rs/u-schedule)
[![CI](https://github.com/iyulab/u-schedule/actions/workflows/ci.yml/badge.svg)](https://github.com/iyulab/u-schedule/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

u-schedule provides domain models, constraints, validation, dispatching rules, and a greedy scheduler for scheduling problems. It builds on `u-metaheur` for metaheuristic algorithms and `u-numflow` for mathematical primitives.

## Modules

| Module | Description |
|--------|-------------|
| `models` | Domain types: `Task`, `Activity`, `Resource`, `Schedule`, `Assignment`, `Calendar`, `Constraint`, `TransitionMatrix` |
| `problem` | `Problem` — tasks and resources that passed validation; the only input the solvers take |
| `validation` | Input integrity checks: duplicate IDs, DAG cycle detection, resource reference validation |
| `dispatching` | Priority dispatching rules and rule engine |
| `scheduler` | Greedy scheduler and KPI evaluation |
| `ga` | GA-based scheduling with OSV/MAV dual-vector encoding |
| `cp` | CP formulation (`ScheduleCpBuilder`) — solved exactly by u-metaheur's `BranchAndBoundSolver`, an activity with several candidate resources choosing among them (the flexible job shop) |

## Dispatching Rules

| Rule | Description |
|------|-------------|
| SPT | Shortest Processing Time — of the imminent operation |
| LPT | Longest Processing Time — of the imminent operation |
| EDD | Earliest Due Date |
| FIFO | First In First Out — by arrival in the queue (release, or end of the previous operation) |
| MST | Minimum Slack Time — `due − now − work left` |
| CR | Critical Ratio — `(due − now) / work left` |
| S/RO | Slack per Remaining Operations |
| ATC | Apparent Tardiness Cost — `(w/p)·exp(−max(0, due − now − work left) / (k·p̄))`, `p̄` the average imminent operation of the jobs competing (reads `Task::weight`) |
| WSPT | Weighted Shortest Processing Time — `weight / processing time`, reads `Task::weight` |
| MWKR | Most Work Remaining — this operation and the ones after it |
| LWKR | Least Work Remaining |
| WINQ | Work In Next Queue — how many other waiting jobs want a machine the job's next operation can use |
| LPUL | Least Planned Utilization Level — booked time over the planned span, per unit of capacity |
| PRIORITY | Job Priority — reads `Task::priority`, higher first |

`Task::weight` (default 1, finite and `> 0`) is the weight `w_j` of the
weighted rules; `Task::priority` is a separate ordinal read only by `PRIORITY`.

### When a rule decides

`SimpleScheduler` (and `run_schedule`) dispatch **operation by operation,
without delay**: whenever a machine could start work, the rule chooses among
the operations that can start then, and judges them on that moment's
`SchedulingContext` — the clock, each job's imminent operation, the work left,
when it joined the queue, the queue at its next machine, and machine load. A
machine never idles while work it can start is waiting; no slot is held for a
job that arrives later. Without a rule engine, `Task::priority` decides (higher
first, ties in input order).

A `SchedulingContext` you build yourself for `RuleEngine::sort_indices` may
leave fields empty; each rule then reads the task as a whole (SPT the total
processing time, LWKR the total work, WINQ 0).

Measured on Taillard's 40 job-shop instances `ta01`–`ta40` (15×15 to 30×15),
makespan gap to the optimum or best known upper bound, mean over instances
(`examples/taillard.rs`):

| Rule | Gap |
|------|-----|
| MWKR | 21.9 % |
| FIFO | 28.4 % |
| SPT | 30.3 % |
| WINQ | 34.3 % |
| LPUL | 42.0 % |
| LPT | 44.2 % |
| LWKR | 47.9 % |
| GA (population 50, 200 generations, seed 1) | 4.5 % |

The rules take milliseconds; the GA about 12 s per instance here. Its first
population holds the rules' schedules (MWKR, SPT, LPT, FIFO, LWKR, and EDD and
ATC when tasks have deadlines) beside random ones, and it keeps its best, so it
never ends worse than the best rule — on all 40 instances it was at least as
good as MWKR. A chromosome is decoded two ways and the shorter schedule is
kept: in the order its OSV gives, and as an *active* schedule
(Giffler–Thompson) that lets an activity use idle time before a longer one
when that delays nothing. Decoding only in the OSV's order left the same GA at
19.7 %. Each child of a crossover and each mutated chromosome is then improved
by a local search on the critical path: swapping the first two or the last two
activities of a block of the longest chain on one machine is the only
rearrangement that can shorten it (Nowicki & Smutnicki 1996), so the search
tries those and makes the best, even when it lengthens the schedule, forbidding
the undoing of a swap for six steps (taboo search) so that it leaves the local
optimum a plain descent stops in; it writes the best order seen back. Without
the local search (`local_search_steps` 0) the same GA was at 16.4 %, with a
descent that stops at its first optimum (20 swaps) at 10.0 %. The run is
reproducible: the same seed gives the same schedule.

Where activities choose among resources (a flexible job shop), crossover also
recombines the machine assignments gene by gene, a load-balanced chromosome
puts each activity where it would end the least loaded given its time there,
and the local search may move a critical activity to another of its
candidates -- where on that resource the longest path through it is shortest
(Mastrolilli & Gambardella 2000) -- the assignment written back with the
order. On Brandimarte's ten instances (`examples/taillard.rs` reads the
`.fjs` format; 200 generations, seed 1) the mean gap to the best known
makespans is 0.5 %, seven of the ten at the best known.

## GA Encoding

The GA module uses dual-vector encoding for job-shop scheduling:

- **OSV (Operation Sequence Vector)** — Permutation encoding that determines operation processing order
- **MAV (Machine Assignment Vector)** — Integer vector that assigns each operation to a specific machine (for flexible job shops)

`SchedulingGaProblem::decode` turns a chromosome into a schedule — the shorter, by the fitness the
search minimizes, of `decode_semi_active` (each activity as early as its machine and its job allow,
in the OSV's order) and `decode_active` (the Giffler–Thompson active schedule that order prioritizes).
It holds one machine per operation and reads no calendars.

### Searching the dispatching order

`DispatchSearch` searches the order `SimpleScheduler` takes the tasks in, with a biased random-key
genetic algorithm (u-metaheur's BRKGA): each task gets a key, and a chromosome is decoded by the
scheduler with the keys as its rule — the lower key wins each decision. Every schedule it considers
is one the dispatcher builds, so it keeps everything the dispatcher keeps: calendars, several
requirements at once, usages, companions, transport times, releases. It returns the better of the
best keyed schedule and the scheduler's own (Hartmann 1998; Gonçalves, Resende & Mendes 2011).

```rust
use u_metaheur::brkga::BrkgaConfig;
use u_schedule::models::{Activity, Resource, ResourceRequirement, Task};
use u_schedule::scheduler::{DispatchSearch, SimpleScheduler};
use u_schedule::Problem;

let on = |id: &str, task: &str, ms: i64| {
    Activity::new(id, task, 0)
        .with_process_time(ms)
        .with_requirement(ResourceRequirement::new("Machine"))
};
let problem = Problem::new(
    vec![
        Task::new("LONG").with_activity(on("L", "LONG", 10_000)),
        Task::new("URGENT").with_activity(on("U", "URGENT", 1_000)).with_deadline(1_000),
    ],
    vec![Resource::new("M", "Machine")],
)
.expect("valid input");
let config = BrkgaConfig::new(2).with_population_size(20).with_max_generations(20).with_seed(7);
let found = DispatchSearch::new(SimpleScheduler::new(), config)
    .with_tardiness_weight(1.0)
    .search(&problem, 0)
    .expect("a valid configuration");
assert_eq!(found.dispatching_cost, 10_000.0); // in input order, URGENT ends 10 s late
assert_eq!(found.cost, 0.0);
```

## Quick Start

```toml
[dependencies]
u-schedule = "0.32"
```

```rust
use u_schedule::models::{Activity, ActivityDuration, Resource, ResourceRequirement, Task};
use u_schedule::scheduler::SimpleScheduler;
use u_schedule::Problem;

// Two jobs, one operation each, both on machine M1. Times are milliseconds.
let job = |id: &str, ms: i64| {
    Task::new(id).with_activity(
        Activity::new(format!("{id}-op1"), id, 0)
            .with_duration(ActivityDuration::fixed(ms))
            .with_requirement(
                ResourceRequirement::new("Machine").with_candidates(vec!["M1".into()]),
            ),
    )
};
let tasks = vec![job("J1", 30_000), job("J2", 20_000)];
let resources = vec![Resource::new("M1", "Machine")];

// Duplicate ids, unknown resources and precedence cycles are refused here, so
// the solvers never see them: they take only a `Problem`.
let problem = Problem::new(tasks, resources).expect("valid input");

let schedule = SimpleScheduler::new().schedule(&problem, 0);
assert!(schedule.is_valid());
assert_eq!(schedule.makespan_ms(), 50_000); // the two jobs run back to back
```

### Rules chosen decision by decision

`SimpleScheduler::with_rule_policy` takes a `RulePolicy` — any
`Fn(&Decision) -> RuleEngine` — and asks it for the rules at every conflict.
The `Decision` carries the tasks, which of them wait and which compete now,
and the `SchedulingContext` (the time, each resource's utilization, …), so a
caller can dispatch by one rule early in the horizon and another once its
tasks run late, without the scheduler knowing the conditions:

```rust
use u_schedule::dispatching::{rules, Decision, RuleEngine};
use u_schedule::scheduler::SimpleScheduler;

let scheduler = SimpleScheduler::new().with_rule_policy(|d: &Decision<'_>| {
    let late = d.waiting.iter().any(|&i| {
        d.tasks[i].deadline.is_some_and(|due| due < d.context.current_time_ms)
    });
    let engine = RuleEngine::new();
    if late { engine.with_rule(rules::Edd) } else { engine.with_rule(rules::Spt) }
});
# let _ = scheduler;
```

### Which resources fill a requirement

A `ResourceRequirement` and a `Resource` share one type vocabulary — free-form
names such as `"Machine"` or `"Operator"`, compared exactly. A requirement is
filled by:

- **its listed `candidates`**, when it lists any — the list wins, and a listed
  resource is accepted whatever its type;
- **otherwise, every resource whose `resource_type` equals the requirement's.**

Either way the resource must hold every `required_skills` entry (skills are
names, matched exactly). `SimpleScheduler`, the GA, the CP builder and the
feasibility checker all apply this one rule, `ResourceRequirement::accepts`.

```rust
use u_schedule::models::{Activity, ActivityDuration, Resource, ResourceRequirement, Task};
use u_schedule::scheduler::SimpleScheduler;
use u_schedule::Problem;

let task = Task::new("J1").with_activity(
    Activity::new("J1-op1", "J1", 0)
        .with_duration(ActivityDuration::fixed(1_000))
        // No candidates: any machine that can mill.
        .with_requirement(ResourceRequirement::new("Machine").with_skill("milling")),
);
let resources = vec![
    Resource::new("W1", "Operator").with_skill("milling"), // not a machine
    Resource::new("M1", "Machine"),                        // cannot mill
    Resource::new("M2", "Machine").with_skill("milling"),
];
let problem = Problem::new(vec![task], resources).expect("valid input");
let schedule = SimpleScheduler::new().schedule(&problem, 0);
let placed = schedule.assignment_for_activity("J1-op1").expect("placed");
assert_eq!(placed.resource_id, "M2");
```

### Time that depends on the resource

An activity's `duration` is its time on a resource with nothing of its own.
On a particular resource it can instead run at a **speed** — processing time
÷ speed, setup and teardown unchanged (a faster machine, a worker's
efficiency: Pinedo's uniform machines `Q`) — or take a **duration** of its own
there (unrelated machines `R`, `p_ij`):

- on several resources at once (a machine and its operator), the speeds
  multiply; a resource with a duration of its own sets the whole time;
- the time is rounded to the nearest millisecond.

`Activity::duration_on(resources)` is the one rule `SimpleScheduler`, the GA
decoders and local search, the CP builder and the feasibility checker read.
`SimpleScheduler` stays non-delay — the earliest start wins, and at the same
start the resource that processes faster. A speed that is not finite and
`> 0`, a negative duration, or a timing for a resource the problem does not
have is refused by `Problem::new`.

```rust
use u_schedule::models::{Activity, ActivityDuration, Resource, ResourceRequirement, Task};
use u_schedule::scheduler::SimpleScheduler;
use u_schedule::Problem;

let job = |id: &str| {
    Task::new(id).with_activity(
        Activity::new(format!("{id}-1"), id, 0)
            .with_process_time(60_000)
            .with_requirement(ResourceRequirement::new("Machine"))
            .with_speed_on("FAST", 2.0) // 30 s there
            .with_duration_on("OLD", ActivityDuration::fixed(90_000)),
    )
};
let resources = vec![Resource::new("OLD", "Machine"), Resource::new("FAST", "Machine")];
let problem = Problem::new(vec![job("A"), job("B")], resources).expect("valid input");
let schedule = SimpleScheduler::new().schedule(&problem, 0);
let a = schedule.assignment_for_activity("A-1").expect("placed");
let b = schedule.assignment_for_activity("B-1").expect("placed");
assert_eq!((a.resource_id.as_str(), a.end_ms), ("FAST", 30_000)); // free at 0 and faster
assert_eq!((b.resource_id.as_str(), b.end_ms), ("OLD", 90_000)); // free at 0: not left idle
```

### Working time and pauses

A resource with a `Calendar` works inside its availability windows and
outside its blocked periods. By default an activity's whole span — setup,
processing, teardown — must lie in one stretch of working time on every
resource it uses. An activity marked `resumable()` may instead pause while its
resources are not all working (a break, a night) and resume after; the
resources stay held across the pause, and the work advances only in the time
**every** one of them works (`joint_work_end`). Its assignment spans the
pauses: it starts in working time and ends when the work is done.

```rust
use u_schedule::models::{Activity, Calendar, Resource, ResourceRequirement, Task};
use u_schedule::scheduler::SimpleScheduler;
use u_schedule::Problem;

const H: i64 = 3_600_000;
let shift = Calendar::new("day").with_window(8 * H, 12 * H).with_window(13 * H, 17 * H);
let task = Task::new("A").with_activity(
    Activity::new("A-1", "A", 0)
        .with_process_time(3 * H)
        .with_requirement(ResourceRequirement::new("Machine"))
        .resumable(),
);
let resources = vec![Resource::new("M", "Machine").with_calendar(shift)];
let problem = Problem::new(vec![task], resources).expect("valid input");
let a = SimpleScheduler::new().schedule(&problem, 11 * H).assignments.remove(0);
assert_eq!((a.start_ms, a.end_ms), (11 * H, 15 * H)); // 11–12, lunch, 13–15
```

An activity can have a calendar of its own — `Activity::with_calendar` — that
it runs within on top of its resources' calendars. An activity that holds **no
resource** (a cure, a wait for a carrier) still takes its time: it starts when
its task reaches it, within its own calendar if it has one, and the task's
next activity starts after it. It competes for nothing, holds nothing, and is
reported in `Schedule::time_only` (a `TimeOnlySpan` each) rather than as an
assignment; `makespan_ms` and `task_completion_time` count it.

```rust
use u_schedule::models::{Activity, Calendar, Resource, ResourceRequirement, Task};
use u_schedule::scheduler::SimpleScheduler;
use u_schedule::Problem;

const H: i64 = 3_600_000;
let machine = |id: &str, step: i32| {
    Activity::new(id, "J", step)
        .with_process_time(H)
        .with_requirement(ResourceRequirement::new("Machine"))
};
let cure = Activity::new("J-2", "J", 1)
    .with_process_time(3 * H)
    .with_calendar(Calendar::new("plant").with_window(8 * H, 20 * H))
    .resumable();
let task = Task::new("J")
    .with_activity(machine("J-1", 0))
    .with_activity(cure)
    .with_activity(machine("J-3", 2));
let problem = Problem::new(vec![task], vec![Resource::new("M", "Machine")]).expect("valid input");
let schedule = SimpleScheduler::new().schedule(&problem, 6 * H);
let span = schedule.time_only_for_activity("J-2").unwrap();
assert_eq!((span.start_ms, span.end_ms), (8 * H, 11 * H)); // waits for the plant
assert_eq!(schedule.assignment_for_activity("J-3").unwrap().start_ms, 11 * H);
```

### Resources held together

A resource can name **companions** it holds whenever an activity runs on it —
a robot that occupies its cell, a machine that needs its fixture:
`Resource::new("ROBOT", "Machine").with_companion("CELL")`. A candidate is
taken only when it and its companions are free for the whole span; the
companion gets an assignment of its own (same span, no setup), takes no unit
of any requirement and does not change the activity's time — but its calendar
counts for a resumable activity. The feasibility check reports a resource used
without its companion.

### How much of a resource an activity takes

A resource's `capacity` is how much it carries at once, and a requirement's
`usage` (default 1) how much of it each activity takes: an operator at
capacity 10 runs two activities that each need half of them
(`ResourceRequirement::new("Operator").with_usage(5)`) side by side, and a
third waits. `quantity` is a different count — how many different resources a
requirement takes. `SimpleScheduler` and the feasibility check read usages;
the GA decoders and the CP builder hold each resource for one activity at a
time, so their schedules stay within any capacity without sharing it.

### Moving work between resources

Work can take time to move from one resource to another — between machines
on different floors, from one site to the next:
`Resource::new("M1", "Machine").with_transport_to("M2", 60_000)`. When a
task's next activity runs on `M2`, it starts no earlier than the previous
activity's end plus the minute; a pair not listed takes none, and over several
resources the longest pair counts. Every scheduler reads it (the GA on the
resource it chose, the CP builder on the ones it may choose -- an activity
chooses only when every candidate moves the work in the same time), and the
feasibility check reports an activity that starts before its work arrives as a
precedence violation.

```rust
use u_schedule::models::{Activity, Resource, ResourceRequirement, Task};
use u_schedule::scheduler::SimpleScheduler;
use u_schedule::Problem;

let on = |id: &str, step: i32, machine: &str| {
    Activity::new(id, "J1", step)
        .with_process_time(1_000)
        .with_requirement(ResourceRequirement::new("Machine").with_candidates(vec![machine.into()]))
};
let problem = Problem::new(
    vec![Task::new("J1").with_activity(on("J1-1", 0, "M1")).with_activity(on("J1-2", 1, "M2"))],
    vec![
        Resource::new("M1", "Machine").with_transport_to("M2", 5_000),
        Resource::new("M2", "Machine"),
    ],
)
.expect("valid input");
let schedule = SimpleScheduler::new().schedule(&problem, 0);
assert_eq!(schedule.assignment_for_activity("J1-2").unwrap().start_ms, 6_000);
```

### Overlapping steps

A task's steps run in `sequence` order, each after the one before ends —
unless it overlaps it, as when the work moves on in transfer batches:
`with_overlap(Overlap::new(0.25))` lets a step start once a quarter of the
previous step's scheduled span has passed, and it still ends no earlier than
the previous step (plus `with_finish_after(lag_ms)`); a transport time adds to
both bounds. A short step is held back so that it does not finish first. Every
scheduler reads it — the CP builder as an end-to-start delay that may be
negative — and the feasibility check reports either bound broken as a
precedence violation.

```rust
use u_schedule::models::{Activity, Overlap, Resource, ResourceRequirement, Task};
use u_schedule::scheduler::SimpleScheduler;
use u_schedule::Problem;

let on = |id: &str, step: i32, ms: i64, machine: &str| {
    Activity::new(id, "J1", step)
        .with_process_time(ms)
        .with_requirement(ResourceRequirement::new("Machine").with_candidates(vec![machine.into()]))
};
let problem = Problem::new(
    vec![Task::new("J1")
        .with_activity(on("cut", 0, 1_000, "M1"))
        .with_activity(on("bend", 1, 1_000, "M2").with_overlap(Overlap::new(0.25)))],
    vec![Resource::new("M1", "Machine"), Resource::new("M2", "Machine")],
)
.expect("valid input");
let schedule = SimpleScheduler::new().schedule(&problem, 0);
let bend = schedule.assignment_for_activity("bend").unwrap();
assert_eq!((bend.start_ms, bend.end_ms), (250, 1_250));
assert!(schedule.violations.is_empty());
```

### Splitting an activity into parts

`with_split(Split::new(3))` lets an activity's processing be split into up to
three parts that run apart — on several resources at once, or at different
times (`with_min_part(ms)` bounds a part from below, `with_part_overhead(ms)`
adds time to each part after the first). Each part is an assignment carrying
its `Part { index, work_ms }`. `SimpleScheduler` splits when that ends the
activity sooner: for each number of parts it tries equal parts and parts that
end together (a resource free later, or slower, does less), and keeps the
placement that ends first. Splitting is allowed, not required — the GA
decoders and the CP builder place an activity whole, which the feasibility
check accepts as well.

```rust
use u_schedule::models::{Activity, Resource, ResourceRequirement, Split, Task};
use u_schedule::scheduler::SimpleScheduler;
use u_schedule::Problem;

let cut = Activity::new("cut", "J1", 0)
    .with_process_time(1_000)
    .with_requirement(ResourceRequirement::new("Machine"))
    .with_split(Split::new(2));
let problem = Problem::new(
    vec![Task::new("J1").with_activity(cut)],
    vec![Resource::new("M1", "Machine"), Resource::new("M2", "Machine")],
)
.expect("valid input");
let schedule = SimpleScheduler::new().schedule(&problem, 0);
assert_eq!(schedule.assignments_for_activity_all("cut").len(), 2);
assert_eq!(schedule.makespan_ms(), 500);
assert!(schedule.violations.is_empty());
```

`SimpleScheduler` and the feasibility check read calendars, activity
calendars, activities without resources and companions; the GA decoders and the CP
builder do not, so on calendared resources their schedules are checked, not
built, against working time, and an activity without resources has no time
in them (breaches appear in `Schedule::violations`).

## Build & Test

```bash
cargo build
cargo test
# Taillard measurement (instances downloaded separately — see the file header)
cargo run --release --example taillard -- <dir> [bks.csv] [SPT,MWKR,...,GA]
```

## Academic References

- Pinedo (2016), *Scheduling: Theory, Algorithms, and Systems*
- Brucker (2007), *Scheduling Algorithms*
- Blazewicz et al. (2019), *Handbook on Scheduling*
- Haupt (1989), *A Survey of Priority Rule-Based Scheduling*

## Dependencies

- [u-metaheur](https://github.com/iyulab/u-metaheur) — Metaheuristic algorithms (GA, SA, ALNS, CP)
- [u-numflow](https://github.com/iyulab/u-numflow) — Mathematical primitives (statistics, RNG)
- `serde` 1.0 — Serialization
- `rand` 0.9 — Random number generation

## License

MIT License — see [LICENSE](LICENSE).

## WebAssembly / npm

Available as an npm package via [wasm-pack](https://rustwasm.github.io/wasm-pack/).

```bash
npm install @iyulab/u-schedule
```

### TypeScript

Every exported function declares its parameter and return types, and the
declarations are generated from the same structs the binding reads and
serialises, so they cannot drift from what it actually accepts and returns:

```ts
export function solve_jobshop(problem: JobShopInput): JobShopOutput;
```

An absent optional value is declared `T | undefined`, which is what the binding
sends. Nothing needs an `as` cast -- and a wrong assumption about a result's
shape is a compile error rather than something that fails at run time.

The same holds on the way in: a dispatching rule, crossover or mutation the
binding does not know (`"spt"` for `"SPT"`, `"PMX"`) does not compile. The
binding still validates every input at the boundary, for JavaScript callers
and for values that reach it through a cast, and a rejected one says what was
wrong.

### Quick Start

```javascript
import { run_schedule, solve_jobshop } from '@iyulab/u-schedule';

const result = run_schedule({
  jobs: [
    { id: "A", processing_time: 5.0, due_date: 10.0 },
    { id: "B", processing_time: 3.0, due_date: 8.0 },
  ],
  config: { rule: "EDD" }
});
```

### Functions

#### `run_schedule(input) -> ScheduleOutput`

Priority dispatching on a flat job list (single or parallel machines). Supports 12 rules: SPT, LPT, EDD, FCFS, CR, WSPT, MST, S/RO, ATC, LWKR, MWKR, PRIORITY.
Whenever a machine frees, the rule picks among the jobs released by then —
judged at that moment, so CR, MST, S/RO and ATC see the current time — and a
machine never waits for a job that is not yet released while another is
([When a rule decides](#when-a-rule-decides)).

**Input:**
```json
{
  "jobs": [
    { "id": "A", "processing_time": 5.0, "due_date": 10.0, "release_time": 0.0, "weight": 1.0, "priority": 0 }
  ],
  "config": { "rule": "ATC", "num_machines": 1, "atc_k": 2.0 }
}
```

**Output:**
```json
{
  "schedule": [{ "id": "A", "start": 0.0, "end": 5.0, "tardiness": 0.0, "machine": 0 }],
  "makespan": 5.0,
  "total_tardiness": 0.0,
  "machine_utilization": [{ "machine": 0, "busy_time": 5.0, "utilization": 1.0 }]
}
```

Times are in seconds. The engine schedules in whole milliseconds, so every time you send (`processing_time`, `due_date`, `release_time`) is rounded to the nearest millisecond — `0.0004` becomes `0` and `0.0005` becomes `0.001`. `machine_utilization` has one entry per machine (`num_machines`, one included): its busy time and that time over the makespan.
`weight` (default 1, `> 0`) is the WSPT/ATC weight — a heavier job goes earlier;
`priority` (integer, default 0) is read only by `PRIORITY`, higher first.
`processing_time` must be `>= 0` and `num_machines` at least 1.
`atc_k` (`> 0`, default 2) is the lookahead `k` of `ATC` and is read by `ATC`
only: given with any other rule it is refused as `invalid_option`, not ignored.
`available_rules()` returns which rule reads which setting -- the table that
refusal is made from -- so a form can offer `atc_k` only where it has an effect:

```js
import { available_rules } from '@iyulab/u-schedule';

const rules = available_rules();
// [{ name: "SPT", reads: ["num_machines"] }, ..., { name: "ATC", reads: ["num_machines", "atc_k"] }, ...]
console.log(rules.filter((r) => r.reads.includes('atc_k')).map((r) => r.name)); // ["ATC"]
```

#### `available_rules() -> RuleInfo[]`

Every rule `run_schedule` takes, once each (`"SRO"` is accepted as another spelling of
`"S/RO"`), with the `config` settings it reads (`{ name, reads }`).

#### `search_schedule(input, search?) -> SearchOutput`

Searches the order `run_schedule` dispatches in. A biased random-key genetic
algorithm gives each job a key, and each set of keys is decoded by the same
dispatcher with the keys deciding (the lower key first) — so every schedule it
considers is one `run_schedule` could build, and the answer is the better of
the best keyed schedule and `config.rule`'s own: never worse than `run_schedule`
on the same input ([Searching the dispatching order](#searching-the-dispatching-order)).

```js
import { search_schedule } from '@iyulab/u-schedule';

const found = search_schedule(
  {
    jobs: [
      { id: 'long', processing_time: 10, due_date: 30 },
      { id: 'urgent', processing_time: 1, due_date: 1 },
    ],
    config: { rule: 'FCFS' },
  },
  { seed: 7, tardiness_weight: 1 },
);
console.log(found.dispatching_cost, found.cost); // 10 0 -- FCFS runs the urgent job second
```

`input` is `run_schedule`'s. `search` (optional) takes `population_size` (at least 3,
default 40), `max_generations` (at least 1, default 100), `stagnation_limit`
(generations without a better schedule before stopping, at least 1, default 30),
`seed` (a whole number 0 to 2^64 − 1 — the same input and seed give the same
schedule when no `time_limit_ms` is set; absent: fresh entropy), `time_limit_ms`
(at least 1) and `tardiness_weight` (`w` in `[0, 1]`, default 0). The output is
`run_schedule`'s plus `cost` — `(1 − w)` × makespan + `w` × total tardiness, in
seconds — `dispatching_cost` (the same for `config.rule` alone; `cost` is never
above it), `generations` and `stop_reason` (`"iteration_limit"`,
`"no_improvement"`, `"time_limit"`, or `"completed"` for fewer than two jobs).

#### `codes() -> Codes`

Every refusal `code` this package's functions throw, with the fields that can come
with it, and every violation `code` `check_jobshop` reports:
`{ refusals: [{ code, fields }], violations: [{ code, fields, checkers }] }`. It is
built from the tables the refusals and violations come from; the tables under
[Errors](#errors) and `check_jobshop` list the same codes with their meaning.

#### `solve_jobshop(input) -> JobShopOutput`

GA-based job-shop scheduling with multi-machine routing and precedence constraints.

**Input:**
```json
{
  "jobs": [
    {
      "id": "J1",
      "operations": [
        { "machine": "M1", "processing_time": 3.0 },
        { "machines": ["M2", "M3"], "processing_time": 2.0, "machine_times": { "M3": 1.5 },
          "release_time": 4.0, "overlap": { "start_after": 0.5, "finish_after": 0.2 } }
      ],
      "due_date": 15.0
    }
  ],
  "num_machines": 3,
  "transport_times": [{ "from": "M1", "to": "M3", "time": 0.5 }],
  "ga_config": {
    "population_size": 100, "max_generations": 200,
    "mutation_rate": 0.1, "seed": 42,
    "tardiness_weight": 0.5,
    "crossover": "POX", "mutation": "Swap"
  }
}
```

An operation runs on one of its `machines` (or its one `machine`) for
`processing_time` seconds — or, on a machine named in `machine_times`, for
that machine's own time (a flexible job shop's per-machine times; each key one
of the operation's machines, each time ≥ 0).

An operation's `release_time` (optional, seconds) holds it back on top of its
job's: it starts no earlier than that (`check_jobshop` reports an earlier
start as `early_start`).

An operation's `overlap` (optional, not on a job's first operation) lets it
overlap the job's previous operation, as when the work moves on in transfer
batches: it starts once `start_after` (a share in `[0, 1]`) of the previous
operation's scheduled span has passed, and ends no earlier than the previous
operation's end plus `finish_after` seconds (≥ 0, default 0) — so a short
operation is held back until the last batch reaches it. A transport time
between their machines adds to both bounds. `check_jobshop` reports either bound
broken as `precedence_violation`.

`transport_times` (optional) is the time a job's work takes to move between
machines: when a job's next operation runs on `to`, it starts no earlier than
the previous one's end on `from` plus `time` seconds. A pair not listed takes
none; each pair is given once, between two different machines the operations
name, with a time ≥ 0. `check_jobshop` reads the same field and reports a step
that starts before its work arrives as `precedence_violation`.

**Output:**
```json
{
  "schedule": [
    { "job_id": "J1", "operation": 1, "machine": "M1", "start": 0.0, "end": 3.0 },
    { "job_id": "J1", "operation": 2, "machine": "M2", "start": 3.0, "end": 5.0 }
  ],
  "jobs": [{ "id": "J1", "completion": 5.0, "tardiness": 0.0 }],
  "makespan": 5.0,
  "total_tardiness": 0.0,
  "fitness": 5.0,
  "generations": 200,
  "fitness_history": [10.0, 8.0, 5.0],
  "stop_reason": "iteration_limit"
}
```

`stop_reason` says why the search stopped: `"iteration_limit"` when it ran its
`max_generations`, `"no_improvement"` when it stalled first (`"completed"` for a
problem with no jobs).

`operation` counts from 1 in the order the job's `operations` array lists them,
not the order they run on the timeline.

`jobs` lists every job in request order with its `completion` (when its last step
ends) and `tardiness` (`max(0, completion − due_date)`, 0 without a `due_date`);
`total_tardiness` is their sum. It is the tardiness the search weighs against
the makespan (`tardiness_weight`), and the same measure as `run_schedule`'s.

Crossover types: `"POX"` | `"LOX"` | `"JOX"`. Mutation types: `"Swap"` | `"Insert"` | `"Invert"`.

#### `check_jobshop({ problem, solution }) -> JobShopCheckOutput`

Checks a job-shop schedule against the problem it answers — each step of each
job placed once, on a machine it can run on, for its processing time, after the
step ahead of it and the job's release, no two steps on one machine at once,
and the reported `makespan` equal to the latest end. `problem` is the
`solve_jobshop` input, read the same way (the same refusals); `solution` is
`{ schedule, makespan? }`. A schedule that breaks the problem is a result, not
an error.

```js
import { solve_jobshop, check_jobshop } from '@iyulab/u-schedule';

const problem = {
  jobs: [
    { id: 'J1', operations: [{ machine: 'M1', processing_time: 3 }, { machine: 'M2', processing_time: 2 }] },
    { id: 'J2', operations: [{ machine: 'M2', processing_time: 4 }, { machine: 'M1', processing_time: 1 }] },
  ],
  ga_config: { seed: 1 },
};
const answer = solve_jobshop(problem);
const plan = { schedule: answer.schedule, makespan: answer.makespan };
console.log(check_jobshop({ problem, solution: plan }).feasible); // true

plan.schedule[0].end = plan.schedule[0].start + 1;
console.log(check_jobshop({ problem, solution: plan }).violations.map((v) => v.code));
// ["duration_too_short", ...]
```

```json
{ "feasible": false, "violations": [{ "code": "duration_too_short", "entity": "J1_1", "message": "..." }] }
```

Each violation carries its `code`; every code but `makespan_mismatch` also
carries `entity` (the step `"J1_2"` — job and step number —, job or machine)
and a readable `message`; `makespan_mismatch` carries `reported` and `actual`.

| `code` | The schedule breaks |
|---|---|
| `unknown_activity` | a job or step the problem does not have, or a step placed twice |
| `requirement_unfilled` | a step left out, or put on a machine it cannot run on |
| `capacity_exceeded` | two steps on one machine at once |
| `precedence_violation` | a step that starts before the step ahead of it ends |
| `duration_too_short` | a step shorter than its processing time |
| `early_start` | a step that starts before its job's `release_time` |
| `deadline_miss` | a job that ends after its `due_date` |
| `makespan_mismatch` | a reported `makespan` other than the latest end |
| `resource_unavailable` · `time_window_violation` | a step outside its machine's availability or its window (not produced by job-shop problems today) |

**GA config constraints:**

| Parameter | Constraint | Default |
|-----------|-----------|---------|
| `population_size` | >= 2 | 100 |
| `max_generations` | >= 1 | 200 |
| `local_search_steps` | >= 0 (0 turns the local search off) | 60 |
| `mutation_rate` | 0.0 -- 1.0 | 0.1 |
| `tardiness_weight` | 0.0 -- 1.0 | 0.5 |
| `seed` | optional, a whole number from 0 to 2^64 − 1 | random |

#### Errors

Every function is synchronous and throws an `Error` whose `message` is
readable text and which carries a `code` naming the reason, next to the values
behind it — so a program can point at what to change without parsing the
message:

```js
import { solve_jobshop } from '@iyulab/u-schedule';

try {
  solve_jobshop({
    jobs: [{ id: 'J1', operations: [{ machine: 'M1', processing_time: 3 }] }],
    ga_config: { population_size: 1 },
  });
} catch (err) {
  console.log(err.code, err.parameter, err.min, err.got); // parameter_out_of_range ga_config.population_size 2 1
}
```

| `code` | Fields | Meaning |
|---|---|---|
| `duplicate_id` | `entity`, `id` (and `first`, `second` for jobs and transport times; `task` for two steps of one task with one sequence, not produced by today's inputs) | Two jobs share an `id` (at positions `first` and `second`, from 0), a `transport_times` pair is given twice (`entity` `transport_time`, `id` `"M1->M2"`), or two tasks, activities or resources do |
| `unknown_option` | `parameter`, `got`, `expected` | A `rule`, `ga_config.crossover` or `ga_config.mutation` that names none of the supported values |
| `parameter_out_of_range` | `parameter`, `min`, `max`, `exclusive_min`, `exclusive_max`, `got` (and `index`, `id` for a job) | Times that add up to more than about 2.3e15 seconds (`parameter: "total_time"`, `min: 0`, `max` that bound — the engine counts milliseconds in 64 bits: every processing time, plus the latest release, plus a job shop's longest transport time once per operation), a `ga_config` value outside the table above, a `search` value outside its range (`search.population_size` below 3, `search.tardiness_weight` outside `[0, 1]`, …), `config.num_machines` below 1, a job-shop `num_machines` below the number of machines the operations name, a job's `processing_time` below 0 or `weight` not above 0, a job-shop `machine_times` value below 0 (`parameter` `machine_times`, `activity`, `resource`), a `transport_times` time below 0 (`parameter` `transport_times`, `resource` and `to` the pair), an operation's `overlap.start_after` outside `[0, 1]` or `overlap.finish_after` below 0 (`parameter` `overlap.start_after` / `overlap.finish_after`, `activity` the step, `J1_2`), or `config.atc_k` not above 0. A count or seed is checked as the whole number it is, so a negative one is refused like a too-small one: `config.num_machines: -1` is `{ min: 1, max: null, got: -1 }`, as `0` is. `min` / `max` are the closed (included) bounds and are always present — `null` on a side that is open or unbounded; `exclusive_min` / `exclusive_max` are the open (excluded) bounds, present only when that end is open: a `processing_time` of -1 is refused with `min: 0`, a `weight` of 0 with `min: null, exclusive_min: 0` |
| `missing_machine` | `job`, `operation` | A job-shop operation with neither `machine` nor `machines` |
| `no_machines` | — | A job-shop request whose operations name no machine at all |
| `empty_input` | `parameter` (`"operations"`), `task` | A job with no operations |
| `invalid_option` | `parameter` (and `rule` for `config.atc_k`, `activity` for `overlap`) | GA or search settings the runner refuses only in combination (`ga_config.elite_ratio` leaving no elite in the population), `config.atc_k` given with a `rule` other than `ATC` — `rule` names the chosen one — or an `overlap` on a job's first operation, which has no previous one (`parameter` `overlap`, `activity` the step); a single setting outside its range is `parameter_out_of_range` |
| `value_not_finite` | `parameter`, `index` | A NaN or ±Infinity anywhere in an argument — `parameter` is the path to it (`config.nodes[1]`), `index` its position in that array, or `null` |
| `malformed_input` | `parameter` | An argument of the wrong shape or type (a missing or unknown key), a count or seed that is not a whole number (`config.num_machines: 2.5` — `parameter` is the field), or a JSON string |
| `invalid_resource_reference` | `activity`, `resource`, `companion`, `to` | A job-shop operation's `machine_times` naming a machine that is not one of its `machines` (`activity` is the step, `J1_2`; `resource` the machine), or a `transport_times` pair whose `from` or `to` no operation runs on, or that moves a machine to itself (`resource` is `from`). Also a step naming a machine the problem does not have, or a machine whose companion is unknown or itself — not produced by today's inputs (the machines are the ones the operations name, with no companions) |
| `invalid_predecessor` | `activity`, `predecessor` | A step that waits on a step the problem does not have (not produced by today's inputs) |
| `cyclic_dependency` | `activity` | Steps that wait on each other in a cycle (not produced by today's inputs: a job's steps run in their listed order) |
| `invalid_calendar_period` | `parameter`, `resource`, `activity`, `index`, `start`, `end` | A machine's (`resource`) or a step's own (`activity`) calendar period that ends before it starts (not produced by today's inputs: they carry no calendars) |
| `internal` | — | A fault in this package rather than in the request — please report it |

`codes()` returns this table and the `check_jobshop` violations below as data —
`{ refusals: [{ code, fields }], violations: [{ code, fields, checkers }] }` — built
from the same tables the refusals and violations are, so a caller that translates
codes can compare its list with the package's.

When the problem check finds several things wrong (two jobs without operations, a
duplicate id and a cycle, …), `code` and its fields are the first finding's, and
`err.errors` lists every finding the same way — `code`, `message` and its fields —
so all of them can be shown at once.

## npm (WebAssembly)

```bash
npm install @iyulab/u-schedule
```

The package resolves per environment via a conditional `exports` map:

| Environment | Entry |
|---|---|
| Bundlers (webpack, Vite, …) | ESM + WebAssembly ESM-integration (`default` condition) |
| Node.js — `require()`, ESM `import`, CJS TS runners (`tsx`, `ts-node`) | CJS glue loading the wasm from the filesystem (`node` condition) — no loader hooks or flags |

A browser **without** a bundler is not supported: the package loads its `.wasm`
file with an ES module import, which browsers refuse (`application/wasm` is not a
module script type), so `<script type="module">` from a CDN fails, and CDN
re-bundling services fail on the same import. Use a bundler or Node.

## Related

- [u-numflow](https://github.com/iyulab/u-numflow) — Mathematical primitives
- [u-metaheur](https://github.com/iyulab/u-metaheur) — Metaheuristic optimization (GA, SA, ALNS, CP)
- [u-geometry](https://github.com/iyulab/u-geometry) — Computational geometry
- [u-nesting](https://github.com/iyulab/U-Nesting) — 2D/3D nesting and bin packing
