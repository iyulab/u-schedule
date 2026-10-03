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
| `cp` | CP-based scheduling formulation |

## Dispatching Rules

| Rule | Description |
|------|-------------|
| SPT | Shortest Processing Time |
| LPT | Longest Processing Time |
| EDD | Earliest Due Date |
| FIFO | First In First Out |
| MST | Minimum Slack Time |
| CR | Critical Ratio |
| S/RO | Slack per Remaining Operations |
| ATC | Apparent Tardiness Cost (reads `Task::weight`) |
| WSPT | Weighted Shortest Processing Time — `weight / processing time`, reads `Task::weight` |
| MWKR | Most Work Remaining |
| LWKR | Least Work Remaining |
| WINQ | Work In Next Queue |
| LPUL | Least Planned Utilization Level |
| PRIORITY | Job Priority — reads `Task::priority`, higher first |

`Task::weight` (default 1, finite and `> 0`) is the weight `w_j` of the
weighted rules; `Task::priority` is a separate ordinal read only by `PRIORITY`.

## GA Encoding

The GA module uses dual-vector encoding for job-shop scheduling:

- **OSV (Operation Sequence Vector)** — Permutation encoding that determines operation processing order
- **MAV (Machine Assignment Vector)** — Integer vector that assigns each operation to a specific machine (for flexible job shops)

## Quick Start

```toml
[dependencies]
u-schedule = "0.12"
```

```rust
use u_schedule::models::{Activity, ActivityDuration, Resource, ResourceRequirement, ResourceType, Task};
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
let resources = vec![Resource::new("M1", ResourceType::Primary)];

// Duplicate ids, unknown resources and precedence cycles are refused here, so
// the solvers never see them: they take only a `Problem`.
let problem = Problem::new(tasks, resources).expect("valid input");

let schedule = SimpleScheduler::new().schedule(&problem, 0);
assert!(schedule.is_valid());
assert_eq!(schedule.makespan_ms(), 50_000); // the two jobs run back to back
```

## Build & Test

```bash
cargo build
cargo test
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

**Input:**
```json
{
  "jobs": [
    { "id": "A", "processing_time": 5.0, "due_date": 10.0, "release_time": 0.0, "weight": 1.0, "priority": 0 }
  ],
  "config": { "rule": "SPT", "num_machines": 1, "atc_k": 2.0 }
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

Times are in seconds. `machine_utilization` is present only when `num_machines > 1`.
`weight` (default 1, `> 0`) is the WSPT/ATC weight — a heavier job goes earlier;
`priority` (integer, default 0) is read only by `PRIORITY`, higher first.
`processing_time` must be `>= 0` and `num_machines` at least 1.

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
        { "machines": ["M2", "M3"], "processing_time": 2.0 }
      ],
      "due_date": 15.0
    }
  ],
  "num_machines": 3,
  "ga_config": {
    "population_size": 100, "max_generations": 200,
    "mutation_rate": 0.1, "seed": 42,
    "tardiness_weight": 0.5,
    "crossover": "POX", "mutation": "Swap"
  }
}
```

**Output:**
```json
{
  "schedule": [
    { "job_id": "J1", "operation": 1, "machine": "M1", "start": 0.0, "end": 3.0 },
    { "job_id": "J1", "operation": 2, "machine": "M2", "start": 3.0, "end": 5.0 }
  ],
  "makespan": 5.0,
  "fitness": 5.0,
  "generations": 200,
  "fitness_history": [10.0, 8.0, 5.0]
}
```

`operation` counts from 1 in the order the job's `operations` array lists them,
not the order they run on the timeline.

Crossover types: `"POX"` | `"LOX"` | `"JOX"`. Mutation types: `"Swap"` | `"Insert"` | `"Invert"`.

**GA config constraints:**

| Parameter | Constraint | Default |
|-----------|-----------|---------|
| `population_size` | >= 2 | 100 |
| `max_generations` | >= 1 | 200 |
| `mutation_rate` | 0.0 -- 1.0 | 0.1 |
| `tardiness_weight` | 0.0 -- 1.0 | 0.5 |
| `seed` | optional u64 | random |

**Errors:** both functions are synchronous and throw an `Error` whose `message` is
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
| `duplicate_id` | `entity`, `id` (and `first`, `second` for jobs) | Two jobs share an `id` (at positions `first` and `second`, from 0), or two tasks, activities or resources do |
| `unknown_option` | `parameter`, `got`, `expected` | A `rule`, `ga_config.crossover` or `ga_config.mutation` that names none of the supported values |
| `parameter_out_of_range` | `parameter`, `min`, `max` (or `null`), `got` (and `index`, `id` for a job) | A `ga_config` value outside the table above, `config.num_machines` below 1, or a job's `processing_time` below 0 or `weight` not above 0 |
| `missing_machine` | `job`, `operation` | A job-shop operation with neither `machine` nor `machines` |
| `no_machines` | — | A job-shop request whose operations name no machine at all |
| `empty_task` | `task` | A job with no operations |
| `invalid_option` | `parameter` | GA settings the runner itself refuses |
| `value_not_finite` | `parameter`, `index` | A NaN or ±Infinity anywhere in an argument — `parameter` is the path to it (`config.nodes[1]`), `index` its position in that array, or `null` |
| `malformed_input` | `parameter` | An argument of the wrong shape or type (a missing or unknown key), or a JSON string |

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
