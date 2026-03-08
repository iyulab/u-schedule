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
| SLACK | Minimum Slack Time |
| CR | Critical Ratio |
| ATC | Apparent Tardiness Cost |
| WSPT | Weighted Shortest Processing Time |
| MWKR | Most Work Remaining |
| LWKR | Least Work Remaining |
| MOPNR | Most Operations Remaining |
| PRIORITY | Job Priority |
| RANDOM | Random Selection |

## GA Encoding

The GA module uses dual-vector encoding for job-shop scheduling:

- **OSV (Operation Sequence Vector)** — Permutation encoding that determines operation processing order
- **MAV (Machine Assignment Vector)** — Integer vector that assigns each operation to a specific machine (for flexible job shops)

## Quick Start

```toml
[dependencies]
u-schedule = { git = "https://github.com/iyulab/u-schedule" }
```

```rust
use u_schedule::models::{Task, Activity, Resource};
use u_schedule::validation::validate_input;
use u_schedule::dispatching::{DispatchingEngine, Rule};

// Define tasks with activities
let task = Task::new("T1")
    .with_activity(Activity::new("A1", 30_000)); // 30 seconds

let resource = Resource::new("R1", "Machine 1");

// Validate input
let errors = validate_input(&[task], &[resource]);
assert!(errors.is_empty());
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

### Quick Start

```javascript
import init, { run_schedule, solve_jobshop } from '@iyulab/u-schedule';

await init();
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
    { "id": "A", "processing_time": 5.0, "due_date": 10.0, "release_time": 0.0, "weight": 1.0 }
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
  "schedule": [{ "job_id": "J1", "operation": 1, "machine": "M1", "start": 0.0, "end": 3.0 }],
  "makespan": 5.0,
  "fitness": 5.0,
  "generations": 200,
  "fitness_history": [10.0, 8.0, 5.0]
}
```

Crossover types: `"POX"` | `"LOX"` | `"JOX"`. Mutation types: `"Swap"` | `"Insert"` | `"Invert"`.

**GA config constraints:**

| Parameter | Constraint | Default |
|-----------|-----------|---------|
| `population_size` | >= 2 | 100 |
| `max_generations` | >= 1 | 200 |
| `mutation_rate` | 0.0 -- 1.0 | 0.1 |
| `tardiness_weight` | 0.0 -- 1.0 | 0.5 |
| `seed` | optional u64 | random |

**Error handling:** Invalid parameters return a JS error string (not a thrown exception). Check the return value:

```javascript
try {
  const result = solve_jobshop({ jobs: [...], ga_config: { population_size: 0 } });
} catch (e) {
  console.error("Scheduling error:", e); // "ga_config.population_size must be >= 2, got 0"
}
```

## Related

- [u-numflow](https://github.com/iyulab/u-numflow) — Mathematical primitives
- [u-metaheur](https://github.com/iyulab/u-metaheur) — Metaheuristic optimization (GA, SA, ALNS, CP)
- [u-geometry](https://github.com/iyulab/u-geometry) — Computational geometry
- [u-nesting](https://github.com/iyulab/U-Nesting) — 2D/3D nesting and bin packing
