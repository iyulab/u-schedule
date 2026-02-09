# u-schedule

**Scheduling framework in Rust**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)

## Overview

u-schedule provides domain models, constraints, validation, dispatching rules, and a greedy scheduler for scheduling problems. It builds on `u-metaheur` for metaheuristic algorithms and `u-optim` for mathematical primitives.

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
- [u-optim](https://github.com/iyulab/u-optim) — Mathematical primitives (statistics, RNG)
- `serde` 1.0 — Serialization
- `rand` 0.9 — Random number generation

## License

MIT License — see [LICENSE](LICENSE).

## Related

- [u-optim](https://github.com/iyulab/u-optim) — Mathematical primitives
- [u-metaheur](https://github.com/iyulab/u-metaheur) — Metaheuristic optimization (GA, SA, ALNS, CP)
- [u-geometry](https://github.com/iyulab/u-geometry) — Computational geometry
- [u-nesting](https://github.com/iyulab/U-Nesting) — 2D/3D nesting and bin packing
