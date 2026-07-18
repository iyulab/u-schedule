# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Maintained from 0.2.3 onward; earlier entries list release dates only (see git history).

## [Unreleased]

## [0.4.0] - 2026-07-18

Closes the model-solver enforcement gap surfaced by a consumer-side
runtime probe: models expressed multi-resource requirements, calendars,
capacities, and duration components, but no solver enforced them and
`Schedule::is_valid()` was vacuously true.

### Added

- `scheduler::ResourceTimeline` — capacity- and calendar-aware booking
  ledger (`earliest_fit`/`fits`/`book`), verified against brute-force by
  property tests.
- `scheduler::{check_schedule, annotate_schedule, FeasibilityInput}` —
  post-hoc feasibility validation filling `Schedule::violations`
  (requirement coverage, simultaneity, skills, capacity incl.
  `Constraint::Capacity` tightening, calendars, precedence, deadlines,
  `TimeWindow`/`NoOverlap`/`Synchronize`).
- `Calendar::interval_fits` — whole-interval working-time check
  (single-window containment semantics).
- `ViolationType::{RequirementUnfilled, TimeWindowViolation,
  SynchronizeViolation}` + `Violation` constructors.
- `Schedule::assignments_for_activity_all` — full assignment set of a
  multi-resource activity.
- `ScheduleRequest::constraints` + `with_constraints` — validated
  post-hoc on `schedule_request` output.
- `TransitionMatrixCollection::has_matrix`.
- Integration + property test suite `tests/enforcement.rs`
  (proptest dev-dependency).

### Changed

- **`SimpleScheduler` now enforces the model** (serial SGS fixed-point,
  Kolisch & Hartmann 1999): an activity books **all** its resource
  requirements (`quantity` units each) for one simultaneous interval;
  calendars and capacities are honored; the occupied span is
  `max(setup) + process + teardown` where setup comes from a resource's
  transition matrix when defined, else `ActivityDuration::setup_ms`.
  Consequences: multi-requirement activities now yield one `Assignment`
  per held resource (`assignment_for_activity` returns the first);
  schedules may differ from 0.3.x where declared constraints were being
  silently ignored. Unfillable requirements leave the activity
  unassigned and are reported as `RequirementUnfilled` instead of being
  silently dropped.
- `SimpleScheduler` output self-annotates via the feasibility checker —
  `Schedule::is_valid()` is now meaningful on this path.

### Unchanged

- GA decode and CP builder enforcement (single-resource, no calendar) —
  see the solver enforcement matrix in the crate docs; run
  `check_schedule` on their output for honest violation reports.
- WASM surface (`run_schedule`, `solve_jobshop`) — separate dispatching
  DTOs, no schema change.

## [0.3.2] - 2026-07-05

### Fixed

- npm: expose the `./package.json` subpath in the `exports` map so tools
  that `require('<pkg>/package.json')` (license scanners, version
  reporters) keep working alongside the conditional exports introduced in
  the previous release (`ERR_PACKAGE_PATH_NOT_EXPORTED`).

## [0.3.1] - 2026-07-05

### Fixed

- **npm packaging — Node-compatible entry.** The npm package previously
  shipped only the wasm-bindgen *bundler*-target output, whose static
  `.wasm` import fails on Node's CJS path (`tsx`/`ts-node` in non-ESM
  packages) with an opaque `SyntaxError: Invalid or unexpected token`.
  The package now additionally ships the *nodejs*-target CJS glue under
  `node/` and routes Node consumers to it via a conditional `exports`
  map (`node` → CJS with filesystem wasm loading, `default` → bundler
  ESM). `require()`, native ESM `import`, and CJS TS runners all work
  without loader hooks. A pre-publish smoke test (CJS `require` + ESM
  `import`) now guards this path in CI. Rust API unchanged.

### Changed

- `u-numflow` dependency `^0.2` → `^0.3` (compatible; 0.3.0 publishes the
  previously-unreleased `wasm` feature and input-validation hardening —
  no API used by this crate changed).


## [0.3.0] - 2026-06-12

### Changed — BREAKING (WASM)

- WASM input/config objects (`run_schedule`, `solve_jobshop` — including nested
  job, operation, and `ga_config` objects) now **reject unknown keys** with an
  explicit `unknown field` error instead of silently ignoring them
  (`serde(deny_unknown_fields)`). Remove any extra keys when upgrading.

### Changed

- Dependency: `u-metaheur` `^0.2` → `^0.3`.

### Fixed

- Latent test defect (wasm feature only): minimal jobshop GA test used
  population 4 with the default elite ratio, flooring to 0 elites, which
  `GaConfig::validate` rejects.

## [0.2.3] - 2026-06-10

### Changed

- WASM: dropped legacy `*_json` parameter-name suffixes — exported functions
  take native JS objects/arrays, and JSON-string arguments are now rejected
  early with a descriptive error.

## Earlier releases

- 0.2.2 — 2026-03-09
- 0.2.1 — 2026-03-08
- 0.2.0 — 2026-03-08
