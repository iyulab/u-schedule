# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Maintained from 0.2.3 onward; earlier entries list release dates only (see git history).

## [Unreleased]

### Fixed

- **`PertEstimate` percentiles stay within the estimate.**
  `duration_at_confidence` (and so `p85`, `p95` and
  `DurationDistribution::Pert`) used the textbook normal approximation
  without bounds, so a low confidence level returned a duration shorter than
  the optimistic estimate: for O = M − s, the 0.1 % point was M − 1.03 s. It
  is now clamped to `[O, P]`, and `probability_of_completion` is 0 before O
  and 1 from P on -- including a zero-width estimate, which divided by zero.
  Inside the interval the textbook values are unchanged; the type
  documentation now says it is the textbook convention, not the Beta-PERT
  quantile.

## [0.7.1] - 2026-09-20

### Added

- **Every exported WASM function declares its return type.** They were typed
  `(...) => any`, with the output's field *names* in the doc comment and the
  element types only in the README -- so a consumer's wrong assumption about a
  result's shape compiled and shipped. `as` is the only thing that can be
  written against `any`, and it is exactly the construct that silences this.

  The declarations are derived from the structs the binding already
  serialises, so there is no second copy to drift: `tsify` emits the interface
  and `unchecked_return_type` names it in the signature. The runtime path is
  unchanged -- same serializer, same bytes. An optional field is declared
  `T | undefined`, which is what the binding sends.

  A publish-path check (`scripts/check-typed-dts.sh`) fails the release if any
  exported function returns `any`, or if a declaration names a type the file
  does not declare. It runs before publishing rather than beside it in CI,
  because the two run on the same push.

  Inputs remain `any`; they are validated at the boundary.

## [0.7.0] - 2026-09-16

### Fixed

- **A job-shop schedule says which step of its job each entry is.** Every row
  of `solve_jobshop`'s `schedule` reported `operation: 1`, whatever step it
  actually was. Machines, start/end times and makespan were all correct, so
  nothing downstream could detect it.

  The cause was two things covering for each other. `ActivityInfo`, which the
  genetic decoder works from, dropped the activity's own id -- so with nothing
  else to hand, the decoder put the *task* id in each assignment's
  `activity_id`, the same string for every step of a job. The WebAssembly
  binding then recovered the step by parsing a trailing number out of that id
  and falling back to `1` when it could not. The CP solver never had the
  defect; it passes the activity id.

### Changed (breaking)

- **`Assignment` carries `sequence: Option<i32>`** -- which step of its task
  the activity is, counting from 1 in the order the task lists them. Both
  solvers set it; `Assignment::new` leaves it `None` and `with_sequence` sets
  it. It is an `Option` rather than a number because a schedule that reports
  every entry as step 1 is exactly the failure above.

- **`ActivityInfo` carries `id`** -- the activity's own id, which it used to
  discard.

- `solve_jobshop`'s `operation` is read from `sequence` rather than parsed out
  of an id. The value and its base are unchanged (counting from 1, in the order
  the job's `operations` array lists them); the README now says so.

## [0.6.1] - 2026-09-15

### Fixed

- Confidence-based durations and on-time probabilities use `u-numflow` 0.6's
  normal quantile and CDF: the quantile was accurate only to 4.5e-4
  (Abramowitz & Stegun 26.2.23) and the CDF to an absolute 7.5e-8; both are now
  accurate to double precision, so these values change in their trailing digits.

## [0.6.0] - 2026-09-07

### Changed (breaking)

- **`rand` is now 0.10** (previously 0.9). `rand::Rng` appears in this crate's
  public signatures, so the two versions are not interchangeable at the boundary
  and callers must move to `rand` 0.10 as well. The generated sequences for a
  given seed are unchanged, so seeded runs reproduce the previous release's
  results.
- **The minimum supported Rust version is now declared as 1.87** and is verified
  by building on that exact toolchain; 1.86 and below fail. The requirement comes
  from this crate's own use of `unsigned_is_multiple_of`, stabilised in 1.87.
  The crate previously declared no `rust-version` at all.
- **`u-metaheur` is now required at 0.4 and `u-numflow` at 0.4** (previously 0.3
  for both), following those crates' own `rand` 0.10 breaks.

### Changed

- **`getrandom` is now 0.4** on WebAssembly targets, reaching the browser entropy
  source through its `wasm_js` crate feature alone. The
  `RUSTFLAGS --cfg getrandom_backend="wasm_js"` that 0.3 required is no longer
  needed.

## [0.5.0] - 2026-07-19

Recorded retroactively: 0.5.0 was released without a changelog entry, and this
one is reconstructed from the release commits (`9ec48f6`, `7576eaf`, `9fca2d8`,
`30ed271`). Dates and contents come from those commits, not from a contemporary
note.

### Added

- `SimpleScheduler::with_fixed_assignments` — seeds the schedule with
  assignments that are fixed in advance ("pins"). At most one pin applies per
  `(activity, resource)` pair. The serial SGS honours a pin all-or-nothing: an
  activity whose pin cannot be placed is skipped rather than partially placed,
  and pins that conflict with each other are reported through the existing
  feasibility annotation as violations rather than being silently dropped.

### Known limitations

- Setup time from `TransitionMatrix` does not participate in pin accounting, so
  a pinned start does not reserve its own changeover window.

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
