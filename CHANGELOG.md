# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Maintained from 0.2.3 onward; earlier entries list release dates only (see git history).

## [Unreleased]

### Changed

- **Breaking (Rust API):** `SchedulingGaProblem::with_tardiness_weight`
  returns `Result` and refuses a weight that is not a number in `[0, 1]`
  (`TardinessWeightOutOfRange`, code `parameter_out_of_range`). It used to
  clamp, so 1.5 ran as pure tardiness and a NaN reached every fitness.
- **Breaking (Rust API):** `Skill::new` and `Resource::with_skill` store the
  level as given, and `validate_input` refuses a level that is not a number in
  `[0, 1]` (`SkillLevelOutOfRange`). A level of 2 used to run as 1.

### Documentation

- The WebAssembly README states that times in seconds are rounded to the
  nearest millisecond, the engine's unit.

## [0.12.0] - 2026-10-03

### Changed

- Depends on u-metaheur 0.6.

- **Breaking:** `PertEstimate::duration_at_confidence` and
  `DurationDistribution::duration_at_confidence` return `Option<i64>`, `None`
  for a confidence level outside [0, 1] (NaN included), or 1 for a log-normal.
  They used to extrapolate past the range (uniform), return an end of it
  (triangular), or the centre (PERT, NaN).

### Fixed

- `validate_input` (and so `Problem::new`) refuses an activity with a negative
  setup, process or teardown time and a resource with capacity below 1
  (`parameter_out_of_range`). Capacity was floored to 1 without a word.
- WASM `solve_jobshop`:
  - a `num_machines` smaller than the number of machines the operations name
    is refused; it used to be ignored. Padding to `num_machines` no longer
    repeats a machine id the operations already use.
  - `ga_config` is checked even when `jobs` is empty; an unknown crossover or
    a population of 0 used to return an empty schedule.
  - a negative `processing_time` is refused (`parameter_out_of_range`, with the
    `activity`).

## [0.11.0] - 2026-10-03

### Added

- `Task::weight` (default 1) and `Task::with_weight` — the weight `w_j` that
  WSPT and ATC read. `validate_input` refuses a weight that is not finite and
  greater than 0 (`ValidationErrorKind::WeightOutOfRange`, code
  `parameter_out_of_range`).
- WASM `run_schedule`: a job's `priority` (integer, default 0), read by the
  `PRIORITY` rule.

### Fixed

- **Breaking:** WSPT and ATC ranked a heavier job later. They derived the weight
  from `Task::priority` as `1000 / (priority + 1)` — lower priority meaning more
  important — while `Task::priority` and the `PRIORITY` rule treat a higher
  value as more important, and the WASM binding stored `weight` as
  `priority = 1000 × weight`. Every unequal pair of weights came out in the
  reverse of Smith's order, and a weight below 0.001 rounded to the most
  important job. Both rules now read `Task::weight`; `priority` no longer
  affects them. Rust callers that encoded importance in `priority` for WSPT/ATC
  set `with_weight` instead; JS callers that compensated by passing `1 / w`
  pass `w` again.
- **Breaking:** WASM `run_schedule` refuses `config.num_machines: 0`, which
  was silently treated as 1, and a job with a negative `processing_time` or a
  `weight` that is not above 0, with `parameter_out_of_range` naming the job by
  `index` and `id`. An unknown `rule` or zero machines is refused even when
  `jobs` is empty.
- **Breaking (WASM):** the `PRIORITY` rule reads the new `priority` field
  instead of `weight`.
- WASM: a NaN or ±Infinity anywhere in an argument is refused with
  `value_not_finite`, with `parameter` the path to it and `index` its position
  in that array. JSON has no such numbers, so it used to reach the wire schema
  as `null` and be refused as `malformed_input` ("invalid type: null, expected
  f64") — the wrong reason, and the library's own non-finite checks behind the
  binding could not be reached.
- WASM `run_schedule` with `config` left out was refused as an unknown rule
  `""`: the omitted object took a derived default that skipped the field
  defaults. It now means the same as `config: {}` — SPT on one machine.

### Documentation

- The README's rule table listed rules that do not exist (SLACK, MOPNR,
  RANDOM) and left out MST, S/RO, WINQ and LPUL.

## [0.10.0] - 2026-09-30

### Added

- `Problem` — tasks and resources that passed `validate_input`. `Problem::new`
  runs the check once and returns every finding.
- `ValidationErrorKind::code()` and `Entity`; `ValidationError` implements
  `Display` and `std::error::Error`.

### Changed

- Depends on u-numflow 0.7 and u-metaheur 0.5.

- **Breaking:** the solvers take only a checked `Problem`:
  `SchedulingGaProblem::new(&problem)`, `ScheduleCpBuilder::new(&problem)`,
  `SimpleScheduler::schedule(&problem, start_time_ms)` and
  `ScheduleRequest::new(problem)` (its `tasks`/`resources` fields are now one
  `problem` field). The check existed but nothing made a caller run it, and the
  solvers look entities up by id: two tasks sharing an id were solved as one,
  and the other was missing from the schedule with nothing saying so. An
  activity naming a resource that does not exist is now refused up front rather
  than scheduled and reported as `RequirementUnfilled`.
- **Breaking:** `ValidationErrorKind` variants carry the ids that locate the
  problem (`DuplicateId { entity, id }`, `InvalidResourceReference { activity,
  resource }`, ...).
- **Breaking:** the WebAssembly functions throw an `Error` carrying a stable
  `code` and the values behind it (`duplicate_id`, `unknown_option`,
  `parameter_out_of_range`, `missing_machine`, ...) instead of a bare string —
  `err.message` reads as before, but `String(err)` now starts with `Error: `.
  The README lists every code and its fields.

- The README says a browser without a bundler is not supported (the package
  loads its `.wasm` through an ES module import, which browsers refuse), instead
  of listing only the environments that work.

## [0.9.0] - 2026-09-30

### Changed

- The publishing workflow runs the README's JavaScript examples against the
  built package before it publishes, so an example that throws is caught
  before a reader copies it.

### Fixed

- The README's JavaScript example imported a default `init` and called
  `await init()`. This package has no default export -- it initialises when it
  is imported, in Node and in bundlers alike -- so the example threw
  `init is not a function` on its first line. It now imports the functions
  directly.
- **A job `id` given to two jobs is refused** by `run_schedule` and
  `solve_jobshop`, naming the id and both positions. `solve_jobshop` used to
  drop one of the two jobs from the schedule without a word; `run_schedule`
  listed both under one name. Input that used to be accepted is now refused.
- The README said a rejected input is returned rather than thrown. Both
  functions throw the message string.
- The README's Rust example did not compile: it imported a
  `DispatchingEngine` and `Rule` that do not exist, called `Activity::new`
  and `Resource::new` with the wrong arguments, and read `validate_input`'s
  `Result` as a list. It now validates and schedules two jobs with
  `SimpleScheduler`. The Quick Start pointed at the git repository instead of
  the published crate.
  The README's Rust examples are now compiled and run with the doc-tests,
  so an example that stops matching the API fails CI.

## [0.8.0] - 2026-09-29

### Changed

- **Every exported WASM function declares its parameter types.** Inputs were
  typed `any`; `run_schedule` takes `ScheduleInput` and `solve_jobshop`
  `JobShopInput`, declared from the structs the binding deserialises, with the
  rule, crossover and mutation as unions of the names accepted. **TypeScript
  code that passed a wrong shape or an unknown name now fails to compile**;
  the runtime path is unchanged.
- The publishing workflow now also fails if an exported function takes a
  parameter typed `any` (`check-typed-dts.sh`).

## [0.7.2] - 2026-09-25

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
