# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Maintained from 0.2.3 onward; earlier entries list release dates only (see git history).

## [Unreleased]

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
