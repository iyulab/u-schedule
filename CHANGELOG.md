# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Maintained from 0.2.3 onward; earlier entries list release dates only (see git history).

## [Unreleased]

## [0.2.3] - 2026-06-10

### Changed

- WASM: dropped legacy `*_json` parameter-name suffixes — exported functions
  take native JS objects/arrays, and JSON-string arguments are now rejected
  early with a descriptive error.

## Earlier releases

- 0.2.2 — 2026-03-09
- 0.2.1 — 2026-03-08
- 0.2.0 — 2026-03-08
