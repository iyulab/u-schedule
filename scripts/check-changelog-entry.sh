#!/usr/bin/env bash
#
# The version in Cargo.toml must have a matching heading in CHANGELOG.md.
#
# Every other release check compares one version string against another, and all
# of them can agree while the changelog says nothing at all — a bump whose entry
# is still sitting under `## [Unreleased]` publishes a version whose consumers
# have no record of what they upgraded into. That gap is what this checks, and
# it is the one that actually kept happening: two releases went out with no
# changelog entry, and two more were nearly graded as patches because the
# additive API they carried was still filed under `[Unreleased]`.
#
# Run locally before pushing a version bump:
#   bash scripts/check-changelog-entry.sh
#
# Enforced in CI (.github/workflows/ci.yml, "Changelog Entry" job). A mismatch is
# emitted as a GitHub Actions ::error:: annotation so it surfaces inline.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

version="$(grep -m1 -E '^version = ' Cargo.toml | sed -E 's/version = "(.*)"/\1/')"
if [[ -z "${version}" ]]; then
  echo "::error file=Cargo.toml::could not read [package] version"
  exit 1
fi
echo "crate version: ${version}"

# Escape the dots so `0.1.0` cannot match a heading like `0X1Y0`.
if grep -qE "^## \[${version//./\.}\]" CHANGELOG.md; then
  echo "Changelog has an entry for ${version}"
  exit 0
fi

echo "::error file=CHANGELOG.md::no '## [${version}]' heading — add the entry for this release in the same commit as the version bump (move it out of '## [Unreleased]')"
exit 1
