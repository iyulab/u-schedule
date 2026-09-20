#!/usr/bin/env bash
# Fails when a generated TypeScript declaration does not describe what the
# binding returns:
#
#   1. an exported function whose return type is `any`
#   2. a type named in a signature or a field that the file never declares
#
# (1) is the reported harm: `as` is the only thing a consumer can write against
# `any`, and `as` is exactly the construct that silences a wrong assumption
# about a result's shape -- a renderer read a per-point boolean mask as an
# array of indices, compiled, shipped, and printed the same subgroup twelve
# times. (2) is how (1) comes back disguised: the declaration names a type that
# does not exist, because the struct behind it is missing its derive, and the
# consumer's build then fails on our file instead of on their mistake.
#
# This runs on the publish path, not only in CI: the two run side by side on
# the same push, so a check that lives only in CI cannot stop a publish.
#
# Usage: check-typed-dts.sh <file.d.ts> [more.d.ts ...]
set -uo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: $(basename "$0") <file.d.ts> [more.d.ts ...]" >&2
    exit 2
fi

# Types a declaration may name without declaring them.
builtins='number|string|boolean|any|void|bigint|symbol|unknown|never|null|undefined|object|Array|Promise|Record|Map|Set|Date|Uint8Array|Uint16Array|Uint32Array|Int8Array|Int16Array|Int32Array|Float32Array|Float64Array|BigInt64Array|BigUint64Array|ArrayBuffer|Function|Error|readonly'

status=0
for dts in "$@"; do
    if [ ! -f "$dts" ]; then
        echo "FAIL: $dts does not exist" >&2
        status=1
        continue
    fi

    total=$(grep -c '^export function' "$dts")
    if [ "$total" -eq 0 ]; then
        echo "FAIL: $dts declares no exported functions -- generated from the wrong build?" >&2
        status=1
        continue
    fi

    untyped=$(grep '^export function' "$dts" | grep -E '\):[[:space:]]*any;')
    if [ -n "$untyped" ]; then
        count=$(printf '%s\n' "$untyped" | wc -l | tr -d ' ')
        echo "FAIL: $dts returns \`any\` from $count of $total exported function(s):" >&2
        printf '%s\n' "$untyped" | sed 's/^/  /' >&2
        echo "  Derive the declaration from the result struct (tsify) and name it in" >&2
        echo "  #[wasm_bindgen(unchecked_return_type = \"...\")]." >&2
        status=1
    fi

    # Every capitalised name a signature or a field mentions has to be declared
    # in the same file.
    declared=$(grep -oE '^export (interface|type|class|enum) [A-Za-z_][A-Za-z_0-9]*' "$dts" | awk '{print $3}' | sort -u)
    returns=$(grep -E '^export function' "$dts" | sed -E 's/.*\):[[:space:]]*//')
    fields=$(grep -E '^    [a-z_][a-z_0-9]*\??:' "$dts" | sed -E 's/^[^:]*:[[:space:]]*//')
    referenced=$(printf '%s\n%s\n' "$returns" "$fields" \
        | grep -oE '[A-Za-z_][A-Za-z_0-9]*' \
        | grep -E '^[A-Z]' \
        | grep -vE "^($builtins)$" \
        | sort -u)

    # Membership by hand: `comm` wants process substitution, which is not
    # dependable in every shell this has to run in.
    missing=''
    for name in $referenced; do
        if ! printf '%s\n' "$declared" | grep -Fxq "$name"; then
            missing="$missing $name"
        fi
    done
    if [ -n "${missing// /}" ]; then
        echo "FAIL: $dts names types it does not declare:" >&2
        for name in $missing; do
            echo "  $name" >&2
        done
        echo "  A struct reached through a declared one needs the derive too." >&2
        status=1
    fi

    if [ -z "$untyped" ] && [ -z "${missing// /}" ]; then
        echo "OK: $dts -- $total exported function(s), every return type declared"
    fi
done

exit "$status"
