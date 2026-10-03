#!/usr/bin/env bash
# Replay every Terminal-Bench task's own solution and report the solve rate.
#
# Each Harbor task ships solution/solve.sh, and the `oracle` agent runs it. The
# score this produces is the ceiling on every model score measured against the
# same dataset afterwards: whatever it falls short of 100% by is a defect in the
# environment provider or in the cluster, not in a model. Run it before you
# spend anything on a real evaluation.
#
# Defaults to Terminal-Bench 2.1, which is the better dataset to *debug* against
# — almost every task is a single container with a shared verifier, so a failure
# is a translation-layer bug rather than an unsupported feature. It is not the
# better dataset to *trust*: 2.1 exercises the separate verifier environment in
# 2 of 90 tasks, while 4.0 uses it in all 66. Run a 4.0 slice too before you
# believe the provider works (see the bottom of this file).
#
# Usage:
#   ./scripts/harbor_oracle_check.sh                       # TB 2.1, 32-way
#   ./scripts/harbor_oracle_check.sh terminal-bench/terminal-bench@4.0.0 16
#   ./scripts/harbor_oracle_check.sh <dataset> <concurrency> --task hello-world
#
# Requires SANDBOX_BASE_URL / SANDBOX_API_KEY, and `harbor` on PATH with
# harbor-orchard importable from the same environment (see harbor_orchard/README).
set -euo pipefail

# @latest is right for a sanity check; pin a version when you intend to compare
# two numbers later.
DATASET="${1:-terminal-bench/terminal-bench-2-1@latest}"
CONCURRENCY="${2:-32}"

: "${SANDBOX_BASE_URL:?SANDBOX_BASE_URL must point at your orchestrator}"

if ! command -v harbor >/dev/null 2>&1; then
    echo "The 'harbor' CLI is not on PATH. From the repository root:" >&2
    echo "  python -m pip install harbor" >&2
    echo "  python -m pip install -e orchard_env -e orchard_eval/harbor_orchard" >&2
    exit 1
fi

echo "Dataset:     ${DATASET}"
echo "Concurrency: ${CONCURRENCY}"
echo "Agent:       oracle (replays each task's solution/solve.sh)"
echo

# --threshold makes this usable as a CI gate. 0.98 on 2.1, where essentially
# every task is supported; lower it to ~0.80 for 4.0, whose compose and GPU
# tasks are correctly declined rather than run.
exec orchard-eval harbor \
    --dataset "${DATASET}" \
    --agent oracle \
    --concurrency "${CONCURRENCY}" \
    --job-name "oracle-$(date +%Y%m%d-%H%M%S)" \
    --threshold "${ORACLE_THRESHOLD:-0.98}" \
    "${@:3}"

# Once this is green, cover the path 2.1 barely touches:
#
#   ORACLE_THRESHOLD=0.80 ./scripts/harbor_oracle_check.sh \
#       terminal-bench/terminal-bench@4.0.0 16
