#!/usr/bin/env bash
# Run SWE-bench Pro end to end for one harness.
#
# Usage:
#   ./scripts/run_swebench_pro.sh codex
#   ./scripts/run_swebench_pro.sh pi 32
#   ./scripts/run_swebench_pro.sh mini-swe-agent 64 dataset.limit=50
#
# The harness config supplies the agent and its credentials; configs/swebench-pro.yaml
# is layered on top and supplies everything that is a property of the benchmark
# (image registry, /app workdir, timeouts). Order matters — the overlay must come
# second so its sandbox settings win.
#
# Requires SANDBOX_BASE_URL / SANDBOX_API_KEY for the orchestrator, plus the
# model credential the chosen harness reads (see configs/<harness>.yaml).
#
# This is one of two paths to SWE-bench Pro. `run_swebench_pro_harbor.sh` runs
# the same 731 instances through Harbor instead, which owns the trial end to
# end; `compare_swebench_pro.py` pairs the two up per instance. Where they
# disagree, one of them is wrong — which is the only way to find out that
# either is.
set -euo pipefail

HARNESS="${1:-mini-swe-agent}"
CONCURRENCY="${2:-16}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/../configs/${HARNESS}.yaml"
OVERLAY="${SCRIPT_DIR}/../configs/swebench-pro.yaml"

if [[ ! -f "${CONFIG}" ]]; then
    echo "No config for harness '${HARNESS}' at ${CONFIG}" >&2
    echo "Available: $(ls "${SCRIPT_DIR}/../configs" | sed 's/\.yaml$//' | tr '\n' ' ')" >&2
    exit 1
fi

: "${SANDBOX_BASE_URL:?SANDBOX_BASE_URL must point at your orchestrator}"

echo "Harness:     ${HARNESS}"
echo "Config:      ${CONFIG} + ${OVERLAY}"
echo "Concurrency: ${CONCURRENCY}"
echo

exec orchard-eval run \
    --config "${CONFIG}" \
    --config "${OVERLAY}" \
    --run-name "${HARNESS}-swebench-pro" \
    --concurrency "${CONCURRENCY}" \
    "${@:3}"
