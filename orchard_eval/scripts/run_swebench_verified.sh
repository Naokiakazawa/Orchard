#!/usr/bin/env bash
# Run SWE-bench Verified end to end for one harness.
#
# Usage:
#   ./scripts/run_swebench_verified.sh codex
#   ./scripts/run_swebench_verified.sh pi 64
#   ./scripts/run_swebench_verified.sh mini-swe-agent 128
#
# Requires SANDBOX_BASE_URL / SANDBOX_API_KEY for the orchestrator, plus the
# model credential the chosen harness reads (see configs/<harness>.yaml).
set -euo pipefail

HARNESS="${1:-codex}"
CONCURRENCY="${2:-32}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/../configs/${HARNESS}.yaml"

if [[ ! -f "${CONFIG}" ]]; then
    echo "No config for harness '${HARNESS}' at ${CONFIG}" >&2
    echo "Available: $(ls "${SCRIPT_DIR}/../configs" | sed 's/\.yaml$//' | tr '\n' ' ')" >&2
    exit 1
fi

: "${SANDBOX_BASE_URL:?SANDBOX_BASE_URL must point at your orchestrator}"

echo "Harness:     ${HARNESS}"
echo "Config:      ${CONFIG}"
echo "Concurrency: ${CONCURRENCY}"
echo

# Everything after the config path is a dotted override, so this script stays a
# thin wrapper rather than a second configuration system.
exec orchard-eval run \
    --config "${CONFIG}" \
    --concurrency "${CONCURRENCY}" \
    "${@:3}"
