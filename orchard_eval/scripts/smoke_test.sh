#!/usr/bin/env bash
# Smoke-test the whole pipeline on a handful of instances before committing to
# a full 500-instance run. Checks the orchestrator, the harness binary, the
# model credential, grading, and reporting in about the time of one rollout.
#
# Usage:
#   ./scripts/smoke_test.sh codex
#   ./scripts/smoke_test.sh mini-swe-agent 5
set -euo pipefail

HARNESS="${1:-codex}"
N="${2:-3}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/../configs/${HARNESS}.yaml"

: "${SANDBOX_BASE_URL:?SANDBOX_BASE_URL must point at your orchestrator}"

echo "Smoke test: ${HARNESS} on ${N} instance(s)"

orchard-eval run \
    --config "${CONFIG}" \
    --limit "${N}" \
    --concurrency "${N}" \
    --run-name "smoke-${HARNESS}" \
    --no-resume \
    --log-level "${LOG_LEVEL}" \
    "${@:3}"

echo
echo "Agent output per instance:"
echo "  results/smoke-${HARNESS}/<run_id>/instances/*/agent.log"
echo "  results/smoke-${HARNESS}/<run_id>/instances/*/trajectory.raw.jsonl"
