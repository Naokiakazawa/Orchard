#!/usr/bin/env bash
# Bracket DeepSWE 1.1 before spending anything on a model.
#
# Two runs, in this order, because a benchmark number is only meaningful
# between them:
#
#   ceiling  `oracle` replays each task's solution/solve.sh, which applies
#            solution/solution.patch and commits it. Expect ~100%. Whatever it
#            falls short by is a defect in this provider or in the cluster, and
#            it caps every model score measured on the same dataset afterwards.
#
#   floor    an agent that does nothing. Expect ~0%. Anything it solves is a
#            task whose verifier passes on the untouched repository — a task
#            that hands free points to every model, forever.
#
# DeepSWE makes both checks sharper than Terminal-Bench does, because grading
# happens in a separate pristine container: the oracle's commits are extracted
# by the [[verifier.collect]] hook, applied to a fresh image built from
# tests/Dockerfile, and graded there. A ceiling below 100% therefore also
# exercises the collect-and-transfer path, not just the build.
#
# Usage:
#   ./scripts/deep_swe_oracle_check.sh                # both, 16-way
#   ./scripts/deep_swe_oracle_check.sh 16 --limit 5   # a slice first
#   CHECK=ceiling ./scripts/deep_swe_oracle_check.sh  # just one of them
#
# Environment:
#   SANDBOX_BASE_URL   required — your orchestrator
#   SANDBOX_API_KEY    required if the orchestrator enforces auth
#   ORACLE_THRESHOLD   minimum ceiling solve rate (default 0.95)
#   FLOOR_MAX          maximum floor solve rate (default 0.02)
#   NULL_AGENT         Harbor's do-nothing agent (default nop; `harbor run
#                      --help` lists the names this build ships)
#   CHECK              ceiling | floor | both (default both)
#   DEEP_SWE_DATASET   override the dataset
#   ORCHARD_HARBOR_IMAGE_REMAP
#                      Task images come from wenlinyao/deep-swe rather than
#                      public.ecr.aws by default; ECR's anonymous pull quota
#                      surfaces as EnvironmentStartTimeoutError, not as a rate
#                      limit. Empty restores the dataset's own registry.
#   ORCHARD_HARBOR_ALLOW_INTERNET
#                      1 skips the network-policy switch entirely, which is
#                      what to use against an orchestrator that serves neither
#                      PUT nor POST /sandboxes/{id}/network. Neither check calls a
#                      model or writes a patch of its own, so both numbers
#                      stay meaningful; only the isolation goes untested.
set -euo pipefail

CONCURRENCY="${1:-16}"
DATASET="${DEEP_SWE_DATASET:-datacurve/deep-swe-1-1@latest}"
CHECK="${CHECK:-both}"
NULL_AGENT="${NULL_AGENT:-nop}"
STAMP="$(date +%Y%m%d-%H%M%S)"

: "${SANDBOX_BASE_URL:?SANDBOX_BASE_URL must point at your orchestrator}"

if ! command -v harbor >/dev/null 2>&1; then
    echo "The 'harbor' CLI is not on PATH. From the repository root:" >&2
    echo "  python -m pip install harbor" >&2
    echo "  python -m pip install -e orchard_env -e orchard_eval/harbor_orchard" >&2
    exit 1
fi

# Neither run calls a model, so seal the sandbox completely. This is also the
# cheapest proof that the air-gap works: if a task somehow passes with no route
# out of the pod and no patch, the grader is the problem.
export ORCHARD_HARBOR_MODEL_EGRESS=0
# The tasks allow the agent 10800s; the oracle needs seconds, but the verifier
# budget is real and the provider must not cut either one off. Capped at the
# orchestrator's SANDBOX_TTL_HOURS (2h), past which the pod is gone anyway.
export ORCHARD_HARBOR_EXEC_TIMEOUT="${ORCHARD_HARBOR_EXEC_TIMEOUT:-7200}"

echo "Dataset:     ${DATASET}"
echo "Concurrency: ${CONCURRENCY}"
echo "Check:       ${CHECK}"
if [[ "${ORCHARD_HARBOR_ALLOW_INTERNET:-0}" != "0" ]]; then
    echo "Isolation:   OFF (ORCHARD_HARBOR_ALLOW_INTERNET) — pods keep egress."
    echo "             Both numbers are still valid: neither agent calls a model"
    echo "             or writes its own patch. Only the air-gap is untested."
fi
echo

status=0

if [[ "${CHECK}" == "ceiling" || "${CHECK}" == "both" ]]; then
    echo "== ceiling: oracle replays each task's own solution =="
    orchard-eval harbor \
        --dataset "${DATASET}" \
        --agent oracle \
        --concurrency "${CONCURRENCY}" \
        --job-name "deepswe-oracle-${STAMP}" \
        --threshold "${ORACLE_THRESHOLD:-0.95}" \
        "${@:2}" || status=1
fi

if [[ "${CHECK}" == "floor" || "${CHECK}" == "both" ]]; then
    echo "== floor: an agent that does nothing =="
    orchard-eval harbor \
        --dataset "${DATASET}" \
        --agent "${NULL_AGENT}" \
        --concurrency "${CONCURRENCY}" \
        --job-name "deepswe-floor-${STAMP}" \
        --max-solve-rate "${FLOOR_MAX:-0.02}" \
        "${@:2}" || status=1
fi

exit "${status}"
