#!/usr/bin/env bash
# Evaluate a model on DeepSWE 1.1, on Orchard Env sandboxes.
#
# Same shape as run_terminal_bench.sh — Harbor owns the trial, this repository
# owns only the environment — but DeepSWE differs from Terminal-Bench in three
# ways that decide whether a run means anything:
#
#   1. Every task is air-gapped (`network_mode = "no-network"` on both the
#      agent and the verifier). The provider honours that by isolating the pod.
#      Every harness here runs its model loop INSIDE the sandbox, so each one
#      needs the endpoint it talks to allowlisted — which is what MODEL_BASE_URL
#      and ORCHARD_HARBOR_EGRESS_ALLOW arrange, and is on by default.
#
#      To bring the benchmark up before that plumbing is right, set
#      ORCHARD_HARBOR_ALLOW_INTERNET=1: the pod gets full egress and the task's
#      air-gap is ignored. Correctness testing only. DeepSWE's images gc the
#      repository's future history so the reference solution cannot leak from
#      git; egress puts it one `git clone` away, so a score measured this way
#      is not comparable with an air-gapped one or with the leaderboard.
#
#   2. Grading happens in a separate verifier container. The agent commits its
#      work, a [[verifier.collect]] hook turns those commits into a patch, and
#      the patch is applied in a pristine image built from tests/Dockerfile.
#      Nothing the agent did to its own container can reach the verifier, so a
#      task that "passes" in the agent pod still has to pass there.
#
#   3. The agent gets 10800s per task, but the orchestrator reaps a sandbox at
#      SANDBOX_TTL_HOURS (2h), so ORCHARD_HARBOR_EXEC_TIMEOUT is capped there
#      rather than at the task budget. A loop cut off is reported as exit -1 —
#      indistinguishable, in the summary, from a crash.
#
# Usage:
#   ./scripts/run_deep_swe.sh                              # codex, 32-way
#   ./scripts/run_deep_swe.sh pi 32 --limit 5              # smoke test first
#   ./scripts/run_deep_swe.sh oracle 16                    # the ceiling
#   ./scripts/run_deep_swe.sh pi 32 -- --override-cpus 8
#
# Arguments after the concurrency go to `orchard-eval harbor`; anything after a
# bare `--` is forwarded verbatim to `harbor run`.
#
# Environment:
#   SANDBOX_BASE_URL   required — your orchestrator
#   SANDBOX_API_KEY    required if the orchestrator enforces auth
#   MODEL_NAME         required for every agent but `oracle`; the served id
#   MODEL_BASE_URL     OpenAI-compatible /v1 root, for a self-hosted fleet
#   MODEL_BASE_URL_REPLICAS
#                      Read MODEL_BASE_URL as the first of N consecutive ports.
#                      Each trial is pinned to one server for its whole loop.
#   MODEL_ROUTING      sticky (default, hashes the task name), random, or
#                      session (one scripts/session_router.py fronts the fleet)
#   API_KEY            credential for that endpoint
#   MODEL_ID           full litellm model string, overriding openai/${MODEL_NAME}
#   ORCHARD_HARBOR_EGRESS_ALLOW
#                      Extra destinations an isolated pod keeps, comma
#                      separated: hostnames, URLs or CIDRs. Needed when the
#                      fleet sits behind a load balancer whose address set is
#                      wider than one DNS answer.
#   ORCHARD_HARBOR_MODEL_EGRESS
#                      0 seals the sandbox completely. Correct for oracle,
#                      which never calls a model.
#   ORCHARD_HARBOR_ALLOW_INTERNET
#                      1 ignores the task's air-gap entirely. For correctness
#                      testing; the resulting score is not comparable.
#   ATTEMPTS           trials per task, for pass@k (default 1)
#   DEEP_SWE_DATASET   override the dataset
#   ORCHARD_HARBOR_IMAGE_REMAP
#                      Task images are pulled from wenlinyao/deep-swe rather
#                      than public.ecr.aws by default, because ECR's anonymous
#                      pull quota turns into EnvironmentStartTimeoutError at
#                      any real concurrency. Point this at your own mirror
#                      (see mirror_deep_swe_images.sh) or set it empty to use
#                      the registry the dataset names.
#
# Run ./scripts/deep_swe_oracle_check.sh first. Its score is the ceiling on
# whatever this reports.
set -euo pipefail

AGENT="${1:-codex}"
CONCURRENCY="${2:-32}"

DATASET="${DEEP_SWE_DATASET:-datacurve/deep-swe-1-1@latest}"
ATTEMPTS="${ATTEMPTS:-1}"
JOBS_DIR="${JOBS_DIR:-./results/harbor}"

# The agent budget the orchestrator can actually honour: sandboxes are reaped
# at SANDBOX_TTL_HOURS (2h), so a larger value only parks a trial on a pod that
# no longer exists. Raise both together to give the agent its full 10800s.
export ORCHARD_HARBOR_EXEC_TIMEOUT="${ORCHARD_HARBOR_EXEC_TIMEOUT:-7200}"

: "${SANDBOX_BASE_URL:?SANDBOX_BASE_URL must point at your orchestrator}"

if ! command -v harbor >/dev/null 2>&1; then
    echo "The 'harbor' CLI is not on PATH. From the repository root:" >&2
    echo "  python -m pip install harbor" >&2
    echo "  python -m pip install -e orchard_env -e orchard_eval/harbor_orchard" >&2
    exit 1
fi

# harbor_orchard is imported by the harbor process, not by this script. The cd
# keeps orchard_eval/harbor_orchard/ — a project directory, not a package — off
# sys.path, where it would answer the import with an empty namespace package
# instead of failing.
if ! (cd /tmp && python -c "import harbor_orchard as m; assert m.__file__") 2>/dev/null; then
    echo "harbor_orchard is not importable from the environment 'harbor' runs in." >&2
    echo "  python -m pip install -e orchard_env -e orchard_eval/harbor_orchard" >&2
    exit 1
fi

MODEL_ARGS=()
if [[ "${AGENT}" == "oracle" ]]; then
    # Nothing in an oracle trial talks to a model, so leave no route out of the
    # pod at all. This is also the run that proves the isolation works.
    export ORCHARD_HARBOR_MODEL_EGRESS="${ORCHARD_HARBOR_MODEL_EGRESS:-0}"
else
    : "${MODEL_NAME:?MODEL_NAME must be set for agent '${AGENT}' (use --agent oracle for the ceiling check)}"
    MODEL_ARGS=(--model "${MODEL_ID:-openai/${MODEL_NAME}}")
    if [[ -n "${MODEL_BASE_URL:-}" ]]; then
        export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${MODEL_BASE_URL}}"
        export OPENAI_API_BASE="${OPENAI_API_BASE:-${MODEL_BASE_URL}}"
    fi
    if [[ -n "${API_KEY:-}" ]]; then
        export OPENAI_API_KEY="${OPENAI_API_KEY:-${API_KEY}}"
    fi
    export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
fi

JOB_NAME="${JOB_NAME:-${AGENT}-deepswe1.1-$(date +%Y%m%d-%H%M%S)}"

echo "Dataset:     ${DATASET}"
echo "Agent:       ${AGENT}"
echo "Model:       ${MODEL_ID:-${MODEL_NAME:-n/a}}"
echo "Endpoint:    ${OPENAI_BASE_URL:-default} (routing=${MODEL_ROUTING:-sticky}, replicas=${MODEL_BASE_URL_REPLICAS:-1})"
echo "Egress:      model=${ORCHARD_HARBOR_MODEL_EGRESS:-1} extra=${ORCHARD_HARBOR_EGRESS_ALLOW:-none}"
if [[ "${ORCHARD_HARBOR_ALLOW_INTERNET:-0}" != "0" ]]; then
    echo
    echo "  !! ORCHARD_HARBOR_ALLOW_INTERNET is set. The task's air-gap is being"
    echo "     ignored, so the agent can reach the upstream repository this task"
    echo "     was cut from. Correctness testing only — this score is not"
    echo "     comparable with an air-gapped run or with the leaderboard."
    echo
fi
echo "Exec budget: ${ORCHARD_HARBOR_EXEC_TIMEOUT}s"
echo "Concurrency: ${CONCURRENCY}  (attempts per task: ${ATTEMPTS})"
echo "Job:         ${JOBS_DIR}/${JOB_NAME}"
echo

# The ${a[@]+...} guard keeps `set -u` from tripping over the empty array the
# oracle agent leaves behind, which bash 3.2 still treats as unbound.
exec orchard-eval harbor \
    --dataset "${DATASET}" \
    --agent "${AGENT}" \
    ${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"} \
    --concurrency "${CONCURRENCY}" \
    --attempts "${ATTEMPTS}" \
    --jobs-dir "${JOBS_DIR}" \
    --job-name "${JOB_NAME}" \
    "${@:3}"
