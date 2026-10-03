#!/usr/bin/env bash
# Evaluate a model on Terminal-Bench 2.1, on Orchard Env sandboxes.
#
# The SWE-bench scripts drive `orchard-eval run`, which owns the rollout. This
# one drives `orchard-eval harbor`, which does not: Harbor owns the trial
# semantics (agent, verifier, reward files) and this repository owns only the
# environment they run in, via the harbor_orchard provider. So the knobs here
# are Harbor's, not configs/<harness>.yaml's — there is no config file to layer.
#
# Usage:
#   ./scripts/run_terminal_bench.sh                        # codex, 32-way
#   ./scripts/run_terminal_bench.sh terminus-2 16
#   ./scripts/run_terminal_bench.sh codex 32 --limit 10    # smoke test first
#   ./scripts/run_terminal_bench.sh codex 32 --task hello-world --log-level DEBUG
#
# Arguments after the concurrency are forwarded to `orchard-eval harbor`. To
# forward something to `harbor run` itself, put it after a bare `--`:
#   ./scripts/run_terminal_bench.sh codex 32 -- --ak max_turns=100
#
# Environment:
#   SANDBOX_BASE_URL   required — your orchestrator
#   SANDBOX_API_KEY    required if the orchestrator enforces auth
#   MODEL_NAME         required for every agent but `oracle`; the served id
#   MODEL_BASE_URL     OpenAI-compatible /v1 root, for a self-hosted fleet
#   MODEL_BASE_URL_REPLICAS
#                      Read MODEL_BASE_URL as the first of N consecutive ports
#                      (a serve_sglang_fleet.sh / frpc_fleet.sh fleet). Each
#                      trial is then pinned to one server for its whole agent
#                      loop, so that server's prefix KV cache survives its turns.
#   MODEL_ROUTING      sticky (default, hashes the task name), random, or
#                      session (one scripts/session_router.py fronts the fleet)
#   API_KEY            credential for that endpoint (any placeholder if unset
#                      on the server)
#   MODEL_ID           full litellm model string, overriding openai/${MODEL_NAME}
#   ATTEMPTS           trials per task, for pass@k (default 1)
#   TB_DATASET         override the dataset (default Terminal-Bench 2.1)
#
# Run ./scripts/harbor_oracle_check.sh before this. The oracle score is the
# ceiling on whatever this reports; a model number measured under a provider
# that cannot build 10% of the tasks is 10% wrong and looks like a model result.
set -euo pipefail

AGENT="${1:-codex}"
CONCURRENCY="${2:-32}"

# 2.1 pinned by name rather than @latest: this script produces a number you
# intend to compare with someone else's, and the tag has moved before.
DATASET="${TB_DATASET:-terminal-bench/terminal-bench-2-1@latest}"
ATTEMPTS="${ATTEMPTS:-1}"
JOBS_DIR="${JOBS_DIR:-./results/harbor}"

: "${SANDBOX_BASE_URL:?SANDBOX_BASE_URL must point at your orchestrator}"

if ! command -v harbor >/dev/null 2>&1; then
    echo "The 'harbor' CLI is not on PATH. From the repository root:" >&2
    echo "  python -m pip install harbor" >&2
    echo "  python -m pip install -e orchard_env -e orchard_eval/harbor_orchard" >&2
    exit 1
fi

# harbor_orchard is imported by the harbor process, not by this script, so a
# missing install shows up as a mid-run "no attribute OrchardEnvironment"
# hundreds of pods later. The cd keeps orchard_eval/harbor_orchard/ — a project
# directory, not a package — off sys.path, where it would answer the import with
# an empty namespace package instead of failing.
if ! (cd /tmp && python -c "import harbor_orchard as m; assert m.__file__") 2>/dev/null; then
    echo "harbor_orchard is not importable from the environment 'harbor' runs in." >&2
    echo "  python -m pip install -e orchard_env -e orchard_eval/harbor_orchard" >&2
    exit 1
fi

MODEL_ARGS=()
if [[ "${AGENT}" != "oracle" ]]; then
    : "${MODEL_NAME:?MODEL_NAME must be set for agent '${AGENT}' (use --agent oracle for the ceiling check)}"
    # litellm splits on the first "/" to choose a provider, so a self-hosted
    # OpenAI-compatible server needs the openai/ prefix that MODEL_NAME must
    # not itself carry. MODEL_ID escapes this for anything else.
    MODEL_ARGS=(--model "${MODEL_ID:-openai/${MODEL_NAME}}")
    # Harbor's agents resolve credentials through litellm's environment, which
    # is a different convention from the SWE-bench configs' MODEL_BASE_URL /
    # API_KEY. Translate rather than asking for both spellings.
    if [[ -n "${MODEL_BASE_URL:-}" ]]; then
        export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${MODEL_BASE_URL}}"
        export OPENAI_API_BASE="${OPENAI_API_BASE:-${MODEL_BASE_URL}}"
    fi
    if [[ -n "${API_KEY:-}" ]]; then
        export OPENAI_API_KEY="${OPENAI_API_KEY:-${API_KEY}}"
    fi
    # A local server started without --api-key still needs the variable set:
    # litellm raises before it ever sends the request otherwise.
    export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
fi

JOB_NAME="${JOB_NAME:-${AGENT}-tb2.1-$(date +%Y%m%d-%H%M%S)}"

echo "Dataset:     ${DATASET}"
echo "Agent:       ${AGENT}"
echo "Model:       ${MODEL_ID:-${MODEL_NAME:-n/a}}"
echo "Endpoint:    ${OPENAI_BASE_URL:-default} (routing=${MODEL_ROUTING:-sticky}, replicas=${MODEL_BASE_URL_REPLICAS:-1})"
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
