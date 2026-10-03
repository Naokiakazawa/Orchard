#!/usr/bin/env bash
# Evaluate a model on SWE-bench Pro *through Harbor*, on Orchard Env sandboxes.
#
# This is the second path to the same benchmark. `run_swebench_pro.sh` drives
# `orchard-eval run`, which owns the rollout, the patch extraction and the
# grading; this one drives `orchard-eval harbor`, which owns none of them —
# Harbor supplies the task, the instruction, the agent and the verifier, and
# this repository supplies only the environment they run in. Two independent
# implementations of one benchmark is the point: where they disagree, one of
# them is wrong, and the disagreement names which instances to open.
#
#   scripts/compare_swebench_pro.py   pairs the two runs up per instance.
#
# Where the two paths differ, and why a few points of drift are expected:
#
#   1. Grading container. `orchard-eval run` grades in a *fresh* pod the agent
#      never touched, by extracting `git diff` and replaying it. Harbor's Pro
#      tasks ship no `tests/Dockerfile`, so the verifier runs in the *same*
#      container the agent worked in. Anything the agent left behind — an
#      installed package, a stray build artifact, an edited test file — is
#      still there when the tests run. Harbor is the more permissive of the
#      two, and an instance that passes only there is worth reading.
#
#   2. Agent resources. The tasks declare 1 CPU and 4096 MB, which is thinner
#      than configs/swebench-pro.yaml's 4 CPUs and 16 GiB. The overrides below
#      restore the parity deliberately; leave them in or the comparison is
#      measuring the pod, not the model.
#
#   3. Agent budget. Harbor honours the task's own 3000s; the YAML path uses
#      harness.timeout 2400. Pass `--agent-timeout-multiplier 0.8` after a bare
#      `--` to match exactly. Note that the agent and the verifier share one
#      pod here, so 3000s + 3000s runs close to the orchestrator's
#      SANDBOX_TTL_HOURS (2h) — raise the TTL before raising the budget.
#
#   4. Harness configuration. Harbor's `mini-swe-agent` runs mini's builtin
#      `mini.yaml`; configs/mini-swe-agent.yaml runs `benchmarks/swebench.yaml`
#      with a 250-step limit. Same CLI, different prompts and different limits.
#      `--ak config_file=...` is the knob if you want them identical.
#
#      For `codex` this is the largest divergence of the six, and it is not a
#      few points of drift: Harbor's codex agent defaults `reasoning_effort`
#      to `high` (`-c model_reasoning_effort=high`, plus `--enable
#      unified_exec`), and configs/codex.yaml sets no effort at all. Measured
#      on one model over the same fleet and the same pods — 34.47% here
#      against 43.09% through Harbor, on 2,394 reasoning tokens per rollout
#      against 6,050, and 67 shell commands against 94. pi and mini-swe-agent
#      have no such asymmetry and land within 3 points of their Harbor selves.
#
#      Against a Qwen3.8-27B fleet the matching arm is the *other* direction:
#      that chat template takes xhigh/medium/low and raises on `high`, so
#      Harbor's default 4xxs every request and unset is the top of the scale.
#      `--ak reasoning_effort=null` drops it, which is what run_all_evals.sh
#      passes; see "Reasoning effort" in the README.
#
#   5. Prompt. Both paths hand over problem_statement + requirements +
#      interface. Harbor's instruction.md wraps them in the SWE-agent framing
#      ("I've uploaded a code repository in /app ..."); the YAML path
#      concatenates them the way upstream's scaffolds do.
#
# Unlike DeepSWE, every one of the 731 tasks sets `allow_internet = true` and
# declares no `network_mode`, so there is no air-gap to honour and no egress
# allowlist to get right. The pod has the network the in-pod agent needs.
#
# `pi` and `mini-swe-agent` run through harbor_orchard.agents rather than
# Harbor's own classes, because Harbor's fetch their CLI into the task image at
# trial time and that cannot work on the 88 Alpine images (nvm serves no musl
# Node build) nor, for mini, on a good number of the glibc ones. Set
# ORCHARD_HARBOR_STOCK_AGENTS=1 to compare against upstream behaviour.
#
# Usage:
#   ./scripts/run_swebench_pro_harbor.sh                          # codex, 32-way
#   ./scripts/run_swebench_pro_harbor.sh mini-swe-agent 32 --limit 5
#   ./scripts/run_swebench_pro_harbor.sh oracle 16 --limit 20     # the ceiling
#   ./scripts/run_swebench_pro_harbor.sh pi 32 -- --agent-timeout-multiplier 0.8
#   TASK_FILE=configs/swebench-pro-musl.txt ./scripts/run_swebench_pro_harbor.sh pi 24
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
#   ATTEMPTS           trials per task, for pass@k (default 1)
#   PRO_DATASET        override the dataset
#   OVERRIDE_CPUS / OVERRIDE_MEMORY_MB
#                      Pod size, defaulting to configs/swebench-pro.yaml's
#                      4 CPUs / 16 GiB so both paths measure the same thing.
#   TASK_FILE          Run only the task names in this file, one per line.
#                      configs/swebench-pro-musl.txt is the 88 Alpine images;
#                      scripts/harbor_failed_tasks.py writes one per cause.
#   EXCLUDE_TASK_FILE  The negation, for scoring what is known to be runnable.
#
# Run the ceiling check first; its score caps whatever this reports:
#   ORACLE_THRESHOLD=0.90 ./scripts/harbor_oracle_check.sh scale-ai/swe-bench-pro@2 16 --limit 25
set -euo pipefail

AGENT="${1:-codex}"
CONCURRENCY="${2:-32}"

# Pinned, not @latest. `@2` is a *revision number*, which is the only immutable
# way to name a Hub version: harbor parses a ref as a tag, a pure-integer
# revision, or a sha256: digest — never as a version string. The Hub displays
# this revision as "2.0.0", but `@2.0.0` parses as a tag, finds none, and fails
# with "Dataset not found". Revision 2 is what `latest` resolves to today and
# therefore what every number measured on this path so far used; naming it
# explicitly is what keeps that true when a revision 3 appears.
#
# Revisions 1 and 2 hold the same 731 task names with a different
# task_version_id on every one — a full republish, not a task-list change.
#
# Note this dataset is *not* SWE-bench Pro V2 (2026-09-22). V2 is supported now,
# on its own path: upstream ships it as a task directory in its own git
# repository — 642 tasks, images on ghcr.io — so it is fetched rather than
# named, and run with --path.
#
#     ./scripts/fetch_swebench_pro_v2.sh
#     ./scripts/run_swebench_pro_v2.sh mini-swe-agent 24
#
# Prefer it for anything you intend to report. V1's images carry the fixing
# commit in /app's git history and the task name carries its SHA, so an agent
# can read the answer: over the 2026-09-21 sweep 701 of these 731 trials did,
# and the 219 that copied gold into a non-test source file scored 90% against
# 71% for the trials that never looked. V2 rebuilds every image from a sanitised
# bundle. See "SWE-bench Pro V2, via Harbor" in the README; the two are not
# comparable with each other.
DATASET="${PRO_DATASET:-scale-ai/swe-bench-pro@2}"
ATTEMPTS="${ATTEMPTS:-1}"
JOBS_DIR="${JOBS_DIR:-./results/harbor}"

# Parity with configs/swebench-pro.yaml, not with the task's own declaration.
OVERRIDE_CPUS="${OVERRIDE_CPUS:-4}"
OVERRIDE_MEMORY_MB="${OVERRIDE_MEMORY_MB:-16384}"

# The agent gets 3000s and the verifier another 3000s in the same pod, so this
# has to outlast either one on its own. Capped at the orchestrator's
# SANDBOX_TTL_HOURS (2h), past which the pod is gone regardless.
export ORCHARD_HARBOR_EXEC_TIMEOUT="${ORCHARD_HARBOR_EXEC_TIMEOUT:-7200}"
# Pro's images are several GB and `npm ci` / `go mod download` run at build.
export ORCHARD_HARBOR_CREATE_TIMEOUT="${ORCHARD_HARBOR_CREATE_TIMEOUT:-1800}"

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
    # pod for one. The tasks are not air-gapped, so this is hygiene, not a
    # requirement.
    export ORCHARD_HARBOR_MODEL_EGRESS="${ORCHARD_HARBOR_MODEL_EGRESS:-0}"
else
    : "${MODEL_NAME:?MODEL_NAME must be set for agent '${AGENT}' (use --agent oracle for the ceiling check)}"
    # What the agent expects in --model differs by agent, and getting it wrong
    # fails on turn 1 in a way that reads as a model problem:
    #
    #   pi     validates --model against its own catalog and exits "Model not
    #          found" for anything absent from it. harbor_orchard stages that
    #          catalog (models.json) declaring MODEL_NAME under the provider
    #          `orchard` as the alias `orchard-model`, so the alias is the name
    #          pi accepts — the served id is not.
    #   codex  strips the provider prefix itself (model_name.split("/")[-1])
    #          and routes through the staged config.toml's `orchard` provider,
    #          so openai/ is a harmless carrier.
    #   others go through litellm, which splits on the first "/" to pick a
    #          provider — hence the openai/ prefix MODEL_NAME must not itself
    #          carry.
    #
    # MODEL_ID overrides all of it.
    if [[ "${AGENT}" == "pi" ]]; then
        DEFAULT_MODEL="orchard/orchard-model"
    else
        DEFAULT_MODEL="openai/${MODEL_NAME}"
    fi
    MODEL_ARGS=(--model "${MODEL_ID:-${DEFAULT_MODEL}}")
    if [[ -n "${MODEL_BASE_URL:-}" ]]; then
        export OPENAI_BASE_URL="${OPENAI_BASE_URL:-${MODEL_BASE_URL}}"
        export OPENAI_API_BASE="${OPENAI_API_BASE:-${MODEL_BASE_URL}}"
    fi
    if [[ -n "${API_KEY:-}" ]]; then
        export OPENAI_API_KEY="${OPENAI_API_KEY:-${API_KEY}}"
    fi
    # A local server started without --api-key still needs the variable set:
    # litellm raises before it ever sends the request otherwise. Harbor's
    # mini-swe-agent reads its own spelling and refuses to start without it.
    export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
    export MSWEA_API_KEY="${MSWEA_API_KEY:-${OPENAI_API_KEY}}"
fi

JOB_NAME="${JOB_NAME:-${AGENT}-swebench-pro-harbor-$(date +%Y%m%d-%H%M%S)}"

# Task selection, if this run is a subset rather than the whole 731.
TASK_ARGS=()
[[ -n "${TASK_FILE:-}" ]] && TASK_ARGS+=(--task-file "${TASK_FILE}")
[[ -n "${EXCLUDE_TASK_FILE:-}" ]] && TASK_ARGS+=(--exclude-task-file "${EXCLUDE_TASK_FILE}")

echo "Dataset:     ${DATASET}  (731 tasks)"
echo "Agent:       ${AGENT}"
echo "Model:       ${MODEL_ID:-${MODEL_NAME:-n/a}}"
echo "Endpoint:    ${OPENAI_BASE_URL:-default} (routing=${MODEL_ROUTING:-sticky}, replicas=${MODEL_BASE_URL_REPLICAS:-1})"
echo "Pod:         ${OVERRIDE_CPUS} CPU / ${OVERRIDE_MEMORY_MB} MB  (task declares 1 / 4096)"
echo "Exec budget: ${ORCHARD_HARBOR_EXEC_TIMEOUT}s"
echo "Concurrency: ${CONCURRENCY}  (attempts per task: ${ATTEMPTS})"
echo "Tasks:       ${TASK_FILE:-all 731}${EXCLUDE_TASK_FILE:+ minus ${EXCLUDE_TASK_FILE}}"
echo "Job:         ${JOBS_DIR}/${JOB_NAME}"
echo

# Split the caller's trailing arguments at the first bare `--`, so the pod
# overrides can be spliced into the `harbor run` side rather than appended after
# a second separator — argparse strips only the first, and the leftover would
# reach `harbor run` as a literal argument.
EVAL_ARGS=()
HARBOR_ARGS=(--override-cpus "${OVERRIDE_CPUS}" --override-memory-mb "${OVERRIDE_MEMORY_MB}")
separated=0
for arg in "${@:3}"; do
    if [[ ${separated} -eq 0 && "${arg}" == "--" ]]; then
        separated=1
    elif [[ ${separated} -eq 1 ]]; then
        # After the defaults, so a caller repeating one of them wins: click
        # keeps the last occurrence of a non-repeatable option.
        HARBOR_ARGS+=("${arg}")
    else
        EVAL_ARGS+=("${arg}")
    fi
done

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
    ${TASK_ARGS[@]+"${TASK_ARGS[@]}"} \
    ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"} \
    -- "${HARBOR_ARGS[@]}"
