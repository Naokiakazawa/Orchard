#!/usr/bin/env bash
# Start N independent sglang servers on consecutive ports, one per GPU group.
#
# This replaces a single `--dp N` server. Data parallelism inside one server
# load-balances every *request*, so consecutive turns of the same agent loop
# land on different workers and each turn re-prefills the whole conversation.
# Separate servers give the eval suite something it can pin a rollout to
# (`model.base_url_replicas` / `--base-url-replicas`), which is what makes the
# prefix KV cache worth anything.
#
# Memory footprint is unchanged: `--dp 8 --tp 1` already holds eight full copies
# of the weights, one per GPU. This just puts a port in front of each of them.
#
# Usage:
#   ./scripts/serve_sglang_fleet.sh
#   REPLICAS=4 BASE_PORT=8000 MODEL_PATH=/models/Qwen3.5-35B-A3B \
#       ./scripts/serve_sglang_fleet.sh
#
# Then put ./scripts/session_router.py in front of them and expose that one
# port with ./scripts/frpc_fleet.sh:
#
#   python scripts/session_router.py --replicas 8 --base-port 8000 --port 8100 \
#       --api-key token-abc123
#   REPLICAS=1 BASE_PORT=8100 REMOTE_BASE_PORT=30021 ./scripts/frpc_fleet.sh
#
# and run the suite with --routing session. Without the router, publish every
# port and pin by hashing instead: --base-url-replicas <REPLICAS>.
#
# The router has to hold the same key as the engines below, or its load probes
# are turned away and it places on session count alone. Both sides default to
# $API_KEY, so exporting that once before either covers it; the printed command
# at the end fills in whichever key this run actually used.
#
# The router breaks placement ties on what each engine reports about its own
# queue, reading /get_server_info first and /metrics second. Neither is needed
# — a fleet answering neither places on session count alone — but passing
# --enable-metrics here gives it the second source if this sglang is old enough
# to lack the first. /router/stats reports `load_in_use` either way.
#
# ─────────────────────────── sampling parameters ────────────────────────────
#
# Every argument after the script's own is forwarded to sglang.launch_server,
# and argparse keeps the last occurrence of a flag — so a per-model recipe is a
# command line, not a copy of this file. Qwen3.8-27B in thinking mode:
#
#   MODEL_PATH=Qwen/Qwen3.8-27B \
#       ./scripts/serve_sglang_fleet.sh --preferred-sampling-params \
#       '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"repetition_penalty":1.0}'
#
# `--preferred-sampling-params` is merged *under* the request
# (`{**preferred, **request}` in tokenizer_manager) and the OpenAI path writes
# a value for every field it knows whether the caller sent one or not. So on
# /v1/chat/completions the preferred values win for exactly temperature, top_p,
# top_k, min_p and repetition_penalty, and are silently discarded for
# presence_penalty and frequency_penalty — sgl-project/sglang#21816. The two
# that do not work default to 0.0, which is what Qwen3.8-27B asks for anyway.
#
# That only matters if you want to *change* them. The model card suggests
# raising presence_penalty (0-2) "to reduce endless repetition", and repetition
# is not hypothetical here: 95 of 500 pi rollouts on SWE-bench Verified ended
# inside a degenerate enumeration loop that ran out the output-token cap.
# Setting it here would do nothing — it has to come from the client, or from
# generation_config.json in MODEL_PATH, which `--sampling-defaults model`
# (SGLang's default) reads before any request is served.
#
# A client that sends its own temperature/top_p overrides all of this. These
# are defaults for callers that send none.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-35B-A3B}"
REPLICAS="${REPLICAS:-8}"
BASE_PORT="${BASE_PORT:-8000}"
# GPUs per server. TP>1 shards one model across TP GPUs, so REPLICAS*TP GPUs
# are needed in total.
TP="${TP:-1}"
HOST="${HOST:-127.0.0.1}"
SERVE_API_KEY="${API_KEY:-token-abc123}"
LOG_DIR="${LOG_DIR:-/tmp/sglang-fleet}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
PYTHON="${PYTHON:-python}"

# Both of these are cheap, and both otherwise surface as eight processes that
# exit within seconds followed by a long silent wait for servers that are
# already gone.
if [ ! -d "${MODEL_PATH}" ] && [[ "${MODEL_PATH}" == /* ]]; then
    echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
    echo "(a HuggingFace repo id is fine too, but an absolute path has to be real)" >&2
    exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT="$(nvidia-smi --list-gpus 2>/dev/null | wc -l)"
    if [ "${GPU_COUNT}" -gt 0 ] && [ $((REPLICAS * TP)) -gt "${GPU_COUNT}" ]; then
        echo "REPLICAS=${REPLICAS} x TP=${TP} needs $((REPLICAS * TP)) GPUs, but this host has ${GPU_COUNT}." >&2
        echo "Lower REPLICAS, or raise TP and lower REPLICAS to match." >&2
        exit 1
    fi
fi

mkdir -p "${LOG_DIR}"
PID_FILE="${LOG_DIR}/pids"
: > "${PID_FILE}"

# Import the entry point once before starting REPLICAS copies of it. A broken
# environment — most often a dependency pinned out from under sglang by an
# unrelated `pip install` into the same interpreter — otherwise shows up as
# eight identical tracebacks in eight log files, reported one at a time.
if ! PREFLIGHT="$("${PYTHON}" -c 'import sglang.launch_server' 2>&1)"; then
    echo "${PYTHON} cannot import sglang.launch_server, so no server would start:" >&2
    echo >&2
    echo "${PREFLIGHT}" | tail -n 15 >&2
    echo >&2
    echo "Serving and evaluating share an interpreter here only by accident; a" >&2
    echo "virtualenv for orchard-eval keeps its resolver away from this one." >&2
    exit 1
fi

echo "model:    ${MODEL_PATH}"
echo "replicas: ${REPLICAS} (tp=${TP}, $((REPLICAS * TP)) GPUs)"
echo "ports:    ${BASE_PORT}..$((BASE_PORT + REPLICAS - 1))"
echo "logs:     ${LOG_DIR}"
echo

for ((i = 0; i < REPLICAS; i++)); do
    port=$((BASE_PORT + i))
    gpus=""
    for ((g = 0; g < TP; g++)); do
        gpus+="${gpus:+,}$((i * TP + g))"
    done

    CUDA_VISIBLE_DEVICES="${gpus}" nohup "${PYTHON}" -m sglang.launch_server \
        --model-path "${MODEL_PATH}" \
        --tp "${TP}" \
        --port "${port}" \
        --host "${HOST}" \
        --api-key "${SERVE_API_KEY}" \
        --tool-call-parser qwen3_coder \
        --reasoning-parser qwen3 \
        --preferred-sampling-params '{"temperature":0.95,"top_p":0.95}' \
        "$@" \
        > "${LOG_DIR}/sglang-${port}.log" 2>&1 &

    echo "$!" >> "${PID_FILE}"
    echo "started replica ${i} on GPU(s) ${gpus}, port ${port} (pid $!)"
done

echo
echo "waiting for all ${REPLICAS} servers to answer /v1/models ..."
echo "(loading ${REPLICAS} copies of the weights is I/O bound; follow one with"
echo " tail -f ${LOG_DIR}/sglang-${BASE_PORT}.log)"

# Read rather than `mapfile`, which is bash 4+ only and absent on macOS.
PIDS=()
while IFS= read -r _pid; do PIDS+=("${_pid}"); done < "${PID_FILE}"
started=${SECONDS}
deadline=$((SECONDS + READY_TIMEOUT))
declare -a READY=()
ready_count=0
last_report=${SECONDS}

while ((ready_count < REPLICAS)); do
    ready_count=0
    for ((i = 0; i < REPLICAS; i++)); do
        port=$((BASE_PORT + i))
        if [ -n "${READY[i]:-}" ]; then
            ready_count=$((ready_count + 1))
            continue
        fi
        if curl -sf -o /dev/null \
            -H "Authorization: Bearer ${SERVE_API_KEY}" \
            "http://${HOST}:${port}/v1/models"; then
            READY[i]=1
            ready_count=$((ready_count + 1))
            echo "  ready: http://${HOST}:${port}/v1  ($((SECONDS - started))s)"
            continue
        fi
        # A server that has exited is never going to answer, and polling a dead
        # pid for the rest of READY_TIMEOUT is indistinguishable from a slow
        # load. Say so now, with the reason, instead of 30 minutes from now.
        if ! kill -0 "${PIDS[i]}" 2>/dev/null; then
            echo >&2
            echo "replica ${i} (port ${port}, pid ${PIDS[i]}) exited before it became ready." >&2
            echo "last 30 lines of ${LOG_DIR}/sglang-${port}.log:" >&2
            echo >&2
            tail -n 30 "${LOG_DIR}/sglang-${port}.log" >&2
            exit 1
        fi
    done
    ((ready_count >= REPLICAS)) && break

    if ((SECONDS > deadline)); then
        echo >&2
        echo "only ${ready_count}/${REPLICAS} servers came up within ${READY_TIMEOUT}s." >&2
        for ((i = 0; i < REPLICAS; i++)); do
            [ -n "${READY[i]:-}" ] || echo "  still down: port $((BASE_PORT + i))" >&2
        done
        echo "see ${LOG_DIR}/sglang-*.log, or raise READY_TIMEOUT." >&2
        exit 1
    fi

    # Silence for half an hour reads as a hang, so say something periodically.
    if ((SECONDS - last_report >= 60)); then
        last_report=${SECONDS}
        echo "  ... ${ready_count}/${REPLICAS} ready after $((SECONDS - started))s"
    fi
    sleep 5
done

echo
echo "All ${REPLICAS} servers up. Stop them with:"
echo "  kill \$(cat ${PID_FILE})"
echo
echo "Next: put one router in front of them, so one frp tunnel serves them all."
echo "  python scripts/session_router.py --replicas ${REPLICAS} --base-port ${BASE_PORT} --port 8100 \\"
echo "      --api-key ${SERVE_API_KEY}"
echo
echo "The key is not decoration: without it the router's load probes are"
echo "rejected, every engine still looks healthy, and placement silently falls"
echo "back to session count alone (/router/stats: load_in_use false)."
