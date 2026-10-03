#!/usr/bin/env bash
# Expose a fleet of local sglang ports through one frpc process.
#
# frpc multiplexes every proxy in its config over a single connection to the
# frps server, so N tunnels cost one process, not N. Local port BASE_PORT+i is
# published as REMOTE_BASE_PORT+i, which keeps both ranges consecutive — that is
# what lets the eval suite address the fleet as one base URL plus a replica
# count instead of eight literal URLs.
#
# With scripts/session_router.py in front of the fleet there is only one local
# port to publish, which is the preferred setup: one remote port instead of
# eight, so `frps.toml` needs one `allowPorts` entry and a refused tunnel
# cannot take out an eighth of a run.
#
# Usage:
#   # one tunnel, in front of the session router (preferred)
#   FRP_SERVER_ADDR=<frps-public-ip> FRP_TOKEN=... \
#       REPLICAS=1 BASE_PORT=8100 REMOTE_BASE_PORT=30021 ./scripts/frpc_fleet.sh
#
#   # one tunnel per engine, for model.routing=sticky
#   FRP_SERVER_ADDR=<frps-public-ip> FRP_TOKEN=... ./scripts/frpc_fleet.sh
#   REPLICAS=4 REMOTE_BASE_PORT=30021 ./scripts/frpc_fleet.sh
#
# The token is read from the environment and written only to CONFIG_PATH, which
# is created with 0600 permissions.
set -euo pipefail

REPLICAS="${REPLICAS:-8}"
BASE_PORT="${BASE_PORT:-8000}"
REMOTE_BASE_PORT="${REMOTE_BASE_PORT:-30021}"
# No apostrophes in a ${VAR:?...} message: bash quote-processes the word even
# inside double quotes, so one would open a string that never closes.
FRP_SERVER_ADDR="${FRP_SERVER_ADDR:?set it to the public IP of your frps VM}"
FRP_SERVER_PORT="${FRP_SERVER_PORT:-7000}"
FRP_TOKEN="${FRP_TOKEN:?set it to the auth.token from frps.toml}"
FRPC_BIN="${FRPC_BIN:-/tmp/frpc}"
CONFIG_PATH="${CONFIG_PATH:-/tmp/frpc.toml}"
PROXY_PREFIX="${PROXY_PREFIX:-sglang}"

# `exec` on a missing binary reports only "No such file or directory" against
# the script's own line number, which says nothing about what to install or
# where it was looked for.
if [ ! -x "${FRPC_BIN}" ]; then
    echo "frpc is not at ${FRPC_BIN} (override with FRPC_BIN=...)." >&2
    if command -v frpc >/dev/null 2>&1; then
        echo "There is one on PATH; use it with:" >&2
        echo "  FRPC_BIN=\$(command -v frpc) $0" >&2
    else
        echo "Fetch one — v0.52.0 or newer, since this writes TOML config, and" >&2
        echo "close to the version your frps runs:" >&2
        echo >&2
        echo "  FRP_VERSION=0.61.1" >&2
        echo "  curl -sSL https://github.com/fatedier/frp/releases/download/v\${FRP_VERSION}/frp_\${FRP_VERSION}_linux_amd64.tar.gz \\" >&2
        echo "    | tar -xz -C /tmp --strip-components=1 frp_\${FRP_VERSION}_linux_amd64/frpc" >&2
        echo "  chmod +x /tmp/frpc" >&2
    fi
    exit 1
fi

umask 077
{
    printf 'serverAddr = "%s"\n' "${FRP_SERVER_ADDR}"
    printf 'serverPort = %s\n\n' "${FRP_SERVER_PORT}"
    printf 'auth.method = "token"\n'
    printf 'auth.token = "%s"\n' "${FRP_TOKEN}"

    for ((i = 0; i < REPLICAS; i++)); do
        printf '\n[[proxies]]\n'
        printf 'name = "%s-%d"\n' "${PROXY_PREFIX}" "$((REMOTE_BASE_PORT + i))"
        printf 'type = "tcp"\n'
        printf 'localIP = "127.0.0.1"\n'
        printf 'localPort = %d\n' "$((BASE_PORT + i))"
        printf 'remotePort = %d\n' "$((REMOTE_BASE_PORT + i))"
    done
} > "${CONFIG_PATH}"

echo "wrote ${CONFIG_PATH}: ${REPLICAS} tunnel(s)"
for ((i = 0; i < REPLICAS; i++)); do
    echo "  127.0.0.1:$((BASE_PORT + i))  ->  ${FRP_SERVER_ADDR}:$((REMOTE_BASE_PORT + i))"
done
echo
echo "MODEL_BASE_URL=http://${FRP_SERVER_ADDR}:${REMOTE_BASE_PORT}/v1  (replicas=${REPLICAS})"
echo

exec "${FRPC_BIN}" -c "${CONFIG_PATH}"
