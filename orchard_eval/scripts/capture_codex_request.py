"""Capture the exact request codex puts on the wire.

Two uses. The first is the one it was written for: reading the body a
self-hosted server rejected. The second is confirming that a staged setting
*arrives* — the config and the request body are not the same claim, and
codex decides some fields from model metadata it could not resolve:

    python scripts/capture_codex_request.py --reasoning-effort high

The parameters of the first captured request are printed whatever the server
answered, so an upstream that 404s ``/v1/responses`` still settles what codex
was about to send.

The config staged here is the harness's own, so what is probed is what a run
would send.

Bisecting a payload by hand only works while you can guess its shape. When the
server answers with `untagged enum ResponseInput` it names neither the item nor
the field, so past a couple of rounds the cheaper move is to read the bytes
codex actually put on the wire.

This runs codex inside a sandbox behind a recording proxy: codex's configured
`base_url` points at localhost, the proxy logs each request body and forwards it
upstream, and whichever request comes back non-200 is dumped in full. The pod is
the only place this can run — the model endpoint is typically reachable from
nowhere else — and it is also the only place codex is installed.

Usage:
    export MODEL_BASE_URL=... API_KEY=... MODEL_NAME=...
    export SANDBOX_BASE_URL=... SANDBOX_API_KEY=...
    python scripts/capture_codex_request.py [--image IMAGE] [--prompt TEXT]
        [--reasoning-effort high] [--reasoning-summaries] [--wire-api chat]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from orchard_env import SandboxClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchard_evalkit.config import ModelConfig  # noqa: E402
from orchard_evalkit.harnesses.installed_cli import (  # noqa: E402
    CODEX_HOME,
    REASONING_EFFORTS,
    CodexHarness,
)

PROXY_PORT = 8999
CAPTURE_DIR = "/tmp/codex_capture"

# A task that cannot be answered without one tool call, because the rejection
# only happens on the turn *after* codex has something to replay.
DEFAULT_PROMPT = (
    "Run `ls /` with your shell tool, then tell me how many entries there were. "
    "Do not do anything else."
)

# Recording proxy. Stdlib only, and deliberately buffering: holding the whole
# upstream response lets it be forwarded with an accurate Content-Length, which
# keeps codex's keep-alive connection valid. Streaming is lost, which costs
# nothing here since only the request bodies matter.
PROXY = r'''
import http.server, json, os, socketserver, threading, urllib.error, urllib.request

UPSTREAM = os.environ["UPSTREAM"].rstrip("/")
CAPTURE = os.environ["CAPTURE_DIR"]
os.makedirs(CAPTURE, exist_ok=True)


def upstream_url(path):
    """Where to forward a request codex addressed to the proxy.

    MODEL_BASE_URL is the /v1 root everywhere in this repo, and the staged
    config points codex at the proxy's own /v1, so an incoming path repeats
    the prefix. Concatenating asks the server for /v1/v1/responses, which
    answers `{"detail": "Not Found"}` — a 404 that reads like the route is
    missing from the server rather than invented by the proxy, and that codex
    then retries five times before failing the turn.
    """
    if UPSTREAM.endswith("/v1") and path.startswith("/v1/"):
        return UPSTREAM + path[len("/v1"):]
    return UPSTREAM + path

_lock = threading.Lock()
_n = [0]

# Hop-by-hop headers plus the two the proxy must recompute. Identity encoding is
# forced so the captured body is readable rather than a gzip blob.
SKIP = {"host", "content-length", "accept-encoding", "connection"}


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):
        self._proxy("GET")

    def do_POST(self):
        self._proxy("POST")

    def _proxy(self, method):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None

        with _lock:
            _n[0] += 1
            index = _n[0]

        req = urllib.request.Request(
            upstream_url(self.path), data=body, method=method
        )
        for key, value in self.headers.items():
            if key.lower() not in SKIP:
                req.add_header(key, value)
        req.add_header("Accept-Encoding", "identity")

        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                status, payload = r.status, r.read()
                headers = [(k, v) for k, v in r.getheaders()
                           if k.lower() not in ("content-length", "transfer-encoding",
                                                "connection", "content-encoding")]
        except urllib.error.HTTPError as e:
            status, payload = e.code, e.read()
            headers = [("Content-Type", "application/json")]
        except Exception as e:
            status = 502
            payload = json.dumps({"proxy_error": "%s: %s" % (type(e).__name__, e)}).encode()
            headers = [("Content-Type", "application/json")]

        if body is not None:
            name = "%s/%03d-%d.json" % (CAPTURE, index, status)
            with open(name, "wb") as fh:
                fh.write(body)
            if status != 200:
                with open("%s/%03d-%d.response" % (CAPTURE, index, status), "wb") as fh:
                    fh.write(payload)

        self.send_response(status)
        for key, value in headers:
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


Server(("127.0.0.1", int(os.environ["PORT"])), Handler).serve_forever()
'''

# Reads the capture directory after the run and reports the rejected request.
ANALYZE = r'''
import glob, json, os, sys

CAPTURE = os.environ["CAPTURE_DIR"]
files = sorted(glob.glob(CAPTURE + "/*.json"))
print("captured %d request(s)" % len(files))
for path in files:
    print("  %s" % os.path.basename(path))

if files:
    first = json.load(open(files[0]))
    print("\n=== parameters of the first request (%s) ==="
          % os.path.basename(files[0]))
    for key in sorted(k for k in first if k not in ("input", "tools", "instructions")):
        print("  %-22s %s" % (key, json.dumps(first[key])[:200]))
    # Printed even when absent, because whether `reasoning` survived codex's
    # model-family gate is the whole question when nothing was rejected.
    print("  %-22s %s" % ("reasoning", json.dumps(first.get("reasoning"))))

bad = [p for p in files if not os.path.basename(p).endswith("-200.json")]
if not bad:
    print("\nno request was rejected -- codex completed without a 4xx")
    raise SystemExit(0)

path = bad[0]
print("\n=== rejected request: %s ===" % os.path.basename(path))
resp = path.rsplit(".json", 1)[0] + ".response"
if os.path.exists(resp):
    print(open(resp).read()[:600])

body = json.load(open(path))
items = body.get("input")
if not isinstance(items, list):
    print("\ninput is not a list: %r" % type(items).__name__)
    raise SystemExit(1)

print("\ninput items (%d):" % len(items))
for i, item in enumerate(items):
    keys = ",".join(sorted(item.keys())) if isinstance(item, dict) else "?"
    kind = item.get("type", "<no type>") if isinstance(item, dict) else "?"
    print("  [%2d] %-24s %s" % (i, kind, keys))

print("\ntools: %s" % [t.get("type") for t in body.get("tools") or []])

# The server reports a byte offset into the body; map it back to an item so the
# suspect is identified rather than guessed.
offset = int(os.environ.get("ERROR_COLUMN") or 0)
if offset:
    prefix = json.dumps(body, separators=(",", ":"))[:offset]
    print("\nerror offset %d falls after %d input items"
          % (offset, prefix.count('"type":')))

print("\n=== last 3 input items, verbatim ===")
for item in items[-3:]:
    print(json.dumps(item, indent=2)[:2000])
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="python:3.11-slim")
    ap.add_argument("--model", default=os.environ.get("MODEL_NAME", ""))
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument(
        "--reasoning-effort",
        default="",
        choices=("", *REASONING_EFFORTS),
        help="harness.params.reasoning_effort to stage (default: unset, as "
        "the harness ships). Captured 2026-09-14: `high` reaches the body as "
        '`\"reasoning\": {\"effort\": \"high\"}` even with summaries off.',
    )
    ap.add_argument(
        "--reasoning-summaries",
        action="store_true",
        help="Stage model_supports_reasoning_summaries = true. Not needed for "
        "the effort to be sent, and it lets codex replay summaries that break "
        "turn 2 on some servers — for testing that claim, not for runs",
    )
    ap.add_argument("--wire-api", default="responses", choices=("responses", "chat"))
    args = ap.parse_args()

    if "MODEL_BASE_URL" not in os.environ:
        print("MODEL_BASE_URL is not set", file=sys.stderr)
        return 2
    upstream = os.environ["MODEL_BASE_URL"].rstrip("/")
    model = args.model.removeprefix("openai/")

    # The harness's own config, not a copy of it: a probe that stages a second
    # hand-written config answers a question about the probe.
    config = CodexHarness(
        model=ModelConfig(name=model, base_url=f"http://127.0.0.1:{PROXY_PORT}/v1"),
        params={
            "wire_api": args.wire_api,
            "reasoning_effort": args.reasoning_effort,
            "reasoning_summaries": args.reasoning_summaries,
        },
        timeout=600,
    )._render_config()
    print("=== staged config.toml ===")
    print(config)

    with SandboxClient() as client:
        print(f"orchestrator health: {client.health()}")
        # Sandboxes have no egress by default, so this must be False.
        with client.create_sandbox(image=args.image, block_network=False) as sandbox:
            print(f"sandbox ready: {sandbox.sandbox_id}\n")

            sandbox.upload_content(PROXY.encode(), "/tmp/proxy.py")
            sandbox.upload_content(ANALYZE.encode(), "/tmp/analyze.py")
            sandbox.upload_content(args.prompt.encode(), "/tmp/prompt.txt")
            sandbox.exec(f"mkdir -p {CODEX_HOME}", timeout=60)
            sandbox.upload_content(config.encode(), f"{CODEX_HOME}/config.toml")

            env = {
                "UPSTREAM": upstream,
                "CAPTURE_DIR": CAPTURE_DIR,
                "PORT": str(PROXY_PORT),
                "CODEX_HOME": CODEX_HOME,
                "OPENAI_API_KEY": os.environ.get("API_KEY", "") or "EMPTY",
                "RUST_LOG": "error",
            }

            # One shell invocation: the proxy has to outlive its own launch and
            # still be a child of the command codex runs under.
            script = (
                "python /tmp/proxy.py & PROXY=$!; sleep 2; "
                "codex exec --json --dangerously-bypass-approvals-and-sandbox "
                f"--skip-git-repo-check -m {model!r} - < /tmp/prompt.txt; "
                "kill $PROXY 2>/dev/null; true"
            )
            print("running codex behind the recording proxy ...\n")
            result = sandbox.exec(script, timeout=900, env=env)
            print(result.stdout[-4000:], end="")
            if result.stderr:
                print(f"--- stderr ---\n{result.stderr[-2000:]}")

            print("\n" + "=" * 70)
            analysis = sandbox.exec(
                "python /tmp/analyze.py",
                timeout=120,
                env={"CAPTURE_DIR": CAPTURE_DIR},
            )
            print(analysis.stdout, end="")
            if analysis.stderr:
                print(f"--- stderr ---\n{analysis.stderr}")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
