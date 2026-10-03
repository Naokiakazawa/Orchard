"""Find which part of a codex Responses payload a self-hosted server rejects.

The endpoint is usually reachable only from inside a sandbox pod, so the probe
is uploaded and executed there rather than run from the launching machine.

Codex fails the whole rollout on the first request the server will not parse,
and the error names a Rust enum rather than the offending item, so the useful
question is *which piece* of the payload is unsupported. This sends a series of
requests that differ by one element each — one tool type, then one input item —
and reports the status of each, so the answer is read off a table instead of
inferred from a single opaque 400.

Both dimensions matter because they fail at different layers: `tools` is
rejected before the model is ever called, while an input item is only rejected
on the second turn, once codex replays its own history.

Usage:
    export MODEL_BASE_URL=... API_KEY=... MODEL_NAME=...
    export SANDBOX_BASE_URL=... SANDBOX_API_KEY=...
    python scripts/probe_responses_input.py [--image IMAGE]
"""

from __future__ import annotations

import argparse
import os
import sys

from orchard_env import SandboxClient

# Runs inside the pod. Stdlib only: `curl` is absent from python:*-slim images,
# and a missing binary would look identical to a blocked network.
POD_PROBE = r'''
import json, os, time, urllib.error, urllib.request

BASE = os.environ["MODEL_BASE_URL"].rstrip("/")
KEY = os.environ.get("MODEL_API_KEY", "")
MODEL = os.environ["MODEL_NAME"]
HDRS = {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}

USER = {"type": "message", "role": "user",
        "content": [{"type": "input_text", "text": "Say hi."}]}
FUNCTION_TOOL = {
    "type": "function",
    "name": "update_plan",
    "description": "Record a plan.",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}

# Every item codex replays on turn 2, each isolated so one rejection does not
# mask the others. `reasoning` appears several times because its shape varies
# with what the server advertises — summary text, raw content and encrypted
# content are independently optional, and a server can accept one but not
# another.
CASES = [
    ("baseline text input", {"input": "Say hi."}),
    ("message item", {"input": [USER]}),
    ("reasoning (content)", {"input": [USER, {
        "type": "reasoning", "id": "rs_1", "summary": [],
        "content": [{"type": "reasoning_text", "text": "thinking"}]}]}),
    ("reasoning (encrypted)", {"input": [USER, {
        "type": "reasoning", "id": "rs_1", "summary": [],
        "encrypted_content": "gAAAAAB0ZXN0"}]}),
    ("reasoning (summary_text)", {"input": [USER, {
        "type": "reasoning", "id": "rs_1",
        "summary": [{"type": "summary_text", "text": "Locating the code."}]}]}),
    ("reasoning (full codex)", {"input": [USER, {
        "type": "reasoning", "id": "rs_1", "status": "completed",
        "summary": [{"type": "summary_text", "text": "Locating the code."}],
        "content": [{"type": "reasoning_text", "text": "thinking"}],
        "encrypted_content": "gAAAAAB0ZXN0"}]}),
    ("developer message", {"input": [{
        "type": "message", "role": "developer",
        "content": [{"type": "input_text", "text": "Be terse."}]}, USER]}),
    ("assistant message", {"input": [USER, {
        "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": "hi", "annotations": []}]}]}),
    ("function_call", {"input": [USER, {
        "type": "function_call", "id": "fc_1", "call_id": "call_1",
        "name": "update_plan", "arguments": "{}"}]}),
    ("function_call (status)", {"input": [USER, {
        "type": "function_call", "id": "fc_1", "call_id": "call_1",
        "status": "completed", "name": "update_plan", "arguments": "{}"}]}),
    ("function_call_output", {"input": [USER, {
        "type": "function_call", "id": "fc_1", "call_id": "call_1",
        "name": "update_plan", "arguments": "{}"}, {
        "type": "function_call_output", "call_id": "call_1", "output": "ok"}]}),
    ("fc_output (parts)", {"input": [USER, {
        "type": "function_call", "id": "fc_1", "call_id": "call_1",
        "name": "shell", "arguments": "{}"}, {
        "type": "function_call_output", "call_id": "call_1",
        "output": [{"type": "output_text", "text": "ok"}]}]}),
    ("fc_output (object)", {"input": [USER, {
        "type": "function_call", "id": "fc_1", "call_id": "call_1",
        "name": "shell", "arguments": "{}"}, {
        "type": "function_call_output", "call_id": "call_1",
        "output": {"content": "ok", "success": True}}]}),
    ("fc_output (status)", {"input": [USER, {
        "type": "function_call", "id": "fc_1", "call_id": "call_1",
        "name": "shell", "arguments": "{}"}, {
        "type": "function_call_output", "id": "fco_1", "call_id": "call_1",
        "status": "completed", "output": "ok"}]}),
    ("custom_tool_call", {"input": [USER, {
        "type": "custom_tool_call", "id": "ctc_1", "call_id": "call_1",
        "name": "apply_patch", "input": "*** Begin Patch"}]}),
    ("local_shell_call", {"input": [USER, {
        "type": "local_shell_call", "id": "lsc_1", "call_id": "call_1",
        "status": "completed",
        "action": {"type": "exec", "command": ["ls"]}}]}),
    ("tools: function", {"input": "Say hi.", "tools": [FUNCTION_TOOL]}),
    ("tools: namespace", {"input": "Say hi.", "tools": [{
        "type": "namespace", "name": "multi_agent_v1",
        "description": "Tools for spawning sub-agents.",
        "tools": [FUNCTION_TOOL]}]}),
    ("tools: web_search", {"input": "Say hi.", "tools": [{"type": "web_search"}]}),
]


def post(path, body, timeout=120):
    """Return (status, body). A transport failure is reported as status 0."""
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(), headers=HDRS
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        return 0, "%s: %s" % (type(e).__name__, e)


def get(path, timeout=30):
    req = urllib.request.Request(BASE + path, headers=HDRS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        return 0, "%s: %s" % (type(e).__name__, e)


def summarize(payload):
    """Pull the server's message out of an error envelope, else say nothing."""
    try:
        err = json.loads(payload).get("error")
    except Exception:
        return payload[:200].replace("\n", " ")
    if isinstance(err, dict):
        return str(err.get("message", err))[:300].replace("\n", " ")
    if err:
        return str(err)[:300].replace("\n", " ")
    return ""


print("target %s  model %r\n" % (BASE, MODEL))

# `sglang.launch_server` serves /get_server_info; the Rust gateway does not.
# The distinction decides whether these rejections are worth working around at
# all, since the two implementations validate very differently.
status, body = get("/models")
print("[GET /v1/models] HTTP %s" % status)
if status == 200:
    try:
        print("  served: %s" % [m["id"] for m in json.loads(body).get("data", [])])
    except Exception:
        print("  body: %s" % body[:200])

root = BASE.rsplit("/v1", 1)[0]
req = urllib.request.Request(root + "/get_server_info", headers=HDRS)
try:
    with urllib.request.urlopen(req, timeout=15) as r:
        print("[GET /get_server_info] HTTP %s -> python sglang server" % r.status)
except urllib.error.HTTPError as e:
    print("[GET /get_server_info] HTTP %s -> likely the Rust gateway" % e.code)
except Exception as e:
    print("[GET /get_server_info] %s: %s" % (type(e).__name__, e))

print("\n%-24s %-6s %s" % ("case", "http", "error"))
print("-" * 100)
failures = []
for name, extra in CASES:
    # Explicit: the responses->chat conversion forwards a missing `stream`
    # as None into a field typed bool, so every case 400s on the default.
    body = {"model": MODEL, "max_output_tokens": 64, "stream": False, **extra}
    t0 = time.monotonic()
    status, payload = post("/responses", body)
    note = "" if status == 200 else summarize(payload)
    print("%-24s %-6s %s" % (name, status, note))
    if status != 200:
        failures.append(name)

print("\n%d/%d rejected" % (len(failures), len(CASES)))
if failures:
    print("rejected: %s" % ", ".join(failures))
raise SystemExit(1 if failures else 0)
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="python:3.11-slim")
    ap.add_argument("--model", default=os.environ.get("MODEL_NAME", ""))
    args = ap.parse_args()

    if "MODEL_BASE_URL" not in os.environ:
        print("MODEL_BASE_URL is not set", file=sys.stderr)
        return 2

    with SandboxClient() as client:
        print(f"orchestrator health: {client.health()}")
        # Sandboxes have no egress by default, so this must be False.
        with client.create_sandbox(image=args.image, block_network=False) as sandbox:
            print(f"sandbox ready: {sandbox.sandbox_id}\n")
            sandbox.upload_content(POD_PROBE.encode(), "/tmp/probe.py")
            result = sandbox.exec(
                "python /tmp/probe.py",
                timeout=600,
                env={
                    "MODEL_BASE_URL": os.environ["MODEL_BASE_URL"].rstrip("/"),
                    "MODEL_API_KEY": os.environ.get("API_KEY", ""),
                    "MODEL_NAME": args.model.removeprefix("openai/"),
                },
            )
            print(result.stdout, end="")
            if result.stderr:
                print(f"--- stderr ---\n{result.stderr}")
            return 0 if result.exit_code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
