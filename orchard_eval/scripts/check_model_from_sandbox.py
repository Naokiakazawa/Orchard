"""Check whether a remote Orchard sandbox can reach the sglang endpoint.

Runs three probes, so a failure can be attributed to a specific hop:

  1. local machine -> sglang  -- is the model endpoint up at all?
  2. orchestrator health      -- are SANDBOX_BASE_URL / SANDBOX_API_KEY good?
  3. sandbox pod  -> sglang   -- the thing we actually care about.

The pod probe exercises ``/v1/chat/completions`` *and* ``/v1/responses``,
because the harnesses are split across the two: codex uses Responses only.

It then probes ``/v1/messages`` — the Anthropic Messages API, which is the only
thing claude-code speaks. sglang serves it natively from v0.5.18, but through a
second adapter in front of the same engine, so a working ``/v1/chat/completions``
says nothing about whether that one works. The probe sends a dozen-odd request
shapes rather than one, because Claude Code's *first* request combines them all
— streaming, prompt caching, tools, thinking, reasoning effort — and a 500 on it
cannot otherwise be attributed to the field responsible. Skip with
``--no-messages``.

Usage:
    export MODEL_BASE_URL=... API_KEY=... SANDBOX_BASE_URL=... SANDBOX_API_KEY=...
    python scripts/check_model_from_sandbox.py [--image IMAGE] [--model NAME]
    python scripts/check_model_from_sandbox.py --replicas 8

``--replicas N`` reads MODEL_BASE_URL as the first of N endpoints on
consecutive ports and probes every one of them, which is how a fleet of sglang
servers behind N frp tunnels gets verified before a run is launched against it.
All N are probed from a single pod, so the extra cost is N model calls.

``--session`` probes through the session-router path instead —
``.../session/<id>/v1`` rather than ``.../v1`` — which is what a run with
``--routing session`` actually sends. Worth one call: it is the cheapest way to
find out whether a CLI's URL joining survives a path prefix, and much cheaper
than discovering it 200 rollouts in.
"""

from __future__ import annotations

import argparse
import os
import sys

import requests

from orchard_env import SandboxClient
from orchard_evalkit.config import expand_ports, session_url

PROMPT = "What is the capital of France?"
EXPECT = "Paris"

# Runs inside the pod. Stdlib only: `curl` is absent from python:*-slim images,
# and a missing binary would look identical to a blocked network.
POD_PROBE = r'''
import json, os, socket, sys, time, urllib.error, urllib.request

BASE = os.environ["MODEL_BASE_URL"].rstrip("/")
KEY = os.environ.get("MODEL_API_KEY", "")
MODEL = os.environ["MODEL_NAME"]
PROMPT = os.environ["PROBE_PROMPT"]
EXPECT = os.environ["PROBE_EXPECT"]
HDRS = {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}


def call(name, url, body=None, timeout=120, hdrs=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=hdrs or HDRS)
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = r.read().decode("utf-8", "replace")
            print("[%s] HTTP %s in %.2fs" % (name, r.status, time.monotonic() - t0))
            return payload
    except urllib.error.HTTPError as e:
        print("[%s] HTTP %s in %.2fs" % (name, e.code, time.monotonic() - t0))
        print(e.read().decode("utf-8", "replace")[:800])
    except Exception as e:
        print("[%s] FAILED in %.2fs: %s: %s"
              % (name, time.monotonic() - t0, type(e).__name__, e))
    return None


host = BASE.split("://", 1)[-1].split("/")[0]
hostname, _, port = host.partition(":")
port = int(port or 80)
print("target %s:%s" % (hostname, port))

t0 = time.monotonic()
try:
    socket.create_connection((hostname, port), timeout=15).close()
    print("[tcp] connected in %.2fs" % (time.monotonic() - t0))
except Exception as e:
    print("[tcp] FAILED in %.2fs: %s: %s" % (time.monotonic() - t0, type(e).__name__, e))
    print("      -> pod cannot open a socket to the endpoint "
          "(network policy, NSG, or endpoint down)")
    sys.exit(2)

models = call("models", BASE + "/models", timeout=30)
if models:
    try:
        print("      served:", [m["id"] for m in json.loads(models).get("data", [])])
    except Exception:
        print("      body:", models[:300])

body = {
    "model": MODEL,
    # Reasoning models spend hundreds of tokens thinking before the first
    # answer token, so a small cap truncates the reply into nothing.
    "max_tokens": 512,
    "messages": [{"role": "user", "content": PROMPT}],
    "temperature": 0,
}
print("      prompt: %r" % PROMPT)
chat = call("chat", BASE + "/chat/completions", body)
if not chat:
    sys.exit(3)
choice = json.loads(chat)["choices"][0]
msg = choice["message"]
reasoning = (msg.get("reasoning_content") or "").strip()
content = (msg.get("content") or "").strip()
print("      finish_reason: %s" % choice.get("finish_reason"))
if reasoning:
    print("      reasoning: %s" % reasoning[:400])
print("      answer: %s" % content[:600])
if not (reasoning or content):
    print("      -> connected, but the model returned an empty completion")
    sys.exit(4)
print("      mentions %r: %s"
      % (EXPECT, EXPECT.lower() in (reasoning + content).lower()))

# Decides the harness's `wire_api`: the codex harness defaults to `responses`,
# which needs this route to work.
responses = call("responses", BASE + "/responses",
                 {"model": MODEL, "input": PROMPT, "max_output_tokens": 512})
if responses is None:
    print("      -> /v1/responses is unusable here; run with --wire-api chat "
          "and a codex predating codex#10157")
else:
    print("      -> /v1/responses works; the default --wire-api responses is fine")

if os.environ.get("PROBE_MESSAGES") != "0":
    # The Anthropic Messages API, which is all claude-code speaks.
    #
    # ANTHROPIC_BASE_URL is BASE with the trailing /v1 removed, because the CLI
    # re-adds it before /messages. Rebuilding it the long way here rather than
    # writing BASE + "/messages" is deliberate: the two are equal only if that
    # round trip is right, and this is the cheap place to find out.
    anthropic_base = BASE[:-3].rstrip("/") if BASE.endswith("/v1") else BASE
    MSG_URL = anthropic_base + "/v1/messages"
    # Claude Code also calls this one, to decide when to compact. It is a
    # separate handler in sglang, so a 500 here is not covered by anything
    # posted to /messages -- and from the CLI's side the two look identical.
    COUNT_URL = MSG_URL + "/count_tokens"
    MSG_HDRS = dict(HDRS)
    MSG_HDRS["anthropic-version"] = "2023-06-01"
    print("\n      anthropic base: %s" % anthropic_base)
    print("      posting to:     %s" % MSG_URL)
    print("                      %s" % COUNT_URL)

    def messages(name, body, url=None):
        payload = call(name, url or MSG_URL, body, hdrs=MSG_HDRS)
        if payload is None:
            return False
        # A streaming reply can be HTTP 200 and still carry an error event,
        # which is a failure the status line alone would call a success.
        if '"type": "error"' in payload or '"type":"error"' in payload:
            print("      error event in body: %s" % payload[:400])
            return False
        return True

    base_body = {
        "model": MODEL,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": PROMPT}],
    }
    # Shaped like Claude Code's own tool definitions rather than a minimal
    # one: the $schema key and the nested constraints are exactly what an
    # Anthropic->OpenAI schema converter can raise on, and a raise is a 500.
    tool = {
        "name": "read_file",
        "description": "Read a file from disk",
        "input_schema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path"},
                "limit": {"type": "integer", "exclusiveMinimum": 0},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    }
    stages = [
        ("msg:plain", dict(base_body)),
        # Claude Code always streams; there is no non-streaming mode for it.
        ("msg:stream", dict(base_body, stream=True)),
        # Prompt caching. Claude Code marks its system prompt and tools with
        # cache_control on every request.
        ("msg:cache", dict(base_body, system=[{
            "type": "text",
            "text": "Be brief.",
            "cache_control": {"type": "ephemeral"},
        }])),
        ("msg:tools", dict(base_body, tools=[tool])),
        # --- everything above passed against a server that still 500s the
        # real CLI, so the rest of this list is the actual search space. ---
        #
        # `thinking` has three spellings and an adapter can know only some of
        # them. MAX_THINKING_TOKENS=0 is meant to suppress the field entirely,
        # but "disabled" is what a client sends when it wants it off
        # explicitly, and "adaptive" is what current Claude Code prefers for
        # any model it does not recognise -- which a filesystem path is.
        ("msg:think-adaptive", dict(base_body, thinking={"type": "adaptive"})),
        ("msg:think-disabled", dict(base_body, thinking={"type": "disabled"})),
        ("msg:think-enabled", dict(
            base_body, max_tokens=2048,
            thinking={"type": "enabled", "budget_tokens": 1024})),
        # The real output cap. 64 proves the route; 16384 is what the harness
        # sets, and a cap above what the server allows is a plausible reject.
        ("msg:bigmax", dict(base_body, max_tokens=16384)),
        # Claude Code sends a system array of several blocks, not one.
        ("msg:sys-multi", dict(base_body, system=[
            {"type": "text", "text": "You are a coding agent."},
            {"type": "text", "text": "Be brief.",
             "cache_control": {"type": "ephemeral"}},
        ])),
        ("msg:tool-choice", dict(
            base_body, tools=[tool],
            tool_choice={"type": "auto", "disable_parallel_tool_use": False})),
        ("msg:metadata", dict(base_body, metadata={"user_id": "orchard"})),
        ("msg:stop-seq", dict(base_body, stop_sequences=["</done>"])),
        # One tool was not a fair test: Claude Code ships ~20, whose schemas
        # use constructs a converter is likelier to choke on than the plain
        # object above.
        ("msg:rich-tools", dict(base_body, tools=[tool, {
            "name": "send_message",
            "description": "Send a message",
            "input_schema": {
                "type": "object",
                "properties": {
                    "to": {"allOf": [{"pattern": "^[^\\n]*$"}],
                           "type": "string"},
                    "mode": {"enum": ["fast", "slow"], "type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"},
                             "maxItems": 32},
                    "opts": {"type": "object",
                             "additionalProperties": {"type": "string"},
                             "propertyNames": {"type": "string"}},
                    "who": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
                "required": ["to"],
            },
        }])),
        # The field that actually broke the first claude-code run here. The CLI
        # sends output_config.effort on every request and defaults it to
        # "high"; sglang forwards it as chat-completions `reasoning_effort`
        # (rewriting "xhigh" to "max"), and a chat template that does not know
        # the level raises inside jinja -- uncaught, so it arrives as a 500.
        # Qwen3's template takes xhigh/medium/low only. The harness pins
        # `effort_fallback: medium`, so probe the CLI default and the pinned
        # value separately: they fail for different reasons and have different
        # fixes. Note this probe talks to the endpoint it is given, so against
        # the session router both come back 200 -- the router strips the field.
        ("msg:effort-high", dict(base_body, output_config={"effort": "high"})),
        ("msg:effort-medium",
         dict(base_body, output_config={"effort": "medium"})),
        # Everything at once, which is the shape the CLI actually posts.
        ("msg:combined", dict(
            base_body, max_tokens=16384, stream=True, tools=[tool],
            tool_choice={"type": "auto"},
            metadata={"user_id": "orchard"},
            output_config={"effort": "medium"},
            system=[{"type": "text", "text": "You are a coding agent.",
                     "cache_control": {"type": "ephemeral"}}])),
    ]
    outcomes = [(n, messages(n, b)) for n, b in stages]
    # Its own route, and its own body shape: count_tokens takes no max_tokens,
    # so it cannot just be folded in above.
    outcomes.append(("msg:count-tokens", messages(
        "msg:count-tokens",
        {"model": MODEL, "tools": [tool],
         "messages": [{"role": "user", "content": PROMPT}]},
        COUNT_URL)))
    broken = [n for n, good in outcomes if not good]

    if not broken:
        print("      -> every shape probed here works. The CLI's 500 is "
              "something this probe still does not send -- capture the real "
              "body with the in-pod proxy rather than guessing further.")
    elif len(broken) == len(outcomes):
        print("      -> /v1/messages is unusable here: even a minimal request "
              "fails, so this is the route or the adapter, not the payload.")
        print("         404 means this sglang predates Anthropic support; a 500 "
              "leaves a traceback in the sglang log.")
    else:
        print("      -> /v1/messages works, but these shapes fail: %s"
              % ", ".join(broken))
        print("         Claude Code combines these on its FIRST request, so "
              "any one of them is fatal to the harness unless noted:")
        hints = {
            "msg:stream": "streaming is not optional for claude-code",
            "msg:cache": "cache_control on system/tools -- the harness would "
                         "have to stop sending it",
            "msg:tools": "tool schema conversion -- without tools the agent "
                         "cannot edit a file, so the run is pointless",
            "msg:think-adaptive": "set MAX_THINKING_TOKENS so the CLI cannot "
                                  "ask for adaptive thinking",
            "msg:think-disabled": "the adapter rejects an explicit 'off'; the "
                                  "field has to be absent entirely",
            "msg:think-enabled": "extended thinking is unsupported here -- "
                                 "max_thinking_tokens: 0 already covers it",
            "msg:bigmax": "lower harness.params.max_tokens below 16384",
            "msg:sys-multi": "multi-block system array",
            "msg:tool-choice": "tool_choice",
            "msg:metadata": "metadata.user_id",
            "msg:stop-seq": "stop_sequences",
            "msg:rich-tools": "a richer tool schema (allOf/anyOf/enum/"
                              "propertyNames) than the plain one that passed",
            "msg:effort-high": "the CLI's default reasoning effort. Expected "
                               "to fail on a Qwen chat template, which is why "
                               "harness.params.effort pins a level -- harmless "
                               "here as long as msg:effort-medium passed.",
            "msg:effort-medium": "the level harness.params.effort pins, so "
                                 "every request would 500. Try effort: low, "
                                 "or \"\" to drop the flag entirely.",
            "msg:combined": "only the combination fails, so it is a limit "
                            "(size, token count) rather than one field",
            "msg:count-tokens": "/v1/messages/count_tokens is broken while "
                                "/v1/messages works. The CLI calls it to "
                                "decide when to compact, so this alone is "
                                "enough to 500 every rollout.",
        }
        for name in broken:
            if name in hints:
                print("         %-20s %s" % (name, hints[name]))

# Advisory only. Three of the four harnesses never touch /v1/messages, so a
# broken Messages API must not fail the check they depend on.
sys.exit(0)
'''


def banner(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def resolve_model(raw: str) -> str:
    """Strip the litellm provider prefix; the server sees only the bare name."""
    return raw.removeprefix("openai/") if raw else raw


def check_local(base_urls: list[str], api_key: str, model: str) -> bool:
    banner("1. Local machine -> sglang")
    ok = True
    for base_url in base_urls:
        print(f"-- {base_url}")
        try:
            r = requests.get(
                f"{base_url}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=2,
            )
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"unreachable from here: {type(e).__name__}: {e}")
            print("(continuing — the pod may have a route your laptop lacks)")
            ok = False
            continue
        ids = [m["id"] for m in r.json().get("data", [])]
        print(f"reachable, served models: {ids}")
        if model and ids and model not in ids:
            print(f"WARNING: requesting {model!r}, server offers {ids}")
    return ok


def check_sandbox(
    image: str,
    base_urls: list[str],
    api_key: str,
    model: str,
    prompt: str,
    messages: bool = True,
) -> bool:
    banner("2. Remote Azure sandbox -> sglang")
    ok = True
    with SandboxClient() as client:
        print(f"orchestrator health: {client.health()}")
        print(f"creating sandbox from {image} (block_network=False) ...")
        # Sandboxes have no egress by default, so this must be False.
        with client.create_sandbox(image=image, block_network=False) as sandbox:
            print(f"sandbox ready: {sandbox.sandbox_id}\n")
            sandbox.upload_content(POD_PROBE.encode(), "/tmp/probe.py")
            # One pod, N probes: what is under test is the N tunnels, not the
            # pod, so spending a pod per endpoint would buy nothing.
            for base_url in base_urls:
                print(f"-- {base_url}")
                result = sandbox.exec(
                    "python /tmp/probe.py",
                    timeout=300,
                    env={
                        "MODEL_BASE_URL": base_url,
                        "MODEL_API_KEY": api_key,
                        "MODEL_NAME": model,
                        "PROBE_PROMPT": prompt,
                        "PROBE_EXPECT": EXPECT,
                        "PROBE_MESSAGES": "1" if messages else "0",
                    },
                )
                print(result.stdout, end="")
                if result.stderr:
                    print(f"--- stderr ---\n{result.stderr}")
                print(f"\nstatus={result.status} exit_code={result.exit_code}\n")
                ok = ok and result.exit_code == 0
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="python:3.11-slim")
    ap.add_argument("--model", default=os.environ.get("MODEL_NAME", ""))
    ap.add_argument(
        "--replicas",
        type=int,
        default=int(os.environ.get("MODEL_BASE_URL_REPLICAS", "1")),
        help="Probe N endpoints on consecutive ports starting at MODEL_BASE_URL",
    )
    ap.add_argument(
        "--session",
        nargs="?",
        const="check-model-from-sandbox",
        default=None,
        metavar="ID",
        help=(
            "Probe through the session-router path, .../session/<ID>/v1, which "
            "is what --routing session sends. Verifies the prefix survives each "
            "CLI's URL joining."
        ),
    )
    ap.add_argument(
        "--sessions",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Probe N distinct session ids rather than one, which is how you "
            "check that the router spreads sessions across the fleet. Implies "
            "--session. Read the result on the GPU host with "
            "curl -s http://127.0.0.1:8100/router/stats | jq .spread"
        ),
    )
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument(
        "--no-messages",
        action="store_true",
        help=(
            "Skip the /v1/messages probes. They are four extra calls per "
            "endpoint, which adds up under --replicas N --sessions M, and "
            "nothing but claude-code uses that route."
        ),
    )
    args = ap.parse_args()

    if "MODEL_BASE_URL" not in os.environ:
        print("MODEL_BASE_URL is not set", file=sys.stderr)
        return 2
    model_base_url = os.environ["MODEL_BASE_URL"].rstrip("/")
    base_urls = (
        expand_ports(model_base_url, args.replicas)
        if args.replicas > 1
        else [model_base_url]
    )
    if args.sessions < 1:
        print("--sessions must be >= 1", file=sys.stderr)
        return 2
    if args.session or args.sessions > 1:
        # Distinct ids, because the router places a session once and pins it
        # after that — sixteen probes sharing one id prove stickiness, not
        # balance. Sequential probing is enough to show the spread: a session
        # keeps counting toward its engine's load for the router's whole
        # active TTL, so earlier probes are still weighing when later ones are
        # placed.
        stem = args.session or "check-model-from-sandbox"
        base_urls = [
            session_url(url, stem if args.sessions == 1 else f"{stem}-{i}")
            for url in base_urls
            for i in range(args.sessions)
        ]
    api_key = os.environ.get("API_KEY", "")
    model = resolve_model(args.model)
    if len(base_urls) > 4:
        print(f"model={model!r}  endpoints={len(base_urls)}, from {base_urls[0]}")
    else:
        print(f"model={model!r}  endpoints={base_urls}")

    local_ok = check_local(base_urls, api_key, model)
    ok = check_sandbox(
        args.image, base_urls, api_key, model, args.prompt, not args.no_messages
    )
    banner(
        "RESULT\n"
        f"  local   -> sglang: {'SUCCESS' if local_ok else 'FAILED'}\n"
        f"  sandbox -> sglang: {'SUCCESS' if ok else 'FAILED'}"
    )
    if not local_ok:
        # Diagnostic only: every harness calls the model from the pod, so a
        # dead local hop usually just means this machine is off the network
        # that serves the endpoint.
        print(
            "No harness calls the model from THIS machine — they all call it "
            "from inside the pod — so a failing local hop is informational "
            "unless the sandbox hop failed too."
        )
    # Exit code still tracks the sandbox hop, which is what this script is for.
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
