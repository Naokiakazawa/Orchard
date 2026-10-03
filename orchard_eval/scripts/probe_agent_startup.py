#!/usr/bin/env python3
"""Find what an agent CLI blocks on while starting inside an air-gapped pod.

On DeepSWE 1.1, mini-swe-agent spends 39-59 minutes between the start of the
agent phase and its first model call, in most trials but not all. The time is
not inference (~25 output tok/s, 13s median call-to-call) and not the sandbox
(a whole rollout moves ~5KB of command output); it is spent inside the agent
process, before it asks the model anything. Against a 120-minute
``SANDBOX_TTL_HOURS`` that is most of the budget.

``LITELLM_LOCAL_MODEL_COST_MAP=True`` was the first suspect, because every
stalled trial logged litellm's cost-map fetch failing against
raw.githubusercontent.com and made its first model call ~10s later. Setting it
removed the log line and changed nothing else: the warning marked the *end* of
the stall, not its cause.

So this measures rather than guesses. It starts one sandbox under the same
egress policy a real trial gets — isolated, with only the model endpoint
allowed — and times each step of startup separately, so a step that blocks
names itself instead of disappearing into one total.

The DNS probes are the point of comparison. ``getent`` is the same
``getaddrinfo`` the agent's HTTP client calls, and its timeout is *not* bounded
by any client-side timeout: httpx resolves before it connects, and a Python
timeout cannot interrupt glibc. Whether a blocked lookup costs 20 seconds or 45
minutes is exactly the question.

Usage:
    export SANDBOX_BASE_URL=... SANDBOX_API_KEY=...
    export MODEL_BASE_URL=http://<frps-public-ip>:30021/v1
    python scripts/probe_agent_startup.py

    # one step only, or a different image
    python scripts/probe_agent_startup.py --only dns --image mirror.gcr.io/...
    python scripts/probe_agent_startup.py --cap 600     # give up on a step sooner

Prints one ``step | elapsed | status | detail`` row per check. Nothing here
writes to the pod's filesystem, and the sandbox is deleted on the way out.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
import time

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harbor_orchard")
)

from harbor_orchard.network import allow_rules_for, host_of  # noqa: E402
from harbor_orchard.settings import session_url  # noqa: E402
from orchard_env import SandboxClient  # noqa: E402

#: Seconds to wait for a pod. A DeepSWE image on a cold node is minutes.
DEFAULT_CREATE_TIMEOUT = 1800

#: Seconds any one step may take before it is abandoned. Deliberately longer
#: than the longest stall observed (59 min), so the measurement is the step's
#: own duration rather than this number.
DEFAULT_CAP = 3900

#: Where the orchestrator mounts the agent CLIs.
TOOLS_DIR = "/opt/sandbox-tools"

#: A DeepSWE 1.1 image, taken from a trial that stalled. Any image from the set
#: works: the agent CLI is mounted by the orchestrator, not baked in, so what is
#: under test here is the pod's network, not the task.
DEFAULT_IMAGE = (
    "mirror.gcr.io/wenlinyao/deep-swe:kh71t2fb7qvx4y0svvv77e5p3182hnvq-v1.1"
)

#: Hosts worth timing, and why each one is a suspect.
#:
#: ``raw.githubusercontent.com`` is litellm's cost map — the known stall, kept
#: as the control now that it is supposed to be switched off.
#: ``openaipublic.blob.core.windows.net`` is where tiktoken fetches a BPE
#: encoding on first use if it is not already cached, and tiktoken is on
#: litellm's import path.
#: ``pypi.org`` stands in for any version or update check.
#: ``example.invalid`` cannot resolve anywhere, by RFC 2606, so it measures what
#: the resolver costs when the *answer* is a definite no rather than a silence.
#: A large gap between it and the others means packets are being dropped rather
#: than refused, which is what turns 20 seconds into 45 minutes.
PROBE_HOSTS = (
    "raw.githubusercontent.com",
    "openaipublic.blob.core.windows.net",
    "pypi.org",
    "example.invalid",
)


def probe_script(cap: int, only: str | None, extra_host: str | None) -> str:
    """A POSIX-sh probe printing one ``name|seconds|status|detail`` row per step.

    POSIX sh and ``printf`` rather than bash and ``echo``: part of the image set
    is busybox-only. Each step is timed around its own ``date``, because the
    whole point is to find which one is slow — a single total would say only
    what we already know.
    """
    want = (lambda group: only in (None, group))
    lines = [
        "have() { command -v \"$1\" >/dev/null 2>&1; }",
        # Without timeout(1) a hung step would run to the exec deadline and take
        # the rest of the probe with it. Every image seen so far has it, from
        # coreutils or busybox; if one does not, steps run uncapped and say so.
        'if have timeout; then CAP="timeout %d"; else CAP=""; fi' % cap,
        'printf "%%s|%%s|%%s|%%s\\n" capped 0 ok "$([ -n "$CAP" ] && echo %ds || echo none)"'
        % cap,
        "step() {",
        '    name=$1; shift',
        '    s=$(date +%s)',
        '    out=$($CAP "$@" 2>&1); st=$?',
        '    e=$(date +%s)',
        # 124 is timeout(1)'s "killed it"; worth distinguishing from a command
        # that failed on its own, which is the expected result for every lookup.
        '    [ $st -eq 124 ] && lab=CAPPED || { [ $st -eq 0 ] && lab=ok || lab="exit$st"; }',
        '    printf "%s|%s|%s|%s\\n" "$name" "$((e-s))" "$lab" '
        '"$(printf "%s" "$out" | tr "\\n" " " | cut -c1-90)"',
        "}",
        # Same timing, but the output is reproduced in full underneath instead
        # of squeezed onto the row. For the steps where *why* it failed is the
        # measurement — a one-line summary of a traceback says nothing.
        "dump() {",
        '    name=$1; shift',
        '    s=$(date +%s)',
        '    out=$($CAP "$@" 2>&1); st=$?',
        '    e=$(date +%s)',
        '    printf "%s|%s|%s|%s\\n" "$name" "$((e-s))" "exit$st" "--- head/tail ---"',
        # mini echoes the whole system+user prompt before it does anything, so
        # a plain dump buries the one line that matters under a screenful of
        # template. The failure is always at the end.
        '    printf "%s\\n" "$out" | head -4 | sed "s|^|      |"',
        '    printf "%s\\n" "$out" | tail -25 | sed "s|^|      |"',
        "}",
    ]

    if want("dns"):
        # resolv.conf first: `options timeout:N attempts:M`, the nameserver
        # count and the search list together predict what an unanswered lookup
        # costs, and are the thing to compare against the measured seconds.
        lines.append(
            'printf "%s|%s|%s|%s\\n" resolv 0 ok '
            '"$(tr "\\n" " " < /etc/resolv.conf | cut -c1-90)"'
        )
        hosts = list(PROBE_HOSTS)
        if extra_host and extra_host not in hosts:
            hosts.insert(0, extra_host)
        for host in hosts:
            # getent, not curl: this isolates name resolution from the connect
            # that follows it, and the two fail for different reasons.
            lines.append(f'step "dns:{host}" getent hosts {host}')

    if want("python"):
        # `mini` is a /bin/sh wrapper, not a console script, so its shebang
        # names the shell — the first version of this probe dutifully ran
        # `/bin/sh -c "import litellm"`. Read the wrapper instead, and go
        # looking for the interpreter it eventually reaches.
        lines += [
            f'dump wrapper cat {TOOLS_DIR}/bin/mini',
            f'dump tools:bin ls -l {TOOLS_DIR}/bin',
            f'dump pythons find {TOOLS_DIR} -maxdepth 4 -name "python3*" -not -type d',
        ]

    if want("mini"):
        binary = f"{TOOLS_DIR}/bin/mini"
        lines += [
            f'if [ ! -x "{binary}" ]; then',
            f'    printf "%s|%s|%s|%s\\n" mini 0 MISSING "no {binary}"',
            "else",
            # --help is the floor: it proves the binary and its loader are
            # healthy. It was instant on the first run, which is why it is not
            # the answer — argparse exits before mini builds a model backend,
            # so the whole path under suspicion is skipped.
            f'    step mini:help:1 "{binary}" --help',
            f'    step mini:help:2 "{binary}" --help',
            "fi",
        ]

    if want("real"):
        # Harbor's own invocation, taken verbatim from a stalled trial.log and
        # given a task that needs no thought. This is the only step that
        # exercises what actually stalls: model-backend construction, the
        # litellm import under mini's interpreter, and the first request. If
        # 39-59 minutes reproduces anywhere it reproduces here; if it does not,
        # then a trial has something this does not, which is worth knowing too.
        lines += [
            'printf "%s|%s|%s|%s\\n" real:endpoint 0 ok "$OPENAI_BASE_URL"',
            'printf "%s|%s|%s|%s\\n" real:costmap 0 ok '
            '"LITELLM_LOCAL_MODEL_COST_MAP=$LITELLM_LOCAL_MODEL_COST_MAP"',
            f'dump real:mini-run {TOOLS_DIR}/bin/mini --yolo "--model=$MINI_MODEL" '
            '"--task=Run exactly one command: echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"',
        ]

    if want("model"):
        # What is left once startup is cleared: the first request itself. A
        # trial's opening prompt is ~50k tokens and sixteen of them arrive at
        # once, so the interesting number is not whether the endpoint answers
        # but how long it queues a realistic prefill under the load the run is
        # already putting on it. Run this *while* a job is going.
        #
        # Through mini's bundled interpreter and its private loader, because
        # that is the only Python in the pod, and with urllib rather than curl
        # because the image set cannot be relied on to have curl.
        lines += [
            f'T={TOOLS_DIR}',
            'MINIPY="$T/glibc/ld-linux-x86-64.so.2 --library-path $T/glibc '
            '$T/mini/python/bin/python3"',
            '[ -x "$T/glibc/ld-linux-x86-64.so.2" ] || MINIPY=python3',
            "cat > /tmp/ping.py <<'PING_EOF'",
            "import json, os, sys, time, urllib.request",
            'url = os.environ["OPENAI_BASE_URL"].rstrip("/") + "/chat/completions"',
            'model = os.environ["MINI_MODEL"].split("openai/", 1)[-1]',
            "words = int(sys.argv[1])",
            '# A filler prompt of roughly `words` tokens, to stand in for the',
            '# repository context a real first turn carries.',
            'prompt = ("the quick brown fox jumps over the lazy dog. " * (words // 9 + 1))',
            'body = json.dumps({"model": model,',
            '    "messages": [{"role": "user", "content": prompt}],',
            '    "max_tokens": 8}).encode()',
            'req = urllib.request.Request(url, data=body, headers={',
            '    "Content-Type": "application/json",',
            '    "Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "EMPTY")})',
            "t = time.time()",
            "try:",
            "    with urllib.request.urlopen(req, timeout=3600) as r:",
            "        data = r.read()",
            '    print("status=%s elapsed=%.1fs prompt_words=%d" % (r.status, time.time() - t, words))',
            '    print(data[:400].decode("utf-8", "replace"))',
            "except Exception as exc:",
            '    print("FAILED after %.1fs: %r" % (time.time() - t, exc))',
            "PING_EOF",
            'step model:chat:tiny $MINIPY /tmp/ping.py 10',
            'dump model:chat:50k $MINIPY /tmp/ping.py 50000',
        ]

    return "\n".join(lines)


def check_clock(sandbox) -> None:
    """Compare the pod's clock with this host's, around one round trip.

    Every stall figure so far is a subtraction across two machines: a model
    call's ``created`` (the inference server) or a log line's timestamp (the
    sandbox node) minus ``agent_execution.started_at`` (the Harbor host). If
    those clocks disagree by ~45 minutes, that subtraction invents a stall
    that nothing inside the pod can reproduce — which is exactly the shape of
    what this probe keeps failing to find. Measured inside one pod with
    elapsed time, everything is fast; measured across the two, it is not.

    The round trip bounds the error: the pod's reading was taken somewhere
    inside it, so a skew larger than the trip is real.
    """
    before = time.time()
    result = sandbox.exec("date +%s", timeout=120)
    after = time.time()
    raw = (result.stdout or "").strip().splitlines()
    if not raw or not raw[-1].isdigit():
        print(f"\nclock: could not read the pod's date: {result.stdout!r} {result.stderr!r}")
        return
    pod = int(raw[-1])
    skew = pod - (before + after) / 2
    print(
        f"\nclock: pod {pod} vs host {(before + after) / 2:.0f} -> "
        f"skew {skew:+.0f}s ({skew / 60:+.1f} min), round trip {after - before:.1f}s"
    )
    if abs(skew) > max(60.0, (after - before) * 2):
        print(
            "       ^ larger than the round trip. Every cross-machine timing in "
            "the run analysis is off by this much, in this direction."
        )


def run_probe(sandbox, label: str, script: str, timeout: int) -> None:
    print(f"\n--- {label} {'-' * (70 - len(label))}")
    result = sandbox.exec(script, timeout=timeout)
    if not result.stdout:
        print(f"  probe produced nothing: exit={result.exit_code} {result.stderr}")
        return
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) != 4:
            print(f"  {line}")
            continue
        name, seconds, status, detail = parts
        print(f"  {name:<34} {seconds:>6}s  {status:<8} {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Image to start")
    parser.add_argument(
        "--only",
        choices=("dns", "python", "mini", "real", "model"),
        help="Run one group of steps instead of all three",
    )
    parser.add_argument(
        "--session",
        nargs="?",
        const="probe-" + os.urandom(4).hex(),
        help=(
            "Address the model through the session router, as every trial "
            "does: MODEL_BASE_URL becomes .../session/<key>/v1. Takes an "
            "optional key; a fresh random one by default, because reusing a "
            "trial's key lands on a session that already exists. This is the "
            "one production path a bare-endpoint probe never exercises"
        ),
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get(
            "MODEL_NAME", "Qwen/Qwen3.8-27B"
        ),
        help=(
            "Model as mini is given it, without the `openai/` prefix this adds. "
            "Defaults to $MODEL_NAME"
        ),
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=DEFAULT_CAP,
        help=(
            "Seconds any one step may take before it is abandoned "
            f"(default: {DEFAULT_CAP}, longer than the 59-minute stall)"
        ),
    )
    parser.add_argument(
        "--create-timeout", type=int, default=DEFAULT_CREATE_TIMEOUT,
        help="Seconds to wait for the pod, dominated by the image pull",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help=(
            "Leave the pod's egress open. The control: if a step is slow here "
            "too, the network policy is not what it is waiting on"
        ),
    )
    args = parser.parse_args()

    if not os.environ.get("SANDBOX_BASE_URL"):
        print("SANDBOX_BASE_URL is not set", file=sys.stderr)
        return 2

    model_url = os.environ.get("MODEL_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if not model_url and not args.open:
        print(
            "MODEL_BASE_URL is not set. An isolated pod needs the same "
            "allowlist a trial gets, or this measures a stricter policy than "
            "the one that stalls. Pass --open to probe without isolation.",
            file=sys.stderr,
        )
        return 2

    if args.session and model_url:
        # Built with the provider's own helper, so the path this probe hits is
        # byte-identical to the one `pinned <trial> to ...` reports.
        model_url = session_url(model_url, args.session)
    rules, unresolved = allow_rules_for([model_url]) if model_url else ([], [])
    if unresolved:
        print(f"could not resolve {', '.join(unresolved)}", file=sys.stderr)
        return 2

    isolated = not args.open
    print(f"image      {args.image}")
    if args.session:
        print(f"session    {args.session} -> {model_url}")
    print(f"egress     {'allowlist ' + str(rules) if isolated else 'open (--open)'}")
    print(f"step cap   {args.cap}s")

    # The same variables _endpoint_env exports into every exec, so the `real`
    # step runs under the environment a trial runs under rather than a
    # sanitised one. Written as a prologue instead of exec(env=...) so the
    # script stays reproducible by hand inside the pod.
    api_key = os.environ.get("SANDBOX_MODEL_API_KEY") or os.environ.get(
        "API_KEY", "EMPTY"
    )
    prologue = "\n".join(
        f"export {name}={value}"
        for name, value in (
            ("OPENAI_BASE_URL", shlex.quote(model_url or "")),
            ("OPENAI_API_BASE", shlex.quote(model_url or "")),
            ("OPENAI_API_KEY", shlex.quote(api_key)),
            ("MSWEA_API_KEY", shlex.quote(api_key)),
            ("LITELLM_LOCAL_MODEL_COST_MAP", "True"),
            # Both from harbor/agents/installed/mini_swe_agent.py. Without the
            # first, mini has no global config, runs its interactive first-run
            # wizard, finds stdin is not a terminal and aborts in under a
            # second — which is what the first replay measured. The second is
            # what lets an unpriced self-hosted model through cost tracking.
            ("MSWEA_CONFIGURED", "true"),
            ("MSWEA_COST_TRACKING", "ignore_errors"),
            ("MINI_MODEL", shlex.quote(f"openai/{args.model_name}")),
        )
    )
    script = prologue + "\n" + probe_script(
        args.cap, args.only, host_of(model_url) if model_url else None
    )
    # The exec has to outlast every step it contains, or the probe is cut off by
    # the thing it is trying to measure. Three caps of headroom covers the
    # groups running back to back.
    exec_timeout = args.cap * 3 + 600

    started = time.monotonic()
    print(f"\nstarting a sandbox (up to {args.create_timeout}s for the pull) ...", flush=True)
    with SandboxClient() as client:
        with client.create_sandbox(
            image=args.image, block_network=isolated, timeout=args.create_timeout
        ) as sandbox:
            print(f"ready in {time.monotonic() - started:.0f}s", flush=True)
            if isolated and rules:
                # Creation takes only a boolean, so the allowlist is a second
                # call — the same two-step the provider does, for the same
                # reason: the policy resolves against a running pod.
                sandbox.disable_network(allowlist=rules)
                print(f"egress narrowed to {rules}", flush=True)
            check_clock(sandbox)
            run_probe(sandbox, "startup", script, exec_timeout)
    print(f"\ntotal {time.monotonic() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
