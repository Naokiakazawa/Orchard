#!/usr/bin/env python3
"""Check that the sandbox-tools payload actually runs inside a given image.

Every sandbox gets the agent CLIs mounted at ``/opt/sandbox-tools`` and shimmed
into ``/usr/local/bin``, so ``command -v pi`` succeeds in every pod. That says
nothing about whether ``pi`` *runs*: most of the payload is dynamically linked
against glibc, and part of the SWE-bench Pro image set is Alpine/musl, where the
loader is missing and ``execve`` fails. The two look identical from outside —
one is a working agent, the other is a trial that dies before the model is
called — so this runs each CLI rather than looking for it.

Worth running after every ``sandbox-tools`` push, against at least one musl
image. The defaults are three known-musl SWE-bench Pro images, taken from the
trials that first exposed the problem.

Usage:
    export SANDBOX_BASE_URL=... SANDBOX_API_KEY=...
    python scripts/probe_sandbox_tools.py
    python scripts/probe_sandbox_tools.py --image alpine:latest --image debian:12-slim
    python scripts/probe_sandbox_tools.py --tool pi --tool mini --timeout 3600

Exits non-zero if any probed CLI failed to run, or if an image could not be
probed at all, so it can gate a run.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from orchard_env import SandboxClient

#: Seconds to wait for a pod. Dominated by the image pull, not by scheduling:
#: a Pro image is several GB and `teleport` is among the largest, so a cold node
#: takes minutes. The SDK's own default is an hour, which looks like a hang.
DEFAULT_CREATE_TIMEOUT = 1800

#: Where the orchestrator mounts the payload (SANDBOX_TOOLS_MOUNT_PATH).
TOOLS_DIR = "/opt/sandbox-tools"

#: What the payload ships. `mini` has no --version flag; --help exercises the
#: same interpreter and import path, which is what is under test here.
TOOLS: dict[str, str] = {
    "codex": "--version",
    "claude": "--version",
    "pi": "--version",
    "opencode": "--version",
    "hermes": "--version",
    "mini": "--help",
}

#: Known-musl SWE-bench Pro images. Any one of them reproduces the failure; all
#: three are here because they come from three different upstream bases.
DEFAULT_IMAGES = (
    "mirror.gcr.io/jefzda/sweap-images:protonmail.webclients-protonmail__webclients-cba6ebbd0707caa524ffee51c62b197f6122c902",
    "mirror.gcr.io/jefzda/sweap-images:gravitational.teleport-gravitational__teleport-2b15263e49da5625922581569834eec4838a9257-vee9b09fb20c43af7e520f57e9239bbcf46b7113d",
    "mirror.gcr.io/jefzda/sweap-images:future-architect.vuls-future-architect__vuls-e049df50fa1eecdccc5348e27845b5c783ed7c76-v73dc95f6b90883d8a87e01e5e9bb6d3cc32add6d",
)


def probe_script(tools: dict[str, str]) -> str:
    """A POSIX-sh probe printing one ``name|status|detail`` line per check.

    POSIX sh, not bash, and ``printf`` rather than ``echo``: an Alpine image has
    busybox and nothing else, and busybox ``echo`` does not expand ``\\t``.
    """
    lines = [
        "libc=glibc",
        "[ -f /etc/alpine-release ] && libc=musl",
        "ldd --version 2>&1 | head -1 | grep -qi musl && libc=musl",
        'printf "%s|%s|%s\\n" libc "$libc" "$(ldd --version 2>&1 | head -1)"',
        f'printf "%s|%s|%s\\n" payload '
        f'"$([ -d {TOOLS_DIR} ] && echo present || echo MISSING)" {TOOLS_DIR}',
    ]
    for tool, flag in tools.items():
        binary = f"{TOOLS_DIR}/bin/{tool}"
        lines += [
            f'if [ ! -x "{binary}" ]; then',
            f'    printf "%s|%s|%s\\n" {tool} MISSING "no {binary}"',
            "else",
            # Capture status separately: `$?` after a pipeline through head(1)
            # is head's, and head exits 0 even when the tool crashed.
            f'    out="$("{binary}" {flag} 2>&1)"; status=$?',
            f'    printf "%s|%s|%s\\n" {tool} '
            f'"$([ $status -eq 0 ] && echo ok || echo FAILED)" '
            '"$(printf "%s" "$out" | head -1)"',
            "fi",
        ]
    return "\n".join(lines)


def probe_image(
    client: SandboxClient,
    image: str,
    tools: dict[str, str],
    create_timeout: int = DEFAULT_CREATE_TIMEOUT,
) -> bool:
    print(f"\n{'=' * 78}\n{image}\n{'=' * 78}")
    # Said before the call, not after: creating the pod blocks on the image
    # pull, which on a cold node is minutes of no output and reads as a hang.
    print(
        f"  starting a sandbox (pulling the image if the node is cold; "
        f"giving up after {create_timeout}s) ...",
        flush=True,
    )
    started = time.monotonic()
    # The payload is mounted by the orchestrator at pod creation, so nothing
    # here needs egress; block_network keeps the probe honest about that.
    with client.create_sandbox(
        image=image, block_network=True, timeout=create_timeout
    ) as sandbox:
        print(f"  ready in {time.monotonic() - started:.0f}s\n", flush=True)
        result = sandbox.exec(probe_script(tools), timeout=300)
    if result.exit_code != 0 and not result.stdout:
        print(f"probe did not run: exit={result.exit_code} {result.stderr}")
        return False

    ok = True
    for line in result.stdout.splitlines():
        name, _, rest = line.partition("|")
        status, _, detail = rest.partition("|")
        print(f"  {name:<10} {status:<8} {detail}")
        if status in ("FAILED", "MISSING") and name in tools:
            ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Image to probe. Repeatable. Defaults to three known-musl Pro images",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_CREATE_TIMEOUT,
        help=(
            "Seconds to wait for each pod before giving up. The image pull "
            f"dominates this (default: {DEFAULT_CREATE_TIMEOUT})"
        ),
    )
    parser.add_argument(
        "--tool",
        action="append",
        default=[],
        choices=sorted(TOOLS),
        help="Probe only these CLIs. Repeatable. Defaults to all of them",
    )
    args = parser.parse_args()

    if not os.environ.get("SANDBOX_BASE_URL"):
        print("SANDBOX_BASE_URL is not set", file=sys.stderr)
        return 2

    images = args.image or list(DEFAULT_IMAGES)
    tools = {name: TOOLS[name] for name in (args.tool or TOOLS)}

    # Kept apart on purpose: an image whose payload is broken is a finding, an
    # image that never started is a gap in the evidence. Reporting them as one
    # number would let a slow pull read as a failed CLI.
    unusable: list[str] = []
    unprobed: list[str] = []
    with SandboxClient() as client:
        print(f"orchestrator health: {client.health()}")
        for image in images:
            try:
                if not probe_image(client, image, tools, args.timeout):
                    unusable.append(image)
            except KeyboardInterrupt:
                # Ctrl-C during a slow pull should skip that image, not throw
                # away the results already printed for the ones before it.
                print("\n  interrupted; skipping this image")
                unprobed.append(image)
            except Exception as exc:  # noqa: BLE001 - one bad image is not fatal
                print(f"  could not probe: {type(exc).__name__}: {exc}")
                unprobed.append(image)

    print(f"\n{'=' * 78}")
    probed = len(images) - len(unprobed)
    if unusable:
        print(f"{len(unusable)}/{probed} probed images have an unusable payload:")
        for image in unusable:
            print(f"  {image}")
        print(
            "\nA CLI that is present but FAILED is the glibc-payload-on-musl case: "
            "the wrapper resolves and execve does not. Agents run from the "
            "payload will die before the model is called on every task using "
            "such an image."
        )
    elif probed:
        print(f"every probed CLI ran on all {probed} images")
    for image in unprobed:
        print(f"NOT PROBED (no result either way): {image}")
    # An image that never started is not a pass: the question went unanswered.
    return 1 if (unusable or unprobed or not probed) else 0


if __name__ == "__main__":
    sys.exit(main())
