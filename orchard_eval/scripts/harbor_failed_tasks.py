#!/usr/bin/env python3
"""Group a finished Harbor job's failures by cause, and name the tasks in each.

``orchard-eval harbor`` already prints a failure *category* per trial, but
AGENT_ERROR covers everything from "the agent CLI could not be installed" to
"the pod was evicted", and those want completely different responses. The
distinguishing evidence is in each trial's ``exception.txt`` and ``trial.log``;
this reads both and sorts the trials by what actually went wrong.

The output is a task-name list, so a diagnosis turns straight into the retry
that tests it::

    python scripts/harbor_failed_tasks.py results/harbor/pi-swebench-pro-harbor
    python scripts/harbor_failed_tasks.py results/harbor/pi-... --signature musl > musl.txt
    orchard-eval harbor -d scale-ai/swe-bench-pro@2 --agent pi --task-file musl.txt

With no ``--signature`` it prints the breakdown instead: one line per cause with
a count, which is the first thing worth looking at after a run.

``--signature`` takes a comma-separated list, because one agent's problem is
often several causes wearing different hats: mini-swe-agent's failed installs
are ``uv-build,python-too-old``. Entries may also be any literal substring, for
a cause that has no preset yet -- ``--signature "No space left on device"``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path

#: One attempt of a Harbor job. Duplicated from
#: ``orchard_evalkit.harbor_bridge.newest_attempt`` rather than imported,
#: because importing it pulls in the whole eval package — and this script is
#: meant to run on a cluster shell with nothing but the standard library.
RUN_ID_RE = re.compile(r"^\d{8}-\d{6}$")


def newest_attempt(job_dir: Path) -> Path:
    """The newest ``YYYYmmdd-HHMMSS`` attempt that recorded something.

    Preferring the newest outright would hide the last real run behind an
    attempt that crashed before its first trial.
    """
    try:
        attempts = [
            d for d in job_dir.iterdir() if d.is_dir() and RUN_ID_RE.match(d.name)
        ]
    except OSError:
        return job_dir
    if not attempts:
        return job_dir
    recorded = [d for d in attempts if any(d.rglob("result.json"))]
    return max(recorded or attempts, key=lambda d: d.name)

#: Causes worth telling apart. A trial is attributed to the first one that
#: matches, and the order encodes that precedence: what stopped the agent from
#: ever starting, then Harbor's own verdict on a phase, then what the provider
#: noticed about the pod. A trial often shows more than one — a pod reaped
#: during teardown leaves "no longer exists" in the log of a trial that had
#: already timed out — and the first match is the one worth acting on.
SIGNATURES: dict[str, tuple[str, str]] = {
    "musl": (
        "linux-x64-musl",
        "Alpine image: nvm found no musl Node tarball, so the agent CLI "
        "was never installed (exit 127)",
    ),
    "uv-build": (
        "maturin",
        "`uv tool install` fell back to a Rust sdist and the image has no "
        "Rust toolchain",
    ),
    "python-too-old": (
        "cannot import name 'NotRequired'",
        "the agent was installed against the image's system Python, which is "
        "older than its dependencies need",
    ),
    "agent-timeout": (
        "AgentTimeoutError",
        "the agent phase hit the task's own timeout_sec (raise it with "
        "`harbor run --agent-timeout-multiplier`)",
    ),
    "verifier-timeout": (
        "VerifierTimeoutError",
        "the verifier hit the task's own timeout_sec",
    ),
    "build-failed": (
        "BuildError",
        "the task's Dockerfile could not be replayed — a translation-layer bug",
    ),
    "pod-gone": (
        "no longer exists",
        "the sandbox pod disappeared mid-trial — evicted, OOM-killed or reaped",
    ),
    "exec-timeout": (
        "did not finish within",
        "the provider stopped waiting on one exec (ORCHARD_HARBOR_EXEC_TIMEOUT)",
    ),
}

UNCLASSIFIED = "other"


def trials(job_dir: Path) -> Iterator[tuple[Path, str]]:
    """Each failed trial in *job_dir*, as ``(trial_dir, evidence)``.

    Harbor writes ``exception.txt`` only for a trial that raised, so its
    presence is the filter — a trial that merely scored zero is not a failure
    this script has anything to say about.

    The trial log is read alongside it because the two hold different halves of
    the story. A pod taken away mid-trial raises inside Harbor's exec, so the
    exception says "exit -1" and blames the provider's exec timeout; only the
    trial log carries the provider's own "the pod ... no longer exists", which
    is the cause.
    """
    for path in sorted(job_dir.glob("*/exception.txt")):
        try:
            evidence = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"skipping unreadable {path}: {exc}", file=sys.stderr)
            continue
        log = path.parent / "trial.log"
        if log.exists():
            try:
                evidence += log.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass  # the exception alone still classifies most causes
        yield path.parent, evidence


def classify(exception: str) -> str:
    for name, (needle, _) in SIGNATURES.items():
        if needle in exception:
            return name
    return UNCLASSIFIED


def task_name(trial_dir: Path) -> str:
    """The dataset-qualified name Harbor filters on.

    ``--include-task-name`` matches against ``PackageTaskId.get_name()``, which
    is ``org/name`` — exactly the ``task_name`` recorded in the trial result —
    so this round-trips without globbing. The directory name does not: Harbor
    truncates it and appends a per-trial suffix.
    """
    try:
        data = json.loads((trial_dir / "result.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(data.get("task_name") or "")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("job_dir", type=Path, help="A Harbor job directory")
    parser.add_argument(
        "--signature",
        help=(
            "Print only the tasks matching these causes, comma-separated: any "
            f"of {', '.join(SIGNATURES)}, or any literal substring of the "
            "exception text. Omit it for the breakdown."
        ),
    )
    args = parser.parse_args()

    if not args.job_dir.is_dir():
        print(f"{args.job_dir} is not a directory", file=sys.stderr)
        return 2

    # Re-runs nest each attempt under the job directory. `trials()` globs one
    # level, so a job directory passed straight in would match nothing and
    # report a clean run — descend to the newest attempt instead.
    attempt = newest_attempt(args.job_dir)
    if attempt != args.job_dir:
        print(f"# reading attempt {attempt.name}", file=sys.stderr)
        args.job_dir = attempt

    wanted: list[str] | None = None
    if args.signature:
        asked = [part.strip() for part in args.signature.split(",") if part.strip()]
        wanted = [SIGNATURES[name][0] if name in SIGNATURES else name for name in asked]
        # A name that is not a preset is still valid — it is used as a literal
        # substring — but it is also what a typo looks like, and the two differ
        # only in whether the output comes back empty. Say which it was.
        freeform = [name for name in asked if name not in SIGNATURES]
        if freeform:
            print(
                f"note: {', '.join(repr(n) for n in freeform)} "
                "is not a preset, matching it as a literal substring",
                file=sys.stderr,
            )

    grouped: dict[str, list[str]] = {}
    unnamed = 0
    for trial_dir, exception in trials(args.job_dir):
        if wanted is not None and not any(needle in exception for needle in wanted):
            continue
        name = task_name(trial_dir)
        if not name:
            # Without a task name the row cannot be retried, and silently
            # dropping it would make the list look complete when it is not.
            unnamed += 1
            continue
        grouped.setdefault(classify(exception), []).append(name)

    if wanted is not None:
        # One task per line and nothing else, so the output is a --task-file.
        # Deduplicated: with --attempts > 1 a task fails once per attempt, and
        # naming it twice would run it twice again.
        names = sorted({n for group in grouped.values() for n in group})
        if not names:
            # Redirected to a file, an empty list is indistinguishable from a
            # clean run, and `--task-file` on it fails much later with Harbor's
            # own "no tasks matched". Fail here, where the cause is known.
            print(
                f"no failed trial in {args.job_dir} matches "
                f"--signature {args.signature!r}.\n"
                f"Presets: {', '.join(SIGNATURES)}. Run without --signature to "
                "see what this job actually failed on.",
                file=sys.stderr,
            )
            return 1
        print("\n".join(names))
    else:
        total = sum(len(group) for group in grouped.values())
        print(f"{total} failed trials in {args.job_dir}\n")
        for cause, group in sorted(grouped.items(), key=lambda item: -len(item[1])):
            help_text = SIGNATURES.get(cause, ("", "no preset matched"))[1]
            print(f"  {len(group):>4}  {cause:<17} {help_text}")
        print(
            "\nRe-run one of them with:\n"
            f"  python {Path(__file__).name} {args.job_dir} "
            "--signature <cause> > tasks.txt\n"
            "  orchard-eval harbor -d <dataset> --agent <agent> --task-file tasks.txt"
        )

    if unnamed:
        print(
            f"\n{unnamed} failed trials have no readable result.json and are "
            "not listed above",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
