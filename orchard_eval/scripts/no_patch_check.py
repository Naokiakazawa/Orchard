#!/usr/bin/env python3
"""Sanity check: with no patch at all, does SWE-bench Verified score ~0%?

The companion to ``gold_patch_check.py``. That one measures the ceiling — the
human fix must resolve ~100%. This one measures the floor: run the benchmark's
own tests against the *untouched* repository and nothing should resolve, because
every instance's FAIL_TO_PASS tests are failing at ``base_commit`` by
definition.

A resolve rate materially above zero here is the grading path giving away
points: an eval script selecting tests that already pass before the fix, a log
parser scoring an unparsed log as success, or rows with an empty FAIL_TO_PASS
list that grade as vacuously resolved. None of that shows up in a gold run,
where those same defects only push an already-100% number to 100%.

The check has a second half that matters as much as the rate. An instance that
scores zero because its eval pod fell over proves nothing, so this also reports
how many instances actually produced parseable test results, and how many of
those kept PASS_TO_PASS green — the tests unrelated to the fix, which must pass
at ``base_commit`` on a healthy image.

Usage:
    python scripts/no_patch_check.py --limit 50
    python scripts/no_patch_check.py                      # all 500
    python scripts/no_patch_check.py --limit 20 --max-rate 0.0
    python scripts/no_patch_check.py --benchmark swe-bench-pro --limit 20
    python scripts/no_patch_check.py --benchmark swe-bench-multilingual --limit 20
    python scripts/no_patch_check.py --report results/noop-swebench-verified/<run_id>
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections import Counter
from pathlib import Path

from orchard_evalkit.config import RunConfig, load_config
from orchard_evalkit.runner import load_existing_records, run_eval

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

CONFIGS = {
    "swe-bench-verified": [CONFIG_DIR / "noop.yaml"],
    "swe-bench-pro": [CONFIG_DIR / "noop-pro.yaml"],
    # Layered, not substituted: `dataset.name=swe-bench-multilingual` alone
    # leaves the verified pod sizes and timeouts in place, and a cargo or
    # gradle suite on 2 CPUs / 8Gi times out instead of reporting a floor.
    "swe-bench-multilingual": [
        CONFIG_DIR / "noop.yaml",
        CONFIG_DIR / "swe-bench-multilingual.yaml",
    ],
}

#: The dataset overlays carry no run_name, so results would otherwise land in
#: the verified run directory.
RUN_NAMES = {"swe-bench-multilingual": "noop-swebench-multilingual"}

#: Above this, a model's score on this suite includes points it did not earn.
DEFAULT_MAX_RATE = 0.02

#: Below this share of instances actually running their tests, the rate above
#: is measuring dead pods rather than the grading path.
DEFAULT_MIN_GRADED = 0.90

#: Why an instance produced no test results. The tests are supposed to run and
#: fail here, so every one of these is a hole in the measurement, not a zero.
DIAGNOSES = {
    "EMPTY_PATCH": "grading short-circuited — set grading.grade_empty_patch=true",
    "EVAL_INFRA_ERROR": "eval pod or its agent failed — not a test result",
    "EVAL_TIMEOUT": "tests exceeded grading.eval_timeout",
    "EVAL_RESET_FAILED": "eval script could not reset the repo to base_commit",
    "PATCH_APPLY_FAILED": "eval script could not apply the *test* patch",
    "EVAL_SCRIPT_MISSING": "SWE-bench Pro run_script.sh/parser.py unavailable",
    "infra_error": "rollout pod could not be created",
}


def split_counts(record, key: str) -> tuple[int, int]:
    """Return ``(passed, total)`` for one test split of a graded instance."""
    split = (record.tests_status or {}).get(key) or {}
    passed = len(split.get("success") or [])
    failed = len(split.get("failure") or [])
    return passed, passed + failed


def summarize_ungraded(records) -> None:
    ungraded = [r for r in records if split_counts(r, "FAIL_TO_PASS")[1] == 0]
    if not ungraded:
        return

    print(f"\n{len(ungraded)} instances produced no FAIL_TO_PASS verdicts:\n")
    reasons = Counter(r.resolved_status or r.exit_status for r in ungraded)
    for reason, count in reasons.most_common():
        hint = DIAGNOSES.get(reason, "tests ran but nothing was parsed — see eval.log")
        print(f"  {count:>4}  {reason:<22} {hint}")

    print("\n  instances:")
    for record in ungraded[:40]:
        detail = (record.error or "").splitlines()[:1]
        print(
            f"    {record.instance_id:<40} {record.resolved_status:<20}"
            f" {detail[0][:120] if detail else ''}"
        )
    if len(ungraded) > 40:
        print(f"    ... and {len(ungraded) - 40} more")


def summarize_scorers(records) -> None:
    """Name the instances that scored without a fix — the whole point."""
    scored = [r for r in records if r.resolved]
    if not scored:
        return

    print(f"\n{len(scored)} resolved WITHOUT a patch:\n")
    for record in scored[:40]:
        f2p_pass, f2p_total = split_counts(record, "FAIL_TO_PASS")
        note = (
            "FAIL_TO_PASS is empty — vacuously resolved"
            if f2p_total == 0
            else f"{f2p_pass}/{f2p_total} FAIL_TO_PASS already passing at base_commit"
        )
        print(f"    {record.instance_id:<40} {note}")
    if len(scored) > 40:
        print(f"    ... and {len(scored) - 40} more")


def verdict(records, max_rate: float, min_graded: float) -> int:
    total = len(records)
    if not total:
        print("No records — nothing was run.")
        return 1

    resolved = sum(1 for r in records if r.resolved)
    rate = resolved / total

    graded = [r for r in records if split_counts(r, "FAIL_TO_PASS")[1] > 0]
    graded_fraction = len(graded) / total
    p2p_green = 0
    for record in graded:
        p2p_passed, p2p_total = split_counts(record, "PASS_TO_PASS")
        if p2p_passed == p2p_total:
            p2p_green += 1

    print("\n" + "=" * 62)
    print(f"  no-patch resolve rate     {resolved}/{total} = {rate * 100:.2f}%")
    print(
        f"  tests actually ran        {len(graded)}/{total} = "
        f"{graded_fraction * 100:.2f}%"
    )
    if graded:
        print(
            f"  PASS_TO_PASS green        {p2p_green}/{len(graded)} = "
            f"{p2p_green / len(graded) * 100:.2f}%"
        )
    print("=" * 62)

    summarize_scorers(records)
    summarize_ungraded(records)

    failures = []
    if rate > max_rate:
        failures.append(
            f"resolve rate {rate * 100:.2f}% is above the {max_rate * 100:.0f}% "
            "ceiling. The tests are passing before the fix exists, so every "
            "model measured here is credited with instances it did not solve. "
            "Check `FAIL_TO_PASS` on the rows listed above and the eval.log "
            "they produced."
        )
    if graded_fraction < min_graded:
        failures.append(
            f"only {graded_fraction * 100:.2f}% of instances produced test "
            f"results (floor {min_graded * 100:.0f}%). The rate above is "
            "measuring broken pods, not grading — the resolve rate it reports "
            "is not evidence of anything until this is fixed."
        )

    if not failures:
        print(
            f"\nPASS — at or below the {max_rate * 100:.0f}% ceiling, with the "
            "tests actually running."
        )
        return 0

    for reason in failures:
        print(f"\nFAIL — {reason}")
    print(
        "\n`python scripts/regrade.py <run_dir>` re-derives each verdict from "
        "the saved eval.log and says whether the wrapper or the environment is "
        "at fault."
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Check at most N instances")
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--instance-id", action="append", default=[])
    parser.add_argument(
        "--benchmark",
        choices=sorted(CONFIGS),
        default="swe-bench-verified",
        help="Which benchmark to check (default: swe-bench-verified)",
    )
    parser.add_argument(
        "--max-rate",
        type=float,
        default=DEFAULT_MAX_RATE,
        help=f"Maximum acceptable resolve rate (default {DEFAULT_MAX_RATE})",
    )
    parser.add_argument(
        "--min-graded",
        type=float,
        default=DEFAULT_MIN_GRADED,
        help=(
            "Minimum share of instances that must produce test results "
            f"(default {DEFAULT_MIN_GRADED})"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Skip the run and judge an existing run directory instead",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "overrides", nargs="*", help="Dotted config overrides, e.g. concurrency=8"
    )
    args = parser.parse_args()

    if args.report:
        records = load_existing_records(Path(args.report) / "results.jsonl")
        return verdict(list(records.values()), args.max_rate, args.min_graded)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    overrides: list[str] = []
    # First, so an explicit `run_name=` on the command line still wins.
    if args.benchmark in RUN_NAMES:
        overrides.append(f"run_name={RUN_NAMES[args.benchmark]}")
    overrides += list(args.overrides)
    if args.limit is not None:
        overrides.append(f"dataset.limit={args.limit}")
    if args.concurrency is not None:
        overrides.append(f"concurrency={args.concurrency}")
    if args.instance_id:
        ids = [i for arg in args.instance_id for i in arg.split(",") if i]
        overrides.append("dataset.instance_ids=[" + ", ".join(ids) + "]")
    if args.no_resume:
        overrides.append("resume=false")
    # Neither is overridable here: a floor measured with a harness that
    # produces patches, or with grading skipping empty ones, is not a floor.
    overrides.append("harness.name=noop")
    overrides.append("grading.grade_empty_patch=true")

    config: RunConfig = load_config(
        [str(path) for path in CONFIGS[args.benchmark]], overrides
    )
    print(f"No-patch check -> {config.run_dir}\n")

    records = asyncio.run(run_eval(config))
    code = verdict(records, args.max_rate, args.min_graded)

    print(f"\n  summary   {config.run_dir / 'summary.json'}")
    print(f"  logs      {config.instances_dir}/<instance_id>/eval.log")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
