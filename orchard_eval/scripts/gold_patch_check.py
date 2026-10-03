#!/usr/bin/env python3
"""Sanity check: does SWE-bench Verified's own patch resolve its own tests?

The dataset ships the human fix for every instance, so replaying it should
resolve ~100%. Whatever it falls short by is a defect in this suite or in the
cluster, and it is subtracted from every model score measured afterwards --
a model cannot beat the harness it is measured with.

This runs the ordinary pipeline with ``harness.name=gold``: same two pods, same
patch extraction, same eval script, same parser. Then it prints a verdict and
exits non-zero when the rate is below ``--threshold``, so it can gate a run.

Usage:
    python scripts/gold_patch_check.py --limit 50
    python scripts/gold_patch_check.py                      # all 500
    python scripts/gold_patch_check.py --limit 20 --threshold 1.0
    python scripts/gold_patch_check.py --benchmark swe-bench-pro --limit 20
    python scripts/gold_patch_check.py --benchmark swe-bench-multilingual --limit 20
    python scripts/gold_patch_check.py --report results/gold-swebench-verified/<run_id>
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
    "swe-bench-verified": [CONFIG_DIR / "gold.yaml"],
    "swe-bench-pro": [CONFIG_DIR / "gold-pro.yaml"],
    # Layered, not substituted: `dataset.name=swe-bench-multilingual` alone
    # leaves the verified pod sizes and timeouts in place, and a cargo or
    # gradle suite on 2 CPUs / 8Gi times out instead of reporting a ceiling.
    "swe-bench-multilingual": [
        CONFIG_DIR / "gold.yaml",
        CONFIG_DIR / "swe-bench-multilingual.yaml",
    ],
}

#: The dataset overlays carry no run_name, so results would otherwise land in
#: the verified run directory.
RUN_NAMES = {"swe-bench-multilingual": "gold-swebench-multilingual"}

#: Below this, the numbers a real run produces are not worth reading.
DEFAULT_THRESHOLD = 0.98

#: How a shortfall should be read. The gold patch is correct by construction,
#: so every one of these points at the machinery, not at the fix.
DIAGNOSES = {
    "EMPTY_PATCH": "extraction lost the patch (git diff / excludes)",
    "PATCH_APPLY_FAILED": "image and dataset row disagree on the base commit",
    "EVAL_INFRA_ERROR": "eval pod or its agent failed — not a test result",
    "EVAL_TIMEOUT": "tests exceeded grading.eval_timeout",
    "EVAL_RESET_FAILED": "eval script could not reset the repo to base_commit",
    "EVAL_SCRIPT_MISSING": "SWE-bench Pro run_script.sh/parser.py unavailable",
    "infra_error": "rollout pod could not be created",
    "agent_error": "gold patch would not apply, or extraction returned nothing",
}


def summarize_failures(records) -> None:
    failures = [r for r in records if not r.resolved]
    if not failures:
        return

    print(f"\n{len(failures)} unresolved:\n")
    reasons = Counter(r.resolved_status or r.exit_status for r in failures)
    for reason, count in reasons.most_common():
        hint = DIAGNOSES.get(reason, "tests failed — inspect eval.log")
        print(f"  {count:>4}  {reason:<22} {hint}")

    print("\n  instances:")
    for record in failures[:40]:
        detail = (record.error or "").splitlines()[:1]
        print(
            f"    {record.instance_id:<40} {record.resolved_status:<20}"
            f" {detail[0][:120] if detail else ''}"
        )
    if len(failures) > 40:
        print(f"    ... and {len(failures) - 40} more")


def verdict(records, threshold: float) -> int:
    total = len(records)
    if not total:
        print("No records — nothing was run.")
        return 1

    resolved = sum(1 for r in records if r.resolved)
    rate = resolved / total

    print("\n" + "=" * 62)
    print(f"  gold patch resolve rate   {resolved}/{total} = {rate * 100:.2f}%")
    print("=" * 62)

    summarize_failures(records)

    if rate >= threshold:
        print(f"\nPASS — at or above the {threshold * 100:.0f}% floor.")
        return 0

    print(
        f"\nFAIL — below the {threshold * 100:.0f}% floor. Every point missing "
        "here is a point no model in this suite can score. Start with "
        "`python scripts/regrade.py <run_dir>`, which re-derives each verdict "
        "from the saved eval.log and says whether the wrapper or the "
        "environment is at fault."
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
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Minimum acceptable resolve rate (default {DEFAULT_THRESHOLD})",
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
        return verdict(list(records.values()), args.threshold)

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
    # The harness is not overridable here: a gold check that ran something else
    # would report a number that means nothing.
    overrides.append("harness.name=gold")

    config: RunConfig = load_config(
        [str(path) for path in CONFIGS[args.benchmark]], overrides
    )
    print(f"Gold-patch check -> {config.run_dir}\n")

    records = asyncio.run(run_eval(config))
    code = verdict(records, args.threshold)

    print(f"\n  summary   {config.run_dir / 'summary.json'}")
    print(f"  logs      {config.instances_dir}/<instance_id>/eval.log")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
