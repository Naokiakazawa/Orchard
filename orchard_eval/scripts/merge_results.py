#!/usr/bin/env python3
"""Fold a targeted re-run back into the run it was carved out of.

``resume`` only re-runs instances recorded as ``infra_error``
(``runner.py:load_existing_records`` plus ``retry_infra_on_resume``), which is
deliberate — an agent that failed is a result. But an *environment* fault the
suite misclassified as an agent failure is not, and those records have to be
replaced rather than retried in place. The recipe is:

    # 1. list the instances to redo
    python scripts/merge_results.py --select "cannot execute in this image" \\
        results/<TAG>/pi-swebench-pro/<RUN_ID> > /tmp/redo.txt

    # 2. redo just those, into a run of their own
    orchard-eval run -c configs/pi.yaml -c configs/swebench-pro.yaml \\
        --output-dir results/<TAG> --run-name pi-swebench-pro-rerun \\
        --instance-id "$(paste -sd, /tmp/redo.txt)" --no-resume

    # 3. fold the redo over the original
    python scripts/merge_results.py results/<TAG>/pi-swebench-pro/<RUN_ID> \\
        --overlay results/<TAG>/pi-swebench-pro-rerun/<RUN_ID>

Later records win per ``instance_id``. The merged run is written to a new
directory by default so the original stays intact and the two remain
comparable; ``--in-place`` overwrites instead.

``summary.json``, ``report.txt`` and ``preds.json`` are regenerated with the
same functions ``orchard-eval run`` and ``orchard-eval report`` use, so the
merged artifacts are indistinguishable in shape from a single clean run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from orchard_evalkit.models import InstanceRecord
from orchard_evalkit.report import (
    format_report,
    format_summary,
    summarize,
    write_predictions,
    write_report,
    write_summary,
)
from orchard_evalkit.runner import load_existing_records


def _load(run_dir: Path) -> dict[str, InstanceRecord]:
    records = load_existing_records(run_dir / "results.jsonl")
    if not records:
        raise SystemExit(f"No records in {run_dir / 'results.jsonl'}")
    return records


def _select(records: dict[str, InstanceRecord], needle: str) -> list[str]:
    """Instance ids whose recorded error contains ``needle``."""
    return sorted(
        instance_id
        for instance_id, record in records.items()
        if needle.lower() in str(record.error or "").lower()
    )


def _write(run_dir: Path, records: list[InstanceRecord], source: Path) -> None:
    """Write the merged run, mirroring what ``EvalRunner`` leaves behind."""
    run_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda r: r.instance_id)

    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for record in ordered:
            handle.write(record.model_dump_json() + "\n")

    # summarize() recovers run_name/harness/model from the records, but not
    # dataset/benchmark — those only ever came from the config. Carry them from
    # the run being merged into, so the merged summary stays quotable.
    previous = source / "summary.json"
    benchmark, dataset = "", ""
    if previous.exists():
        old = json.loads(previous.read_text(encoding="utf-8"))
        benchmark, dataset = old.get("benchmark", ""), old.get("dataset", "")

    summary = summarize(ordered, benchmark=benchmark)
    summary.dataset = dataset
    write_summary(summary, run_dir / "summary.json")
    write_predictions(ordered, run_dir / "preds.json")
    write_report(format_report(summary, ordered), run_dir / "report.txt")
    print(format_summary(summary))
    print(f"  merged run written to {run_dir}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir", type=Path, help="results/<name>/<run_id> holding results.jsonl"
    )
    parser.add_argument(
        "--overlay",
        type=Path,
        action="append",
        default=[],
        help="A re-run directory to fold over it. Repeatable; later wins.",
    )
    parser.add_argument(
        "--select",
        help=(
            "Instead of merging, print the instance ids whose error contains "
            "this text — the input for `orchard-eval run --instance-id`."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Where to write the merged run (default: <run_dir>-merged).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite run_dir instead of writing a new directory.",
    )
    args = parser.parse_args()

    base = _load(args.run_dir)

    if args.select:
        selected = _select(base, args.select)
        print("\n".join(selected))
        # Nothing matched is worth an exit code: it usually means the run was
        # fine, or that the wording moved.
        return 0 if selected else 1

    if not args.overlay:
        raise SystemExit("Pass --overlay DIR at least once, or use --select")

    replaced, added = 0, 0
    for overlay_dir in args.overlay:
        for instance_id, record in _load(overlay_dir).items():
            if instance_id in base:
                replaced += 1
            else:
                added += 1
            base[instance_id] = record

    print(f"  {replaced} record(s) replaced, {added} added")
    out = (
        args.run_dir
        if args.in_place
        else (args.out or args.run_dir.parent / f"{args.run_dir.name}-merged")
    )
    _write(out, list(base.values()), source=args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
