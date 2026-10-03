"""Aggregating instance records into a run summary.

Two artifacts come out of a run besides the raw records:

* ``summary.json`` — the numbers you actually quote (resolve rate, exit-status
  histogram, cost, timing);
* ``preds.json`` — SWE-bench's own prediction format, so any run can be
  re-graded with the official harness independently of this suite. That
  second file is what makes a claimed score checkable by someone who does not
  trust this code.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean

from orchard_evalkit.config import RunConfig
from orchard_evalkit.datasets import detect_benchmark
from orchard_evalkit.models import InstanceRecord, RunSummary


def summarize(
    records: list[InstanceRecord],
    config: RunConfig | None = None,
    wall_clock_s: float = 0.0,
    benchmark: str = "",
) -> RunSummary:
    """Compute aggregate statistics over instance records."""
    total = len(records)
    resolved = sum(1 for r in records if r.resolved)
    exit_statuses = Counter(r.exit_status for r in records)

    def _mean(values: list[float]) -> float:
        return round(mean(values), 2) if values else 0.0

    summary = RunSummary(
        total=total,
        resolved=resolved,
        resolve_rate=round(resolved / total, 4) if total else 0.0,
        empty_patches=sum(1 for r in records if not r.patch.strip()),
        errors=sum(1 for r in records if r.error),
        missing_trajectories=sum(1 for r in records if not r.n_messages),
        length_truncated=sum(1 for r in records if r.metrics.get("length_stops")),
        mean_messages=_mean([float(r.n_messages) for r in records]),
        exit_statuses=dict(sorted(exit_statuses.items())),
        total_cost=round(
            sum(float(r.metrics.get("cost", 0.0) or 0.0) for r in records), 4
        ),
        mean_rollout_s=_mean([r.timings.rollout_s for r in records]),
        mean_grade_create_s=_mean([r.timings.grade_create_s for r in records]),
        mean_grade_s=_mean([r.timings.grade_s for r in records]),
        mean_total_s=_mean([r.timings.total_s for r in records]),
        wall_clock_s=round(wall_clock_s, 2),
    )

    if config is not None:
        summary.run_name = config.run_name
        summary.harness = config.harness.name
        summary.model = config.model.name
        summary.dataset = config.dataset.name
        # config.dataset.benchmark is often "auto"; report what it resolves to.
        summary.benchmark = benchmark or detect_benchmark(config.dataset)
    elif records:
        summary.run_name = records[0].run_name
        summary.harness = records[0].harness
        summary.model = records[0].model
        summary.benchmark = benchmark

    return summary


def write_summary(summary: RunSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2), encoding="utf-8"
    )


def write_report(text: str, path: Path) -> None:
    """Persist the human-readable console report next to the JSON artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_predictions(
    records: list[InstanceRecord], path: Path, model_name: str = ""
) -> None:
    """Write predictions in the format SWE-bench's own harness consumes."""
    payload = {
        record.instance_id: {
            "instance_id": record.instance_id,
            "model_name_or_path": model_name or record.model or "orchard-eval",
            "model_patch": record.patch,
        }
        for record in records
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def format_duration(seconds: float) -> str:
    """Render seconds as ``1h 02m 03s`` / ``2m 03s`` / ``3.4s``."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def format_summary(summary: RunSummary) -> str:
    """Render a summary as a short console table."""
    lines = [
        "",
        "=" * 62,
        f"  {summary.run_name or 'run'}",
        "=" * 62,
        f"  harness         {summary.harness}",
        f"  model           {summary.model or '<unset>'}",
        f"  dataset         {summary.dataset or '<unset>'}",
        f"  benchmark       {summary.benchmark or 'swebench'}",
        "-" * 62,
        f"  instances       {summary.total}",
        f"  resolved        {summary.resolved}",
        f"  resolve rate    {summary.resolve_rate * 100:.2f}%",
        f"  empty patches   {summary.empty_patches}",
        f"  errors          {summary.errors}",
        f"  mean messages   {summary.mean_messages:.1f}",
    ]
    if summary.missing_trajectories:
        lines.append(
            f"  ⚠ no trajectory {summary.missing_trajectories}"
            f"  (of {summary.total})"
        )
    if summary.length_truncated:
        lines.append(
            f"  ⚠ truncated     {summary.length_truncated}"
            f"  (hit the output-token cap mid-message)"
        )
    if summary.total_cost:
        lines.append(f"  total cost      ${summary.total_cost:.2f}")
    lines += [
        "-" * 62,
        f"  mean rollout    {summary.mean_rollout_s:.1f}s",
        f"  mean eval pod   {summary.mean_grade_create_s:.1f}s",
        f"  mean grading    {summary.mean_grade_s:.1f}s",
        f"  mean total      {summary.mean_total_s:.1f}s",
    ]
    if summary.exit_statuses:
        lines += ["-" * 62, "  exit statuses"]
        for status, count in summary.exit_statuses.items():
            lines.append(f"    {status:<28} {count}")
    if summary.wall_clock_s:
        lines += [
            "-" * 62,
            f"  total eval time {format_duration(summary.wall_clock_s)}",
        ]
    lines += ["=" * 62, ""]
    return "\n".join(lines)


def format_instances(records: list[InstanceRecord]) -> str:
    """Render the per-instance breakdown: verdict, phase timings, cost, error."""
    if not records:
        return ""

    ordered = sorted(records, key=lambda r: r.instance_id)
    id_width = max(len("instance"), max(len(r.instance_id) for r in ordered))
    status_width = max(len("exit"), max(len(r.exit_status) for r in ordered))

    header = (
        f"  {'instance':<{id_width}}  {'ok':<3}  {'exit':<{status_width}}  "
        f"{'msgs':>5}  {'create':>8}  {'setup':>8}  {'rollout':>8}  "
        f"{'evalpod':>8}  {'grade':>8}  {'total':>8}  {'cost':>8}  {'try':>3}"
    )
    lines = ["per-instance results", "-" * len(header), header, "-" * len(header)]

    for record in ordered:
        t = record.timings
        cost = float(record.metrics.get("cost", 0.0) or 0.0)
        lines.append(
            f"  {record.instance_id:<{id_width}}  "
            f"{'yes' if record.resolved else 'no':<3}  "
            f"{record.exit_status:<{status_width}}  "
            f"{record.n_messages:>5}  "
            f"{t.create_s:>7.1f}s  {t.setup_s:>7.1f}s  {t.rollout_s:>7.1f}s  "
            f"{t.grade_create_s:>7.1f}s  {t.grade_s:>7.1f}s  {t.total_s:>7.1f}s  "
            f"{('$' + format(cost, '.4f')) if cost else '-':>8}  "
            f"{record.rollout_attempts:>3}"
        )
    lines.append("-" * len(header))

    failures = [r for r in ordered if r.error]
    if failures:
        lines += ["", "errors", "-" * len(header)]
        for record in failures:
            lines.append(f"  {record.instance_id}  [{record.exit_status}]")
            for line in str(record.error).splitlines() or [""]:
                lines.append(f"      {line}")
            lines.append("")

    empty = [r for r in ordered if not r.patch.strip()]
    if empty:
        lines += ["empty patches", "-" * len(header)]
        lines += [f"  {r.instance_id}  [{r.exit_status}]" for r in empty]
        lines.append("")

    return "\n".join(lines)


def format_report(summary: RunSummary, records: list[InstanceRecord]) -> str:
    """Full report: the console summary plus the per-instance breakdown."""
    return f"{format_summary(summary)}\n{format_instances(records)}"


__all__ = [
    "format_duration",
    "format_instances",
    "format_report",
    "format_summary",
    "summarize",
    "write_predictions",
    "write_report",
    "write_summary",
]
