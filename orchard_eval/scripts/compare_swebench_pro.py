#!/usr/bin/env python3
"""Pair a SWE-bench Pro run against the same benchmark run through Harbor.

The suite can measure SWE-bench Pro two ways, and they share almost nothing
below the dataset: ``orchard-eval run`` owns the rollout, extracts a ``git
diff`` and grades it in a *fresh* pod, while ``orchard-eval harbor`` hands the
whole trial to Harbor, whose Pro tasks grade in the *same* container the agent
worked in. Two independent implementations agreeing is evidence; one number on
its own is not.

    python scripts/compare_swebench_pro.py \\
        results/<tag>/mini-swe-agent-swebench-pro \\
        results/<tag>/harbor/mini-swe-agent-swebench-pro-harbor

Either path may be a run directory or the file/directory inside it — a run
directory resolves to its newest attempt, exactly as ``results_table.py`` does.

What the output is for: the headline rates say *whether* the paths agree, and
the per-instance breakdown says *where* they do not. A one-sided disagreement
that is entirely "harbor only" is the expected shape — Harbor's verifier sees
everything the agent left in the container — and one that is spread across both
directions is not, and is worth opening.

Instance identity is the join key. A Harbor task is named ``scale-ai/<id>``
where ``<id>`` is the dataset's ``instance_id`` verbatim, so the mapping is a
prefix strip and nothing here has to know the naming scheme.

To make the comparison exact on a slice rather than on the full 731, run one
path first and pin the other to the instances it actually finished:

    python scripts/compare_swebench_pro.py --emit-harbor-tasks \\
        results/<tag>/mini-swe-agent-swebench-pro
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchard_evalkit.harbor_bridge import collect_outcomes  # noqa: E402

#: Harbor prefixes every task in this dataset with its publishing org.
HARBOR_ORG = "scale-ai"

#: Disagreements printed before the list is truncated. ``--list`` removes it.
DEFAULT_LIST_LIMIT = 20


def instance_id_of(task_name: str) -> str:
    """``scale-ai/instance_ansible__ansible-abc-vdef`` -> ``instance_...``.

    Splitting on the last ``/`` rather than stripping a literal ``scale-ai/``
    keeps this working if the dataset is ever republished under another org, or
    if a future Harbor drops the prefix from ``task_name`` altogether.
    """
    return task_name.rsplit("/", 1)[-1]


def join_key(instance_id: str) -> str:
    """The identity two paths are paired on, case-folded.

    Harbor lower-cases a task name when it publishes it, and the dataset does
    not: ``instance_NodeBB__NodeBB-...`` arrives from Harbor as
    ``instance_nodebb__nodebb-...``. Joining on the raw id silently drops every
    one of NodeBB's 44 instances from the comparison — a sixth of the
    disagreements — and reports the loss as "both paths ran 687 instances",
    which reads like coverage rather than a bug.
    """
    return instance_id.lower()


def newest_attempt(path: Path) -> Path:
    """Resolve a run directory to the attempt its results live in.

    ``orchard-eval run`` writes ``<run-name>/<run-id>/results.jsonl`` and run
    ids are ``YYYYmmdd-HHMMSS``, so the last one sorted is the newest. Passing
    the attempt directory or the file itself is also accepted, because that is
    what a path copied out of a log looks like.
    """
    if path.is_file():
        return path
    candidate = path / "results.jsonl"
    if candidate.is_file():
        return candidate
    attempts = sorted(p for p in path.iterdir() if p.is_dir()) if path.is_dir() else []
    for attempt in reversed(attempts):
        if (attempt / "results.jsonl").is_file():
            return attempt / "results.jsonl"
    raise SystemExit(f"No results.jsonl under {path}")


def read_run(path: Path) -> dict[str, bool]:
    """``instance_id -> resolved`` from an ``orchard-eval run`` results file.

    A resumed run appends rather than rewrites, so an instance can appear more
    than once; the last record wins, which is the rule ``load_existing_records``
    applies when deciding what to skip. A half-written final line is skipped
    rather than fatal, so this can be read while the run is still going.
    """
    resolved: dict[str, bool] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            instance_id = row.get("instance_id")
            if instance_id:
                resolved[str(instance_id)] = bool(row.get("resolved"))
    return resolved


def read_harbor(path: Path) -> tuple[dict[str, bool], int]:
    """``instance_id -> solved`` from a Harbor job directory, plus trial count.

    A task solved by *any* of its trials counts as solved, which is pass@k and
    is only distinguishable from pass@1 when the job was run with ``--attempts
    > 1``. The trial count is returned so the caller can say which was measured
    instead of leaving it implied.
    """
    outcomes = collect_outcomes(path)
    if not outcomes:
        raise SystemExit(
            f"No trial results under {path}\n"
            "  Harbor writes one result.json per trial; point at the job "
            "directory, e.g. results/<tag>/harbor/<job-name>."
        )
    solved: dict[str, bool] = {}
    for outcome in outcomes:
        instance_id = instance_id_of(outcome.task)
        solved[instance_id] = solved.get(instance_id, False) or outcome.solved
    return solved, len(outcomes)


def compare(run: dict[str, bool], harbor: dict[str, bool]) -> dict:
    """Cross-tabulate the two paths over the instances both of them ran.

    Paired on :func:`join_key`, but reported under the run's own spelling of
    the id — that is what ``results.jsonl`` and the ``instances/`` directories
    are named after, so it is the string a reader can grep for.
    """
    run_ids = {join_key(i): i for i in run}
    harbor_ids = {join_key(i): i for i in harbor}
    keys = sorted(set(run_ids) & set(harbor_ids))
    name = {key: run_ids[key] for key in keys}
    shared = [name[key] for key in keys]
    verdicts = {
        name[key]: (run[run_ids[key]], harbor[harbor_ids[key]]) for key in keys
    }
    both = [i for i in shared if all(verdicts[i])]
    neither = [i for i in shared if not any(verdicts[i])]
    run_only = [i for i in shared if verdicts[i] == (True, False)]
    harbor_only = [i for i in shared if verdicts[i] == (False, True)]
    return {
        "shared": shared,
        "both": both,
        "neither": neither,
        "run_only": run_only,
        "harbor_only": harbor_only,
        "agreement": (len(both) + len(neither)) / len(shared) if shared else 0.0,
        "run_only_in_run": sorted(run_ids[k] for k in set(run_ids) - set(harbor_ids)),
        "run_only_in_harbor": sorted(
            harbor_ids[k] for k in set(harbor_ids) - set(run_ids)
        ),
        # (run, harbor) per shared instance, so the report can score the
        # shared subset without redoing the join.
        "verdicts": verdicts,
    }


def _rate(solved: int, total: int) -> str:
    return f"{solved:>4}/{total:<4} {solved / total:>7.2%}" if total else "   -"


def format_report(
    run: dict[str, bool],
    harbor: dict[str, bool],
    result: dict,
    *,
    run_label: str,
    harbor_label: str,
    trials: int,
    list_all: bool,
) -> str:
    shared = result["shared"]
    scored = [result["verdicts"][i] for i in shared]
    lines = [
        "",
        "  SWE-bench Pro, measured twice",
        "  " + "-" * 72,
        f"  {'orchard-eval run':<22} {_rate(sum(run.values()), len(run))}"
        "   graded in a fresh pod",
        f"  {'orchard-eval harbor':<22} {_rate(sum(harbor.values()), len(harbor))}"
        "   graded in the agent's pod",
        "",
        f"  run:     {run_label}",
        f"  harbor:  {harbor_label}  ({trials} trials)",
        "",
    ]

    if not shared:
        lines += [
            "  The two runs share no instances, so there is nothing to compare.",
            "  Pin the second path to what the first one ran:",
            "    python scripts/compare_swebench_pro.py --emit-harbor-tasks "
            f"{run_label}",
            "",
        ]
        return "\n".join(lines)

    lines += [
        f"  Both paths ran {len(shared)} instances.",
        "  " + "-" * 72,
        f"  {'restricted to those:':<22} {'run':<7}"
        f"{_rate(sum(ran for ran, _ in scored), len(shared))}",
        f"  {'':<22} {'harbor':<7}"
        f"{_rate(sum(hit for _, hit in scored), len(shared))}",
        "",
        f"  {len(result['both']):>5}  both resolved",
        f"  {len(result['neither']):>5}  both unresolved",
        f"  {len(result['run_only']):>5}  run only      "
        "(the fresh-pod grader passed it, Harbor's did not)",
        f"  {len(result['harbor_only']):>5}  harbor only   "
        "(Harbor passed it, the fresh-pod grader did not)",
        "",
        f"  agreement: {result['agreement']:.2%}",
    ]

    lines += _format_disagreements(result, list_all=list_all)
    lines += _format_coverage(result)
    return "\n".join(lines) + "\n"


def _format_disagreements(result: dict, *, list_all: bool) -> list[str]:
    """The instances to actually open, grouped by which path scored them."""
    lines: list[str] = []
    groups = (
        (
            "harbor only",
            result["harbor_only"],
            "expected where the agent left state behind — an installed "
            "package,\n           an edited test file — that the fresh pod "
            "never sees",
        ),
        (
            "run only",
            result["run_only"],
            "harder to explain: the extracted patch scored where the agent's "
            "own\n           container did not. Read the trial's verifier "
            "log first",
        ),
    )
    for label, items, why in groups:
        if not items:
            continue
        lines += ["", f"  {label}  ({len(items)}) — {why}"]
        shown = items if list_all else items[:DEFAULT_LIST_LIMIT]
        lines += [f"      {item}" for item in shown]
        if len(shown) < len(items):
            lines.append(f"      ... and {len(items) - len(shown)} more (--list)")
    return lines


def _format_coverage(result: dict) -> list[str]:
    """Instances only one path ran at all, which no rate above accounts for."""
    only_run = result["run_only_in_run"]
    only_harbor = result["run_only_in_harbor"]
    if not only_run and not only_harbor:
        return []
    lines = ["", "  Not comparable — only one path ran these:"]
    if only_run:
        lines.append(f"      {len(only_run):>5}  in the run only")
    if only_harbor:
        lines.append(f"      {len(only_harbor):>5}  in the harbor job only")
    return lines


def emit_harbor_tasks(run: dict[str, bool]) -> str:
    """``--task`` arguments pinning a Harbor job to what a run already did.

    Harbor filters with ``fnmatch`` over the full ``org/name``, so the exact
    name matches exactly one task and needs no escaping.
    """
    return " \\\n".join(
        # Harbor's own spelling of the name, which is the id lower-cased;
        # see `join_key`. fnmatch is case-sensitive, so the dataset's
        # `instance_NodeBB__NodeBB-...` would pin the job to nothing.
        f"  --task {HARBOR_ORG}/{join_key(instance_id)}"
        for instance_id in sorted(run)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare a SWE-bench Pro run with the same benchmark run "
        "through Harbor.",
    )
    parser.add_argument(
        "run",
        help="An `orchard-eval run` directory, attempt directory, or file",
    )
    parser.add_argument(
        "harbor",
        nargs="?",
        help="A Harbor job directory, e.g. results/<tag>/harbor/<job-name>",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help=f"Print every disagreement, not the first {DEFAULT_LIST_LIMIT}",
    )
    parser.add_argument(
        "--emit-harbor-tasks",
        action="store_true",
        help="Print `--task` arguments for the instances the run covered, so "
        "the Harbor job can be pinned to exactly those, and exit",
    )
    parser.add_argument("--json", help="Also write the pairing to this path")
    args = parser.parse_args(argv)

    run_path = newest_attempt(Path(args.run))
    run = read_run(run_path)
    if not run:
        raise SystemExit(f"No records in {run_path}")

    if args.emit_harbor_tasks:
        print(emit_harbor_tasks(run))
        return 0

    if not args.harbor:
        parser.error(
            "the harbor job directory is required unless --emit-harbor-tasks"
        )

    harbor_path = Path(args.harbor)
    harbor, trials = read_harbor(harbor_path)
    result = compare(run, harbor)

    print(
        format_report(
            run,
            harbor,
            result,
            run_label=str(run_path),
            harbor_label=str(harbor_path),
            trials=trials,
            list_all=args.list,
        )
    )

    if args.json:
        payload = {
            "run": {"path": str(run_path), "resolved": run},
            "harbor": {"path": str(harbor_path), "trials": trials, "resolved": harbor},
            **{
                key: value
                for key, value in result.items()
                if key not in ("shared", "verdicts")
            },
            "shared": result["shared"],
        }
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  pairing saved to {args.json}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
