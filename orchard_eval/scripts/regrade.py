#!/usr/bin/env python3
"""Re-grade a finished run from its saved logs, using swebench directly.

The official ``run_evaluation`` CLI needs a Docker daemon, which a Kubernetes
worker does not have. This calls the same grading functions swebench itself
calls -- ``get_logs_eval`` and ``get_eval_report`` -- against the ``eval.log``
each instance already wrote, so:

* a disagreement with ``results.jsonl`` means our grading wrapper is at fault;
* agreement means the zero came from the patch or from the environment, and the
  diagnosis column says which.

Usage:
    python scripts/regrade.py results/smoke-mini-swe-agent
    python scripts/regrade.py results/smoke-mini-swe-agent --show astropy__astropy-12907
    python scripts/regrade.py results/gold-swebench-pro/<run_id> --dataset swe-bench-pro

SWE-bench Pro has no ``swebench`` package to check against: its verdict is a
set comparison over the ``output.json`` its own per-instance parser produced.
Re-deriving that from the saved artifact still catches the failure that matters
— a run whose recorded score does not follow from the test results it captured.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path
from typing import Any

from orchard_evalkit.config import DatasetConfig
from orchard_evalkit.datasets import (
    BENCHMARK_SWEBENCH_PRO,
    detect_benchmark,
    resolve_dataset_name,
)
from orchard_evalkit.datasets.swebench import load_records
from orchard_evalkit.datasets.swebench_pro import (
    COL_FAIL_TO_PASS,
    COL_PASS_TO_PASS,
    parse_string_list,
)
from orchard_evalkit.grading.swebench import build_test_spec
from orchard_evalkit.grading.swebench_pro import grade_from_output

#: Symptoms that mean the sandbox was broken rather than the patch being wrong.
ENV_SYMPTOMS = (
    ("import failure", r"ModuleNotFoundError|ImportError"),
    ("collection error", r"ERROR collecting|errors? during collection"),
    ("missing C extension", r"cannot import name '_|undefined symbol|\.so: cannot open"),
    ("conda missing", r"conda: command not found|EnvironmentNameNotFound"),
    ("no such file", r"No such file or directory"),
)


def official_grade(
    test_spec: Any, instance_id: str, patch: str, log_text: str
) -> tuple[dict[str, str], bool, dict[str, Any]]:
    from swebench.harness.grading import get_eval_report, get_logs_eval

    with tempfile.NamedTemporaryFile(
        "w", suffix=".log", delete=False, encoding="utf-8"
    ) as handle:
        handle.write(log_text)
        path = handle.name
    try:
        status_map, found = get_logs_eval(test_spec, path)
        prediction = {
            "instance_id": instance_id,
            "model_name_or_path": "orchard-eval",
            "model_patch": patch,
        }
        report = get_eval_report(test_spec, prediction, path, True).get(instance_id, {})
        return dict(status_map or {}), found, report
    finally:
        Path(path).unlink(missing_ok=True)


def marker_diagnosis(test_spec: Any, log_text: str) -> str:
    """Say whether the test output actually sits between swebench's markers."""
    from swebench.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT
    from swebench.harness.log_parsers import PARSER_REGISTRY

    if START_TEST_OUTPUT not in log_text or END_TEST_OUTPUT not in log_text:
        return "markers absent"
    parser = PARSER_REGISTRY[test_spec.log_parser]
    sliced = log_text.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]
    if parser(sliced, test_spec):
        return ""
    if parser(log_text, test_spec):
        return "results OUTSIDE markers (stream ordering)"
    return "no results anywhere"


def env_diagnosis(log_text: str) -> str:
    hits = [
        f"{label} x{len(re.findall(pattern, log_text))}"
        for label, pattern in ENV_SYMPTOMS
        if re.search(pattern, log_text)
    ]
    return "; ".join(hits)


def load_our_records(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "results.jsonl"
    if not path.exists():
        raise SystemExit(f"no results at {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def regrade_pro(
    run_dir: Path,
    records: list[dict[str, Any]],
    rows: dict[str, dict[str, Any]],
    show: str | None,
) -> int:
    """Re-derive Pro verdicts from each instance's saved ``output.json``."""
    print(f"{'instance':<60}{'ours':<7}{'rederived':<11}{'f2p':<9}{'p2p':<9}note")
    print("-" * 120)

    agree = disagree = 0
    ours_resolved = rederived_resolved = 0

    for record in sorted(records, key=lambda r: r["instance_id"]):
        instance_id = record["instance_id"]
        mine = bool(record.get("resolved"))
        ours_resolved += int(mine)
        output_path = run_dir / "instances" / instance_id / "output.json"
        row = rows.get(instance_id)

        if row is None:
            note = "not in dataset"
        elif not output_path.exists():
            note = (
                record.get("resolved_status")
                or record.get("error")
                or "no output.json"
            )
        else:
            note = ""

        if note:
            print(
                f"{instance_id:<60}{str(mine):<7}{'n/a':<11}{'-':<9}{'-':<9}{note}"
            )
            continue

        assert row is not None
        output = json.loads(output_path.read_text(encoding="utf-8"))
        # The saved test log is what tells "the suite never ran" apart from
        # "the suite ran and nothing passed", so hand it over when it is there.
        log_path = output_path.with_name("tests_stdout.log")
        stdout_tail = (
            log_path.read_text(encoding="utf-8", errors="replace")
            if log_path.exists()
            else ""
        )
        grade = grade_from_output(
            output,
            parse_string_list(row.get(COL_FAIL_TO_PASS)),
            parse_string_list(row.get(COL_PASS_TO_PASS)),
            stdout_tail=stdout_tail,
        )
        theirs = grade.resolved
        rederived_resolved += int(theirs)
        agree, disagree = (
            (agree + 1, disagree) if mine == theirs else (agree, disagree + 1)
        )

        f2p = grade.tests_status["FAIL_TO_PASS"]
        p2p = grade.tests_status["PASS_TO_PASS"]
        f2p_cell = f"{len(f2p['success'])}/{len(f2p['success']) + len(f2p['failure'])}"
        p2p_cell = f"{len(p2p['success'])}/{len(p2p['success']) + len(p2p['failure'])}"
        print(
            f"{instance_id:<60}{str(mine):<7}{str(theirs):<11}"
            f"{f2p_cell:<9}{p2p_cell:<9}{grade.error or ''}"
        )

        if show == instance_id:
            print("\n--- tests the parser reported " + "-" * 60)
            for name, status in sorted(grade.status_map.items()):
                print(f"  {status:<9}{name}")
            print("--- missing FAIL_TO_PASS " + "-" * 65)
            for name in f2p["failure"]:
                print(f"  {name}")
            print("--- end " + "-" * 82 + "\n")

    total = len(records)
    print("-" * 120)
    print(f"ours:      {ours_resolved}/{total} resolved")
    print(f"rederived: {rederived_resolved}/{total} resolved")
    print(f"agree on {agree}, disagree on {disagree}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="results/<run_name>/<run_id>, the directory holding results.jsonl",
    )
    parser.add_argument("--dataset", default="swe-bench-verified")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--revision",
        default="",
        help=(
            "Dataset git revision (tag/branch/sha). Pass the same value the run "
            "used — for SWE-bench Pro that is `v1.0`, since the default branch "
            "now serves V2 and would regrade against different fail_to_pass sets."
        ),
    )
    parser.add_argument("--benchmark", default="auto")
    parser.add_argument("--show", help="dump the parsed region for one instance")
    args = parser.parse_args()

    records = load_our_records(args.run_dir)
    wanted = {r["instance_id"] for r in records}
    dataset = DatasetConfig(
        name=args.dataset,
        split=args.split,
        benchmark=args.benchmark,
        revision=args.revision,
    )
    all_rows = load_records(
        dataset.model_copy(update={"name": resolve_dataset_name(dataset.name)})
    )
    rows = {
        row["instance_id"]: row for row in all_rows if row["instance_id"] in wanted
    }

    if detect_benchmark(dataset, all_rows) == BENCHMARK_SWEBENCH_PRO:
        return regrade_pro(args.run_dir, records, rows, args.show)

    print(f"{'instance':<34}{'ours':<7}{'swebench':<10}{'f2p':<9}{'p2p':<9}note")
    print("-" * 110)

    agree = disagree = 0
    ours_resolved = official_resolved = 0

    for record in sorted(records, key=lambda r: r["instance_id"]):
        instance_id = record["instance_id"]
        log_path = args.run_dir / "instances" / instance_id / "eval.log"
        mine = bool(record.get("resolved"))
        ours_resolved += int(mine)

        if not log_path.exists():
            note = record.get("resolved_status") or record.get("error") or "no eval.log"
            print(f"{instance_id:<34}{str(mine):<7}{'n/a':<10}{'-':<9}{'-':<9}{note}")
            continue

        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        row = rows.get(instance_id)
        if row is None:
            print(f"{instance_id:<34}{str(mine):<7}{'?':<10}{'-':<9}{'-':<9}not in dataset")
            continue

        test_spec = build_test_spec(row)
        status_map, found, report = official_grade(
            test_spec, instance_id, record.get("patch", ""), log_text
        )
        theirs = bool(report.get("resolved"))
        official_resolved += int(theirs)
        agree, disagree = (agree + 1, disagree) if mine == theirs else (agree, disagree + 1)

        tests = report.get("tests_status", {})
        f2p = tests.get("FAIL_TO_PASS", {"success": [], "failure": []})
        p2p = tests.get("PASS_TO_PASS", {"success": [], "failure": []})
        f2p_cell = f"{len(f2p['success'])}/{len(f2p['success']) + len(f2p['failure'])}"
        p2p_cell = f"{len(p2p['success'])}/{len(p2p['success']) + len(p2p['failure'])}"

        notes = [n for n in (marker_diagnosis(test_spec, log_text),) if n]
        if not found:
            notes.append("swebench found no parseable output")
        if not status_map:
            notes.append("empty status map")
        if reason := report.get("infra_failure_reason"):
            notes.append(f"infra: {reason}")
        if symptoms := env_diagnosis(log_text):
            notes.append(symptoms)

        print(
            f"{instance_id:<34}{str(mine):<7}{str(theirs):<10}"
            f"{f2p_cell:<9}{p2p_cell:<9}{'; '.join(notes)}"
        )

        if args.show == instance_id:
            from swebench.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT

            print("\n--- region swebench parses " + "-" * 60)
            if START_TEST_OUTPUT in log_text and END_TEST_OUTPUT in log_text:
                region = log_text.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]
            else:
                region = "(markers absent; showing tail)\n" + log_text[-4000:]
            print(region[:8000])
            print("--- end " + "-" * 74 + "\n")

    total = len(records)
    print("-" * 110)
    print(f"ours:     {ours_resolved}/{total} resolved")
    print(f"swebench: {official_resolved}/{total} resolved")
    print(f"agree on {agree}, disagree on {disagree}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
