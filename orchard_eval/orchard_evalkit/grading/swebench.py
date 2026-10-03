"""SWE-bench grading, run in a fresh sandbox the agent never had access to.

Grading uses the **official** ``swebench`` package to build the evaluation
script and to parse its output, so a resolve rate produced here means the same
thing as one produced by the upstream harness. What differs is only *where* the
script runs: in an Orchard Env pod started from the instance image, instead of
in a local Docker container started from the same image.

The runner hands this function a pod created *after* the agent's pod was
deleted, mirroring upstream ``run_evaluation``: the environment a patch is
scored in is built from the image and nothing else, so a rollout cannot
influence its own grade through anything git does not track — an extra ``pip
install``, a warmed cache, a stray file outside the repository.

The sequence per instance is the standard one:

1. reset the repository to ``base_commit`` — cheap insurance on a fresh pod, and
   the guarantee that a patch which does not apply scores zero rather than
   silently benefiting from leftover working-tree state;
2. apply the candidate patch;
3. run the instance's evaluation script, which restores the official test files,
   applies the test patch, and runs the FAIL_TO_PASS / PASS_TO_PASS commands;
4. parse the log and compute the resolution status.

``swebench`` reorganized substantially at 5.0: ``make_test_spec`` moved, and the
evaluation script became a *dataset column* rather than something synthesized
from ``repo``/``version`` constants. Both layouts are supported here — see
:func:`build_test_spec` — because pinning users to one release of a benchmark
harness is not a reasonable thing for an evaluation suite to do.
"""

from __future__ import annotations

import inspect
import logging
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from orchard_evalkit.models import GradeResult, TaskInstance
from orchard_evalkit.sandbox import EvalSandbox, SandboxGoneError, is_transport_error

logger = logging.getLogger(__name__)

#: Where the generated evaluation script is staged inside the sandbox.
EVAL_SCRIPT_PATH = "/tmp/orchard_eval_run_tests.sh"

#: Path SWE-bench bakes into its images and eval scripts.
CANONICAL_WORKDIR = "/testbed"

#: Delimiters the eval script prints around the test command's output. Mirrors
#: swebench's constants of the same name, which have been stable across releases.
START_TEST_OUTPUT = ">>>>> Start Test Output"
END_TEST_OUTPUT = ">>>>> End Test Output"

#: Markers swebench's own harness prints when a step before the tests failed.
#: ``get_logs_eval`` refuses to parse a log containing one, and reports that
#: the same way it reports a log with no test output at all — so they have to
#: be told apart here or every one of them looks like a failed test run.
SWEBENCH_FAILURE_MARKERS: tuple[tuple[str, str, str], ...] = (
    (
        ">>>>> Patch Apply Failed",
        "PATCH_APPLY_FAILED",
        "the eval script could not apply the test patch",
    ),
    (
        ">>>>> Reset Failed",
        "EVAL_RESET_FAILED",
        "the eval script could not reset the repository to base_commit",
    ),
    (
        ">>>>> Tests Timed Out",
        "EVAL_TIMEOUT",
        "the test command timed out inside the eval script",
    ),
    (
        ">>>>> Tests Errored",
        "RESOLVED_NO",
        "the test command errored before producing results",
    ),
)

# Prediction dict keys. These are the literal values of swebench's ``KEY_*``
# constants, which moved between releases while the strings never changed.
PRED_INSTANCE_ID = "instance_id"
PRED_MODEL = "model_name_or_path"
PRED_PATCH = "model_patch"

#: Columns swebench >= 5.0 expects to find on the dataset row itself.
MODERN_ROW_COLUMNS = ("eval_script", "log_parser", "eval_type")


def _require_swebench() -> None:
    try:
        import swebench  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "Grading needs the `swebench` package. Install it with: "
            "pip install -e 'orchard_eval[swebench]' — or disable grading with "
            "grading.enabled=false to collect patches and trajectories only."
        ) from exc


def resolve_make_test_spec() -> tuple[Callable[..., Any], bool]:
    """Locate ``make_test_spec``, reporting whether it is the >= 5.0 flavour."""
    _require_swebench()
    try:
        # swebench >= 5.0
        from swebench.harness.utils import make_test_spec

        return make_test_spec, True
    except ImportError:
        pass
    # swebench 3.x / 4.x
    from swebench.harness.test_spec.test_spec import make_test_spec

    return make_test_spec, False


def build_test_spec(record: dict[str, Any], image: str = "") -> Any:
    """Build a swebench ``TestSpec`` from a dataset row.

    Args:
        record: The raw dataset row, exactly as loaded.
        image: Image resolved by the suite, injected when the row lacks one
            (swebench >= 5.0 requires it; older datasets do not carry it).
    """
    make_test_spec, is_modern = resolve_make_test_spec()

    if is_modern:
        missing = [c for c in MODERN_ROW_COLUMNS if c not in record]
        if missing:
            raise RuntimeError(
                f"swebench >= 5.0 needs the columns {missing} on each dataset "
                "row, which the legacy princeton-nlp datasets do not carry. Use "
                "the modern dataset (dataset.name=verified, i.e. "
                "SWE-bench/SWE-bench_Verified) or pin swebench < 5."
            )
        row = dict(record)
        if not row.get("image"):
            row["image"] = image
        return make_test_spec(row)

    # Older releases synthesize the script and accept an architecture.
    kwargs: dict[str, Any] = {}
    if "arch" in inspect.signature(make_test_spec).parameters:
        kwargs["arch"] = "x86_64"
    return make_test_spec(record, **kwargs)


def build_eval_script(
    record: dict[str, Any], workdir: str, image: str = ""
) -> tuple[Any, str]:
    """Return ``(test_spec, eval_script)``, retargeted at ``workdir``."""
    test_spec = build_test_spec(record, image)
    script = test_spec.eval_script
    if workdir != CANONICAL_WORKDIR:
        script = script.replace(CANONICAL_WORKDIR, workdir)
    return test_spec, script


def diagnose_unparsable_log(
    log_text: str, exit_code: int | None = None
) -> tuple[str, str]:
    """Return ``(resolved_status, error)`` for a log swebench declined to parse.

    ``get_logs_eval`` collapses four very different situations into one boolean,
    and three of them are the machinery failing rather than the patch. Naming
    them apart is what keeps a broken pod from being counted as a wrong answer
    — ``EVAL_INFRA_ERROR`` is retried on a fresh pod by the runner.
    """
    for marker, status, explanation in SWEBENCH_FAILURE_MARKERS:
        if marker in log_text:
            return status, f"{explanation} ({marker})"

    stripped = log_text.strip()
    if not stripped:
        return (
            "EVAL_INFRA_ERROR",
            "eval script produced no output at all (exit="
            f"{exit_code}) — the command was killed or the pod's agent stopped "
            "responding",
        )

    if exit_code == 124:
        return (
            "EVAL_TIMEOUT",
            f"eval script was killed at grading.eval_timeout ({len(log_text)} "
            "bytes captured); raise the timeout or narrow the test selection",
        )

    if START_TEST_OUTPUT in log_text and END_TEST_OUTPUT in log_text:
        # The tests ran between the markers, yet the parser found no per-test
        # verdicts. That is the test command printing progress rather than a
        # per-test summary — never a patch that failed.
        return (
            "RESOLVED_NO",
            f"the test command ran (exit={exit_code}) but emitted no per-test "
            "results the row's log_parser recognised — on Python repos this is "
            "usually pytest missing `-rA`, which the image sets up in the "
            "working tree and a reset reverts; on go/cargo/gradle/npm repos "
            "check the parser matches the runner the eval script invokes",
        )

    tail = stripped.splitlines()[-1][:200]
    return (
        "RESOLVED_NO",
        f"eval log has no test-output markers (exit={exit_code}, "
        f"{len(log_text)} bytes) — the eval script stopped before the tests. "
        f"Last line: {tail}",
    )


def parse_eval_log(
    test_spec: Any,
    instance_id: str,
    log_text: str,
    patch: str = "",
    exit_code: int | None = None,
) -> GradeResult:
    """Turn raw test output into a :class:`GradeResult` using swebench's parser."""
    from swebench.harness.constants import ResolvedStatus
    from swebench.harness.grading import (
        get_eval_report,
        get_logs_eval,
        get_resolution_status,
    )

    # swebench writes and re-reads a log file; keep that contract rather than
    # depending on internals that have moved between releases.
    with tempfile.NamedTemporaryFile(
        "w", suffix=".log", delete=False, encoding="utf-8"
    ) as handle:
        handle.write(log_text)
        log_path = handle.name

    try:
        status_map, found = get_logs_eval(test_spec, log_path)
        if not found:
            status, error = diagnose_unparsable_log(log_text, exit_code)
            return GradeResult(
                resolved=False,
                resolved_status=status,
                reward=0.0,
                status_map=dict(status_map or {}),
                eval_exit_code=exit_code,
                error=error,
            )

        prediction = {
            PRED_INSTANCE_ID: instance_id,
            PRED_MODEL: "orchard-eval",
            # swebench >= 5.0 short-circuits on a None patch, and the real patch
            # is what was actually evaluated.
            PRED_PATCH: patch,
        }
        report_map = get_eval_report(
            test_spec, prediction, log_path, include_tests_status=True
        )
        report = report_map.get(instance_id, {})
        tests_status = report.get("tests_status", {})
        resolved = bool(report.get("resolved"))
        resolved_status = get_resolution_status(tests_status) if tests_status else ""
        if not resolved_status:
            resolved_status = (
                ResolvedStatus.FULL.value if resolved else ResolvedStatus.NO.value
            )

        result = GradeResult(
            resolved=resolved,
            resolved_status=str(resolved_status),
            reward=1.0 if resolved else 0.0,
            tests_status=dict(tests_status),
            status_map=dict(status_map or {}),
            eval_exit_code=exit_code,
        )
        if not status_map:
            # The tests ran and the parser understood the log's shape, yet found
            # no per-test verdicts. That is the test command printing progress
            # rather than a summary — never a patch that failed.
            result.error = (
                "eval log parsed but yielded no per-test results — the test "
                "command is not emitting a per-test summary (pytest needs "
                "`-rA`); check that the image's working-tree setup was not "
                "reset, and that the row's log_parser matches its runner"
            )
        # swebench >= 5.0 distinguishes "the agent failed" from "the environment
        # failed". A run full of these is a broken cluster, not a weak model, so
        # it must not vanish silently into the resolve rate.
        if report.get("infra_failure"):
            result.error = "swebench flagged an infrastructure failure"
        return result
    finally:
        Path(log_path).unlink(missing_ok=True)


async def grade_swebench_patch(
    sandbox: EvalSandbox,
    instance: TaskInstance,
    patch: str,
    *,
    eval_timeout: int = 1200,
    apply_timeout: int = 120,
    allow_empty: bool = False,
    save_log: Callable[[str, str], None] | None = None,
) -> GradeResult:
    """Score ``patch`` against the benchmark's own tests.

    Args:
        allow_empty: Run the tests on the untouched repository when ``patch``
            is empty, instead of returning ``EMPTY_PATCH`` without spending a
            pod. This is the lower-bound baseline: the tests must score ~0%
            with no fix applied, and anything higher means the eval script is
            selecting tests that already pass at ``base_commit``.
        save_log: Optional ``callable(name, content)`` used to persist raw logs
            as run artifacts.

    Returns:
        A :class:`GradeResult`. Every failure mode — empty patch, patch that
        will not apply, crashed test run — comes back as an unresolved result
        with a populated ``error`` rather than an exception, because a single
        broken instance must never abort a 500-instance run.
    """
    if not patch.strip() and not allow_empty:
        return GradeResult.unresolved("empty patch", status="EMPTY_PATCH")

    instance_id = instance.instance_id
    try:
        test_spec, eval_script = build_eval_script(
            instance.raw, instance.workdir, instance.image
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[%s] could not build eval script", instance_id)
        return GradeResult.unresolved(f"eval script build failed: {exc}")

    try:
        # The pod is fresh from the image, so the tree is already at
        # base_commit — and it must stay exactly as the image left it: swebench
        # bakes environment setup into uncommitted working-tree edits (sphinx's
        # `-rA` sed on tox.ini), and reverting those disables the log parser.
        await sandbox.prepare_repo(instance.base_commit, reset=False)

        if patch.strip():
            apply_result = await sandbox.apply_patch(patch, timeout=apply_timeout)
            if not apply_result.succeeded:
                if save_log is not None:
                    save_log("apply_patch.log", apply_result.output)
                return GradeResult.unresolved(
                    f"patch did not apply (exit={apply_result.exit_code})",
                    status="PATCH_APPLY_FAILED",
                )

        await sandbox.write_file(eval_script, EVAL_SCRIPT_PATH)
        eval_result = await sandbox.exec(
            f"chmod +x {EVAL_SCRIPT_PATH} && /bin/bash {EVAL_SCRIPT_PATH}",
            timeout=eval_timeout,
            cwd=instance.workdir,
            # swebench's start/end markers are `set -x` traces on stderr while
            # the test output goes to stdout. Parsing reads only the region
            # between the markers, so the streams must stay interleaved.
            merge_stderr=True,
        )
        if save_log is not None:
            save_log("eval.log", eval_result.output)

        result = parse_eval_log(
            test_spec, instance_id, eval_result.output, patch, eval_result.exit_code
        )
        result.eval_exit_code = eval_result.exit_code
        logger.info(
            "[%s] resolved=%s status=%s",
            instance_id,
            result.resolved,
            result.resolved_status,
        )
        return result
    except SandboxGoneError:
        # The runner turns this into an infra failure and retries on a new pod.
        raise
    except Exception as exc:  # noqa: BLE001 - one bad instance must not stop a run
        logger.exception("[%s] grading failed", instance_id)
        if is_transport_error(exc):
            # The orchestrator or the pod's agent failed. That says nothing
            # about the patch, so it must not be scored as a wrong answer.
            return GradeResult.unresolved(
                f"{type(exc).__name__}: {exc}", status="EVAL_INFRA_ERROR"
            )
        return GradeResult.unresolved(f"{type(exc).__name__}: {exc}")


__all__ = [
    "CANONICAL_WORKDIR",
    "EVAL_SCRIPT_PATH",
    "MODERN_ROW_COLUMNS",
    "SWEBENCH_FAILURE_MARKERS",
    "build_eval_script",
    "build_test_spec",
    "diagnose_unparsable_log",
    "grade_swebench_patch",
    "parse_eval_log",
    "resolve_make_test_spec",
]
