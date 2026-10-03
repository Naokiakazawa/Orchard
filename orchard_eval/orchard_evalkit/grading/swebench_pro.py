"""SWE-bench Pro grading, run in a fresh sandbox the agent never had access to.

Pro grades nothing like SWE-bench does. There is no ``swebench`` package to
lean on and no synthesized eval script: each instance ships **its own two
files** in the ``SWE-bench_Pro-os`` repository —

``run_script.sh``
    the repo's test command, in a shape that takes a comma-separated list of
    test files (``go test -run``, ``jest``, ``pytest``, ...);
``parser.py``
    a per-repo parser that turns that command's stdout/stderr into
    ``{"tests": [{"name": ..., "status": "PASSED"|"FAILED"|...}]}``.

Both run **inside the sandbox**, which is where upstream runs them and the only
place where the repo's own toolchain exists. Grading is then a set comparison,
exactly as in upstream's ``swe_bench_pro_eval.py``::

    resolved = (fail_to_pass | pass_to_pass) <= {t for t in tests if PASSED}

The sequence per instance mirrors upstream's entry script:

1. ``git reset --hard base_commit`` — unlike SWE-bench, Pro's images want this;
   their build does not hide environment setup in uncommitted working-tree
   edits, and upstream resets unconditionally before scoring;
2. apply the candidate patch;
3. run ``before_repo_set_cmd`` (the build step: ``npm ci``, ``go mod download``,
   ...), which must happen *after* the patch so the patch is what gets built;
4. run ``run_script.sh`` over ``selected_test_files_to_run``;
5. run ``parser.py`` over the captured logs and read back ``output.json``.

The per-instance scripts are not in the HuggingFace dataset, so they come from
either a local checkout (``grading.pro_scripts_dir``) or GitHub, cached on
disk — see :class:`RunScriptStore`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from orchard_evalkit.datasets.swebench_pro import (
    COL_BEFORE_REPO_SET_CMD,
    COL_FAIL_TO_PASS,
    COL_PASS_TO_PASS,
    COL_SELECTED_TEST_FILES,
    parse_string_list,
)
from orchard_evalkit.models import GradeResult, TaskInstance
from orchard_evalkit.sandbox import EvalSandbox, SandboxGoneError, is_transport_error

logger = logging.getLogger(__name__)

#: Scratch directory inside the sandbox. Upstream mounts ``/workspace`` for the
#: same purpose, and the instance scripts hardcode nothing else.
WORKSPACE = "/workspace"
RUN_SCRIPT_PATH = f"{WORKSPACE}/run_script.sh"
PARSER_PATH = f"{WORKSPACE}/parser.py"
ENTRY_SCRIPT_PATH = f"{WORKSPACE}/entryscript.sh"
STDOUT_LOG = f"{WORKSPACE}/stdout.log"
STDERR_LOG = f"{WORKSPACE}/stderr.log"
OUTPUT_JSON = f"{WORKSPACE}/output.json"

#: The two per-instance files that make up an instance's eval script.
SCRIPT_NAMES = ("run_script.sh", "parser.py")

#: Where those files live when they are not on disk already.
RUN_SCRIPTS_BASE_URL = "https://raw.githubusercontent.com/scaleapi/SWE-bench_Pro-os"
DEFAULT_RUN_SCRIPTS_REF = "main"

#: Cache for downloaded scripts. They are a few kB each and never change for a
#: given ref, so one fetch per instance per machine is enough.
DEFAULT_SCRIPTS_CACHE = Path.home() / ".cache" / "orchard-eval" / "swebench-pro"

#: Bytes of each log kept as a run artifact. Some Pro suites emit tens of MB.
LOG_TAIL_BYTES = 200_000

#: Verdicts, spelled the way ``swebench`` spells them so both benchmarks'
#: results files can be read by the same tooling.
RESOLVED_FULL = "RESOLVED_FULL"
RESOLVED_PARTIAL = "RESOLVED_PARTIAL"
RESOLVED_NO = "RESOLVED_NO"

#: The status ``parser.py`` emits for a test that passed.
STATUS_PASSED = "PASSED"


class RunScriptError(RuntimeError):
    """An instance's ``run_script.sh``/``parser.py`` could not be obtained."""


class RunScriptStore:
    """Supplies each instance's ``run_script.sh`` and ``parser.py``.

    Args:
        scripts_dir: A local ``run_scripts/`` directory — i.e. a checkout of
            ``scaleapi/SWE-bench_Pro-os``. Offline and pinned to whatever the
            checkout is at, which is what a reproducible run wants.
        cache_dir: Where downloads are cached when ``scripts_dir`` is unset.
        ref: Git ref to download from. Pin it for a run you intend to compare
            against later; upstream has revised these scripts more than once.
        allow_download: Set false to fail loudly rather than reach the network.

    Downloads are serialized behind one lock. The files are a few kB, the miss
    only happens once per instance, and the alternative — 32 workers fetching
    concurrently on the first run — is a rate limit rather than a speedup.
    """

    def __init__(
        self,
        scripts_dir: str | Path | None = None,
        *,
        cache_dir: str | Path | None = None,
        ref: str = DEFAULT_RUN_SCRIPTS_REF,
        allow_download: bool = True,
    ):
        self.scripts_dir = Path(scripts_dir).expanduser() if scripts_dir else None
        self.cache_dir = (
            Path(cache_dir).expanduser()
            if cache_dir
            else DEFAULT_SCRIPTS_CACHE / (ref or DEFAULT_RUN_SCRIPTS_REF)
        )
        self.ref = ref or DEFAULT_RUN_SCRIPTS_REF
        self.allow_download = allow_download
        self._lock = threading.Lock()

    @staticmethod
    def _validate(instance_id: str) -> str:
        # The id becomes a path segment and a URL segment. A benchmark row must
        # not be able to reach outside the cache directory.
        if not instance_id or "/" in instance_id or ".." in instance_id:
            raise RunScriptError(f"Refusing unsafe instance id {instance_id!r}")
        return instance_id

    def load(self, instance_id: str) -> dict[str, str]:
        """Return ``{filename: contents}``. Blocking; call via :meth:`aload`."""
        self._validate(instance_id)
        if self.scripts_dir is not None:
            return {
                name: self._read_local(self.scripts_dir / instance_id / name)
                for name in SCRIPT_NAMES
            }
        return {name: self._cached(instance_id, name) for name in SCRIPT_NAMES}

    async def aload(self, instance_id: str) -> dict[str, str]:
        """Async wrapper — the work is filesystem and network I/O."""
        return await asyncio.to_thread(self.load, instance_id)

    def _read_local(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RunScriptError(
                f"{path} is missing. grading.pro_scripts_dir must point at the "
                "`run_scripts/` directory of a scaleapi/SWE-bench_Pro-os "
                "checkout; unset it to download the scripts instead."
            ) from exc

    def _cached(self, instance_id: str, name: str) -> str:
        path = self.cache_dir / instance_id / name
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            pass
        if not self.allow_download:
            raise RunScriptError(
                f"{path} is not cached and downloads are disabled. Point "
                "grading.pro_scripts_dir at a SWE-bench_Pro-os checkout."
            )
        with self._lock:
            # Another worker may have fetched it while this one waited.
            try:
                return path.read_text(encoding="utf-8")
            except OSError:
                content = self._download(instance_id, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Written via a sibling temp file so a killed run never leaves a
            # truncated script that every later run would happily reuse.
            tmp = path.with_suffix(path.suffix + ".partial")
            tmp.write_text(content, encoding="utf-8")
            tmp.replace(path)
        return content

    def _download(self, instance_id: str, name: str) -> str:
        url = "/".join(
            (
                RUN_SCRIPTS_BASE_URL,
                urllib.parse.quote(self.ref, safe=""),
                "run_scripts",
                urllib.parse.quote(instance_id, safe=""),
                urllib.parse.quote(name, safe=""),
            )
        )
        logger.debug("fetching %s", url)
        request = urllib.request.Request(url, headers={"User-Agent": "orchard-eval"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise RunScriptError(
                f"{url} returned HTTP {exc.code}. The instance may not exist at "
                f"ref {self.ref!r}, or the upstream layout has changed."
            ) from exc
        except OSError as exc:
            raise RunScriptError(f"Could not fetch {url}: {exc}") from exc


def before_repo_set_cmd(record: dict[str, Any]) -> str:
    """Return the build command upstream runs before the tests.

    Only the last line is used, matching upstream: the column holds a short
    transcript and everything above the final line is context that has already
    been baked into the image.
    """
    raw = str(record.get(COL_BEFORE_REPO_SET_CMD) or "").strip()
    return raw.splitlines()[-1].strip() if raw else ""


def selected_test_files(record: dict[str, Any]) -> list[str]:
    return parse_string_list(record.get(COL_SELECTED_TEST_FILES))


def build_entry_script(record: dict[str, Any], workdir: str) -> str:
    """Render the script that runs the tests and the parser in the sandbox.

    Deliberately *not* ``set -e``: upstream's entry script is not, and a build
    step that prints to stderr but succeeds, or a test command that exits
    non-zero because tests failed, must still reach the parser. Stopping early
    would turn every red suite into an unparsable log.
    """
    build = before_repo_set_cmd(record)
    files = ",".join(selected_test_files(record))
    # No files means "run everything", which is what run_script.sh does with no
    # argument. Passing an empty argument instead makes it match no test at all.
    test_args = f" {shlex.quote(files)}" if files else ""

    return f"""#!/bin/bash
# Generated by orchard-eval for SWE-bench Pro. Mirrors the upstream entryscript.
mkdir -p {WORKSPACE}
cd {shlex.quote(workdir)} || exit 1

PYTHON="$(command -v python3 || command -v python)"
if [ -z "$PYTHON" ]; then
    echo "orchard-eval: no python interpreter in this image" >&2
    exit 127
fi

{build}
echo "orchard-eval: before_repo_set_cmd exit=$?"

bash {RUN_SCRIPT_PATH}{test_args} > {STDOUT_LOG} 2> {STDERR_LOG}
echo "orchard-eval: run_script exit=$?"

"$PYTHON" {PARSER_PATH} {STDOUT_LOG} {STDERR_LOG} {OUTPUT_JSON}
echo "orchard-eval: parser exit=$?"
"""


def grade_from_output(
    output: dict[str, Any],
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    *,
    stdout_tail: str = "",
) -> GradeResult:
    """Score parsed test results the way upstream's evaluator does.

    Upstream requires *every* ``fail_to_pass`` and ``pass_to_pass`` test to be
    reported ``PASSED``; a test the parser never saw counts as not passing,
    which is what makes a crashed suite score zero instead of vacuously
    resolving.

    ``stdout_tail`` is the captured test log, used only to tell the two very
    different reasons for an empty result set apart. Optional so callers that
    re-score a saved ``output.json`` — ``scripts/regrade.py`` — need not have it.
    """
    tests = output.get("tests") or []
    status_map = {
        str(test.get("name")): str(test.get("status"))
        for test in tests
        if isinstance(test, dict) and test.get("name")
    }
    passed = {name for name, status in status_map.items() if status == STATUS_PASSED}

    def split(names: list[str]) -> dict[str, list[str]]:
        return {
            "success": [name for name in names if name in passed],
            "failure": [name for name in names if name not in passed],
        }

    f2p = split(fail_to_pass)
    p2p = split(pass_to_pass)
    resolved = not f2p["failure"] and not p2p["failure"]

    if resolved:
        status = RESOLVED_FULL
    elif f2p["success"]:
        status = RESOLVED_PARTIAL
    else:
        status = RESOLVED_NO

    result = GradeResult(
        resolved=resolved,
        resolved_status=status,
        reward=1.0 if resolved else 0.0,
        tests_status={"FAIL_TO_PASS": f2p, "PASS_TO_PASS": p2p},
        status_map=status_map,
    )
    if not fail_to_pass:
        # Nothing had to change for this to "pass". Never silently a win.
        result.error = (
            "row has an empty fail_to_pass list — this instance resolves "
            "without any fix and should be excluded from the score"
        )
    elif not status_map and not stdout_tail.strip():
        result.error = (
            "the parser produced no test results at all and the test log is "
            "empty — the test command most likely never ran; see stdout.log / "
            "stderr.log"
        )
    elif not status_map:
        # An empty result set on top of a *non-empty* log is usually the tests
        # failing, not the harness breaking: several of Pro's per-instance
        # parsers only record column-0 `--- PASS:` lines, so a red top-level
        # test legitimately yields `{"tests": []}`. Saying "never ran" here
        # sends the reader hunting for an infrastructure fault that is not
        # there.
        result.error = (
            f"the instance's parser matched no test results in a "
            f"{len(stdout_tail)}-byte test log, so the tests ran and none of "
            "them passed. Pro's Go parsers only record column-0 '--- PASS:' "
            "lines, which makes a failing top-level test look like this; "
            "confirm against tests_stdout.log before suspecting the harness"
        )
    return result


def diagnose_missing_output(
    exit_code: int | None, stdout_tail: str, stderr_tail: str
) -> tuple[str, str]:
    """Return ``(resolved_status, error)`` when ``output.json`` never appeared.

    Separating "the machinery failed" from "the patch failed" is what keeps a
    broken pod out of the resolve rate: ``EVAL_INFRA_ERROR`` is retried on a
    fresh pod by the runner, the rest are recorded as they are.
    """
    if exit_code == 124:
        return (
            "EVAL_TIMEOUT",
            "the eval script was killed at grading.eval_timeout; raise it or "
            "narrow the test selection",
        )
    if not stdout_tail.strip() and not stderr_tail.strip():
        return (
            "EVAL_INFRA_ERROR",
            f"the eval script produced no output at all (exit={exit_code}) — "
            "the command was killed or the pod's agent stopped responding",
        )
    tail = (stderr_tail or stdout_tail).strip().splitlines()[-1][:200]
    return (
        RESOLVED_NO,
        f"the parser wrote no output.json (exit={exit_code}); the test command "
        f"or the parser failed. Last line: {tail}",
    )


async def _tail(sandbox: EvalSandbox, path: str, limit: int = LOG_TAIL_BYTES) -> str:
    """Read the end of a file in the sandbox, tolerating its absence."""
    result = await sandbox.exec(
        f"tail -c {limit} {shlex.quote(path)} 2>/dev/null", timeout=120
    )
    return result.stdout


async def grade_swebench_pro_patch(
    sandbox: EvalSandbox,
    instance: TaskInstance,
    patch: str,
    *,
    scripts: RunScriptStore,
    eval_timeout: int = 1200,
    apply_timeout: int = 120,
    allow_empty: bool = False,
    save_log: Callable[[str, str], None] | None = None,
) -> GradeResult:
    """Score ``patch`` against SWE-bench Pro's tests.

    Args:
        scripts: Source of the instance's ``run_script.sh`` and ``parser.py``.
        allow_empty: Run the tests on the untouched repository when ``patch``
            is empty. This is the lower-bound baseline — the tests must score
            ~0% with no fix applied.
        save_log: Optional ``callable(name, content)`` used to persist raw logs
            as run artifacts.

    Returns:
        A :class:`GradeResult`. Every failure mode comes back as an unresolved
        result with a populated ``error`` rather than an exception, because a
        single broken instance must never abort a 731-instance run.
    """
    if not patch.strip() and not allow_empty:
        return GradeResult.unresolved("empty patch", status="EMPTY_PATCH")

    instance_id = instance.instance_id
    record = instance.raw
    workdir = instance.workdir

    try:
        files = await scripts.aload(instance_id)
    except RunScriptError as exc:
        logger.error("[%s] %s", instance_id, exc)
        return GradeResult.unresolved(str(exc), status="EVAL_SCRIPT_MISSING")

    entry_script = build_entry_script(record, workdir)
    if save_log is not None:
        save_log("entryscript.sh", entry_script)

    try:
        # Upstream resets hard before scoring, and Pro's images tolerate it:
        # unlike SWE-bench, their setup lives in the image's layers rather than
        # in uncommitted edits to the working tree.
        quoted = shlex.quote(workdir)
        commit = shlex.quote(instance.base_commit)
        reset = await sandbox.exec(
            f"git config --global --add safe.directory {quoted} "
            f"&& git reset --hard {commit} && git checkout {commit}",
            timeout=600,
            merge_stderr=True,
        )
        if not reset.succeeded:
            if save_log is not None:
                save_log("reset.log", reset.output)
            return GradeResult.unresolved(
                f"could not reset the repository to {instance.base_commit} "
                f"(exit={reset.exit_code})",
                status="EVAL_RESET_FAILED",
            )

        if patch.strip():
            applied = await sandbox.apply_patch(patch, timeout=apply_timeout)
            if not applied.succeeded:
                if save_log is not None:
                    save_log("apply_patch.log", applied.output)
                return GradeResult.unresolved(
                    f"patch did not apply (exit={applied.exit_code})",
                    status="PATCH_APPLY_FAILED",
                )

        await sandbox.exec(f"mkdir -p {WORKSPACE}", timeout=60)
        await sandbox.write_file(files["run_script.sh"], RUN_SCRIPT_PATH)
        await sandbox.write_file(files["parser.py"], PARSER_PATH)
        await sandbox.write_file(entry_script, ENTRY_SCRIPT_PATH)

        run = await sandbox.exec(
            f"chmod +x {ENTRY_SCRIPT_PATH} && /bin/bash {ENTRY_SCRIPT_PATH}",
            timeout=eval_timeout,
            cwd=workdir,
            merge_stderr=True,
        )
        if save_log is not None:
            save_log("eval.log", run.output)

        stdout_tail = await _tail(sandbox, STDOUT_LOG)
        stderr_tail = await _tail(sandbox, STDERR_LOG)
        if save_log is not None:
            save_log("tests_stdout.log", stdout_tail)
            save_log("tests_stderr.log", stderr_tail)

        try:
            raw_output = await sandbox.read_file(OUTPUT_JSON)
            output = json.loads(raw_output)
        except SandboxGoneError:
            raise
        except Exception as exc:  # noqa: BLE001 - a missing/!json file is data
            if is_transport_error(exc):
                return GradeResult.unresolved(
                    f"{type(exc).__name__}: {exc}", status="EVAL_INFRA_ERROR"
                )
            status, error = diagnose_missing_output(
                run.exit_code, stdout_tail, stderr_tail
            )
            return GradeResult(
                resolved=False,
                resolved_status=status,
                reward=0.0,
                eval_exit_code=run.exit_code,
                error=error,
            )

        if save_log is not None:
            save_log("output.json", raw_output)

        result = grade_from_output(
            output,
            parse_string_list(record.get(COL_FAIL_TO_PASS)),
            parse_string_list(record.get(COL_PASS_TO_PASS)),
            stdout_tail=stdout_tail,
        )
        result.eval_exit_code = run.exit_code
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
            return GradeResult.unresolved(
                f"{type(exc).__name__}: {exc}", status="EVAL_INFRA_ERROR"
            )
        return GradeResult.unresolved(f"{type(exc).__name__}: {exc}")


__all__ = [
    "DEFAULT_RUN_SCRIPTS_REF",
    "DEFAULT_SCRIPTS_CACHE",
    "ENTRY_SCRIPT_PATH",
    "OUTPUT_JSON",
    "PARSER_PATH",
    "RESOLVED_FULL",
    "RESOLVED_NO",
    "RESOLVED_PARTIAL",
    "RUN_SCRIPT_PATH",
    "RunScriptError",
    "RunScriptStore",
    "SCRIPT_NAMES",
    "STDERR_LOG",
    "STDOUT_LOG",
    "WORKSPACE",
    "before_repo_set_cmd",
    "build_entry_script",
    "diagnose_missing_output",
    "grade_from_output",
    "grade_swebench_pro_patch",
    "selected_test_files",
]
