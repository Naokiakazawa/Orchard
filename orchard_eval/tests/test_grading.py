"""SWE-bench grading.

Two layers are tested:

* the *sandbox interaction* — reset, apply, run, parse — with fakes, so the
  ordering guarantees hold without needing swebench installed;
* the *swebench integration* itself, against the real package, because the API
  moved between releases and a compat shim is only worth anything if it is
  exercised against a real ``TestSpec``.
"""

import pytest

from orchard_evalkit.grading.swebench import (
    CANONICAL_WORKDIR,
    EVAL_SCRIPT_PATH,
    grade_swebench_patch,
)
from orchard_evalkit.models import TaskInstance
from orchard_evalkit.sandbox import EvalSandbox
from tests.fakes import FakeJobResult, FakeSandboxInstance

PATCH = "diff --git a/f b/f\n+fixed\n"


def _instance(**kwargs) -> TaskInstance:
    defaults = {
        "instance_id": "astropy__astropy-12907",
        "problem_statement": "p",
        "base_commit": "d16bfe0",
        "image": "docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest",
        "raw": {"instance_id": "astropy__astropy-12907"},
    }
    defaults.update(kwargs)
    return TaskInstance(**defaults)


def _sandbox(responder=None):
    instance = FakeSandboxInstance(responder=responder)
    return EvalSandbox(instance, workdir="/testbed"), instance


async def _no_sleep(_seconds):
    """Collapse retry backoff so a retry test costs nothing."""
    return None


class TestGradeSandboxFlow:
    @pytest.mark.asyncio
    async def test_an_empty_patch_short_circuits(self):
        # No pod work at all: there is nothing to grade.
        sandbox, instance = _sandbox()
        result = await grade_swebench_patch(sandbox, _instance(), "   \n")

        assert result.resolved is False
        assert result.resolved_status == "EMPTY_PATCH"
        assert instance.commands == []

    @pytest.mark.asyncio
    async def test_allow_empty_runs_the_tests_on_the_untouched_tree(self, monkeypatch):
        # The no-patch floor: the eval script has to run so the FAIL_TO_PASS
        # tests can fail. Short-circuiting here would make that check measure
        # nothing at all.
        monkeypatch.setattr(
            "orchard_evalkit.grading.swebench.build_eval_script",
            lambda record, workdir, image="": (object(), "echo test"),
        )
        monkeypatch.setattr(
            "orchard_evalkit.grading.swebench.parse_eval_log",
            lambda spec, iid, log, patch="", exit_code=None: __import__(
                "orchard_evalkit.models", fromlist=["GradeResult"]
            ).GradeResult(),
        )

        sandbox, instance = _sandbox()
        await grade_swebench_patch(sandbox, _instance(), "", allow_empty=True)

        commands = [c["command"] for c in instance.commands]
        assert any(EVAL_SCRIPT_PATH in c for c in commands)
        assert not any("git apply" in c for c in commands)

    @pytest.mark.asyncio
    async def test_a_patch_that_will_not_apply_scores_zero(self, monkeypatch):
        monkeypatch.setattr(
            "orchard_evalkit.grading.swebench.build_eval_script",
            lambda record, workdir, image="": (object(), "echo test"),
        )

        def responder(cmd):
            # Every command in GIT_APPLY_CMDS has to fail, not just the `git apply`
            # ones: the chain ends with a `patch(1)` fallback and then a reverse
            # check, and either succeeding means the patch did land. All of them
            # name the staged patch file, so that is what to match on.
            if "orchard_eval_model.patch" in cmd:
                return FakeJobResult(stderr="error: patch failed", exit_code=1)
            return FakeJobResult()

        sandbox, instance = _sandbox(responder)
        result = await grade_swebench_patch(sandbox, _instance(), PATCH)

        assert result.resolved_status == "PATCH_APPLY_FAILED"
        # The eval script must never run on a tree the patch did not reach.
        assert not any(EVAL_SCRIPT_PATH in c["command"] for c in instance.commands)

    @pytest.mark.asyncio
    async def test_the_image_working_tree_is_never_reverted(self, monkeypatch):
        # SWE-bench images carry uncommitted setup in the working tree — the
        # `-rA` sed on sphinx's tox.ini. Reverting it makes pytest print
        # progress dots instead of a summary, so a passing suite scores zero.
        monkeypatch.setattr(
            "orchard_evalkit.grading.swebench.build_eval_script",
            lambda record, workdir, image="": (object(), "echo test"),
        )
        monkeypatch.setattr(
            "orchard_evalkit.grading.swebench.parse_eval_log",
            lambda spec, iid, log, patch="", exit_code=None: __import__(
                "orchard_evalkit.models", fromlist=["GradeResult"]
            ).GradeResult(resolved=True, reward=1.0),
        )

        sandbox, instance = _sandbox()
        await grade_swebench_patch(sandbox, _instance(), PATCH)

        commands = [c["command"] for c in instance.commands]
        assert not any("git checkout -f" in c for c in commands)
        assert not any("git clean" in c for c in commands)
        apply_at = next(i for i, c in enumerate(commands) if "git apply" in c)
        eval_at = next(i for i, c in enumerate(commands) if EVAL_SCRIPT_PATH in c)
        assert apply_at < eval_at

    @pytest.mark.asyncio
    async def test_a_build_failure_is_a_result_not_an_exception(self, monkeypatch):
        def explode(record, workdir, image=""):
            raise RuntimeError("dataset row is missing eval_script")

        monkeypatch.setattr("orchard_evalkit.grading.swebench.build_eval_script", explode)
        sandbox, _ = _sandbox()
        result = await grade_swebench_patch(sandbox, _instance(), PATCH)

        # A single broken instance must never abort a 500-instance run.
        assert result.resolved is False
        assert "eval_script" in result.error

    @pytest.mark.asyncio
    async def test_logs_are_saved_as_artifacts(self, monkeypatch):
        monkeypatch.setattr(
            "orchard_evalkit.grading.swebench.build_eval_script",
            lambda record, workdir, image="": (object(), "echo test"),
        )
        monkeypatch.setattr(
            "orchard_evalkit.grading.swebench.parse_eval_log",
            lambda spec, iid, log, patch="", exit_code=None: __import__(
                "orchard_evalkit.models", fromlist=["GradeResult"]
            ).GradeResult(),
        )

        saved = {}
        sandbox, _ = _sandbox(lambda cmd: FakeJobResult(stdout="TEST OUTPUT"))
        await grade_swebench_patch(
            sandbox, _instance(), PATCH, save_log=lambda n, c: saved.__setitem__(n, c)
        )
        assert "TEST OUTPUT" in saved["eval.log"]

    @pytest.mark.asyncio
    async def test_a_5xx_from_the_orchestrator_is_an_infra_failure(self, monkeypatch):
        # A broken pod says nothing about the patch; scoring it as a zero would
        # quietly lower every resolve rate the cluster hiccups during.
        monkeypatch.setattr(
            "orchard_evalkit.grading.swebench.build_eval_script",
            lambda record, workdir, image="": (object(), "echo test"),
        )
        monkeypatch.setattr("asyncio.sleep", _no_sleep)

        class ServerError(Exception):
            status = 500

        sandbox, instance = _sandbox()

        async def boom(content, remote_path):
            raise ServerError("Internal Server Error")

        instance.upload_content = boom
        result = await grade_swebench_patch(sandbox, _instance(), PATCH)

        assert result.resolved_status == "EVAL_INFRA_ERROR"

    @pytest.mark.asyncio
    async def test_a_transient_upload_failure_is_retried(self, monkeypatch):
        monkeypatch.setattr("asyncio.sleep", _no_sleep)

        class ServerError(Exception):
            status = 503

        sandbox, instance = _sandbox()
        calls = {"n": 0}
        original = instance.upload_content

        async def flaky(content, remote_path):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ServerError("Service Unavailable")
            return await original(content, remote_path)

        instance.upload_content = flaky
        await sandbox.write_file("hello", "/tmp/x")

        assert calls["n"] == 2
        assert instance.files["/tmp/x"] == b"hello"


swebench = pytest.importorskip("swebench", reason="swebench extra not installed")


#: A modern (swebench >= 5.0) dataset row carries these columns itself.
#: The start/end markers matter: swebench parses only the region between them,
#: and appends the test command's own exit code after the end marker.
MODERN_ROW = {
    "instance_id": "demo__demo-1",
    "image": "swebench/sweb.eval.x86_64.demo_1776_demo-1:latest",
    "repo": "demo/demo",
    "version": "1.0",
    "FAIL_TO_PASS": ["tests/test_demo.py::test_new"],
    "PASS_TO_PASS": ["tests/test_demo.py::test_old"],
    "log_parser": "parse_log_pytest",
    "eval_type": "pass_and_fail",
    "eval_script": (
        "#!/bin/bash\n"
        "set -uxo pipefail\n"
        "cd /testbed\n"
        "git checkout abc123 -- tests/test_demo.py\n"
        ">>>>> Start Test Output\n"
        "pytest tests/test_demo.py\n"
        ">>>>> End Test Output\n"
    ),
}


def _eval_log(f2p_status: str, p2p_status: str, exit_code: int = 0) -> str:
    """Build a realistic eval log with swebench's markers."""
    from swebench.harness.grading import (
        END_TEST_OUTPUT,
        START_TEST_OUTPUT,
        TEST_EXIT_CODE,
    )

    return "\n".join(
        [
            "installing dependencies ...",
            START_TEST_OUTPUT,
            f"{f2p_status} tests/test_demo.py::test_new",
            f"{p2p_status} tests/test_demo.py::test_old",
            END_TEST_OUTPUT,
            f"{TEST_EXIT_CODE}: {exit_code}",
        ]
    )


def _modern_spec():
    from orchard_evalkit.grading.swebench import build_test_spec, resolve_make_test_spec

    _, is_modern = resolve_make_test_spec()
    if not is_modern:
        pytest.skip("installed swebench predates the 5.0 layout")
    return build_test_spec(MODERN_ROW)


class TestSwebenchIntegration:
    def test_a_modern_row_builds_a_test_spec(self):
        spec = _modern_spec()
        assert spec.instance_id == "demo__demo-1"
        assert "pytest tests/test_demo.py" in spec.eval_script

    def test_the_script_records_the_test_commands_exit_code(self):
        # Eval scripts end with a `git checkout`, so the script's own status is
        # the reset's. swebench captures `$?` right after the tests instead.
        assert "SWEBENCH_TEST_EXIT_CODE=$?" in _modern_spec().eval_script

    def test_a_missing_image_is_filled_in_from_the_suite(self):
        from orchard_evalkit.grading.swebench import (
            build_test_spec,
            resolve_make_test_spec,
        )

        _, is_modern = resolve_make_test_spec()
        if not is_modern:
            pytest.skip("installed swebench predates the 5.0 layout")

        row = {k: v for k, v in MODERN_ROW.items() if k != "image"}
        spec = build_test_spec(row, image="myacr.io/swebench/demo:latest")
        assert spec.image == "myacr.io/swebench/demo:latest"

    def test_a_legacy_row_gets_an_actionable_error(self):
        from orchard_evalkit.grading.swebench import (
            build_test_spec,
            resolve_make_test_spec,
        )

        _, is_modern = resolve_make_test_spec()
        if not is_modern:
            pytest.skip("installed swebench predates the 5.0 layout")

        legacy_row = {
            k: v
            for k, v in MODERN_ROW.items()
            if k not in ("eval_script", "log_parser", "eval_type")
        }
        with pytest.raises(RuntimeError) as excinfo:
            build_test_spec(legacy_row)
        # The message must name the fix, not just the symptom.
        message = str(excinfo.value)
        assert "eval_script" in message
        assert "SWE-bench/SWE-bench_Verified" in message

    def test_the_script_is_retargeted_at_a_custom_workdir(self):
        from orchard_evalkit.grading.swebench import build_eval_script

        _modern_spec()  # skips when swebench is too old
        _, script = build_eval_script(MODERN_ROW, "/workspace/repo")
        assert "/workspace/repo" in script
        assert CANONICAL_WORKDIR not in script

    def test_the_canonical_workdir_is_left_alone(self):
        from orchard_evalkit.grading.swebench import build_eval_script

        _modern_spec()
        _, script = build_eval_script(MODERN_ROW, CANONICAL_WORKDIR)
        assert "cd /testbed" in script


class TestResolutionSemantics:
    """The scoring rules, checked against the real swebench parser."""

    def test_all_tests_passing_resolves(self):
        from orchard_evalkit.grading.swebench import parse_eval_log

        result = parse_eval_log(
            _modern_spec(), "demo__demo-1", _eval_log("PASSED", "PASSED"), PATCH
        )
        assert result.resolved is True
        assert result.resolved_status == "RESOLVED_FULL"
        assert result.reward == 1.0

    def test_a_failing_fail_to_pass_test_does_not_resolve(self):
        from orchard_evalkit.grading.swebench import parse_eval_log

        result = parse_eval_log(
            _modern_spec(), "demo__demo-1", _eval_log("FAILED", "PASSED", 1), PATCH
        )
        assert result.resolved is False
        assert result.reward == 0.0

    def test_a_regressed_pass_to_pass_test_does_not_resolve(self):
        # A patch that fixes the bug but breaks the existing suite is not a fix.
        from orchard_evalkit.grading.swebench import parse_eval_log

        result = parse_eval_log(
            _modern_spec(), "demo__demo-1", _eval_log("PASSED", "FAILED", 1), PATCH
        )
        assert result.resolved is False
        assert result.reward == 0.0

    def test_unparseable_output_is_not_scored_as_resolved(self):
        # A crashed container produces output with no markers; treating that as
        # anything but unresolved would silently corrupt a run.
        from orchard_evalkit.grading.swebench import parse_eval_log

        result = parse_eval_log(_modern_spec(), "demo__demo-1", "container died", PATCH)
        assert result.resolved is False
        assert result.reward == 0.0
        assert result.error

    def test_the_per_test_status_map_is_reported(self):
        from orchard_evalkit.grading.swebench import parse_eval_log

        result = parse_eval_log(
            _modern_spec(), "demo__demo-1", _eval_log("PASSED", "PASSED"), PATCH
        )
        assert result.status_map["tests/test_demo.py::test_new"] == "PASSED"
        assert result.tests_status


class TestUnparsableLogDiagnosis:
    """A log swebench will not parse is usually the cluster, not the patch."""

    def test_an_empty_log_is_an_infrastructure_failure(self):
        from orchard_evalkit.grading.swebench import diagnose_unparsable_log

        status, error = diagnose_unparsable_log("", exit_code=-1)
        assert status == "EVAL_INFRA_ERROR"
        assert "no output" in error

    def test_a_killed_command_is_reported_as_a_timeout(self):
        from orchard_evalkit.grading.swebench import diagnose_unparsable_log

        status, _ = diagnose_unparsable_log("partial output\n", exit_code=124)
        assert status == "EVAL_TIMEOUT"

    def test_swebench_own_failure_markers_are_named(self):
        from orchard_evalkit.grading.swebench import diagnose_unparsable_log

        status, error = diagnose_unparsable_log(">>>>> Patch Apply Failed", 1)
        assert status == "PATCH_APPLY_FAILED"
        assert "test patch" in error

    def test_a_real_log_without_markers_keeps_its_last_line(self):
        from orchard_evalkit.grading.swebench import diagnose_unparsable_log

        status, error = diagnose_unparsable_log("setup ok\ntox: not found\n", 127)
        assert status == "RESOLVED_NO"
        assert "tox: not found" in error

    def test_a_suite_that_ran_but_printed_no_summary_says_so(self):
        # swebench reports this identically to "markers missing", but the cause
        # is the opposite: the tests ran fine and the parser had nothing to read.
        from orchard_evalkit.grading.swebench import diagnose_unparsable_log

        log = (
            "+ : '>>>>> Start Test Output'\n"
            "tests/test_demo.py ...       [100%]\n"
            "3 passed in 0.25s\n"
            "+ : '>>>>> End Test Output'\n"
        )
        status, error = diagnose_unparsable_log(log, exit_code=0)
        assert status == "RESOLVED_NO"
        assert "-rA" in error
