"""SWE-bench Pro: dataset normalization, benchmark detection, and grading.

These tests are offline. Nothing here touches HuggingFace, Docker Hub, GitHub,
or an orchestrator — the point is that the parts which *decide* things (which
image, which workdir, which benchmark, resolved or not) are checkable without
any of that.
"""

from __future__ import annotations

import json
import shlex

import pytest
from orchard_evalkit.config import DatasetConfig, SandboxConfig
from orchard_evalkit.datasets import (
    BENCHMARK_SWEBENCH,
    BENCHMARK_SWEBENCH_PRO,
    detect_benchmark,
    normalize_benchmark,
    resolve_image_prefix,
    resolve_workdir,
)
from orchard_evalkit.datasets.swebench_pro import (
    DEFAULT_IMAGE_PREFIX,
    DEFAULT_WORKDIR,
    ELEMENT_WEB_EXCEPTION,
    build_problem_statement,
    dockerhub_tag,
    instances_from_records,
    parse_string_list,
    pro_image_name,
)
from orchard_evalkit.grading.swebench_pro import (
    ENTRY_SCRIPT_PATH,
    OUTPUT_JSON,
    RESOLVED_FULL,
    RESOLVED_NO,
    RESOLVED_PARTIAL,
    RUN_SCRIPT_PATH,
    RunScriptError,
    RunScriptStore,
    before_repo_set_cmd,
    build_entry_script,
    diagnose_missing_output,
    grade_from_output,
    grade_swebench_pro_patch,
)
from orchard_evalkit.sandbox import EvalSandbox

from tests.fakes import FakeJobResult, FakeSandboxInstance


def _run_script_argv(script: str) -> list[str]:
    """The ``run_script.sh`` invocation from the entry script, as a token list.

    Splitting into tokens keeps these assertions about *how many arguments*
    run_script.sh receives, which is what matters, rather than about whether
    ``shlex.quote`` happened to add quotes — it does not for a comma-separated
    list of filenames, since commas and dots are both shell-safe.
    """
    line = next(line for line in script.splitlines() if RUN_SCRIPT_PATH in line)
    return shlex.split(line.split(">", 1)[0])


def pro_record(**overrides):
    """A row shaped like the ones in ``ScaleAI/SWE-bench_Pro``.

    Every list-valued column is a *string* holding a Python literal, which is
    how the dataset actually stores them.
    """
    record = {
        "instance_id": "instance_navidrome__navidrome-0130c6dc",
        "repo": "navidrome/navidrome",
        "base_commit": "0130c6dc13438b48cf0fdfab08a89e357b5517c9",
        "problem_statement": "Playlists do not refresh.",
        "requirements": "The playlist must refresh after import.",
        "interface": "func (p *Playlist) Refresh() error",
        "patch": "diff --git a/a.go b/a.go\n",
        "dockerhub_tag": "navidrome.navidrome-0130c6dc",
        "fail_to_pass": "['TestRefresh', 'TestImport']",
        "pass_to_pass": "['TestExisting']",
        "before_repo_set_cmd": "cd /app\ngo mod download",
        "selected_test_files_to_run": "['playlist_test.go', 'import_test.go']",
    }
    record.update(overrides)
    return record


class TestParseStringList:
    def test_python_literal_lists_are_parsed(self):
        assert parse_string_list("['a', 'b']") == ["a", "b"]

    def test_json_lists_are_parsed(self):
        assert parse_string_list('["a", "b"]') == ["a", "b"]

    def test_real_lists_pass_through(self):
        assert parse_string_list(["a", "b"]) == ["a", "b"]

    def test_empty_and_null_become_empty(self):
        assert parse_string_list(None) == []
        assert parse_string_list("") == []
        assert parse_string_list("[]") == []

    def test_names_with_quotes_survive(self):
        assert parse_string_list('["Test[a\'b]"]') == ["Test[a'b]"]

    def test_a_plain_string_is_one_entry(self):
        assert parse_string_list("TestOne\nTestTwo") == ["TestOne", "TestTwo"]

    def test_arbitrary_code_is_not_executed(self):
        # `eval()` upstream would run this. literal_eval must refuse it and
        # fall back to the line reader rather than importing anything.
        assert parse_string_list("__import__('os').system('true')") == [
            "__import__('os').system('true')"
        ]


class TestDockerhubTag:
    def test_repo_prefix_is_lowercased_and_instance_case_kept(self):
        tag = dockerhub_tag(
            "instance_NodeBB__NodeBB-7b8bffd7-vf2cf3cbd", "NodeBB/NodeBB"
        )
        assert tag == "nodebb.nodebb-NodeBB__NodeBB-7b8bffd7-vf2cf3cbd"

    def test_missing_version_suffix_is_stripped(self):
        tag = dockerhub_tag(
            "instance_navidrome__navidrome-abc-vnan", "navidrome/navidrome"
        )
        assert tag == "navidrome.navidrome-navidrome__navidrome-abc"

    def test_element_web_is_shortened(self):
        tag = dockerhub_tag(
            "instance_element-hq__element-web-abc-vnan", "element-hq/element-web"
        )
        assert tag == "element-hq.element-element-hq__element-web-abc"

    def test_the_one_element_web_exception_keeps_its_full_name_and_suffix(self):
        tag = dockerhub_tag(ELEMENT_WEB_EXCEPTION, "element-hq/element-web")
        assert tag.startswith("element-hq.element-web-")
        assert tag.endswith("-vnan")

    def test_tags_are_truncated_to_what_docker_accepts(self):
        tag = dockerhub_tag(f"instance_x__y-{'a' * 200}", "org/repo")
        assert len(tag) == 128

    def test_a_row_without_a_repo_is_an_error_not_a_wrong_image(self):
        with pytest.raises(ValueError, match="dockerhub_tag"):
            dockerhub_tag("instance_x__y-abc", "")


class TestImageName:
    def test_the_dockerhub_tag_column_wins(self):
        image = pro_image_name(pro_record())
        assert image == f"{DEFAULT_IMAGE_PREFIX}/sweap-images:navidrome.navidrome-0130c6dc"

    def test_the_tag_is_derived_when_the_column_is_missing(self):
        record = pro_record()
        del record["dockerhub_tag"]
        assert pro_image_name(record).endswith(
            ":navidrome.navidrome-navidrome__navidrome-0130c6dc"
        )

    def test_a_custom_registry_is_honoured(self):
        image = pro_image_name(pro_record(), "myacr.azurecr.io/mirror")
        assert image.startswith("myacr.azurecr.io/mirror/sweap-images:")

    def test_an_explicit_image_on_the_row_wins_outright(self):
        image = pro_image_name(pro_record(image="ghcr.io/me/custom:1"))
        assert image == "ghcr.io/me/custom:1"


class TestProblemStatement:
    def test_requirements_and_interface_are_appended(self):
        statement = build_problem_statement(pro_record())
        assert "Playlists do not refresh." in statement
        assert "Requirements:\nThe playlist must refresh after import." in statement
        assert "New interfaces introduced:\nfunc (p *Playlist) Refresh() error" in statement

    def test_empty_sections_are_dropped_rather_than_pasted_as_none(self):
        statement = build_problem_statement(
            pro_record(requirements=None, interface="")
        )
        assert statement == "Playlists do not refresh."

    def test_the_agent_sees_the_full_specification(self):
        # A Pro task is not solvable from the issue text alone; this is the
        # only place the extra columns reach the harness.
        [instance] = instances_from_records([pro_record()])
        assert "Requirements:" in instance.problem_statement


class TestInstanceNormalization:
    def test_workdir_defaults_to_app_not_testbed(self):
        [instance] = instances_from_records([pro_record()])
        assert instance.workdir == DEFAULT_WORKDIR == "/app"

    def test_the_raw_row_is_carried_through_for_grading(self):
        [instance] = instances_from_records([pro_record()])
        assert instance.raw["fail_to_pass"] == "['TestRefresh', 'TestImport']"

    def test_a_row_without_an_instance_id_is_rejected(self):
        with pytest.raises(ValueError, match="instance_id"):
            instances_from_records([{"repo": "a/b"}])


class TestBenchmarkDetection:
    def test_an_explicit_benchmark_always_wins(self):
        cfg = DatasetConfig(name="swe-bench-verified", benchmark="swe-bench-pro")
        assert detect_benchmark(cfg) == BENCHMARK_SWEBENCH_PRO

    def test_full_names_shorthands_and_ids_are_all_recognized(self):
        for name in ("swe-bench-pro", "pro", "SWE-bench Pro", "swebench_pro"):
            assert (
                detect_benchmark(DatasetConfig(name=name)) == BENCHMARK_SWEBENCH_PRO
            ), name
        assert (
            detect_benchmark(DatasetConfig(name="ScaleAI/SWE-bench_Pro"))
            == BENCHMARK_SWEBENCH_PRO
        )

    def test_swebench_stays_the_default(self):
        assert (
            detect_benchmark(DatasetConfig(name="swe-bench-verified"))
            == BENCHMARK_SWEBENCH
        )
        assert detect_benchmark(DatasetConfig(name="verified")) == BENCHMARK_SWEBENCH
        assert (
            detect_benchmark(DatasetConfig(name="SWE-bench/SWE-bench_Verified"))
            == BENCHMARK_SWEBENCH
        )

    def test_a_local_jsonl_is_recognized_from_its_columns(self):
        cfg = DatasetConfig(name="/data/rows.jsonl")
        assert detect_benchmark(cfg, [pro_record()]) == BENCHMARK_SWEBENCH_PRO
        assert (
            detect_benchmark(cfg, [{"instance_id": "astropy__astropy-1", "patch": ""}])
            == BENCHMARK_SWEBENCH
        )

    def test_an_unknown_benchmark_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown benchmark"):
            normalize_benchmark("swe-bench-plus")


class TestSandboxTargets:
    def test_pro_overrides_the_swebench_defaults(self):
        sandbox = SandboxConfig()
        assert (
            resolve_image_prefix(BENCHMARK_SWEBENCH_PRO, sandbox.image_prefix)
            == DEFAULT_IMAGE_PREFIX
        )
        assert resolve_workdir(BENCHMARK_SWEBENCH_PRO, sandbox.workdir) == "/app"

    def test_a_swebench_mirror_is_not_used_for_pro_images(self):
        # configs/gold.yaml pins mirror.gcr.io/swebench; that registry has no
        # Pro images at all, so keeping it would fail every pod creation.
        assert (
            resolve_image_prefix(BENCHMARK_SWEBENCH_PRO, "mirror.gcr.io/swebench")
            == DEFAULT_IMAGE_PREFIX
        )

    def test_a_deliberate_pro_mirror_is_kept(self):
        assert (
            resolve_image_prefix(BENCHMARK_SWEBENCH_PRO, "myacr.azurecr.io/pro")
            == "myacr.azurecr.io/pro"
        )
        assert resolve_workdir(BENCHMARK_SWEBENCH_PRO, "/srv/app") == "/srv/app"

    def test_swebench_runs_are_left_alone(self):
        assert (
            resolve_image_prefix(BENCHMARK_SWEBENCH, "mirror.gcr.io/swebench")
            == "mirror.gcr.io/swebench"
        )
        assert resolve_workdir(BENCHMARK_SWEBENCH, "/testbed") == "/testbed"


class TestEntryScript:
    def test_only_the_last_line_of_the_build_command_is_used(self):
        assert before_repo_set_cmd(pro_record()) == "go mod download"

    def test_selected_test_files_are_passed_as_one_comma_separated_argument(self):
        script = build_entry_script(pro_record(), "/app")
        assert _run_script_argv(script) == [
            "bash",
            RUN_SCRIPT_PATH,
            "playlist_test.go,import_test.go",
        ]

    def test_no_selection_means_no_argument_at_all(self):
        # run_script.sh runs the whole suite when called with no argument, and
        # matches nothing at all when called with an empty one.
        script = build_entry_script(
            pro_record(selected_test_files_to_run="[]"), "/app"
        )
        assert _run_script_argv(script) == ["bash", RUN_SCRIPT_PATH]

    def test_the_build_step_runs_after_the_patch_and_before_the_tests(self):
        script = build_entry_script(pro_record(), "/app")
        assert script.index("go mod download") < script.index("run_script.sh")

    def test_the_parser_runs_over_the_captured_logs(self):
        script = build_entry_script(pro_record(), "/app")
        assert "/workspace/parser.py /workspace/stdout.log" in script
        assert OUTPUT_JSON in script

    def test_the_workdir_is_quoted(self):
        script = build_entry_script(pro_record(), "/path with spaces")
        assert "cd '/path with spaces'" in script


class TestGradeFromOutput:
    def output(self, **statuses):
        return {
            "tests": [
                {"name": name, "status": status} for name, status in statuses.items()
            ]
        }

    def test_all_required_tests_passing_resolves(self):
        grade = grade_from_output(
            self.output(TestRefresh="PASSED", TestExisting="PASSED"),
            ["TestRefresh"],
            ["TestExisting"],
        )
        assert grade.resolved
        assert grade.resolved_status == RESOLVED_FULL
        assert grade.reward == 1.0

    def test_a_failing_pass_to_pass_test_is_a_regression_not_a_pass(self):
        grade = grade_from_output(
            self.output(TestRefresh="PASSED", TestExisting="FAILED"),
            ["TestRefresh"],
            ["TestExisting"],
        )
        assert not grade.resolved
        assert grade.resolved_status == RESOLVED_PARTIAL
        assert grade.tests_status["PASS_TO_PASS"]["failure"] == ["TestExisting"]

    def test_a_test_the_parser_never_saw_counts_as_not_passing(self):
        grade = grade_from_output(self.output(), ["TestRefresh"], [])
        assert not grade.resolved
        assert grade.resolved_status == RESOLVED_NO
        assert grade.tests_status["FAIL_TO_PASS"]["failure"] == ["TestRefresh"]

    def test_a_skipped_test_does_not_count_as_passing(self):
        grade = grade_from_output(
            self.output(TestRefresh="SKIPPED"), ["TestRefresh"], []
        )
        assert not grade.resolved

    def test_an_empty_output_is_flagged_rather_than_scored_silently(self):
        grade = grade_from_output(self.output(), ["TestRefresh"], [])
        assert "no test results" in (grade.error or "")

    def test_an_empty_result_set_over_a_real_log_is_not_called_a_missing_run(self):
        # Pro's Go parsers only record column-0 `--- PASS:` lines, so a red
        # top-level test yields `{"tests": []}` from a perfectly healthy run.
        # Blaming the test command there sends the reader after a fault that is
        # not in the harness at all.
        log = "=== RUN   TestRefresh\n--- FAIL: TestRefresh (0.17s)\nFAIL\n"
        grade = grade_from_output(
            self.output(), ["TestRefresh"], [], stdout_tail=log
        )
        assert not grade.resolved
        assert "never ran" not in (grade.error or "")
        assert "the tests ran and none of them passed" in (grade.error or "")
        assert str(len(log)) in (grade.error or "")

    def test_an_empty_result_set_over_an_empty_log_still_blames_the_run(self):
        grade = grade_from_output(
            self.output(), ["TestRefresh"], [], stdout_tail="   \n"
        )
        assert "never ran" in (grade.error or "")

    def test_an_empty_fail_to_pass_list_is_flagged(self):
        # Vacuously resolved. Upstream scores it as a win; saying so is what
        # keeps the no-patch check able to find these rows.
        grade = grade_from_output(self.output(), [], [])
        assert grade.resolved
        assert "empty fail_to_pass" in (grade.error or "")


class TestDiagnoseMissingOutput:
    def test_a_killed_script_is_a_timeout(self):
        status, _ = diagnose_missing_output(124, "", "")
        assert status == "EVAL_TIMEOUT"

    def test_silence_is_infrastructure_not_a_wrong_answer(self):
        status, error = diagnose_missing_output(-1, "", "")
        assert status == "EVAL_INFRA_ERROR"
        assert "no output" in error

    def test_output_without_a_report_is_an_ordinary_failure(self):
        status, error = diagnose_missing_output(1, "compile error\n", "")
        assert status == RESOLVED_NO
        assert "compile error" in error


class TestRunScriptStore:
    def test_scripts_are_read_from_a_local_checkout(self, tmp_path):
        instance_dir = tmp_path / "instance_a__b-1"
        instance_dir.mkdir()
        (instance_dir / "run_script.sh").write_text("echo run", encoding="utf-8")
        (instance_dir / "parser.py").write_text("print('parse')", encoding="utf-8")

        store = RunScriptStore(tmp_path)
        assert store.load("instance_a__b-1") == {
            "run_script.sh": "echo run",
            "parser.py": "print('parse')",
        }

    def test_a_missing_script_says_what_to_configure(self, tmp_path):
        store = RunScriptStore(tmp_path)
        with pytest.raises(RunScriptError, match="pro_scripts_dir"):
            store.load("instance_a__b-1")

    def test_a_cache_hit_never_reaches_the_network(self, tmp_path):
        cached = tmp_path / "instance_a__b-1"
        cached.mkdir()
        (cached / "run_script.sh").write_text("echo run", encoding="utf-8")
        (cached / "parser.py").write_text("print('parse')", encoding="utf-8")

        store = RunScriptStore(cache_dir=tmp_path, allow_download=False)
        assert store.load("instance_a__b-1")["run_script.sh"] == "echo run"

    def test_a_cache_miss_without_downloads_fails_loudly(self, tmp_path):
        store = RunScriptStore(cache_dir=tmp_path, allow_download=False)
        with pytest.raises(RunScriptError, match="not cached"):
            store.load("instance_a__b-1")

    def test_path_traversal_in_an_instance_id_is_refused(self, tmp_path):
        store = RunScriptStore(cache_dir=tmp_path)
        with pytest.raises(RunScriptError, match="unsafe"):
            store.load("../../etc/passwd")


def store_with(tmp_path, instance_id: str) -> RunScriptStore:
    directory = tmp_path / instance_id
    directory.mkdir(parents=True)
    (directory / "run_script.sh").write_text("echo run", encoding="utf-8")
    (directory / "parser.py").write_text("print('parse')", encoding="utf-8")
    return RunScriptStore(tmp_path)


class TestGradeSwebenchProPatch:
    def instance(self):
        [instance] = instances_from_records([pro_record()])
        return instance

    def sandbox(self, responder=None):
        sandbox_instance = FakeSandboxInstance(responder=responder)
        return EvalSandbox(sandbox_instance, workdir="/app"), sandbox_instance

    @pytest.mark.asyncio
    async def test_an_empty_patch_never_spends_a_pod(self, tmp_path):
        sandbox, _ = self.sandbox()
        grade = await grade_swebench_pro_patch(
            sandbox,
            self.instance(),
            "",
            scripts=store_with(tmp_path, "instance_navidrome__navidrome-0130c6dc"),
        )
        assert grade.resolved_status == "EMPTY_PATCH"

    @pytest.mark.asyncio
    async def test_a_passing_run_is_resolved(self, tmp_path):
        instance = self.instance()
        sandbox, raw = self.sandbox()
        raw.files[OUTPUT_JSON] = json.dumps(
            {
                "tests": [
                    {"name": "TestRefresh", "status": "PASSED"},
                    {"name": "TestImport", "status": "PASSED"},
                    {"name": "TestExisting", "status": "PASSED"},
                ]
            }
        ).encode()

        grade = await grade_swebench_pro_patch(
            sandbox,
            instance,
            "diff --git a/a.go b/a.go\n",
            scripts=store_with(tmp_path, instance.instance_id),
        )
        assert grade.resolved
        assert grade.resolved_status == RESOLVED_FULL
        # The instance's own scripts have to be in the pod for any of this to
        # have meant anything.
        assert "/workspace/run_script.sh" in raw.files
        assert "/workspace/parser.py" in raw.files
        assert ENTRY_SCRIPT_PATH in raw.files

    @pytest.mark.asyncio
    async def test_a_failed_reset_is_not_scored_as_a_wrong_patch(self, tmp_path):
        def responder(command: str):
            if "git reset --hard" in command:
                return FakeJobResult(stderr="unknown revision", exit_code=128)
            return None

        instance = self.instance()
        sandbox, _ = self.sandbox(responder)
        grade = await grade_swebench_pro_patch(
            sandbox,
            instance,
            "diff --git a/a.go b/a.go\n",
            scripts=store_with(tmp_path, instance.instance_id),
        )
        assert grade.resolved_status == "EVAL_RESET_FAILED"

    @pytest.mark.asyncio
    async def test_a_missing_output_json_is_diagnosed_not_scored(self, tmp_path):
        instance = self.instance()
        sandbox, _ = self.sandbox()  # no OUTPUT_JSON staged
        grade = await grade_swebench_pro_patch(
            sandbox,
            instance,
            "diff --git a/a.go b/a.go\n",
            scripts=store_with(tmp_path, instance.instance_id),
        )
        assert not grade.resolved
        assert grade.resolved_status in {"EVAL_INFRA_ERROR", RESOLVED_NO}

    @pytest.mark.asyncio
    async def test_unavailable_scripts_are_their_own_status(self, tmp_path):
        sandbox, _ = self.sandbox()
        grade = await grade_swebench_pro_patch(
            sandbox,
            self.instance(),
            "diff --git a/a.go b/a.go\n",
            scripts=RunScriptStore(tmp_path),
        )
        assert grade.resolved_status == "EVAL_SCRIPT_MISSING"
