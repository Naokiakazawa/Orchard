"""Pairing an ``orchard-eval run`` against the same benchmark run via Harbor.

``scripts/`` is not a package, so the module is loaded by path — the same thing
``python scripts/compare_swebench_pro.py`` does, without the subprocess.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "compare_swebench_pro.py"


def _load():
    spec = importlib.util.spec_from_file_location("compare_swebench_pro", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


compare_swebench_pro = _load()


def write_run(root: Path, rows: list[dict], *, run_id: str = "20260910-120000") -> Path:
    attempt = root / run_id
    attempt.mkdir(parents=True)
    (attempt / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return attempt


def write_trial(job: Path, task: str, reward: float, *, trial: str = "t") -> None:
    directory = job / f"{task.rsplit('/', 1)[-1]}__{trial}"
    directory.mkdir(parents=True)
    (directory / "result.json").write_text(
        json.dumps(
            {
                "task_name": task,
                "trial_name": directory.name,
                "verifier_result": {"rewards": {"reward": reward}},
            }
        ),
        encoding="utf-8",
    )


class TestInstanceIdOf:
    def test_strips_the_publishing_org(self):
        assert (
            compare_swebench_pro.instance_id_of("scale-ai/instance_ansible__x-a-vb")
            == "instance_ansible__x-a-vb"
        )

    def test_leaves_an_unprefixed_name_alone(self):
        assert compare_swebench_pro.instance_id_of("instance_x") == "instance_x"


class TestNewestAttempt:
    def test_resolves_a_run_directory_to_its_newest_attempt(self, tmp_path):
        run = tmp_path / "run"
        write_run(run, [{"instance_id": "a", "resolved": True}], run_id="20260101-000000")
        newest = write_run(
            run, [{"instance_id": "a", "resolved": False}], run_id="20260202-000000"
        )
        assert compare_swebench_pro.newest_attempt(run) == newest / "results.jsonl"

    def test_accepts_the_attempt_directory(self, tmp_path):
        attempt = write_run(tmp_path / "run", [{"instance_id": "a", "resolved": True}])
        assert (
            compare_swebench_pro.newest_attempt(attempt) == attempt / "results.jsonl"
        )

    def test_accepts_the_file_itself(self, tmp_path):
        attempt = write_run(tmp_path / "run", [{"instance_id": "a", "resolved": True}])
        path = attempt / "results.jsonl"
        assert compare_swebench_pro.newest_attempt(path) == path

    def test_names_the_directory_when_there_is_nothing_to_read(self, tmp_path):
        empty = tmp_path / "run"
        empty.mkdir()
        with pytest.raises(SystemExit, match=str(empty)):
            compare_swebench_pro.newest_attempt(empty)


class TestReadRun:
    def test_last_record_wins(self, tmp_path):
        """A resumed run appends; the redo is the record that counts."""
        attempt = write_run(
            tmp_path / "run",
            [
                {"instance_id": "a", "resolved": False},
                {"instance_id": "b", "resolved": True},
                {"instance_id": "a", "resolved": True},
            ],
        )
        assert compare_swebench_pro.read_run(attempt / "results.jsonl") == {
            "a": True,
            "b": True,
        }

    def test_a_half_written_line_is_skipped(self, tmp_path):
        """Readable mid-run, which is when the file has a partial last line."""
        attempt = write_run(tmp_path / "run", [{"instance_id": "a", "resolved": True}])
        path = attempt / "results.jsonl"
        path.write_text(path.read_text() + '{"instance_id": "b", "res')
        assert compare_swebench_pro.read_run(path) == {"a": True}


class TestReadHarbor:
    def test_maps_task_names_onto_instance_ids(self, tmp_path):
        job = tmp_path / "job"
        write_trial(job, "scale-ai/instance_a", 1.0)
        write_trial(job, "scale-ai/instance_b", 0.0)
        solved, trials = compare_swebench_pro.read_harbor(job)
        assert solved == {"instance_a": True, "instance_b": False}
        assert trials == 2

    def test_a_task_counts_as_solved_when_any_attempt_solved_it(self, tmp_path):
        job = tmp_path / "job"
        write_trial(job, "scale-ai/instance_a", 0.0, trial="1")
        write_trial(job, "scale-ai/instance_a", 1.0, trial="2")
        solved, trials = compare_swebench_pro.read_harbor(job)
        assert solved == {"instance_a": True}
        assert trials == 2

    def test_the_job_level_result_is_not_a_trial(self, tmp_path):
        """One ``result.json`` sits at the job root with a different schema."""
        job = tmp_path / "job"
        write_trial(job, "scale-ai/instance_a", 1.0)
        (job / "result.json").write_text(json.dumps({"n_trials": 1}), encoding="utf-8")
        solved, trials = compare_swebench_pro.read_harbor(job)
        assert solved == {"instance_a": True}
        assert trials == 1

    def test_names_the_directory_when_there_are_no_trials(self, tmp_path):
        job = tmp_path / "job"
        job.mkdir()
        with pytest.raises(SystemExit, match=str(job)):
            compare_swebench_pro.read_harbor(job)


class TestCompare:
    def test_cross_tabulates_the_shared_instances(self):
        run = {"a": True, "b": True, "c": False, "d": False}
        harbor = {"a": True, "b": False, "c": True, "d": False}
        result = compare_swebench_pro.compare(run, harbor)
        assert result["both"] == ["a"]
        assert result["run_only"] == ["b"]
        assert result["harbor_only"] == ["c"]
        assert result["neither"] == ["d"]
        assert result["agreement"] == 0.5

    def test_instances_only_one_path_ran_are_reported_separately(self):
        """They are not disagreements, and averaging them in would say they are."""
        result = compare_swebench_pro.compare({"a": True, "x": True}, {"a": True, "y": False})
        assert result["shared"] == ["a"]
        assert result["run_only_in_run"] == ["x"]
        assert result["run_only_in_harbor"] == ["y"]
        assert result["agreement"] == 1.0

    def test_no_overlap_is_zero_agreement_not_a_crash(self):
        result = compare_swebench_pro.compare({"a": True}, {"b": True})
        assert result["shared"] == []
        assert result["agreement"] == 0.0

    def test_harbors_lower_cased_task_name_still_pairs(self):
        # Harbor lower-cases a task name when it publishes it and the dataset
        # does not, which silently dropped all 44 NodeBB instances from every
        # SWE-bench Pro comparison and reported it as coverage.
        run = {"instance_NodeBB__NodeBB-abc-vdef": True}
        harbor = {"instance_nodebb__nodebb-abc-vdef": False}
        result = compare_swebench_pro.compare(run, harbor)
        assert result["run_only"] == ["instance_NodeBB__NodeBB-abc-vdef"]
        assert result["run_only_in_run"] == []
        assert result["run_only_in_harbor"] == []

    def test_the_run_names_the_shared_instances(self):
        """The run's spelling is what results.jsonl and instances/ use."""
        result = compare_swebench_pro.compare({"Ab": True}, {"ab": True})
        assert result["shared"] == ["Ab"]
        assert result["both"] == ["Ab"]


class TestFormatReport:
    def _report(self, run, harbor, **kwargs):
        return compare_swebench_pro.format_report(
            run,
            harbor,
            compare_swebench_pro.compare(run, harbor),
            run_label="results/run",
            harbor_label="results/harbor/job",
            trials=len(harbor),
            list_all=kwargs.get("list_all", False),
        )

    def test_reports_both_rates_and_the_disagreements(self):
        report = self._report({"a": True, "b": True}, {"a": True, "b": False})
        assert "50.00%" in report  # agreement over the two shared instances
        assert "run only" in report
        assert "b" in report

    def test_says_so_when_the_two_runs_share_nothing(self):
        report = self._report({"a": True}, {"b": True})
        assert "share no instances" in report
        assert "--emit-harbor-tasks" in report

    def test_long_lists_are_truncated_unless_asked(self):
        limit = compare_swebench_pro.DEFAULT_LIST_LIMIT
        run = {f"i{n:03d}": True for n in range(limit + 5)}
        harbor = dict.fromkeys(run, False)
        assert "and 5 more" in self._report(run, harbor)
        assert "and 5 more" not in self._report(run, harbor, list_all=True)


class TestEmitHarborTasks:
    def test_names_every_instance_the_run_covered(self):
        emitted = compare_swebench_pro.emit_harbor_tasks({"b": True, "a": False})
        assert emitted.splitlines()[0].strip().startswith("--task scale-ai/a")
        assert "scale-ai/b" in emitted

    def test_emits_harbors_spelling_of_the_name(self):
        # Harbor filters with fnmatch, which is case-sensitive, so the
        # dataset's spelling would pin the job to no tasks at all.
        emitted = compare_swebench_pro.emit_harbor_tasks({"instance_NodeBB__x": True})
        assert "scale-ai/instance_nodebb__x" in emitted


class TestMain:
    def test_end_to_end(self, tmp_path, capsys):
        run = tmp_path / "run"
        write_run(
            run,
            [
                {"instance_id": "instance_a", "resolved": True},
                {"instance_id": "instance_b", "resolved": False},
            ],
        )
        job = tmp_path / "job"
        write_trial(job, "scale-ai/instance_a", 1.0)
        write_trial(job, "scale-ai/instance_b", 1.0)

        pairing = tmp_path / "pairing.json"
        assert compare_swebench_pro.main([str(run), str(job), "--json", str(pairing)]) == 0

        out = capsys.readouterr().out
        assert "harbor only" in out
        assert "instance_b" in out
        payload = json.loads(pairing.read_text())
        assert payload["harbor_only"] == ["instance_b"]
        assert payload["both"] == ["instance_a"]

    def test_emit_mode_needs_no_harbor_job(self, tmp_path, capsys):
        run = tmp_path / "run"
        write_run(run, [{"instance_id": "instance_a", "resolved": True}])
        assert compare_swebench_pro.main([str(run), "--emit-harbor-tasks"]) == 0
        assert "--task scale-ai/instance_a" in capsys.readouterr().out

    def test_the_harbor_job_is_otherwise_required(self, tmp_path):
        run = tmp_path / "run"
        write_run(run, [{"instance_id": "instance_a", "resolved": True}])
        with pytest.raises(SystemExit):
            compare_swebench_pro.main([str(run)])
