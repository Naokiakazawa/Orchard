"""Report aggregation."""

import json

from orchard_evalkit.config import RunConfig
from orchard_evalkit.models import InstanceRecord, Timings
from orchard_evalkit.report import (
    format_summary,
    summarize,
    write_predictions,
    write_summary,
)


def _record(instance_id: str, *, resolved: bool = False, **kwargs) -> InstanceRecord:
    return InstanceRecord(
        instance_id=instance_id,
        harness="codex",
        model="gpt-5",
        resolved=resolved,
        reward=1.0 if resolved else 0.0,
        **kwargs,
    )


class TestSummarize:
    def test_resolve_rate(self):
        summary = summarize(
            [_record("a", resolved=True), _record("b"), _record("c", resolved=True)]
        )
        assert summary.total == 3
        assert summary.resolved == 2
        assert summary.resolve_rate == round(2 / 3, 4)

    def test_empty_run_does_not_divide_by_zero(self):
        assert summarize([]).resolve_rate == 0.0

    def test_truncated_rollouts_are_counted(self):
        # A harness that quits on a truncated completion produces rollouts
        # that look clean and resolve nothing; the count is the only warning.
        summary = summarize(
            [
                _record("a", metrics={"length_stops": 1}),
                _record("b", metrics={"length_stops": 0}),
                _record("c", metrics={}),
            ]
        )
        assert summary.length_truncated == 1
        assert "truncated" in format_summary(summary)

    def test_a_run_with_no_truncation_says_nothing_about_it(self):
        assert "truncated" not in format_summary(summarize([_record("a")]))

    def test_empty_patches_are_counted(self):
        summary = summarize([_record("a", patch=""), _record("b", patch="diff")])
        assert summary.empty_patches == 1

    def test_exit_statuses_are_histogrammed(self):
        summary = summarize(
            [
                _record("a", exit_status="completed"),
                _record("b", exit_status="completed"),
                _record("c", exit_status="agent_error"),
            ]
        )
        assert summary.exit_statuses == {"agent_error": 1, "completed": 2}

    def test_cost_is_summed_from_metrics(self):
        summary = summarize(
            [_record("a", metrics={"cost": 0.5}), _record("b", metrics={"cost": 1.25})]
        )
        assert summary.total_cost == 1.75

    def test_timings_are_averaged(self):
        summary = summarize(
            [
                _record("a", timings=Timings(rollout_s=10, total_s=20)),
                _record("b", timings=Timings(rollout_s=20, total_s=40)),
            ]
        )
        assert summary.mean_rollout_s == 15.0
        assert summary.mean_total_s == 30.0

    def test_config_supplies_run_metadata(self):
        config = RunConfig(
            run_name="r", harness={"name": "pi"}, dataset={"name": "verified"}
        )
        summary = summarize([_record("a")], config)
        assert summary.run_name == "r"
        assert summary.harness == "pi"
        assert summary.dataset == "verified"


class TestWriters:
    def test_summary_is_written_as_json(self, tmp_path):
        path = tmp_path / "nested" / "summary.json"
        write_summary(summarize([_record("a", resolved=True)]), path)
        assert json.loads(path.read_text())["resolved"] == 1

    def test_predictions_use_the_swebench_format(self, tmp_path):
        # This file is what lets someone re-grade the run with the official
        # harness, independently of this code.
        path = tmp_path / "preds.json"
        write_predictions([_record("a__b-1", patch="diff --git a/f b/f")], path, "m1")
        payload = json.loads(path.read_text())
        assert payload["a__b-1"] == {
            "instance_id": "a__b-1",
            "model_name_or_path": "m1",
            "model_patch": "diff --git a/f b/f",
        }


class TestFormatSummary:
    def test_renders_the_headline_numbers(self):
        text = format_summary(summarize([_record("a", resolved=True), _record("b")]))
        assert "resolve rate    50.00%" in text
        assert "instances       2" in text
