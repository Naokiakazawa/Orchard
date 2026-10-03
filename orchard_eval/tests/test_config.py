"""Config layering and override parsing."""

import ast
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pytest

from orchard_evalkit.config import (
    ModelConfig,
    RunConfig,
    anthropic_base_url,
    deep_merge,
    expand_ports,
    load_config,
    parse_overrides,
    session_url,
)

#: harbor_orchard's copy of ``anthropic_base_url``, compiled straight out of the
#: file rather than imported. See ``test_the_harbor_copy_agrees``.
_HARBOR_SETTINGS = (
    Path(__file__).resolve().parent.parent
    / "harbor_orchard"
    / "harbor_orchard"
    / "settings.py"
)


def _load_harbor_anthropic_base_url():
    source = _HARBOR_SETTINGS.read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "anthropic_base_url":
            module = ast.Module(body=[node], type_ignores=[])
            namespace = {"urlsplit": urlsplit, "urlunsplit": urlunsplit}
            exec(compile(module, str(_HARBOR_SETTINGS), "exec"), namespace)
            return namespace["anthropic_base_url"]
    raise AssertionError(
        f"harbor_orchard lost its anthropic_base_url ({_HARBOR_SETTINGS}); the "
        "Harbor arm of claude-code would start requesting /v1/v1/messages"
    )


class TestParseOverrides:
    def test_dotted_keys_become_nested(self):
        assert parse_overrides(["dataset.limit=10"]) == {"dataset": {"limit": 10}}

    def test_values_are_yaml_typed(self):
        parsed = parse_overrides(
            ["resume=false", "concurrency=64", "dataset.instance_ids=[a, b]"]
        )
        assert parsed["resume"] is False
        assert parsed["concurrency"] == 64
        assert parsed["dataset"]["instance_ids"] == ["a", "b"]

    def test_multiple_overrides_share_a_branch(self):
        parsed = parse_overrides(["dataset.limit=5", "dataset.split=dev"])
        assert parsed["dataset"] == {"limit": 5, "split": "dev"}

    @pytest.mark.parametrize("bad", ["nokey", "a..b=1", ".x=1"])
    def test_malformed_overrides_are_rejected(self, bad):
        with pytest.raises(ValueError):
            parse_overrides([bad])


class TestEndpointFleet:
    """Several servers behind one model, one of them per rollout."""

    def test_replicas_expand_to_consecutive_ports(self):
        assert expand_ports("http://h:30021/v1", 3) == [
            "http://h:30021/v1",
            "http://h:30022/v1",
            "http://h:30023/v1",
        ]

    def test_replicas_need_an_explicit_port(self):
        with pytest.raises(ValueError, match="explicit port"):
            ModelConfig(base_url="http://h/v1", base_url_replicas=4)

    def test_a_single_endpoint_is_left_alone(self):
        model = ModelConfig(base_url="http://h:8000/v1")
        assert model.for_key("django__django-11095") is model

    def test_one_instance_always_gets_the_same_endpoint(self):
        # The whole point: turn 2 of a loop must reach the server that cached
        # turn 1, and so must a retry and a resumed run.
        model = ModelConfig(base_url="http://h:30021/v1", base_url_replicas=8)
        chosen = model.for_key("astropy__astropy-12907").base_url
        assert chosen in model.endpoints()
        assert model.for_key("astropy__astropy-12907").base_url == chosen

    def test_instances_spread_across_the_fleet(self):
        model = ModelConfig(base_url="http://h:30021/v1", base_url_replicas=8)
        used = {model.for_key(f"repo__repo-{i}").base_url for i in range(200)}
        assert used == set(model.endpoints())

    def test_explicit_endpoints_win_over_replicas(self):
        model = ModelConfig(
            base_url="http://h:30021/v1",
            base_url_replicas=8,
            base_urls=["http://a:8000/v1", "http://b:8000/v1"],
        )
        assert model.endpoints() == ["http://a:8000/v1", "http://b:8000/v1"]

    def test_unknown_routing_is_rejected(self):
        with pytest.raises(ValueError, match="model.routing"):
            ModelConfig(base_url="http://h:8000/v1", routing="round-robin")


class TestSessionRouting:
    """One router in front of the fleet, addressed by a path segment."""

    def test_the_session_id_lands_in_the_path(self):
        model = ModelConfig(base_url="http://h:30021/v1", routing="session")
        assert (
            model.for_key("django__django-11095").base_url
            == "http://h:30021/session/django__django-11095/v1"
        )

    def test_the_base_url_stays_a_v1_root(self):
        # codex appends its protocol path to whatever it is given, so the
        # segment has to go before /v1 rather than after it.
        model = ModelConfig(base_url="http://h:30021/v1", routing="session")
        assert model.for_key("x").base_url.endswith("/v1")

    def test_a_slash_in_the_key_is_escaped(self):
        # A Harbor task name is `org/task`, and an unescaped slash would be
        # read by the router as the start of the upstream path.
        model = ModelConfig(base_url="http://h:30021/v1", routing="session")
        assert "/session/org%2Ftask/v1" in model.for_key("org/task").base_url

    def test_a_single_router_is_not_short_circuited(self):
        # The regression this mode is one line away from: endpoint_for returns
        # base_url untouched for any fleet of one, which would leave every
        # rollout unkeyed and round-robined while looking like it worked.
        model = ModelConfig(base_url="http://h:30021/v1", routing="session")
        assert model.for_key("x").base_url != model.base_url

    def test_one_instance_always_gets_the_same_url(self):
        model = ModelConfig(base_url="http://h:30021/v1", routing="session")
        first = model.for_key("astropy__astropy-12907").base_url
        assert model.for_key("astropy__astropy-12907").base_url == first

    def test_a_fleet_behind_a_router_is_rejected(self):
        with pytest.raises(ValueError, match="session"):
            ModelConfig(
                base_url="http://h:30021/v1", base_url_replicas=8, routing="session"
            )

    def test_an_already_prefixed_base_url_is_rejected(self):
        # The natural mistake is copying a URL out of a log.
        with pytest.raises(ValueError, match="session"):
            ModelConfig(base_url="http://h:30021/session/a/v1", routing="session")

    def test_session_url_needs_an_absolute_base(self):
        with pytest.raises(ValueError, match="absolute"):
            session_url("/v1", "x")


class TestAnthropicBaseUrl:
    """The ``/v1`` that every other client requires, and Claude Code appends.

    Getting this wrong does not fail loudly: the CLI requests
    ``/v1/v1/messages``, sglang answers 404, and the rollout ends with an empty
    patch that looks exactly like a model that solved nothing.
    """

    def test_it_strips_the_trailing_v1(self):
        assert (
            anthropic_base_url("http://h:30021/session/a/v1")
            == "http://h:30021/session/a"
        )

    def test_it_leaves_a_url_without_one_alone(self):
        assert anthropic_base_url("http://h:30021/session/a") == (
            "http://h:30021/session/a"
        )

    def test_it_is_idempotent(self):
        once = anthropic_base_url("http://h:30021/v1")
        assert anthropic_base_url(once) == once

    def test_it_strips_only_one_segment(self):
        # /v1/v1 is not a path this fleet serves, but stripping greedily would
        # turn a typo into a different wrong URL rather than the same one.
        assert anthropic_base_url("http://h:30021/v1/v1") == "http://h:30021/v1"

    def test_it_keeps_scheme_host_and_query(self):
        assert (
            anthropic_base_url("https://h:30021/session/a/v1?x=1")
            == "https://h:30021/session/a?x=1"
        )

    def test_it_does_not_strip_a_v1_inside_a_segment(self):
        # A served model id can end in anything; `/apiv1` is not `/v1`.
        assert anthropic_base_url("http://h:30021/apiv1") == "http://h:30021/apiv1"

    def test_the_session_url_round_trip(self):
        # The composition that actually runs: session routing builds the URL,
        # this strips it for the one client that re-adds the segment.
        assert anthropic_base_url(session_url("http://h:30021/v1", "org/task")) == (
            "http://h:30021/session/org%2Ftask"
        )

    def test_the_harbor_copy_agrees(self):
        # Duplicated on purpose — harbor_orchard is imported in processes where
        # orchard_evalkit is not installed — so nothing but a test keeps the two
        # honest. Lifted out of the source rather than imported: importing
        # harbor_orchard.settings drags in Harbor itself, which is not installed
        # everywhere this suite runs, and a skipped test would keep the two in
        # step only on the machines that least need it.
        harbor_copy = _load_harbor_anthropic_base_url()
        for url in (
            "http://h:30021/v1",
            "http://h:30021/session/a/v1",
            "http://h:30021/session/a",
            "https://h:30021/session/a/v1?x=1",
            "http://h:30021/apiv1",
            "http://h:30021/v1/v1",
        ):
            assert anthropic_base_url(url) == harbor_copy(url), url


class TestDeepMerge:
    def test_nested_dicts_merge_rather_than_replace(self):
        merged = deep_merge(
            {"sandbox": {"cpu": "2", "memory": "8Gi"}}, {"sandbox": {"cpu": "8"}}
        )
        assert merged == {"sandbox": {"cpu": "8", "memory": "8Gi"}}

    def test_inputs_are_not_mutated(self):
        base = {"a": {"b": 1}}
        deep_merge(base, {"a": {"c": 2}})
        assert base == {"a": {"b": 1}}


class TestLoadConfig:
    def test_later_files_win_over_earlier_ones(self, tmp_path):
        first = tmp_path / "a.yaml"
        first.write_text("concurrency: 4\nharness:\n  name: codex\n")
        second = tmp_path / "b.yaml"
        second.write_text("concurrency: 16\n")

        config = load_config([str(first), str(second)])
        assert config.concurrency == 16
        # Untouched keys from the earlier file survive the merge.
        assert config.harness.name == "codex"

    def test_cli_overrides_beat_files(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("concurrency: 4\n")
        config = load_config([str(path)], ["concurrency=99"])
        assert config.concurrency == 99

    def test_missing_file_is_an_error(self):
        with pytest.raises(FileNotFoundError):
            load_config(["/nonexistent/config.yaml"])


class TestEnvExpansion:
    def test_env_vars_are_substituted(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MODEL_BASE_URL", "http://vllm:8000/v1")
        path = tmp_path / "c.yaml"
        path.write_text("model:\n  base_url: ${MODEL_BASE_URL}\n")
        assert load_config([str(path)]).model.base_url == "http://vllm:8000/v1"

    def test_unset_variable_is_an_error(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MODEL_BASE_URL", raising=False)
        path = tmp_path / "c.yaml"
        path.write_text("model:\n  base_url: ${MODEL_BASE_URL}\n")
        with pytest.raises(ValueError, match="MODEL_BASE_URL"):
            load_config([str(path)])

    def test_expansion_reaches_nested_lists(self, tmp_path, monkeypatch):
        monkeypatch.setenv("STEP_LIMIT", "50")
        path = tmp_path / "c.yaml"
        path.write_text(
            "harness:\n  params:\n    config_overrides:\n"
            "      - agent.step_limit=${STEP_LIMIT}\n"
        )
        config = load_config([str(path)])
        assert config.harness.params["config_overrides"] == ["agent.step_limit=50"]

    def test_cli_overrides_still_win(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MODEL_NAME", "from-env")
        path = tmp_path / "c.yaml"
        path.write_text("model:\n  name: ${MODEL_NAME}\n")
        config = load_config([str(path)], ["model.name=from-cli"])
        assert config.model.name == "from-cli"


class TestRunConfig:
    def test_run_name_defaults_to_harness_and_model(self):
        config = RunConfig(harness={"name": "pi"}, model={"name": "openai/gpt-5"})
        assert config.run_name == "pi__openai-gpt-5"

    def test_explicit_run_name_is_kept(self):
        assert RunConfig(run_name="my-run").run_name == "my-run"

    def test_paths_derive_from_output_dir_run_name_and_run_id(self, tmp_path):
        config = RunConfig(output_dir=tmp_path, run_name="r1", run_id="20260827-120000")
        run_dir = tmp_path / "r1" / "20260827-120000"
        assert config.run_dir == run_dir
        assert config.results_path == run_dir / "results.jsonl"
        assert config.instances_dir == run_dir / "instances"

    def test_run_id_defaults_to_a_timestamp(self, tmp_path):
        config = RunConfig(output_dir=tmp_path, run_name="r1")
        assert re.fullmatch(r"\d{8}-\d{6}", config.run_id)

    def test_a_fresh_run_does_not_reuse_a_previous_attempt(self, tmp_path):
        # Re-running a config must never overwrite the earlier run's artifacts.
        previous = tmp_path / "r1" / "20200101-000000"
        previous.mkdir(parents=True)
        (previous / "results.jsonl").touch()
        config = RunConfig(output_dir=tmp_path, run_name="r1", resume=False)
        assert config.run_dir != previous

    def test_resume_continues_the_newest_attempt(self, tmp_path):
        # Otherwise every resume would open an empty directory beside the run
        # it was meant to continue, and re-run all of it.
        for name in ("20200101-000000", "20260101-000000"):
            attempt = tmp_path / "r1" / name
            attempt.mkdir(parents=True)
            (attempt / "results.jsonl").touch()
        config = RunConfig(output_dir=tmp_path, run_name="r1", resume=True)
        assert config.run_id == "20260101-000000"

    def test_an_attempt_that_recorded_nothing_is_not_resumed(self, tmp_path):
        (tmp_path / "r1" / "20260101-000000").mkdir(parents=True)
        config = RunConfig(output_dir=tmp_path, run_name="r1", resume=True)
        assert config.run_id != "20260101-000000"

    def test_explicit_run_id_is_kept(self, tmp_path):
        config = RunConfig(output_dir=tmp_path, run_name="r1", run_id="pinned")
        assert config.run_id == "pinned"


class TestModelConfig:
    def test_api_key_env_is_resolved_at_use_time(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "secret-value")
        config = RunConfig(model={"api_key_env": "MY_KEY"})
        assert config.model.resolve_api_key() == "secret-value"

    def test_missing_env_var_resolves_to_none(self, monkeypatch):
        monkeypatch.delenv("ABSENT_KEY", raising=False)
        config = RunConfig(model={"api_key_env": "ABSENT_KEY"})
        assert config.model.resolve_api_key() is None
