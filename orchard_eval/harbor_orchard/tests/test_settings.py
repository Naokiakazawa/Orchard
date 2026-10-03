"""Endpoint resolution for a fleet of interchangeable model servers.

Harbor runs every trial in one process, so ``OPENAI_BASE_URL`` cannot vary per
trial there. It varies here instead, and these tests pin the two properties the
KV-cache argument depends on: the mapping is a function of the task name, and
every server gets used.
"""

from __future__ import annotations

import json

import pytest

from harbor_orchard.agent_config import PI_MODEL_ALIAS, codex_config, pi_config
from harbor_orchard.settings import (
    ConfigurationError,
    OrchardSettings,
    agent_deadline,
    expand_ports,
    load_settings,
    task_agent_timeout_sec,
)

FLEET = tuple(f"http://h:{port}/v1" for port in range(30021, 30029))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        "MODEL_BASE_URL",
        "MODEL_BASE_URLS",
        "MODEL_BASE_URL_REPLICAS",
        "MODEL_ROUTING",
        "MODEL_WIRE_API",
        "ORCHARD_HARBOR_EGRESS_ALLOW",
        "ORCHARD_HARBOR_MODEL_EGRESS",
        "ORCHARD_HARBOR_ALLOW_INTERNET",
        "ORCHARD_HARBOR_STEP_TIMEOUT",
        "ORCHARD_HARBOR_EXEC_TIMEOUT",
        "ORCHARD_HARBOR_LIVENESS_INTERVAL",
        "ORCHARD_HARBOR_IMAGE_REMAP",
        "ORCHARD_HARBOR_AGENT_MARGIN",
        "ORCHARD_HARBOR_AGENT_MULTIPLIER",
        "ORCHARD_HARBOR_AGENT_TIMEOUT_SEC",
    ):
        monkeypatch.delenv(name, raising=False)


class TestAgentDeadline:
    """Stopping the in-pod CLI before Harbor stops waiting on it.

    Harbor's deadline is an ``asyncio.wait_for`` around the agent coroutine: it
    cancels the await and leaves the CLI running in the pod, where it blocks
    log collection and the verifier behind it. Measured on a 731-trial
    SWE-bench Pro run, 34% of timed-out trials then sat for a median of 65
    minutes; trials whose agent exited cleanly never did (0 of 209). These
    pin the arithmetic that makes the CLI stop first.
    """

    def test_it_reproduces_harbors_deadline_less_the_margin(self):
        # DeepSWE: 5400s x 1.5 = 8100, which is what Harbor enforces.
        assert (
            agent_deadline(5400, multiplier=1.5, margin=120, exec_timeout=9000) == 7980
        )

    def test_a_multiplier_of_one_is_the_bare_harbor_run(self):
        assert (
            agent_deadline(3000, multiplier=1.0, margin=120, exec_timeout=7200) == 2880
        )

    def test_it_stays_under_the_exec_timeout(self):
        # The orchestrator abandons an exec at exec_timeout, so a deadline
        # beyond it would never fire and the CLI would be orphaned anyway.
        assert (
            agent_deadline(5400, multiplier=1.5, margin=120, exec_timeout=3600) == 3480
        )

    def test_a_zero_margin_turns_it_off(self):
        # The escape hatch: back to the behaviour this replaces.
        assert agent_deadline(5400, multiplier=1.5, margin=0, exec_timeout=9000) is None

    def test_a_task_with_no_agent_timeout_gets_none(self):
        # Harbor leaves such a phase unbounded; so does this.
        assert agent_deadline(None, multiplier=1.5, margin=120, exec_timeout=9000) is None

    def test_a_margin_wider_than_the_budget_gets_none(self):
        # Rather than a negative deadline, which `timeout` reads as "expired"
        # and would kill every rollout on turn 0.
        assert agent_deadline(60, multiplier=1.0, margin=120, exec_timeout=9000) is None


class TestTaskAgentTimeout:
    """Harbor hands an environment its ``environment/`` directory and no other
    fact about the task, so the budget is read from the task.toml beside it."""

    def _task(self, tmp_path, body):
        (tmp_path / "task.toml").write_text(body)
        environment_dir = tmp_path / "environment"
        environment_dir.mkdir()
        return environment_dir

    def test_it_reads_the_agent_budget(self, tmp_path):
        environment_dir = self._task(
            tmp_path, '[agent]\nnetwork_mode = "no-network"\ntimeout_sec = 5400.0\n'
        )
        assert task_agent_timeout_sec(environment_dir) == 5400.0

    def test_a_task_without_one_gets_none(self, tmp_path):
        environment_dir = self._task(tmp_path, '[agent]\nnetwork_mode = "no-network"\n')
        assert task_agent_timeout_sec(environment_dir) is None

    def test_a_missing_task_toml_gets_none(self, tmp_path):
        environment_dir = tmp_path / "environment"
        environment_dir.mkdir()
        assert task_agent_timeout_sec(environment_dir) is None

    def test_unparseable_toml_gives_up_rather_than_failing_the_trial(self, tmp_path):
        # A deadline is an optimisation over letting the CLI run to the exec
        # timeout. Raising here would turn a cosmetic problem into a lost run.
        environment_dir = self._task(tmp_path, "[agent\ntimeout_sec = ")
        assert task_agent_timeout_sec(environment_dir) is None

    def test_no_environment_dir_gets_none(self):
        assert task_agent_timeout_sec(None) is None


class TestAgentDeadlineSettings:
    def test_the_margin_defaults_to_two_minutes(self):
        assert load_settings().agent_deadline_margin == 120

    def test_the_multiplier_defaults_to_one(self):
        # A bare `harbor run` passes no multiplier. Guessing 1.0 can only
        # under-estimate Harbor's deadline, which stops the agent early rather
        # than late — the safe direction to be wrong in.
        assert load_settings().agent_timeout_multiplier == 1.0

    def test_both_come_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("ORCHARD_HARBOR_AGENT_MARGIN", "300")
        monkeypatch.setenv("ORCHARD_HARBOR_AGENT_MULTIPLIER", "1.5")
        settings = load_settings()
        assert settings.agent_deadline_margin == 300
        assert settings.agent_timeout_multiplier == 1.5

    def test_the_override_is_unset_by_default(self):
        assert load_settings().agent_timeout_override is None

    def test_the_override_comes_from_the_environment(self, monkeypatch):
        # Harbor's --agent-timeout replaces the task's own budget. Missing it
        # would derive the in-pod deadline from the task's smaller number and
        # stop every rollout early.
        monkeypatch.setenv("ORCHARD_HARBOR_AGENT_TIMEOUT_SEC", "3600")
        assert load_settings().agent_timeout_override == 3600.0

    def test_an_explicitly_empty_override_is_unset(self, monkeypatch):
        monkeypatch.setenv("ORCHARD_HARBOR_AGENT_TIMEOUT_SEC", "")
        assert load_settings().agent_timeout_override is None

    def test_a_non_numeric_multiplier_is_named(self, monkeypatch):
        monkeypatch.setenv("ORCHARD_HARBOR_AGENT_MULTIPLIER", "1.5x")
        with pytest.raises(ConfigurationError, match="must be a number"):
            load_settings()


class TestRequestTimeout:
    def test_it_outlasts_the_longest_agent_loop(self):
        # DeepSWE allows the agent 10800s. Sizing the transport off
        # step_timeout alone cuts the run off at 3900s, and an HTTP timeout
        # mid-loop reads as a connection bug rather than as a timeout.
        settings = OrchardSettings(base_url="http://o", step_timeout=3600, exec_timeout=10800)
        assert settings.request_timeout == 11100

    def test_a_long_build_step_still_wins_when_it_is_longer(self):
        settings = OrchardSettings(base_url="http://o", step_timeout=7200, exec_timeout=1800)
        assert settings.request_timeout == 7500


class TestEgressAllow:
    def test_it_is_empty_by_default(self):
        assert load_settings().egress_allow == ()

    def test_it_splits_on_commas_and_drops_blanks(self, monkeypatch):
        monkeypatch.setenv(
            "ORCHARD_HARBOR_EGRESS_ALLOW", " 10.4.0.0/16 , sglang.internal ,"
        )
        assert load_settings().egress_allow == ("10.4.0.0/16", "sglang.internal")

    def test_model_egress_is_on_unless_turned_off(self, monkeypatch):
        assert load_settings().model_egress is True
        monkeypatch.setenv("ORCHARD_HARBOR_MODEL_EGRESS", "0")
        assert load_settings().model_egress is False

    def test_ignoring_the_air_gap_is_opt_in(self, monkeypatch):
        # A default of True would silently invalidate every air-gapped score.
        assert load_settings().allow_internet is False
        monkeypatch.setenv("ORCHARD_HARBOR_ALLOW_INTERNET", "1")
        assert load_settings().allow_internet is True


class TestImageRemap:
    def test_deep_swe_is_served_from_the_hub_copy_by_default(self):
        # Unset must not mean "pull from public.ecr.aws": that is the throttled
        # registry whose slow pulls read as EnvironmentStartTimeoutError.
        assert dict(load_settings().image_remap) == {
            "public.ecr.aws/d3j8x8q7/swe-bench-202605": "wenlinyao/deep-swe"
        }

    def test_an_explicit_value_replaces_the_table(self, monkeypatch):
        monkeypatch.setenv("ORCHARD_HARBOR_IMAGE_REMAP", "quay.io/a/b=me/b")
        assert dict(load_settings().image_remap) == {"quay.io/a/b": "me/b"}

    def test_empty_means_pull_from_where_the_dataset_says(self, monkeypatch):
        monkeypatch.setenv("ORCHARD_HARBOR_IMAGE_REMAP", "")
        assert load_settings().image_remap == ()

    def test_a_malformed_value_fails_at_load(self, monkeypatch):
        monkeypatch.setenv("ORCHARD_HARBOR_IMAGE_REMAP", "quay.io/a/b")
        with pytest.raises(ConfigurationError):
            load_settings()

class TestExpandPorts:
    def test_consecutive_ports(self):
        assert expand_ports("http://h:30021/v1", 3) == [
            "http://h:30021/v1",
            "http://h:30022/v1",
            "http://h:30023/v1",
        ]

    def test_a_missing_port_is_an_error(self):
        # Silently defaulting to 80 would put all eight rollouts on one server.
        with pytest.raises(ConfigurationError, match="explicit port"):
            expand_ports("http://h/v1", 8)


class TestLoadSettings:
    def test_replicas_expand(self, monkeypatch):
        monkeypatch.setenv("MODEL_BASE_URL", "http://h:30021/v1")
        monkeypatch.setenv("MODEL_BASE_URL_REPLICAS", "8")
        assert load_settings().model_base_urls == FLEET

    def test_explicit_list_wins(self, monkeypatch):
        monkeypatch.setenv("MODEL_BASE_URL", "http://h:30021/v1")
        monkeypatch.setenv("MODEL_BASE_URL_REPLICAS", "8")
        monkeypatch.setenv("MODEL_BASE_URLS", "http://a:8000/v1, http://b:8000/v1")
        assert load_settings().model_base_urls == (
            "http://a:8000/v1",
            "http://b:8000/v1",
        )

    def test_unset_means_no_pinning(self):
        assert load_settings().model_base_urls == ()
        assert load_settings().endpoint_for("anything") is None

    def test_unknown_routing_is_rejected(self, monkeypatch):
        monkeypatch.setenv("MODEL_ROUTING", "round-robin")
        with pytest.raises(ConfigurationError, match="MODEL_ROUTING"):
            load_settings()


class TestEndpointFor:
    def test_sticky_is_stable(self):
        settings = OrchardSettings(base_url="http://o", model_base_urls=FLEET)
        chosen = settings.endpoint_for("write-compressor")
        assert chosen in FLEET
        # A retried trial must return to the server holding its prefix.
        assert settings.endpoint_for("write-compressor") == chosen

    def test_sticky_spreads_across_the_fleet(self):
        settings = OrchardSettings(base_url="http://o", model_base_urls=FLEET)
        used = {settings.endpoint_for(f"task-{i}") for i in range(200)}
        assert used == set(FLEET)

    def test_single_endpoint_is_used_verbatim(self):
        settings = OrchardSettings(
            base_url="http://o", model_base_urls=("http://h:8000/v1",)
        )
        assert settings.endpoint_for("anything") == "http://h:8000/v1"


class TestSessionRouting:
    """One session_router.py in front of the fleet, addressed by a path."""

    @staticmethod
    def _settings(url: str = "http://h:30021/v1") -> OrchardSettings:
        return OrchardSettings(
            base_url="http://o", model_base_urls=(url,), routing="session"
        )

    def test_the_task_name_lands_in_the_path(self):
        assert (
            self._settings().endpoint_for("write-compressor")
            == "http://h:30021/session/write-compressor/v1"
        )

    def test_a_slash_in_the_task_name_is_escaped(self):
        # Harbor task names can contain "/", which the router would otherwise
        # read as the start of the upstream path.
        assert "/session/org%2Ftask/v1" in self._settings().endpoint_for("org/task")

    def test_a_single_router_is_not_short_circuited(self):
        # The regression this mode is one line away from: endpoint_for returns
        # the base URL untouched for any fleet of one, which would leave every
        # trial unkeyed and round-robined while looking like it worked.
        settings = self._settings()
        assert settings.endpoint_for("x") != settings.model_base_urls[0]

    def test_a_retried_trial_returns_to_the_same_session(self):
        settings = self._settings()
        assert settings.endpoint_for("write-compressor") == settings.endpoint_for(
            "write-compressor"
        )

    def test_a_fleet_behind_a_router_is_rejected(self):
        with pytest.raises(ConfigurationError, match="session"):
            OrchardSettings(base_url="http://o", model_base_urls=FLEET, routing="session")

    def test_an_already_prefixed_base_url_is_rejected(self):
        with pytest.raises(ConfigurationError, match="session"):
            self._settings("http://h:30021/session/a/v1")

    def test_it_is_selectable_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("MODEL_BASE_URL", "http://h:30021/v1")
        monkeypatch.setenv("MODEL_ROUTING", "session")
        settings = load_settings()
        assert settings.routing == "session"
        assert (
            settings.endpoint_for("write-compressor")
            == "http://h:30021/session/write-compressor/v1"
        )


class TestCodexConfig:
    def test_it_declares_the_endpoint_as_a_provider(self):
        # Codex honours OPENAI_BASE_URL only for its built-in provider, which
        # talks to api.openai.com. Only a declared provider reaches a fleet.
        config = codex_config("http://h:30021/v1")
        assert 'model_provider = "orchard"' in config
        assert 'base_url = "http://h:30021/v1"' in config
        assert 'env_key = "OPENAI_API_KEY"' in config
        assert 'wire_api = "responses"' in config

    def test_namespace_tools_are_off(self):
        # Each of these contributes a Responses API "namespace" tool, which a
        # vLLM/SGLang server rejects outright on turn 1.
        config = codex_config("http://h:30021/v1", wire_api="chat")
        for switch in ("multi_agent", "apps", "memories", "remote_plugin"):
            assert f"{switch} = false" in config
        assert 'wire_api = "chat"' in config

    def test_unknown_wire_api_is_rejected(self, monkeypatch):
        monkeypatch.setenv("MODEL_WIRE_API", "grpc")
        with pytest.raises(ConfigurationError, match="MODEL_WIRE_API"):
            load_settings()


class TestPiConfig:
    def test_the_served_id_is_declared_under_an_alias(self):
        # pi reads --model as an optional provider/id, so a served id that is a
        # filesystem path cannot be passed on the command line.
        config = json.loads(
            pi_config("http://h:30021/v1", "/data/models/Qwen3.5-35B-A3B")
        )
        provider = config["providers"]["orchard"]
        assert provider["baseUrl"] == "http://h:30021/v1"
        assert provider["apiKey"] == "$OPENAI_API_KEY"
        assert provider["models"] == [
            {"id": "/data/models/Qwen3.5-35B-A3B", "name": PI_MODEL_ALIAS}
        ]

    def test_sglang_compat_flags_are_set(self):
        # vLLM/SGLang reject the developer role and reasoning_effort that pi
        # otherwise sends to a reasoning-capable model.
        compat = json.loads(pi_config("http://h:30021/v1", "m"))["providers"][
            "orchard"
        ]["compat"]
        assert compat == {
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": False,
        }

    def test_the_output_cap_is_staged_when_asked_for(self):
        # Left to itself pi caps output at 16384 and ends the whole session on
        # the first truncated completion instead of re-prompting. Stating the
        # number here is what makes it a recorded property of a run rather than
        # a default inherited from whichever pi the image happened to ship.
        models = json.loads(
            pi_config("http://h:30021/v1", "m", max_tokens=16384)
        )["providers"]["orchard"]["models"]
        assert models == [{"id": "m", "name": PI_MODEL_ALIAS, "maxTokens": 16384}]

    def test_no_cap_leaves_the_field_out(self):
        # Not "emit pi's default": an absent field is how a caller asks for
        # whatever this pi build considers its own default.
        models = json.loads(pi_config("http://h:30021/v1", "m"))["providers"][
            "orchard"
        ]["models"]
        assert models == [{"id": "m", "name": PI_MODEL_ALIAS}]

    def test_the_staged_config_is_valid_json_either_way(self):
        # The cap is rendered as a text fragment inside a format template, so a
        # missing comma here is a pod that dies on turn 1 with a parse error.
        for max_tokens in (None, 16384, 262144):
            json.loads(pi_config("http://h/v1", "m", max_tokens=max_tokens))


class TestPiMaxTokensSetting:
    """``ORCHARD_HARBOR_PI_MAX_TOKENS`` — the Harbor side of ``pi.yaml``'s
    ``max_tokens``. The two paths have to agree or their scores do not."""

    def test_it_defaults_to_pis_own_cap(self, monkeypatch):
        monkeypatch.delenv("ORCHARD_HARBOR_PI_MAX_TOKENS", raising=False)
        assert load_settings().pi_max_tokens == 16384

    def test_it_can_be_raised(self, monkeypatch):
        monkeypatch.setenv("ORCHARD_HARBOR_PI_MAX_TOKENS", "32768")
        assert load_settings().pi_max_tokens == 32768

    def test_an_empty_value_means_let_pi_decide(self, monkeypatch):
        monkeypatch.setenv("ORCHARD_HARBOR_PI_MAX_TOKENS", "")
        assert load_settings().pi_max_tokens is None

    def test_a_non_number_is_rejected(self, monkeypatch):
        monkeypatch.setenv("ORCHARD_HARBOR_PI_MAX_TOKENS", "lots")
        with pytest.raises(ConfigurationError, match="ORCHARD_HARBOR_PI_MAX_TOKENS"):
            load_settings()


class TestCommitBeforeCollectSetting:
    def test_it_is_on_by_default(self, monkeypatch):
        monkeypatch.delenv("ORCHARD_HARBOR_COMMIT_BEFORE_COLLECT", raising=False)
        assert load_settings().commit_before_collect is True

    def test_it_can_be_turned_off(self, monkeypatch):
        monkeypatch.setenv("ORCHARD_HARBOR_COMMIT_BEFORE_COLLECT", "0")
        assert load_settings().commit_before_collect is False

    def test_the_identity_is_configurable(self, monkeypatch):
        monkeypatch.setenv("ORCHARD_HARBOR_COMMIT_AUTHOR_NAME", "bot")
        monkeypatch.setenv("ORCHARD_HARBOR_COMMIT_AUTHOR_EMAIL", "bot@example.test")
        settings = load_settings()
        assert settings.commit_author_name == "bot"
        assert settings.commit_author_email == "bot@example.test"
