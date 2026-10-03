"""Run configuration: YAML files plus ``key=value`` overrides.

A run is fully described by a :class:`RunConfig`. It is loaded by layering, in
increasing precedence:

1. the defaults declared on the models below
2. one or more YAML files (``--config``)
3. dotted ``key=value`` overrides from the command line

That ordering means a config file is a starting point, never a straitjacket —
``orchard-eval run -c configs/codex.yaml dataset.limit=10`` is a valid way to
smoke-test a full-benchmark config.

String values in a YAML file may also reference the environment as
``${VAR}``; see :func:`expand_env`.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import yaml
from pydantic import BaseModel, Field, model_validator

#: ``${VAR}`` inside a YAML string value.
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

#: Lexicographic order matches chronological order, which is what makes
#: ``max()`` the right way to find the newest attempt.
RUN_ID_FORMAT = "%Y%m%d-%H%M%S"

#: How a rollout is assigned an endpoint when several serve the same model.
ROUTING_MODES = ("sticky", "random", "session")

#: Path segment that names the session to ``scripts/session_router.py``. Must
#: match ``SESSION_PREFIX`` there and ``SESSION_SEGMENT`` in
#: ``harbor_orchard/settings.py``.
SESSION_SEGMENT = "/session/"


def expand_ports(base_url: str, count: int) -> list[str]:
    """``http://h:30021/v1`` with ``count=3`` -> ports 30021, 30022, 30023."""
    parts = urlsplit(base_url)
    if parts.port is None:
        raise ValueError(
            "model.base_url_replicas needs an explicit port in model.base_url; "
            f"got {base_url!r}"
        )
    host = parts.hostname or ""
    if ":" in host:  # IPv6 literals lose their brackets in `hostname`
        host = f"[{host}]"
    userinfo = ""
    if parts.username:
        userinfo = parts.username
        if parts.password:
            userinfo += f":{parts.password}"
        userinfo += "@"
    return [
        urlunsplit(
            (
                parts.scheme,
                f"{userinfo}{host}:{parts.port + offset}",
                parts.path,
                parts.query,
                parts.fragment,
            )
        )
        for offset in range(count)
    ]


def session_url(base_url: str, key: str) -> str:
    """``http://h:30021/v1`` + ``django__django-1`` ->
    ``http://h:30021/session/django__django-1/v1``.

    The session travels in the path because a base URL is the only part of a
    request every agent CLI lets this suite set: codex, pi, litellm and Claude
    Code all expose one, and none of them expose custom request headers.

    The segment goes *before* the existing path, not after it, because the base
    URL has to stay a ``/v1`` root — codex appends its protocol path to
    whatever it is given, and pi and litellm do the same.

    ``safe=""`` so a Harbor task name containing ``/`` becomes ``%2F`` rather
    than a second path segment, which the router would read as the start of the
    upstream path.
    """
    parts = urlsplit(base_url)
    if not parts.scheme or not parts.netloc:
        raise ValueError(
            f"model.routing=session needs an absolute model.base_url; got {base_url!r}"
        )
    # `netloc` verbatim rather than the host/port/userinfo reassembly
    # `expand_ports` does: there is no port arithmetic here, so there is no
    # reason to take the authority apart and risk putting it back wrong.
    path = f"{SESSION_SEGMENT}{quote(key, safe='')}{parts.path}"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def anthropic_base_url(base_url: str) -> str:
    """``http://h:30021/session/a/v1`` -> ``http://h:30021/session/a``.

    Claude Code is the one CLI here that appends ``/v1/messages`` rather than
    ``/chat/completions``, so it needs the ``/v1`` root every other consumer
    wants removed — pointed at ``.../v1`` it would request ``/v1/v1/messages``
    and take a 404 on turn one.

    Only the environment variable gets the stripped form. ``model.base_url``
    keeps its ``/v1`` because :func:`orchard_evalkit.router.close_url` maps it
    back to ``/router/session/<sid>`` at teardown, and because the same config
    has to keep working for codex and pi.

    Idempotent, and a no-op on a URL that does not end in ``/v1``.

    Kept in step with the copy in ``harbor_orchard/settings.py``.
    """
    parts = urlsplit(base_url)
    path = parts.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[: -len("/v1")]
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


class ModelConfig(BaseModel):
    """Which model the harness should drive, and how to reach it.

    ``api_key`` is never written to disk by the suite. Prefer leaving it unset
    and pointing ``api_key_env`` at an environment variable instead.

    A model may be served by several interchangeable endpoints — eight sglang
    processes behind eight ports, say. There are two ways to keep one agent
    loop on one of them, which is what makes that server's prefix KV cache
    worth anything:

    ``routing="sticky"`` declares every endpoint here and hashes the rollout
    key to one of them. No extra process, but it needs one published port per
    server, and a hash spreads binomially rather than evenly.

    ``routing="session"`` points ``base_url`` at one ``scripts/session_router.py``
    in front of the fleet and names the rollout in the URL path instead. One
    published port, and the router places each new session on whichever engine
    is carrying the least work — which an ordinary load balancer could not do,
    because it would spread a single loop's turns across the fleet.
    """

    name: str = ""
    base_url: str | None = None
    #: Endpoints serving the same model, used verbatim. Wins over
    #: ``base_url_replicas``, for a fleet that is not on consecutive ports.
    base_urls: list[str] = Field(default_factory=list)
    #: Read ``base_url`` as the first of this many endpoints on consecutive
    #: ports, which is what a fleet of single-GPU servers normally looks like.
    base_url_replicas: int = 1
    #: How a rollout is pinned to an endpoint. ``sticky`` derives it from the
    #: instance id, so a retried or resumed instance returns to the same server;
    #: ``random`` draws per rollout; ``session`` names the instance in the URL
    #: path and lets ``scripts/session_router.py`` choose.
    routing: str = "sticky"
    api_key: str | None = None
    api_key_env: str | None = None
    #: Free-form knobs forwarded to the harness (temperature, thinking level, ...).
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_endpoints(self) -> ModelConfig:
        if self.routing not in ROUTING_MODES:
            raise ValueError(
                f"model.routing must be one of {', '.join(ROUTING_MODES)}; "
                f"got {self.routing!r}"
            )
        if self.base_url_replicas < 1:
            raise ValueError("model.base_url_replicas must be >= 1")
        self.endpoints()  # fail at load time, not on the first rollout
        if self.routing == "session":
            if len(self.endpoints()) > 1:
                raise ValueError(
                    "model.routing=session expects the single session-router "
                    f"endpoint, but {len(self.endpoints())} were configured. The "
                    "router is what picks an engine; listing the engines here as "
                    "well would pin a rollout twice and defeat both mechanisms."
                )
            if self.base_url and SESSION_SEGMENT in self.base_url:
                raise ValueError(
                    f"model.base_url already contains {SESSION_SEGMENT!r}, so "
                    "session routing would insert a second prefix. Point it at "
                    "the router's /v1 root instead."
                )
        return self

    def resolve_api_key(self) -> str | None:
        """Return the API key, reading ``api_key_env`` from the environment."""
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None

    def endpoints(self) -> list[str]:
        """Every endpoint serving this model, in a stable order."""
        if self.base_urls:
            return list(self.base_urls)
        if not self.base_url:
            return []
        if self.base_url_replicas == 1:
            return [self.base_url]
        return expand_ports(self.base_url, self.base_url_replicas)

    def endpoint_for(self, key: str) -> str | None:
        """The single endpoint ``key`` is allowed to talk to.

        Sticky by default: the choice is a function of the key, so every turn of
        one agent loop — and a retry, and a resumed run — lands on the same
        server. Spreading a loop's turns across a fleet would throw away the
        prefix cache the fleet exists to exploit.
        """
        endpoints = self.endpoints()
        if self.routing == "session":
            # Handled before the single-endpoint shortcut below, which would
            # otherwise return `base_url` unchanged: a fleet behind a router is
            # exactly one URL, so the shortcut would make this mode a silent
            # no-op — every rollout would reach the router unkeyed and be
            # round-robined, which looks like it worked and is not.
            base = endpoints[0] if endpoints else self.base_url
            return session_url(base, key) if base else None
        if len(endpoints) <= 1:
            return endpoints[0] if endpoints else self.base_url
        if self.routing == "random":
            return random.choice(endpoints)
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        return endpoints[int.from_bytes(digest, "big") % len(endpoints)]

    def for_key(self, key: str) -> ModelConfig:
        """This config with :attr:`base_url` pinned to one endpoint."""
        chosen = self.endpoint_for(key)
        if chosen == self.base_url:
            return self
        return self.model_copy(update={"base_url": chosen})


class HarnessConfig(BaseModel):
    """Which agent harness to evaluate, plus harness-specific settings."""

    name: str = "mini-swe-agent"
    #: Seconds allowed for a single rollout before the harness is abandoned.
    timeout: int = 2400
    #: Passed verbatim to the harness constructor.
    params: dict[str, Any] = Field(default_factory=dict)


class SandboxConfig(BaseModel):
    """Per-instance Orchard Env sandbox settings."""

    base_url: str | None = None
    api_key: str | None = None
    prefix: str | None = None
    cpu: str = "2"
    memory: str = "8Gi"
    #: Seconds to wait for a pod to become ready. A pod that has not pulled its
    #: image by now is on a node that will not finish at twice the budget
    #: either, so failing fast and retrying elsewhere is cheaper than waiting.
    create_timeout: int = 600
    #: Agent harnesses call model APIs, so egress is required by default.
    block_network: bool = False
    #: Registry holding the SWE-bench instance images.
    image_prefix: str = "docker.io/swebench"
    #: Where the repository lives inside the SWE-bench images.
    workdir: str = "/testbed"
    #: Seconds for ordinary setup/teardown commands.
    command_timeout: int = 300


class DatasetConfig(BaseModel):
    """Which task instances to run."""

    #: HuggingFace dataset id, a local ``.jsonl``/``.json`` path, or a name
    #: understood by :mod:`orchard_evalkit.datasets` — ``swe-bench-verified``,
    #: ``swe-bench-lite``, ``swe-bench-full``, ``swe-bench-pro``. Spelling is
    #: forgiving: spaces and underscores fold to hyphens, so ``SWE-bench Pro``
    #: works too.
    name: str = "swe-bench-verified"
    #: Which benchmark's loading and grading rules apply: ``swe-bench``,
    #: ``swe-bench-pro``, or ``auto`` to infer it from ``name`` and the rows.
    #: The two differ in image naming, repository path, and grading program, so
    #: this is the one setting a mixed setup must get right.
    benchmark: str = "auto"
    split: str = "test"
    #: Git revision of the HuggingFace dataset — a tag, branch or commit sha.
    #: Empty follows the default branch, which is how a benchmark changes under
    #: a run without anyone asking: ``ScaleAI/SWE-bench_Pro`` replaced its
    #: default config with V2 (642 tasks) on 2026-09-22, and the tasks V2 kept
    #: still had their ``problem_statement``, ``requirements``, ``interface``
    #: and ``fail_to_pass`` rewritten. Pin this for any number you intend to
    #: compare against an earlier run. Ignored for local ``.jsonl`` datasets.
    revision: str = ""
    #: Explicit allow-list; wins over ``filter``/``slice``.
    instance_ids: list[str] = Field(default_factory=list)
    #: Regex matched against ``instance_id``.
    filter: str = ""
    #: Python slice spec, e.g. ``"0:25"``.
    slice: str = ""
    limit: int | None = None
    shuffle: bool = False
    seed: int = 42


class GradingConfig(BaseModel):
    """How the produced patch is scored."""

    enabled: bool = True
    #: Seconds allowed for the benchmark's own test command.
    eval_timeout: int = 1200
    #: Seconds allowed for applying the model patch.
    apply_timeout: int = 120
    #: Run the tests even when the patch is empty, instead of short-circuiting
    #: to ``EMPTY_PATCH``. Only the ``noop`` baseline wants this: it is how the
    #: suite measures what the tests score with no fix at all, which must be
    #: ~0%. For a real run this would burn a pod per agent that produced
    #: nothing, so it stays off.
    grade_empty_patch: bool = False
    # --- SWE-bench Pro only -------------------------------------------------
    #: Local ``run_scripts/`` directory from a ``scaleapi/SWE-bench_Pro-os``
    #: checkout, holding each instance's ``run_script.sh`` and ``parser.py``.
    #: They are not part of the HuggingFace dataset. Empty means download them
    #: from GitHub and cache them under ``pro_scripts_cache``.
    pro_scripts_dir: str = ""
    #: Git ref the scripts are downloaded from. Pin it for a run whose numbers
    #: you intend to compare later — upstream has revised these scripts.
    pro_scripts_ref: str = "main"
    #: Cache directory for downloaded scripts. Empty uses
    #: ``~/.cache/orchard-eval/swebench-pro/<ref>``.
    pro_scripts_cache: str = ""

class RunConfig(BaseModel):
    """Everything needed to reproduce one evaluation run."""

    run_name: str = ""
    #: Timestamped subdirectory of ``output_dir/run_name`` holding one attempt,
    #: so re-running a config never overwrites the previous run's artifacts.
    #: Defaults to now, or to the newest existing attempt when resuming.
    run_id: str = ""
    output_dir: Path = Path("./results")
    concurrency: int = 8
    #: Retries for *infrastructure* failures only. A failed agent is a result,
    #: not an error, and is never retried.
    max_retries: int = 2
    #: Extra whole-rollout attempts when the sandbox is lost mid-instance (pod
    #: reaped, evicted, OOM-killed). Each one starts from a fresh pod and
    #: re-spends the model budget, so it is bounded separately from the cheap
    #: pod-creation retries above.
    rollout_retries: int = 2
    #: Extra attempts for a rollout that spent its whole budget rather than
    #: losing its pod. Bounded separately and lower, because a deadline is
    #: usually a property of the instance: the next pod spends the same half
    #: hour reaching the same wall, and across four runs no instance whose
    #: final attempt hit the deadline was ever rescued by another one. It is
    #: not zero, because a pod that dies in the last seconds of its budget is
    #: indistinguishable from one that ran out of it, and one spare attempt is
    #: cheaper than silently dropping an instance that would have scored.
    timeout_retries: int = 1
    #: Skip instances already present in ``results.jsonl``.
    resume: bool = True
    #: On resume, re-run instances recorded as ``infra_error``. They carry no
    #: score, so keeping them would let a transient cluster fault (a reaped pod,
    #: a refused create) permanently lower the run's denominator.
    retry_infra_on_resume: bool = True
    #: Keep the sandbox alive after the instance finishes (debugging only).
    keep_sandbox: bool = False
    #: Persist per-instance prompts, agent logs, patches, and eval logs.
    save_artifacts: bool = True
    log_level: str = "INFO"

    harness: HarnessConfig = Field(default_factory=HarnessConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    grading: GradingConfig = Field(default_factory=GradingConfig)

    @model_validator(mode="after")
    def _default_run_paths(self) -> RunConfig:
        if not self.run_name:
            model_slug = (self.model.name or "model").replace("/", "-")
            object.__setattr__(self, "run_name", f"{self.harness.name}__{model_slug}")
        if not self.run_id:
            object.__setattr__(self, "run_id", self._resolve_run_id())
        return self

    def _resolve_run_id(self) -> str:
        # Resuming has to continue the newest attempt, not open an empty
        # directory beside it — otherwise every resume re-runs the whole set.
        if self.resume:
            latest = latest_run_id(Path(self.output_dir) / self.run_name)
            if latest:
                return latest
        return time.strftime(RUN_ID_FORMAT)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name / self.run_id

    @property
    def results_path(self) -> Path:
        return self.run_dir / "results.jsonl"

    @property
    def instances_dir(self) -> Path:
        return self.run_dir / "instances"


def latest_run_id(base: Path) -> str:
    """Newest attempt under ``base`` that actually recorded something.

    A directory with no ``results.jsonl`` is an attempt that died before its
    first instance finished; resuming into it would be indistinguishable from
    starting fresh, and would leave the real history orphaned.
    """
    try:
        candidates = [
            d.name
            for d in base.iterdir()
            if d.is_dir() and (d / "results.jsonl").exists()
        ]
    except OSError:
        return ""
    return max(candidates, default="")


def expand_env(value: Any) -> Any:
    """Recursively substitute ``${VAR}`` from the environment.

    An unset variable is an error rather than an empty string: a config that
    silently evaluates to ``base_url: ""`` fails hundreds of instances into a
    run instead of before the first one.
    """
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            try:
                return os.environ[name]
            except KeyError:
                raise ValueError(
                    f"Config references ${{{name}}} but {name} is not set."
                ) from None

        return _ENV_REF.sub(replace, value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def parse_overrides(overrides: list[str]) -> dict[str, Any]:
    """Turn ``["dataset.limit=10", "concurrency=64"]`` into a nested dict.

    Values are parsed as YAML, so ``true``, ``10``, ``[a, b]`` and
    ``{k: v}`` all work; anything unparseable stays a string.
    """
    parsed: dict[str, Any] = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override {override!r}; expected key=value")
        key, raw_value = override.split("=", 1)
        parts = [p for p in key.split(".")]
        if any(not p for p in parts):
            raise ValueError(f"Invalid override {override!r}: empty key segment")
        cursor = parsed
        for part in parts[:-1]:
            nxt = cursor.setdefault(part, {})
            if not isinstance(nxt, dict):
                raise ValueError(f"Conflicting override for {key!r}")
            cursor = nxt
        cursor[parts[-1]] = yaml.safe_load(raw_value)
    return parsed


def load_config(
    config_paths: list[str] | None = None,
    overrides: list[str] | None = None,
) -> RunConfig:
    """Build a :class:`RunConfig` from YAML files plus CLI overrides."""
    merged: dict[str, Any] = {}
    for path in config_paths or []:
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Config file not found: {resolved}")
        loaded = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config file {resolved} must contain a mapping")
        merged = deep_merge(merged, expand_env(loaded))
    merged = deep_merge(merged, parse_overrides(overrides or []))
    return RunConfig(**merged)
