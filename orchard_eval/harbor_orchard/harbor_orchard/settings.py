"""Configuration for the Orchard environment provider.

Read from the process environment rather than from ``task.toml``, because these
are properties of *your cluster*, not of a task: a terminal-bench task must
produce the same score whoever runs it and on whatever hardware.
"""

from __future__ import annotations

import hashlib
import os
import random
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

from harbor_orchard import network
from harbor_orchard.agent_config import WIRE_APIS
from harbor_orchard.images import DEFAULT_MIRROR, DEFAULT_REMAP, parse_remap

#: Bytes per transfer chunk. Bigger is faster and uses more memory on both ends.
DEFAULT_CHUNK_MB = 16

#: What an in-pod agent CLI reads for its OpenAI-compatible endpoint. Both
#: spellings are set because which one a client honours has moved between
#: releases, and a wrong guess sends the trial to api.openai.com.
BASE_URL_ENV_VARS = ("OPENAI_BASE_URL", "OPENAI_API_BASE")

#: How a trial is assigned an endpoint when several serve the same model.
ROUTING_MODES = ("sticky", "random", "session")

#: Path segment that names the session to ``scripts/session_router.py``. Kept
#: in step with ``SESSION_SEGMENT`` in ``orchard_evalkit/config.py``.
SESSION_SEGMENT = "/session/"

#: What the in-pod agent shim reads for its own deadline, in seconds. Exported
#: into every exec by :meth:`environment.OrchardEnvironment.exec`; unset means
#: the shim runs the CLI unwrapped, exactly as it did before.
DEADLINE_ENV_VAR = "ORCHARD_AGENT_DEADLINE"

#: Seconds between the shim's SIGTERM and its SIGKILL, for the CLI to flush
#: whatever it was writing. Matches ``AGENT_KILL_GRACE_S`` in
#: ``orchard_evalkit/harnesses/installed_cli.py``, which solves the same
#: problem on the ``orchard-eval run`` path.
AGENT_KILL_GRACE_S = 10

#: Seconds allowed for snapshotting the agent's work as ``model.patch``. It is
#: a ``git add -A`` plus a diff, which on the largest task repositories here
#: (element-web, teleport) is tens of seconds rather than the sub-second it
#: looks like. Matches the budget upstream's own locked agents give the same
#: command; past it the capture is abandoned and the trial carries on, because
#: a missing patch costs a re-grade and a failed agent phase costs the rollout.
MODEL_PATCH_TIMEOUT_S = 300

#: Where the orchestrator mounts the sandbox-tools payload
#: (``SANDBOX_TOOLS_MOUNT_PATH``). Re-exported as ``agents.TOOLS_DIR``. Every
#: payload CLI runs from under here, which is what makes it a precise enough
#: pattern to stop an abandoned agent with and nothing else in a task image.
PAYLOAD_DIR = "/opt/sandbox-tools"


#: Seconds allowed for the pre-flight ``git status``. Generous because it runs
#: once, before the model is paid for anything, on repositories as large as
#: element-web; past it the check is abandoned and the trial carries on.
CLEAN_TREE_TIMEOUT_S = 120

#: Set to 0 to let a trial start on a repository that is already modified.
#:
#: The default refuses, because a dirty tree means this trial is not the same
#: experiment as its neighbours — see :data:`shell.CLEAN_TREE_ROUTINE` for how
#: 96 trials of the 2026-09-24 V2 sweep came to start on another agent's
#: finished work. Worth turning off only for a dataset whose images ship
#: modifications deliberately; a run made that way is not comparable with one
#: that was not.
REQUIRE_CLEAN_TREE_ENV_VAR = "ORCHARD_HARBOR_REQUIRE_CLEAN_TREE"


def require_clean_tree() -> bool:
    """Whether a modified checkout ends the trial before the agent starts.

    Read per call rather than captured at import, so a test — and a shell that
    exports it between two runs — sees the value it set.
    """
    return _bool_env(REQUIRE_CLEAN_TREE_ENV_VAR, True)


class ConfigurationError(RuntimeError):
    """The provider is not configured well enough to start a sandbox."""


def expand_ports(base_url: str, count: int) -> list[str]:
    """``http://h:30021/v1`` with ``count=3`` -> ports 30021, 30022, 30023."""
    parts = urlsplit(base_url)
    if parts.port is None:
        raise ConfigurationError(
            "MODEL_BASE_URL_REPLICAS needs an explicit port in MODEL_BASE_URL; "
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
    """``http://h:30021/v1`` + ``write-compressor`` ->
    ``http://h:30021/session/write-compressor/v1``.

    Duplicated from ``orchard_evalkit.config`` for the same reason
    :func:`expand_ports` is: Harbor imports this package in its own process,
    where the eval suite is not necessarily installed. Keep the two in step.
    """
    parts = urlsplit(base_url)
    if not parts.scheme or not parts.netloc:
        raise ConfigurationError(
            f"MODEL_ROUTING=session needs an absolute MODEL_BASE_URL; got {base_url!r}"
        )
    # safe="" so a task name containing "/" becomes %2F rather than a second
    # path segment, which the router would read as the upstream path.
    path = f"{SESSION_SEGMENT}{quote(key, safe='')}{parts.path}"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def anthropic_base_url(base_url: str) -> str:
    """``http://h:30021/session/a/v1`` -> ``http://h:30021/session/a``.

    Claude Code appends ``/v1/messages`` to whatever ``ANTHROPIC_BASE_URL``
    holds, so it is the one client here that wants the ``/v1`` root every other
    one requires. Pointed at ``.../v1`` it requests ``/v1/v1/messages`` and
    takes a 404 on turn one.

    Idempotent, and a no-op on a URL that does not end in ``/v1``.

    Duplicated from ``orchard_evalkit.config`` for the same reason
    :func:`session_url` is. Keep the two in step.
    """
    parts = urlsplit(base_url)
    path = parts.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[: -len("/v1")]
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def task_agent_timeout_sec(environment_dir: str | Path | None) -> float | None:
    """``[agent] timeout_sec`` from the task that owns *environment_dir*.

    Harbor hands an environment its own ``environment/`` directory and nothing
    else about the task, so the task's ``task.toml`` is read from the parent —
    the same relationship :meth:`environment.OrchardEnvironment._sticky_key`
    relies on. ``None`` for any task that declares no agent timeout, which is
    also what Harbor does with one: it leaves the phase unbounded.

    Deliberately forgiving. A deadline is an optimisation over letting the CLI
    run to the exec timeout, so an unreadable or surprising ``task.toml`` gives
    up on the deadline rather than failing the trial.
    """
    if not environment_dir:
        return None
    path = Path(environment_dir).parent / "task.toml"
    try:
        with path.open("rb") as handle:
            config = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    agent = config.get("agent")
    if not isinstance(agent, dict):
        return None
    timeout = agent.get("timeout_sec")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
        return None
    return float(timeout) if timeout > 0 else None


def agent_deadline(
    timeout_sec: float | None,
    *,
    multiplier: float,
    margin: int,
    exec_timeout: int,
) -> int | None:
    """Seconds the in-pod CLI gets, or ``None`` to leave it unbounded.

    ``timeout_sec * multiplier`` reproduces Harbor's
    ``Trial._compute_agent_timeout_sec``; subtracting *margin* is what makes
    the CLI stop first. Also clamped under ``exec_timeout``, because the
    orchestrator abandons an exec there and a deadline it never reaches buys
    nothing.

    Harbor's own computation additionally honours ``max_timeout_sec``, which
    this cannot see. That cap can only lower Harbor's deadline, so ignoring it
    risks a deadline that is too *late* — which is no worse than today, where
    there is none at all — and never one that cuts a rollout short.
    """
    if margin <= 0 or timeout_sec is None:
        return None
    deadline = int(timeout_sec * multiplier) - margin
    deadline = min(deadline, exec_timeout - margin)
    return deadline if deadline > 0 else None


@dataclass(frozen=True)
class OrchardSettings:
    base_url: str
    api_key: str | None = None
    prefix: str | None = None
    #: Fallbacks for tasks that declare no resources of their own.
    default_cpu: str = "4"
    default_memory: str = "16Gi"
    #: Seconds to wait for a pod to become ready. Pulls of multi-gigabyte task
    #: images dominate this, not scheduling.
    create_timeout: int = 1800
    #: Seconds for one replayed Dockerfile instruction. Tasks compile toolchains
    #: from source in a single ``RUN``, so this has to be generous.
    step_timeout: int = 3600
    #: Seconds for one agent or verifier command. This bounds a whole agent
    #: loop when Harbor's agent declares no timeout of its own, and a loop cut
    #: off here is reported as an ordinary non-zero exit, so it is generous.
    exec_timeout: int = 3600
    #: Seconds to stop the in-pod agent CLI *before* Harbor's own agent
    #: deadline. ``0`` disables the in-pod deadline entirely.
    #:
    #: Harbor enforces its deadline with ``asyncio.wait_for`` around the agent
    #: coroutine (``harbor/trial/trial.py``), which cancels the *await* and
    #: nothing else: the CLI keeps running in the pod, and every call that
    #: follows — log sync, artifact collection, the verifier — queues behind it
    #: until it finishes on its own or the pod is reaped. Measured on a 731-trial
    #: SWE-bench Pro run: of the trials that timed out, 34% then sat for a median
    #: of 65 minutes, the last model call landing a median of 2687s *after* the
    #: recorded agent end. That is 174 GPU-hours, 28% of the run's agent budget,
    #: and it aged 189 pods past ``SANDBOX_TTL_HOURS`` so they were never graded.
    #: Trials whose agent exited cleanly never did this: 0 of 209.
    #:
    #: Stopping the CLI a margin early instead means the exec returns through
    #: its ordinary path, with stdout drained and the trajectory written.
    agent_deadline_margin: int = 120
    #: Mirror of Harbor's ``--agent-timeout-multiplier``, which is a CLI flag
    #: this package never sees. ``orchard_evalkit.harbor_bridge`` exports it
    #: alongside the flag so the two cannot disagree; a bare ``harbor run``
    #: leaves it at 1.0, which under-estimates the deadline and therefore only
    #: ever stops the agent early, never late.
    agent_timeout_multiplier: float = 1.0
    #: Mirror of Harbor's ``--agent-timeout``, which replaces the task's own
    #: ``[agent] timeout_sec`` before the multiplier is applied. Exported by
    #: ``orchard_evalkit.harbor_bridge`` for the same reason as the multiplier.
    #: Unlike the unreadable ``max_timeout_sec`` cap, ignoring this one is not
    #: safe in the convenient direction: a run that raises the budget would
    #: leave the in-pod deadline derived from the *task's* smaller number and
    #: cut every rollout short.
    agent_timeout_override: float | None = None
    #: Seconds between checks that the pod a call is waiting on still exists.
    #: Nothing fails an in-flight job when the orchestrator reaps its sandbox,
    #: so without this a trial whose pod is taken away blocks for the whole of
    #: ``exec_timeout`` above. ``0`` disables the check.
    liveness_interval: float = 120
    chunk_size: int = DEFAULT_CHUNK_MB * 1024 * 1024
    #: ``None`` disables rewriting, for a cluster with a node-level mirror.
    image_mirror: str | None = DEFAULT_MIRROR
    #: Whole repositories served from a copy you pushed, as ``(source, dest)``
    #: pairs. Empty disables substitution. See :data:`images.DEFAULT_REMAP`.
    image_remap: tuple[tuple[str, str], ...] = tuple(DEFAULT_REMAP.items())
    #: Interchangeable endpoints serving the same model — eight sglang servers
    #: behind eight ports, say. More than one makes each trial talk to exactly
    #: one of them, so that server's prefix KV cache still holds the
    #: conversation on the agent's next turn.
    model_base_urls: tuple[str, ...] = ()
    #: ``sticky`` hashes the task name to one of them; ``random`` draws per
    #: trial; ``session`` expects a single ``scripts/session_router.py`` and
    #: names the task in the URL path so it can place the trial itself.
    routing: str = "sticky"
    #: Credential the in-pod agent presents. A server started without
    #: ``--api-key`` still needs *something*: codex aborts when its ``env_key``
    #: resolves to nothing.
    model_api_key: str | None = None
    #: Protocol a staged codex config declares.
    wire_api: str = "responses"
    #: The id the server actually serves, needed to declare it in pi's catalog.
    model_name: str | None = None
    #: Write an agent provider config into the pod. Off means the agent is left
    #: with whatever Harbor configured, which cannot reach a self-hosted server.
    stage_agent_config: bool = True
    #: Log any exec that takes at least this long, with its duration. The
    #: window between Harbor starting the agent phase and the agent's first
    #: model call has measured 39-59 minutes on DeepSWE while every step of it
    #: measures in seconds when probed alone; this is how that window reports
    #: itself from inside a real run.
    slow_exec_log_sec: float = 60.0
    #: Output-token cap staged in pi's models.json. pi's own default is 16384;
    #: naming it here means the number a run used is recorded rather than
    #: inherited, and can be changed without editing a template. It is not a
    #: free knob: a reasoning model that loops inside ``thinking`` spends the
    #: whole cap on one message, and pi ends the session on the first truncated
    #: completion instead of re-prompting. Raising it does not remove those
    #: rollouts, it converts them into timeouts — measured on SWE-bench
    #: Verified, 16384 -> 32768 moved rollouts-ending-on-length from 95/500 to
    #: 40/500 while rollouts hitting the wall went from 29/500 to 146/500, for
    #: no net change in resolve rate. Treat this and the agent timeout as one
    #: knob. ``None`` leaves the field out and lets pi pick its own default.
    pi_max_tokens: int | None = 16384
    #: Commit the agent's working tree before Harbor runs a collect hook that
    #: diffs against ``HEAD``. DeepSWE 1.1 grades only committed work (its
    #: ``[[verifier.collect]]`` is ``git diff <base> HEAD``, applied in a
    #: separate verifier container), and neither pi nor mini-swe-agent commits
    #: on its own, so without this most rollouts are graded as an empty patch
    #: no matter how much they changed. Off restores Harbor's behaviour, which
    #: is also what the benchmark literally asks the agent to do for itself.
    commit_before_collect: bool = True
    #: Identity the pre-collect commit passes with ``git -c``. Scoped to that
    #: one command on purpose: task images rarely configure an identity, but
    #: exporting ``GIT_AUTHOR_*`` would outrank every task's own ``git config``
    #: on every exec, including verifier runs.
    commit_author_name: str = "orchard-eval"
    commit_author_email: str = "orchard-eval@local"
    #: Keep the pinned model endpoint reachable from an otherwise isolated pod.
    #: Off is the strictest setting and the right one for an agent whose model
    #: loop runs on the host (oracle), which needs nothing from inside the
    #: sandbox. Harbor's own mini-swe-agent is *not* one of these: it is a
    #: BaseInstalledAgent that runs the CLI in the pod, so it is pinned.
    model_egress: bool = True
    #: Extra destinations an isolated pod may still reach, as hostnames, URLs
    #: or CIDRs. For a model fleet behind a load balancer whose address set is
    #: wider than one DNS answer.
    egress_allow: tuple[str, ...] = ()
    #: Ports opened for an allowlisted destination that does not name one of its
    #: own. The orchestrator's allowlist is per port, so something has to be
    #: chosen; a destination given as ``host:port`` or a URL overrides this.
    egress_ports: tuple[int, ...] = network.DEFAULT_PORTS
    #: Hand the in-pod agent the endpoint's address rather than its hostname.
    #: An isolated pod has no DNS — the allowlist is CIDRs and carries no port-53
    #: exemption — so a hostname fails on the first lookup and reads as a model
    #: error. Turn off for a load-balanced name that must not be pinned to one
    #: backend, and allowlist the DNS server instead.
    resolve_endpoint: bool = True
    #: Ignore a task's ``network_mode = "no-network"`` and give the pod full
    #: egress. For bringing a benchmark up, when a build needs the network or
    #: an in-pod agent has no allowlist yet. A score measured this way is not
    #: comparable with an air-gapped one — DeepSWE's images deliberately gc the
    #: repository's future history so the reference solution cannot leak, and
    #: egress puts it one `git clone` away.
    allow_internet: bool = False
    #: Isolate the pod whatever the task declares. The mirror of
    #: ``allow_internet``: some benchmarks ship ``allow_internet = true`` on
    #: every task, which makes an air-gapped measurement impossible to ask for
    #: even though the reference solution is reachable over the network. The
    #: pinned model endpoint stays allowlisted, so an in-pod agent still runs.
    #: A build that downloads anything cannot be replayed under it.
    force_isolation: bool = False
    #: Make ``localhost`` resolve to 127.0.0.1 before ``::1`` inside the pod.
    #:
    #: Benchmark test suites routinely start a server and talk to it over
    #: ``http://localhost:<port>``, and a server that binds the IPv4 wildcard —
    #: NodeBB logs "listening on 0.0.0.0:4568" — is simply not there on ``::1``.
    #: Upstream never sees it because Node asks getaddrinfo with
    #: ``AI_ADDRCONFIG`` and a single-stack container has no global IPv6
    #: address, so glibc drops ``::1`` from the answer. A dual-stack Kubernetes
    #: pod keeps it, RFC 3484 sorts it first, and the connection is refused
    #: before the server is ever consulted.
    #:
    #: Measured on the 2026-09-24 V2 oracle run: both of its two zeros were
    #: ``connect ECONNREFUSED ::1:4568`` against a live 0.0.0.0 listener, and
    #: all 23 passing trials had none. That is the pod grading itself rather
    #: than the model, which is the same reason the Pro stages override CPU and
    #: memory. Off leaves the pod's resolver exactly as the orchestrator built
    #: it.
    prefer_ipv4: bool = True

    def __post_init__(self) -> None:
        """Reject a misconfiguration here, not on trial 200.

        These all look like they work: an extra endpoint simply pins twice, a
        doubled prefix simply 404s on every turn, and two contradictory network
        overrides simply let one win — each of which reads as an agent failure
        rather than as the typo it is.
        """
        if self.allow_internet and self.force_isolation:
            raise ConfigurationError(
                "ORCHARD_HARBOR_ALLOW_INTERNET and ORCHARD_HARBOR_FORCE_ISOLATION "
                "ask for opposite things: one ignores a task's air-gap to open the "
                "pod, the other ignores a task's declared access to close it. Set "
                "at most one."
            )
        if self.routing != "session":
            return
        if len(self.model_base_urls) > 1:
            raise ConfigurationError(
                "MODEL_ROUTING=session expects the single session-router "
                f"endpoint, but {len(self.model_base_urls)} were configured. The "
                "router is what picks an engine; naming the engines as well "
                "would pin a trial twice and defeat both mechanisms."
            )
        if self.model_base_urls and SESSION_SEGMENT in self.model_base_urls[0]:
            raise ConfigurationError(
                f"MODEL_BASE_URL already contains {SESSION_SEGMENT!r}, so session "
                "routing would insert a second prefix. Point it at the router's "
                "/v1 root instead."
            )

    @property
    def request_timeout(self) -> int:
        """How long one HTTP request to the orchestrator may take.

        Exec waits server-side rather than polling, so this has to outlast the
        longest thing the provider asks a pod to do — a build step or a whole
        agent loop, whichever is larger. Sizing it off ``step_timeout`` alone
        cuts off a long agent run at the transport layer, which surfaces as a
        connection error rather than as the timeout it is.
        """
        return max(self.step_timeout, self.exec_timeout) + 300

    def endpoint_for(self, key: str) -> str | None:
        """The single endpoint a trial keyed by *key* may talk to.

        Sticky by default: the choice is a function of the key, so every turn of
        one agent loop — and a retried attempt — lands on the same server.
        Spreading a loop's turns across a fleet re-prefills the whole growing
        conversation on a cold server, which is the cost the fleet exists to
        avoid.
        """
        if not self.model_base_urls:
            return None
        if self.routing == "session":
            # Before the single-endpoint shortcut below, which would otherwise
            # return the router's bare URL and make this mode a silent no-op:
            # the trial would reach the router unkeyed and be round-robined.
            return session_url(self.model_base_urls[0], key)
        if len(self.model_base_urls) == 1:
            return self.model_base_urls[0]
        if self.routing == "random":
            return random.choice(self.model_base_urls)
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest, "big") % len(self.model_base_urls)
        return self.model_base_urls[index]

    def require_base_url(self) -> str:
        if not self.base_url:
            raise ConfigurationError(
                "SANDBOX_BASE_URL is not set. The Orchard environment needs the "
                "orchestrator URL, e.g. SANDBOX_BASE_URL=http://orchestrator.example"
            )
        return self.base_url


def load_settings() -> OrchardSettings:
    mirror = os.environ.get("ORCHARD_HARBOR_IMAGE_MIRROR", DEFAULT_MIRROR).strip()
    return OrchardSettings(
        base_url=os.environ.get("SANDBOX_BASE_URL", "").rstrip("/"),
        api_key=os.environ.get("SANDBOX_API_KEY") or None,
        prefix=os.environ.get("SANDBOX_PREFIX") or None,
        default_cpu=os.environ.get("ORCHARD_HARBOR_DEFAULT_CPU", "4"),
        default_memory=os.environ.get("ORCHARD_HARBOR_DEFAULT_MEMORY", "16Gi"),
        create_timeout=_int_env("ORCHARD_HARBOR_CREATE_TIMEOUT", 1800),
        step_timeout=_int_env("ORCHARD_HARBOR_STEP_TIMEOUT", 3600),
        exec_timeout=_int_env("ORCHARD_HARBOR_EXEC_TIMEOUT", 3600),
        agent_deadline_margin=_int_env("ORCHARD_HARBOR_AGENT_MARGIN", 120),
        agent_timeout_multiplier=_float_env("ORCHARD_HARBOR_AGENT_MULTIPLIER", 1.0),
        agent_timeout_override=_opt_float_env("ORCHARD_HARBOR_AGENT_TIMEOUT_SEC"),
        liveness_interval=_int_env("ORCHARD_HARBOR_LIVENESS_INTERVAL", 120),
        chunk_size=_int_env("ORCHARD_HARBOR_CHUNK_MB", DEFAULT_CHUNK_MB) * 1024 * 1024,
        # An explicitly empty value means "do not rewrite", which is different
        # from the variable being unset.
        image_mirror=mirror or None,
        image_remap=_image_remap(),
        model_base_urls=_model_base_urls(),
        routing=_routing(),
        model_api_key=os.environ.get("API_KEY") or None,
        wire_api=_wire_api(),
        model_name=os.environ.get("MODEL_NAME") or None,
        stage_agent_config=_bool_env("ORCHARD_HARBOR_STAGE_AGENT_CONFIG", True),
        slow_exec_log_sec=float(
            os.environ.get("ORCHARD_HARBOR_SLOW_EXEC_LOG_SEC", "60") or 60
        ),
        pi_max_tokens=_opt_int_env("ORCHARD_HARBOR_PI_MAX_TOKENS", 16384),
        commit_before_collect=_bool_env("ORCHARD_HARBOR_COMMIT_BEFORE_COLLECT", True),
        commit_author_name=os.environ.get(
            "ORCHARD_HARBOR_COMMIT_AUTHOR_NAME", "orchard-eval"
        ),
        commit_author_email=os.environ.get(
            "ORCHARD_HARBOR_COMMIT_AUTHOR_EMAIL", "orchard-eval@local"
        ),
        model_egress=_bool_env("ORCHARD_HARBOR_MODEL_EGRESS", True),
        egress_allow=tuple(
            entry.strip()
            for entry in os.environ.get("ORCHARD_HARBOR_EGRESS_ALLOW", "").split(",")
            if entry.strip()
        ),
        egress_ports=_egress_ports(),
        resolve_endpoint=_bool_env("ORCHARD_HARBOR_RESOLVE_ENDPOINT", True),
        allow_internet=_bool_env("ORCHARD_HARBOR_ALLOW_INTERNET", False),
        force_isolation=_bool_env("ORCHARD_HARBOR_FORCE_ISOLATION", False),
        prefer_ipv4=_bool_env("ORCHARD_HARBOR_PREFER_IPV4", True),
    )


def _egress_ports() -> tuple[int, ...]:
    raw = os.environ.get("ORCHARD_HARBOR_EGRESS_PORTS")
    if raw is None:
        return network.DEFAULT_PORTS
    ports: list[int] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            port = int(entry)
        except ValueError:
            raise ConfigurationError(
                f"ORCHARD_HARBOR_EGRESS_PORTS: {entry!r} is not a port number"
            ) from None
        if not 1 <= port <= 65535:
            raise ConfigurationError(
                f"ORCHARD_HARBOR_EGRESS_PORTS: {port} is out of range"
            )
        ports.append(port)
    # Explicitly empty would allowlist a destination on no port at all, which
    # reads as "allowed" but denies every packet.
    if not ports:
        raise ConfigurationError("ORCHARD_HARBOR_EGRESS_PORTS: no ports given")
    return tuple(ports)


def _image_remap() -> tuple[tuple[str, str], ...]:
    # Unset keeps the built-in table; explicitly empty turns substitution off.
    raw = os.environ.get("ORCHARD_HARBOR_IMAGE_REMAP")
    if raw is None:
        return tuple(DEFAULT_REMAP.items())
    try:
        return tuple(parse_remap(raw).items())
    except ValueError as exc:
        raise ConfigurationError(f"ORCHARD_HARBOR_IMAGE_REMAP: {exc}") from exc


def _model_base_urls() -> tuple[str, ...]:
    explicit = os.environ.get("MODEL_BASE_URLS", "")
    if explicit.strip():
        return tuple(url.strip() for url in explicit.split(",") if url.strip())
    base_url = os.environ.get("MODEL_BASE_URL", "").strip()
    if not base_url:
        return ()
    replicas = _int_env("MODEL_BASE_URL_REPLICAS", 1)
    if replicas < 1:
        raise ConfigurationError("MODEL_BASE_URL_REPLICAS must be >= 1")
    if replicas == 1:
        return (base_url,)
    return tuple(expand_ports(base_url, replicas))


def _routing() -> str:
    routing = os.environ.get("MODEL_ROUTING", "").strip() or "sticky"
    if routing not in ROUTING_MODES:
        raise ConfigurationError(
            f"MODEL_ROUTING must be one of {', '.join(ROUTING_MODES)}; got {routing!r}"
        )
    return routing


def _wire_api() -> str:
    wire_api = os.environ.get("MODEL_WIRE_API", "").strip() or "responses"
    if wire_api not in WIRE_APIS:
        raise ConfigurationError(
            f"MODEL_WIRE_API must be one of {', '.join(WIRE_APIS)}; got {wire_api!r}"
        )
    return wire_api


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no")


def _opt_float_env(name: str) -> float | None:
    """A number, or ``None`` when the variable is unset or explicitly empty."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number, got {raw!r}") from exc


def _opt_int_env(name: str, default: int | None) -> int | None:
    """Like :func:`_int_env`, but an explicit empty value means "unset".

    ``ORCHARD_HARBOR_PI_MAX_TOKENS=`` is how a caller asks for pi's own
    default rather than any number this package would otherwise impose.
    """
    if name not in os.environ:
        return default
    raw = os.environ[name].strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}") from exc


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}") from exc


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number, got {raw!r}") from exc
