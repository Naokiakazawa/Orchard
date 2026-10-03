#!/usr/bin/env python3
"""One local process that keeps an agent's whole conversation on one engine.

Eight sglang servers behind eight frp tunnels worked, but it made the *client*
responsible for affinity: the eval suite hashed an instance id to a port
(``ModelConfig.endpoint_for``), which meant eight published ports, eight
entries in every air-gapped pod's egress allowlist, and a placement decision
made by a process that cannot see how busy any engine is.

This moves the decision to the one place that can see it. Every request names
its session in the path — ``POST /session/<sid>/v1/responses`` — so the router
pins ``<sid>`` to an engine on first sight and sends everything after that to
the same one. The engine's prefix cache therefore still holds turn N-1 when
turn N arrives, which is the entire point of running a fleet of single-GPU
servers instead of one ``--dp 8`` server.

The session travels in the path because a base URL is the only part of a
request every agent CLI lets the harness set: codex, pi, litellm and Claude
Code all expose one, and none of them expose custom request headers.

Placement is by *live* session count rather than by a hash of the id. A hash
spreads binomially, so over 500 instances some engines get materially more work
than others and an engine that finishes early sits idle; counting live sessions
puts each new rollout wherever there is actually room.

A session stops being live when the harness says so — ``DELETE
/router/session/<sid>`` — and falls back to an idle timeout only when that call
never arrives. The difference is larger than it sounds: inferred from silence
alone, an engine's apparent load includes every rollout it has recently
finished, in proportion to how fast it finishes them, so the engine that
retired six short rollouts in the last ten minutes looks like the busiest in
the fleet and is handed nothing.

Ties in session count — six rollouts each across eight engines is the steady
state of a run at concurrency 48, so ties are the normal case — are broken by
what the engines report about their own queues, and by nothing at all when they
report nothing. That fallback is not a formality: ``/get_server_info`` and
``/metrics`` are both optional in sglang, and a fleet offering neither must
place exactly as well as it did before either was asked for.

One request shape is not relayed verbatim. Claude Code puts
``output_config.effort`` in every Anthropic request and offers no way not to,
sglang forwards it as chat-completions ``reasoning_effort``, and this fleet's
chat template accepts three levels that do not include the CLI's default — so
the router drops that one field on ``POST /v1/messages``, which puts the
request on the template's own default and lands claude-code where pi,
mini-swe-agent and codex already are. ``--no-strip-effort`` turns it off. It is
the only body this router reads; see :func:`strip_effort`.

Runs in the sglang venv on the GPU host, so it imports nothing but the standard
library and aiohttp (a core sglang dependency).

Usage:
    python scripts/session_router.py --replicas 8 --base-port 8000 --port 8100
    python scripts/session_router.py -u http://127.0.0.1:8000 -u http://h:8001

The engines' own ``--api-key`` has to reach this too, or the tiebreak above is
off for the whole run: ``--api-key``, or ``$API_KEY`` in the environment, and
both default to what ``scripts/serve_sglang_fleet.sh`` starts the engines with.

Then publish the single port with ./scripts/frpc_fleet.sh:
    REPLICAS=1 BASE_PORT=8100 REMOTE_BASE_PORT=30021 ./scripts/frpc_fleet.sh

That tunnel is for the sandboxes. The harness reaches this router a second
time, on its own behalf, to say a rollout is over — and it may sit somewhere
the tunnel does not answer from. ``ORCHARD_ROUTER_CONTROL_URL`` points that one
call somewhere else, usually straight at ``http://127.0.0.1:8100``; without it
a tunnel the pods can use and the harness cannot leaves every session to the
idle timeout, which is the exact bias the ``DELETE`` exists to remove.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import math
import os
import resource
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import aiohttp
from aiohttp import web
from yarl import URL

#: Path segment that carries the session id. Must match ``SESSION_SEGMENT`` in
#: orchard_evalkit/config.py and harbor_orchard/settings.py.
SESSION_PREFIX = "/session/"

#: Where the harness says a rollout is over: ``DELETE /router/session/<sid>``.
#: Under ``/router/`` because that namespace is registered ahead of the
#: catch-all and so can never be shadowed by a path the engines serve.
CLOSE_PREFIX = "/router/session/"

#: An unauthenticated port with an unbounded dictionary key is the only denial
#: of service surface here, and no real session id comes close to this.
MAX_SID_LEN = 512

#: How long after its last request a session still counts against its engine
#: when nothing closed it. It has to cover the *gap between turns*, which for a
#: coding agent is a test suite run rather than a think — so it is long, and a
#: rollout that ends without a close goes on occupying its engine for this much
#: longer than it ran. That is the price of guessing, and it is why the harness
#: closes its session at teardown rather than leaving this to expire.
ACTIVE_TTL_S = 900.0

#: A closed session returning sooner than this goes back to the same engine;
#: later than this it is placed again. Keying on the task rather than the
#: attempt exists so a retry lands on the engine still holding its prefix, but
#: that only pays while the prefix is there — past some gap the engines have
#: been serving other rollouts for longer than this one was away, and the pin
#: is no longer a cache hit, only an imbalance the router inflicted on itself.
REOPEN_WARM_S = 600.0

#: When a session's bookkeeping is forgotten entirely. Affects memory and the
#: readability of /router/stats only: a returning id is simply placed again,
#: and by then the engine has long since evicted its prefix anyway.
SESSION_TTL_S = 7200.0

#: Never move the same session twice inside this window. A fleet-wide stall
#: would otherwise shuffle every session onto every engine in turn and throw
#: away eight prefix caches instead of one.
REASSIGN_COOLDOWN_S = 30.0

#: A load sample older than this many health intervals is ignored, and once any
#: candidate's sample is stale the tiebreak is dropped for the whole fleet.
#: Half-fresh samples are worse than none: the engine that stopped answering
#: would score zero and attract everything.
LOAD_STALE_FACTOR = 2.5

#: What an engine is asked about itself, in the order the routes are tried.
#: ``/get_server_info`` is unconditional in recent sglang; ``/metrics`` needs
#: ``--enable-metrics``. Neither is required.
LOAD_ROUTES = ("/get_server_info", "/metrics")

#: The key the engines are started with when nobody names one, straight from
#: ``SERVE_API_KEY`` in ``scripts/serve_sglang_fleet.sh``. The two scripts are a
#: pair and have to agree on this: sglang guards ``/get_server_info`` behind
#: ``--api-key`` but leaves ``/v1/models`` answering 401 rather than 5xx, so a
#: router holding the wrong key sees a fleet that is unanimously healthy and
#: unanimously silent about its load — the tiebreak off for the whole run, with
#: nothing in the log to say why. That is not a hypothetical: it is what an
#: unset ``$API_KEY`` did to a DeepSWE sweep.
DEFAULT_API_KEY = "token-abc123"

#: Engine-reported counters, in the spellings sglang has given them. Read as a
#: set of alternatives rather than a single name because these have been
#: renamed across releases, and a rename must cost the tiebreak rather than
#: produce a zero that reads as an idle engine.
RUNNING_KEYS = ("num_running_reqs", "num_running_requests")
QUEUE_KEYS = ("num_queue_reqs", "num_queued_reqs", "num_waiting_reqs")
USAGE_KEYS = ("token_usage",)

#: A turn takes as long as the model needs, so there is no total deadline; what
#: there is instead is a *silence* deadline. Every byte resets it, so for a
#: streamed response this is an idle detector. For a non-streamed
#: /v1/chat/completions there are no intervening bytes at all, so it has to
#: outlast one whole generation — which is why it is not the 60s that looks
#: sufficient.
SOCK_READ_S = 1800.0

#: The engines are on loopback. A connect slower than this is a dead process.
SOCK_CONNECT_S = 10.0

#: Upstream pool idle time. Longer than the gap between an agent's turns, so a
#: loop that spends four minutes running pytest does not pay a reconnect.
UPSTREAM_KEEPALIVE_S = 300.0

#: Client-side keep-alive, for the same reason from the other direction.
CLIENT_KEEPALIVE_S = 620.0

#: Headers describing *this* hop, which must not be copied to the next one
#: (RFC 9110 7.6.1). ``Transfer-Encoding`` is the one that actually bites:
#: aiohttp's client refuses a relayed TE that collides with the framing it
#: picked for a streamed body, and its server would wrap an already-chunked
#: response in a second chunked layer on the way back.
HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)

#: Written by whichever server emits the response; copying the upstream's would
#: put two of each on the wire.
RESPONSE_DROP = frozenset({"date", "server"})

#: The only routes whose body this proxy reads. Anthropic-shaped, so exact
#: matches: the OpenAI routes the other three harnesses use are never buffered.
EFFORT_ROUTES = frozenset({"/v1/messages", "/v1/messages/count_tokens"})

log = logging.getLogger("session-router")


def split_session(raw_path: str, prefix: str = SESSION_PREFIX) -> tuple[str, str]:
    """``/session/abc/v1/responses`` -> ``("abc", "/v1/responses")``.

    Takes *prefix* so the close route splits its path by exactly the same rule
    the proxy splits its own: an id is whatever precedes the first literal
    ``/``, whichever namespace it arrived under. Two spellings of that rule
    would be two chances for a close to name an id the proxy never stored.

    Splits the *raw*, still-percent-encoded path rather than reading
    ``request.match_info``. A session id is an opaque token that can legally
    contain ``%2F`` — a Harbor task name is ``org/task`` — and which aiohttp
    versions decode ``%2F`` before matching has changed more than once.
    Splitting the encoded string and forwarding the remainder untouched means
    the router never re-encodes anything, so no path can be corrupted in
    transit whatever aiohttp does underneath.
    """
    if not raw_path.startswith(prefix):
        return "", raw_path
    sid, slash, suffix = raw_path[len(prefix) :].partition("/")
    if not sid:
        return "", raw_path
    return sid, f"{slash}{suffix}" or "/"


def _hop_by_hop(headers) -> set[str]:
    """Hop-by-hop header names, including whatever ``Connection:`` nominates."""
    drop = set(HOP_BY_HOP)
    for token in headers.get("Connection", "").split(","):
        name = token.strip().lower()
        if name:
            drop.add(name)
    return drop


def request_headers(request: web.Request) -> dict[str, str]:
    """What to send upstream: everything but this hop's own framing.

    ``Authorization`` passes through untouched, which is what keeps sglang's
    ``--api-key`` working through the router. ``Host`` is dropped because
    aiohttp derives it from the target URL.
    """
    drop = _hop_by_hop(request.headers) | {"host"}
    return {k: v for k, v in request.headers.items() if k.lower() not in drop}


def response_headers(upstream: aiohttp.ClientResponse) -> dict[str, str]:
    """What to send back. ``Content-Encoding`` is kept deliberately.

    Keeping it is only correct because the client session is built with
    ``auto_decompress=False``: with decompression on, aiohttp would hand this
    proxy plaintext while the header still claimed gzip, and the agent would
    fail to parse its own response.
    """
    drop = _hop_by_hop(upstream.headers) | RESPONSE_DROP
    return {k: v for k, v in upstream.headers.items() if k.lower() not in drop}


def strip_effort(raw: bytes) -> bytes:
    """Drop ``output_config.effort`` from an Anthropic request body.

    Claude Code puts an effort level on every request and offers no way not to:
    absent ``--effort`` it sends ``high``. sglang's Anthropic adapter forwards
    the level as chat-completions ``reasoning_effort``, rewriting ``xhigh`` to
    ``max`` on the way, and Qwen3.8-27B's chat template accepts only ``xhigh``,
    ``medium`` and ``low`` and raises on anything else. So the CLI's own
    default is a 500, ``xhigh`` is unreachable from the CLI at all, and
    ``medium`` — the highest level that survives — is the one setting that
    injects *no* reasoning instruction. Removing the field puts the request on
    the template's ``default('xhigh')`` branch, which is where the harnesses
    that send no effort at all already sit.

    Returns *raw* itself, not a copy, when there is nothing to strip, so the
    caller can tell by identity whether the body changed.
    """
    try:
        doc = json.loads(raw)
    except (UnicodeDecodeError, ValueError):
        return raw
    if not isinstance(doc, dict):
        return raw
    config = doc.get("output_config")
    if not isinstance(config, dict) or "effort" not in config:
        return raw
    config = {k: v for k, v in config.items() if k != "effort"}
    # Send no output_config rather than an empty one: a request that never had
    # the field is the thing being imitated, and it is the shape the adapter is
    # certain to handle.
    if config:
        doc["output_config"] = config
    else:
        doc.pop("output_config")
    return json.dumps(doc).encode()


def _as_number(value: object) -> float | None:
    """A float, for anything honestly one. ``True`` is not, and neither is a
    NaN — an unordered value in a sort key would make placement arbitrary."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _collect(scopes, keys, reduce):
    """First matching key per scope, combined by *reduce*; ``None`` if absent."""
    values = []
    for scope in scopes:
        for key in keys:
            value = _as_number(scope.get(key))
            if value is not None:
                values.append(value)
                break
    return reduce(values) if values else None


def parse_server_info(payload: bytes) -> tuple[float, float, float] | None:
    """Requests running, requests queued and KV usage from /get_server_info.

    The counters live at the top level in older sglang and inside
    ``internal_states`` since, so both are read. A release that renames them
    again yields ``None``, which turns the tiebreak off rather than reporting
    an idle engine.
    """
    try:
        doc = json.loads(payload)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(doc, dict):
        return None

    scopes = [doc]
    states = doc.get("internal_states")
    if isinstance(states, dict):
        scopes.append(states)
    elif isinstance(states, list):
        # One entry per data-parallel rank, so the counts add and the fraction
        # does not.
        scopes.extend(s for s in states if isinstance(s, dict))

    running = _collect(scopes, RUNNING_KEYS, sum)
    queued = _collect(scopes, QUEUE_KEYS, sum)
    usage = _collect(scopes, USAGE_KEYS, max)
    if running is None and queued is None:
        return None
    return running or 0.0, queued or 0.0, usage or 0.0


def parse_metrics(payload: bytes) -> tuple[float, float, float] | None:
    """The same three numbers out of a Prometheus exposition.

    Deliberately not a full parser: it reads ``name{labels} value`` and ignores
    everything it does not recognise, so an exposition carrying metrics from
    other libraries — which sglang's does — costs nothing to skip.
    """
    running: list[float] = []
    queued: list[float] = []
    usage: list[float] = []
    for raw in payload.decode("utf-8", "replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, brace, rest = line.partition("{")
        if brace:
            _, _, tail = rest.partition("}")
        else:
            name, _, tail = line.partition(" ")
        fields = tail.split()
        # A sample may carry a trailing timestamp, and may be NaN or +Inf while
        # the engine is starting.
        value = _as_number_text(fields[0]) if fields else None
        if value is None:
            continue
        # `sglang:num_running_reqs` -> `num_running_reqs`
        key = name.strip().rpartition(":")[2]
        if key in RUNNING_KEYS:
            running.append(value)
        elif key in QUEUE_KEYS:
            queued.append(value)
        elif key in USAGE_KEYS:
            usage.append(value)
    if not running and not queued:
        return None
    return float(sum(running)), float(sum(queued)), max(usage, default=0.0)


def _as_number_text(text: str) -> float | None:
    try:
        return _as_number(float(text))
    except ValueError:
        return None


def parse_load(route: str, payload: bytes) -> tuple[float, float, float] | None:
    """Whichever parser *route* is served by."""
    if route == "/metrics":
        return parse_metrics(payload)
    return parse_server_info(payload)


def _error(status: int, message: str, kind: str) -> web.Response:
    """An OpenAI-shaped error body.

    Agent CLIs print the response body into their trajectory, so an aiohttp
    HTML error page lands in the transcript as noise. This at least reads as
    the routing failure it is.
    """
    return web.json_response(
        {
            "error": {
                "message": f"session-router: {message}",
                "type": kind,
                "code": status,
            }
        },
        status=status,
    )


def raise_fd_limit() -> int:
    """Lift RLIMIT_NOFILE toward the hard limit and report where it landed.

    Two sockets per in-flight request plus a pooled connection per engine puts
    a 48-agent run in the high hundreds, and the common default is 1024 —
    shared with whatever sglang is already holding. Exhaustion surfaces as
    ``OSError(EMFILE)`` out of ``accept()``, which reads like a network fault.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = min(hard, 65536)
    if soft < want:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
        except (ValueError, OSError):  # pragma: no cover - platform dependent
            pass
    return resource.getrlimit(resource.RLIMIT_NOFILE)[0]


@dataclass
class Session:
    """One agent loop, and the engine holding its prefix."""

    sid: str
    upstream: str
    created: float
    last_seen: float
    inflight: int = 0
    requests: int = 0
    errors: int = 0
    moves: int = 0
    moved_at: float = 0.0
    #: When the harness said this rollout was over. 0.0 while it is open.
    closed_at: float = 0.0

    def live(self, now: float, ttl: float) -> bool:
        """Whether this session still counts against its engine.

        In-flight beats everything else, and that is not a formality: a trial
        Harbor abandoned at its timeout leaves the CLI in the pod generating
        for another three quarters of an hour, and those are real GPU seconds
        on this engine however finished the harness considers the trial.

        Otherwise a closed session frees its slot at once, and one nobody
        closed falls back to the idle window.
        """
        if self.inflight > 0:
            return True
        if self.closed_at:
            return False
        return (now - self.last_seen) <= ttl


@dataclass
class Upstream:
    """One sglang server."""

    url: str
    index: int
    healthy: bool = True
    inflight: int = 0
    requests: int = 0
    failures: int = 0
    assigned: int = 0
    last_error: str = ""
    last_ok: float = 0.0
    #: What the engine last said about itself. ``load_at`` of 0.0 means it has
    #: never answered a load probe, which is a supported state and not a fault.
    running: float = 0.0
    queued: float = 0.0
    token_usage: float = 0.0
    load_at: float = 0.0
    load_source: str = ""
    #: Whether the last load probe was turned away for the key it presented, and
    #: whether that has been said out loud yet. Distinct from "answers neither
    #: route", which is a supported configuration and stays quiet.
    load_denied: bool = False
    #: Sessions placed here since that sample was taken. Without it a burst of
    #: placements would all read one stale number and stampede onto whichever
    #: engine happened to look idle when it was taken.
    since_sample: int = 0

    @property
    def label(self) -> str:
        """``:8003`` — what the logs use, since the host is always the same."""
        port = URL(self.url).port
        return f":{port}" if port else self.url


class Router:
    """The session table, the placement rule, and the relay."""

    def __init__(
        self,
        upstreams: list[str],
        *,
        api_key: str | None = None,
        active_ttl: float = ACTIVE_TTL_S,
        session_ttl: float = SESSION_TTL_S,
        sock_read: float = SOCK_READ_S,
        health_interval: float = 15.0,
        spread_interval: float = 60.0,
        require_session: bool = False,
        assign_log: Path | None = None,
        reopen_warm: float = REOPEN_WARM_S,
        load_probe: bool = True,
        strip_effort: bool = True,
    ) -> None:
        self.upstreams = [Upstream(url=u.rstrip("/"), index=i) for i, u in enumerate(upstreams)]
        self._by_url = {u.url: u for u in self.upstreams}
        self.api_key = api_key
        self.active_ttl = active_ttl
        self.session_ttl = session_ttl
        self.sock_read = sock_read
        self.health_interval = health_interval
        self.spread_interval = spread_interval
        self.require_session = require_session
        self.assign_log = assign_log
        self.reopen_warm = reopen_warm
        self.load_probe = load_probe
        self.strip_effort = strip_effort

        self.sessions: dict[str, Session] = {}
        self.client: aiohttp.ClientSession | None = None
        self.started = time.monotonic()
        self.status_counts: Counter[str] = Counter()
        self.total_requests = 0
        self.no_session = 0
        self.no_session_posts = 0
        self.effort_stripped = 0
        self.evicted = 0
        self.reassigned = 0
        self.closed = 0
        self.reopened = 0
        self.evacuated = 0
        self._cursor = 0
        self._tasks: list[asyncio.Task] = []
        self._assign_fh = None
        self._warned: dict[str, float] = {}

    # ---------------------------------------------------------------- lifecycle

    async def start(self, app: web.Application) -> None:
        """Build the upstream client. Never at import: a session created
        outside the running loop binds to the wrong one."""
        self.client = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(
                # The engines are the queue. A connector limit here would turn
                # engine backpressure into an opaque client-side stall, and
                # aiohttp charges that wait to the connect timeout — so an
                # overload would surface as a timeout from a healthy fleet.
                limit=0,
                limit_per_host=0,
                keepalive_timeout=UPSTREAM_KEEPALIVE_S,
                enable_cleanup_closed=True,
                force_close=False,
            ),
            # ClientSession defaults to ClientTimeout(total=5*60), which is the
            # single most likely way this router breaks a run: it would cut
            # every generation longer than five minutes off mid-SSE, and the
            # agent would report a truncated stream rather than a timeout.
            timeout=aiohttp.ClientTimeout(
                total=None,
                # Also covers waiting for a pooled connection; a queue is not
                # a fault.
                connect=None,
                sock_connect=SOCK_CONNECT_S,
                sock_read=self.sock_read,
            ),
            auto_decompress=False,
            # And therefore: never advertise an encoding the real client did
            # not ask for, or aiohttp hands a gzip body to an agent that never
            # requested one.
            skip_auto_headers=("Accept", "Accept-Encoding", "User-Agent"),
            # HTTP_PROXY happening to be set in the sglang venv would send
            # every rollout through a proxy that cannot reach 127.0.0.1.
            trust_env=False,
            raise_for_status=False,
        )
        assert self.client.timeout.total is None, "the 5-minute default survived"

        if self.assign_log is not None:
            self.assign_log.parent.mkdir(parents=True, exist_ok=True)
            self._assign_fh = self.assign_log.open("a", encoding="utf-8")

        if self.health_interval > 0:
            self._tasks.append(asyncio.create_task(self._health_loop()))
        self._tasks.append(asyncio.create_task(self._sweep_loop()))

    async def stop(self, app: web.Application) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        if self.client is not None:
            await self.client.close()
        log.info("final spread: %s", self._spread_line())
        if self._assign_fh is not None:
            self._assign_fh.close()

    # ---------------------------------------------------------------- placement

    def place(self, now: float) -> Upstream:
        """The engine a new session goes to: fewest live sessions, then the
        shallowest queue the engines report, then fewest in-flight requests,
        then round-robin.

        Load is counted over *live* sessions rather than over every session
        ever assigned, and that difference is the whole design. A 500-instance
        run at concurrency 48 retires a session every few minutes; a cumulative
        counter would converge on equal lifetime totals while saying nothing
        about who is busy now, and a rollout that died on turn 1 would weigh
        the same as one forty turns deep. What an engine is actually paying for
        is the conversations it currently holds, so that is what is counted.

        What makes that count trustworthy is the harness closing its session at
        teardown; see :meth:`close`. Left to the idle window alone, an engine
        carries every rollout it has recently finished as phantom load, in
        proportion to how fast it finishes them — so a wave of rollouts that
        die in two minutes can leave one engine forty-odd phantom sessions deep
        and excluded from placement while it sits idle.

        Recomputed rather than maintained incrementally: a decrement would have
        to fire when a session *stops* being live, which is a timer per session
        and a class of bug — a leaked counter permanently poisons an engine —
        to save iterating a few dozen entries once per session, not per request.

        Synchronous on purpose. It must not yield between reading the table and
        writing the assignment, or two first-requests of one rollout would each
        place it.
        """
        live: Counter[str] = Counter()
        for session in self.sessions.values():
            if session.live(now, self.active_ttl):
                live[session.upstream] += 1

        # An all-unhealthy fleet still gets traffic: refusing to place is
        # strictly worse than placing on an engine that may have recovered
        # since the last probe.
        pool = [u for u in self.upstreams if u.healthy] or self.upstreams
        cursor, self._cursor = self._cursor, self._cursor + 1
        measured = self._load_is_usable(pool, now)
        return min(
            pool,
            key=lambda u: (
                live[u.url],
                # Where most of the discrimination actually happens: 48
                # rollouts over 8 engines is six each, so the primary key ties
                # for most of a run.
                self._load_score(u) if measured else 0.0,
                u.inflight,
                # Rotating, so the first eight sessions of a run land one per
                # engine instead of all eight on upstreams[0].
                (u.index - cursor) % len(self.upstreams),
            ),
        )

    def _load_is_usable(self, pool: list[Upstream], now: float) -> bool:
        """Whether every candidate has answered a load probe recently enough.

        All or nothing. Scoring one engine on a real queue depth and another on
        a sample that stopped arriving would rank the engine that went quiet as
        the emptiest in the fleet, which is the opposite of what a stale sample
        means.
        """
        if not self.load_probe or self.health_interval <= 0:
            return False
        horizon = self.health_interval * LOAD_STALE_FACTOR
        return all(u.load_at and now - u.load_at <= horizon for u in pool)

    def _load_score(self, upstream: Upstream) -> float:
        """How busy an engine says it is, plus what has been sent to it since.

        Requests queued and running rather than tokens: KV occupancy is already
        approximated by the live-session count this breaks ties within, and
        combining the two would need a weight nobody here can justify.
        ``token_usage`` is reported by /router/stats instead of being scored.
        """
        return upstream.running + upstream.queued + upstream.since_sample

    def session_for(self, sid: str, now: float) -> Session:
        """The session for *sid*, placing it if this is its first request."""
        session = self.sessions.get(sid)
        if session is not None:
            cold = bool(
                session.closed_at
                and not session.inflight
                and now - session.closed_at > self.reopen_warm
            )
            session.closed_at = 0.0
            session.last_seen = now
            if cold:
                self._reopen(session, now)
            return session
        upstream = self.place(now)
        upstream.assigned += 1
        upstream.since_sample += 1
        session = Session(sid=sid, upstream=upstream.url, created=now, last_seen=now)
        self.sessions[sid] = session
        self._record_assignment(session, upstream, now, reason="new")
        return session

    def _reopen(self, session: Session, now: float) -> None:
        """Place a closed session again, because it came back cold.

        ``last_seen`` is already updated when this runs, so the session counts
        as live on its old engine and placement weighs against returning there
        rather than for it. A tie still returns it, which is the right answer:
        a cold cache on a busy engine is no better than a cold cache here.
        """
        if now - session.moved_at < REASSIGN_COOLDOWN_S:
            return
        upstream = self.place(now)
        if upstream.url == session.upstream:
            return
        session.upstream = upstream.url
        session.moves += 1
        session.moved_at = now
        upstream.assigned += 1
        upstream.since_sample += 1
        self.reopened += 1
        self._record_assignment(session, upstream, now, reason="reopened")

    def close(self, sid: str, now: float) -> Session | None:
        """Mark *sid* finished, freeing its slot for the next rollout.

        Idempotent, and deliberately not a deletion: the record is what lets a
        retry of the same task go back to the engine still holding its prefix,
        and what makes /router/stats legible after a run. A session closed
        while something is still streaming from it stays counted until its last
        byte — see :meth:`Session.live`.
        """
        session = self.sessions.get(sid)
        if session is None or session.closed_at:
            return session
        session.closed_at = now
        self.closed += 1
        self._record_assignment(
            session, self._by_url[session.upstream], now, reason="closed"
        )
        return session

    def _round_robin(self) -> Upstream:
        """Where an unkeyed request goes. No affinity is owed to one."""
        pool = [u for u in self.upstreams if u.healthy] or self.upstreams
        self._cursor += 1
        return pool[self._cursor % len(pool)]

    def _record_assignment(
        self, session: Session, upstream: Upstream, now: float, *, reason: str
    ) -> None:
        """Log a placement, and append it to the audit file.

        The audit file matters more than it looks. The suite used to record the
        engine per rollout as ``metrics.base_url``, and a run was checked with
        ``jq -r .metrics.base_url results.jsonl | sort | uniq -c``. Behind a
        router every rollout records the same port, so that recipe now reports
        500 unique session URLs and says nothing about spread. This file is the
        replacement, and without it the distribution is not recoverable.
        """
        live = Counter(
            s.upstream for s in self.sessions.values() if s.live(now, self.active_ttl)
        )
        vector = " ".join(str(live[u.url]) for u in self.upstreams)
        log.info(
            "%s sid=%s -> %s  live=[%s] known=%d",
            reason,
            unquote(session.sid),
            upstream.label,
            vector,
            len(self.sessions),
        )
        if self._assign_fh is None:
            return
        self._assign_fh.write(
            json.dumps(
                {
                    "ts": time.time(),
                    "sid": unquote(session.sid),
                    "upstream": upstream.url,
                    "live_at_assign": live[upstream.url],
                    "reason": reason,
                }
            )
            + "\n"
        )
        self._assign_fh.flush()

    # ------------------------------------------------------------------- relay

    async def proxy(self, request: web.Request) -> web.StreamResponse:
        """Relay one request to the engine this session belongs to."""
        # rel_url.raw_path, not request.path: the latter is percent-decoded, so
        # a %2F inside a session id would become a real separator and silently
        # change the shape of the forwarded path.
        raw_path = request.rel_url.raw_path
        sid, suffix = split_session(raw_path)
        now = time.monotonic()

        if sid and len(sid) > MAX_SID_LEN:
            return _error(400, f"session id longer than {MAX_SID_LEN} bytes", "invalid_request_error")

        if sid:
            session = self.session_for(sid, now)
            upstream = self._by_url[session.upstream]
        else:
            session = None
            denied = self._note_unkeyed(request, raw_path)
            if denied is not None:
                return denied
            upstream = self._round_robin()

        self.total_requests += 1
        return await self._relay(request, suffix, session, upstream, now)

    def _note_unkeyed(self, request: web.Request, raw_path: str) -> web.Response | None:
        """Account for a request that named no session, and maybe refuse it.

        ``GET /v1/models`` legitimately has no session — it is what
        serve_sglang_fleet.sh and check_model_from_sandbox.py probe with. A
        *POST* without one is different: ``urljoin("http://h/session/x/v1",
        "/v1/chat/completions")`` yields ``http://h/v1/chat/completions``, so
        any client that joins an absolute path instead of concatenating drops
        its affinity silently. That has to be visible.
        """
        self.no_session += 1
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return None
        self.no_session_posts += 1
        if self.require_session:
            return _error(
                400,
                f"{request.method} {raw_path} named no session; expected "
                f"{SESSION_PREFIX}<id>{raw_path}",
                "invalid_request_error",
            )
        # Rate-limited: one client doing this does it on every turn.
        last = self._warned.get(raw_path, 0.0)
        now = time.monotonic()
        if now - last > 60.0:
            self._warned[raw_path] = now
            log.warning(
                "%s %s named no session, so it is round-robined and its agent "
                "loop will not reuse any engine's prefix cache",
                request.method,
                raw_path,
            )
        return None

    def _strips_effort(self, request: web.Request, suffix: str) -> bool:
        """Whether this request's body is worth buffering and rewriting.

        Narrow on purpose: a POST, to one of two Anthropic routes, whose body
        this router can read as-is. A compressed body is passed through rather
        than decompressed — nothing sends one, and guessing is how a proxy
        corrupts a request.
        """
        return (
            self.strip_effort
            and request.method == "POST"
            and suffix in EFFORT_ROUTES
            and "Content-Encoding" not in request.headers
        )

    async def _relay(
        self,
        request: web.Request,
        suffix: str,
        session: Session | None,
        upstream: Upstream,
        now: float,
    ) -> web.StreamResponse:
        headers = request_headers(request)
        # body_exists is False for a bodyless GET, which stops aiohttp bolting
        # a Content-Length: 0 onto a request that had none.
        body: aiohttp.StreamReader | bytes | None = (
            request.content if request.body_exists else None
        )
        if body is not None and self._strips_effort(request, suffix):
            # The one place this proxy holds a body. Every other request —
            # every OpenAI one included — is still relayed straight from the
            # client socket to the engine. build_app's client_max_size=0 is
            # what makes read() safe for a turn-40 conversation.
            raw = await request.read()
            body = strip_effort(raw)
            if body is not raw:
                self.effort_stripped += 1
            # request_headers copied the original length, which the rewrite has
            # invalidated, and a chunked upload arrives with no length at all.
            headers["Content-Length"] = str(len(body))
        query = request.rel_url.raw_query_string

        response: aiohttp.ClientResponse | None = None
        for attempt in (1, 2):
            target = URL(
                f"{upstream.url}{suffix}" + (f"?{query}" if query else ""),
                # Lossless: yarl would otherwise re-quote and normalize a path
                # this router has no business touching.
                encoded=True,
            )
            upstream.inflight += 1
            if session is not None:
                session.inflight += 1
            try:
                response = await self.client.request(
                    request.method,
                    target,
                    headers=headers,
                    data=body,
                    allow_redirects=False,
                )
                break
            except aiohttp.ClientConnectorError as exc:
                # Raised while establishing the connection, so the body is
                # either still unread or held in full, and retrying cannot
                # truncate it either way. Any later failure is not safe to
                # replay and falls through below.
                upstream.inflight -= 1
                if session is not None:
                    session.inflight -= 1
                self._note_failure(upstream, exc)
                moved = self._reassign(session, upstream, now) if attempt == 1 else None
                if moved is None:
                    return self._connect_error(upstream, session, exc)
                upstream = moved
            except (TimeoutError, aiohttp.ClientError, OSError) as exc:
                upstream.inflight -= 1
                if session is not None:
                    session.inflight -= 1
                self._note_failure(upstream, exc)
                return self._connect_error(upstream, session, exc)

        assert response is not None
        upstream.requests += 1
        if session is not None:
            session.requests += 1
        self.status_counts[str(response.status)] += 1
        if response.status >= 500:
            # Forwarded verbatim, and the session is NOT moved: a 500 from
            # sglang is almost always about *this* request — context length,
            # a tool schema its parser rejects, an OOM on one batch — and
            # migrating a healthy conversation over one bad request discards a
            # warm cache for nothing.
            self._note_failure(upstream, RuntimeError(f"HTTP {response.status}"))

        out = web.StreamResponse(
            status=response.status,
            reason=response.reason,
            headers=response_headers(response),
        )
        try:
            await out.prepare(request)
            # iter_any yields whatever arrived, with no re-chunking and no line
            # parsing, so one SSE frame reaches the agent as soon as sglang
            # emits it. iter_chunked would add a copy and a buffer per token.
            async for chunk in response.content.iter_any():
                # Awaited, not fired and forgotten: the drain is what pushes a
                # slow client back through the tunnel to the engine's socket
                # instead of growing an unbounded buffer in this process.
                await out.write(chunk)
            await out.write_eof()
        except (TimeoutError, aiohttp.ClientError, OSError) as exc:
            # Bytes are already on the wire, so there is no status left to
            # change. Returning the started response lets aiohttp close the
            # connection, which is what tells the agent its stream was cut.
            self._note_failure(upstream, exc)
            if session is not None:
                session.errors += 1
            log.warning("stream from %s broke: %s", upstream.label, exc)
        finally:
            upstream.inflight -= 1
            if session is not None:
                session.inflight -= 1
                session.last_seen = time.monotonic()
            # A no-op once the body reached EOF, and a hard TCP close
            # otherwise. That close is the only lever this router has over a
            # generation nobody wants any more: sglang aborts a streaming
            # request when its client disconnects, so a rollout killed at its
            # timeout stops costing GPU seconds here instead of decoding to
            # max_tokens for nobody. Not release(): release() is about
            # connection reuse, and reuse is what a half-read SSE stream must
            # not get.
            response.close()
        return out

    def _reassign(
        self, session: Session | None, dead: Upstream, now: float
    ) -> Upstream | None:
        """Move *session* off a refusing engine, if that is safe and useful."""
        if session is None:
            healthy = [u for u in self.upstreams if u.healthy and u.url != dead.url]
            return healthy[0] if healthy else None
        if session.inflight > 0:
            return None  # a sibling turn is still streaming from the old engine
        if now - session.moved_at < REASSIGN_COOLDOWN_S:
            return None
        candidates = [u for u in self.upstreams if u.healthy and u.url != dead.url]
        if not candidates:
            return None
        upstream = self.place(now)
        if upstream.url == dead.url:
            upstream = candidates[0]
        session.upstream = upstream.url
        session.moves += 1
        session.moved_at = now
        self.reassigned += 1
        upstream.assigned += 1
        upstream.since_sample += 1
        log.warning(
            "moved session %s %s -> %s after %s; its prefix cache is cold, so "
            "its next turn re-prefills the whole conversation",
            unquote(session.sid),
            dead.label,
            upstream.label,
            dead.last_error or "a connection failure",
        )
        self._record_assignment(session, upstream, now, reason="moved")
        return upstream

    def _connect_error(
        self, upstream: Upstream, session: Session | None, exc: BaseException
    ) -> web.Response:
        self.status_counts["502"] += 1
        if session is not None:
            session.errors += 1
        return _error(502, f"{upstream.label} is unreachable: {exc}", "api_error")

    def _note_failure(self, upstream: Upstream, exc: BaseException) -> None:
        upstream.failures += 1
        upstream.last_error = f"{type(exc).__name__}: {exc}"[:200]

    # ------------------------------------------------------------ housekeeping

    async def _health_loop(self) -> None:
        """Probe /v1/models so new sessions stop landing on a dead engine.

        /v1/models is the cheapest route that proves the *model* is loaded
        rather than that a socket accepts, and it is what serve_sglang_fleet.sh
        already waits on — so a fleet this probe calls healthy is a fleet that
        script would have called ready.

        Probed concurrently, because serially one engine timing out delays the
        next engine's probe by the whole timeout and eight of those outlast the
        interval itself — the fleet would be slowest to notice a failure
        exactly when several engines are in trouble.

        Nothing raised by a probe may end this loop: a router that silently
        stopped checking would keep placing sessions on a dead engine forever,
        and the symptom would be indistinguishable from the engine being fine.
        """
        while True:
            await asyncio.sleep(self.health_interval)
            await asyncio.gather(
                *(self._probe(u) for u in self.upstreams), return_exceptions=True
            )

    async def _probe(self, upstream: Upstream) -> None:
        """One engine's health check, and its load sample if it offers one.

        An engine that fails keeps serving whatever is still streaming from it
        — those sockets cannot be taken back — but its idle sessions are moved.
        The usual argument against moving a session is the warm prefix cache it
        would throw away, and that argument does not survive the engine having
        gone away: whatever comes back has an empty cache regardless.
        """
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        timeout = aiohttp.ClientTimeout(total=5)
        ok = False
        try:
            async with self.client.get(
                f"{upstream.url}/v1/models", headers=headers, timeout=timeout
            ) as resp:
                ok = resp.status < 500
                await resp.read()
        except (TimeoutError, aiohttp.ClientError, OSError) as exc:
            self._note_failure(upstream, exc)

        if not ok:
            if upstream.healthy and upstream.failures >= 2:
                upstream.healthy = False
                now = time.monotonic()
                moved = self._evacuate(upstream, now)
                log.warning(
                    "%s marked unhealthy (%s); new sessions will go elsewhere "
                    "and %d idle one(s) moved off it",
                    upstream.label,
                    upstream.last_error,
                    moved,
                )
            return

        upstream.last_ok = time.monotonic()
        if not upstream.healthy:
            upstream.healthy = True
            log.info("%s is healthy again", upstream.label)
        upstream.failures = 0
        if self.load_probe:
            await self._sample_load(upstream, headers)

    async def _sample_load(self, upstream: Upstream, headers: dict[str, str]) -> None:
        """Ask an engine how busy it is. Advisory in every sense.

        Never touches health. An sglang built without ``/get_server_info`` or
        started without ``--enable-metrics`` answers neither route, and taking
        a working GPU out of the fleet over a statistic would be a far worse
        outcome than placing without one. A failure here only lets the sample
        go stale, which turns the tiebreak off for the whole fleet.

        The one failure worth a word is 401/403, because it is the only one
        that means the fleet *would* answer. sglang guards this route behind
        ``--api-key`` and lets ``/v1/models`` answer 401 rather than 5xx, so a
        wrong key leaves every engine healthy, every load sample missing, and
        ``load_in_use`` false with nothing in the log pointing at the cause.
        """
        timeout = aiohttp.ClientTimeout(total=5)
        # Whatever answered last time first, so a fleet offering only one of
        # the two stops paying for the other after a single cycle.
        routes = sorted(LOAD_ROUTES, key=lambda route: route != upstream.load_source)
        denied = False
        for route in routes:
            try:
                async with self.client.get(
                    f"{upstream.url}{route}", headers=headers, timeout=timeout
                ) as resp:
                    payload = await resp.read()
                    if resp.status != 200:
                        denied = denied or resp.status in (401, 403)
                        continue
            except (TimeoutError, aiohttp.ClientError, OSError) as exc:
                log.debug("%s: %s did not answer: %s", upstream.label, route, exc)
                continue
            sample = parse_load(route, payload)
            if sample is None:
                continue
            upstream.running, upstream.queued, upstream.token_usage = sample
            upstream.load_at = time.monotonic()
            upstream.since_sample = 0
            upstream.load_denied = False
            if upstream.load_source != route:
                log.info("%s reports its load on %s", upstream.label, route)
                upstream.load_source = route
            return

        if denied and not upstream.load_denied:
            upstream.load_denied = True
            log.warning(
                "%s refused the load probe (%s) for the key it was given, so "
                "placement is on session count alone. Start this router with "
                "--api-key matching the engines' --api-key, or export $API_KEY "
                "before it",
                upstream.label,
                " and ".join(LOAD_ROUTES),
            )

    def _evacuate(self, dead: Upstream, now: float) -> int:
        """Move what can still be moved off an engine that just failed.

        Only the idle, open, live ones. A session with bytes in flight is
        attached to a socket this router cannot reclaim; a closed one has
        nothing left to route; an expired one is dead weight whose move would
        only pollute the audit log.

        Silent when no healthy engine remains, because a fleet-wide stall must
        not become a fleet-wide shuffle.
        """
        if not any(u.healthy and u.url != dead.url for u in self.upstreams):
            return 0
        moved = 0
        for session in list(self.sessions.values()):
            if session.upstream != dead.url or session.inflight or session.closed_at:
                continue
            if not session.live(now, self.active_ttl):
                continue
            if self._reassign(session, dead, now) is not None:
                moved += 1
        self.evacuated += moved
        return moved

    async def _sweep_loop(self) -> None:
        """Forget dead sessions, and say how the fleet is loaded."""
        while True:
            await asyncio.sleep(self.spread_interval)
            now = time.monotonic()
            stale = [
                sid
                for sid, s in self.sessions.items()
                if s.inflight == 0 and now - s.last_seen > self.session_ttl
            ]
            for sid in stale:
                del self.sessions[sid]
            self.evicted += len(stale)
            log.info("%s", self._spread_line())

    def _spread_line(self) -> str:
        now = time.monotonic()
        live = Counter(
            s.upstream for s in self.sessions.values() if s.live(now, self.active_ttl)
        )
        cells = " ".join(f"{u.label.lstrip(':')}:{live[u.url]}" for u in self.upstreams)
        inflight = sum(u.inflight for u in self.upstreams)
        return (
            f"spread live={sum(live.values())} {cells} inflight={inflight} "
            f"requests={self.total_requests} closed={self.closed} "
            f"unkeyed={self.no_session}"
        )

    # ------------------------------------------------------------------ routes

    async def stats(self, request: web.Request) -> web.Response:
        now = time.monotonic()
        live = Counter(
            s.upstream for s in self.sessions.values() if s.live(now, self.active_ttl)
        )
        return web.json_response(
            {
                "uptime_s": round(now - self.started, 1),
                "sessions": {
                    "live": sum(live.values()),
                    "known": len(self.sessions),
                    "open": sum(1 for s in self.sessions.values() if not s.closed_at),
                    "closed": self.closed,
                    "reopened": self.reopened,
                    "evicted": self.evicted,
                    "reassigned": self.reassigned,
                    "evacuated": self.evacuated,
                },
                "requests": {
                    "total": self.total_requests,
                    "inflight": sum(u.inflight for u in self.upstreams),
                    "no_session": self.no_session,
                    "no_session_posts": self.no_session_posts,
                    "effort_stripped": self.effort_stripped,
                },
                "status_counts": dict(sorted(self.status_counts.items())),
                "upstreams": [
                    {
                        "url": u.url,
                        "healthy": u.healthy,
                        "live": live[u.url],
                        "assigned": u.assigned,
                        "inflight": u.inflight,
                        "requests": u.requests,
                        "failures": u.failures,
                        "last_error": u.last_error,
                        "last_ok_s": round(now - u.last_ok, 1) if u.last_ok else None,
                        "load": {
                            "source": u.load_source or None,
                            "running": u.running,
                            "queued": u.queued,
                            "token_usage": round(u.token_usage, 3),
                            "age_s": round(now - u.load_at, 1) if u.load_at else None,
                            "since_sample": u.since_sample,
                        },
                    }
                    for u in self.upstreams
                ],
                "spread": {u.label.lstrip(":"): live[u.url] for u in self.upstreams},
                # Whether the "load" blocks above are actually breaking ties,
                # which is not inferable from their presence: one stale sample
                # disables the tiebreak for every engine.
                "load_in_use": self._load_is_usable(
                    [u for u in self.upstreams if u.healthy] or self.upstreams, now
                ),
            }
        )

    async def close_session(self, request: web.Request) -> web.Response:
        """``DELETE /router/session/<sid>`` — the harness saying a trial ended.

        A 200 even for an id this router has never seen. A rollout whose agent
        never reached the model still tears down and still closes, and a 404 in
        the harness log would read as a fault rather than as the no-op it is;
        ``known`` distinguishes the two for anyone who cares.

        The id is taken from the raw path rather than from ``match_info`` for
        the reason :func:`split_session` gives: a Harbor task name is
        ``org/task``, it arrives as ``org%2Ftask``, and which aiohttp versions
        decode that before matching has changed more than once.
        """
        sid, _ = split_session(request.rel_url.raw_path, CLOSE_PREFIX)
        if not sid:
            return _error(
                400, f"expected {CLOSE_PREFIX}<id>", "invalid_request_error"
            )
        session = self.close(sid, time.monotonic())
        if session is None:
            return web.json_response(
                {"sid": unquote(sid), "known": False, "closed": False}
            )
        return web.json_response(
            {
                "sid": unquote(sid),
                "known": True,
                "closed": True,
                "upstream": session.upstream,
                # Non-zero means something is still generating under this id
                # after the harness gave up on it, so the slot is not free yet.
                "inflight": session.inflight,
                "requests": session.requests,
            }
        )

    async def healthz(self, request: web.Request) -> web.Response:
        """Liveness of the *router*, independent of the engines, so that a dead
        fleet behind a live tunnel is distinguishable from nothing listening."""
        return web.json_response(
            {"ok": True, "healthy_upstreams": sum(u.healthy for u in self.upstreams)}
        )


def build_app(router: Router) -> web.Application:
    # client_max_size=0 disables aiohttp's 1 MiB body cap. It is only enforced
    # by request.read()/post(), which this proxy never calls — but a turn-40
    # codex conversation is well over 1 MiB, so leaving the default in place
    # would be leaving a trap for whoever later adds body logging.
    app = web.Application(client_max_size=0)
    # Registered before the catch-all: aiohttp resolves in registration order,
    # so the reserved namespace can never be shadowed by a proxied path.
    app.router.add_get("/router/stats", router.stats)
    app.router.add_get("/router/healthz", router.healthz)
    # `{tail:.*}` rather than `{sid}` so a decoded %2F cannot fail to match;
    # the handler re-splits the raw path itself.
    app.router.add_route("DELETE", "/router/session/{tail:.*}", router.close_session)
    app.router.add_route("*", "/{tail:.*}", router.proxy)
    app.on_startup.append(router.start)
    app.on_cleanup.append(router.stop)
    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-u",
        "--upstream",
        action="append",
        default=[],
        help="Engine URL, repeatable. Wins over --base-port/--replicas",
    )
    parser.add_argument("--base-port", type=int, default=8000)
    parser.add_argument(
        "--replicas",
        type=int,
        default=8,
        help="Engines on consecutive ports from --base-port",
    )
    parser.add_argument("--upstream-host", default="127.0.0.1")
    parser.add_argument("--host", default="127.0.0.1", help="Where the router listens")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("API_KEY") or DEFAULT_API_KEY,
        help=(
            "Presented to the engines on health and load probes. Defaults to "
            f"$API_KEY, then to {DEFAULT_API_KEY!r} — the same fallback "
            "serve_sglang_fleet.sh starts them with. Pass an empty string to "
            "probe unauthenticated"
        ),
    )
    parser.add_argument("--active-ttl", type=float, default=ACTIVE_TTL_S)
    parser.add_argument(
        "--reopen-warm",
        type=float,
        default=REOPEN_WARM_S,
        help="Seconds a closed session keeps its engine if it comes back",
    )
    parser.add_argument(
        "--load-probe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Break placement ties on what the engines report about their own "
            "queues. Harmless where they report nothing"
        ),
    )
    parser.add_argument(
        "--strip-effort",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Drop output_config.effort from POST /v1/messages, which is the "
            "only way a Claude Code rollout reaches this fleet's chat template "
            "on its xhigh default. --no-strip-effort relays bodies verbatim"
        ),
    )
    parser.add_argument("--session-ttl", type=float, default=SESSION_TTL_S)
    parser.add_argument("--read-timeout", type=float, default=SOCK_READ_S)
    parser.add_argument(
        "--health-interval", type=float, default=15.0, help="0 disables probing"
    )
    parser.add_argument("--spread-interval", type=float, default=60.0)
    parser.add_argument(
        "--require-session",
        action="store_true",
        help=(
            "Refuse a POST that names no session instead of round-robining it. "
            "Off by default: it would break /v1/models smoke tests."
        ),
    )
    parser.add_argument(
        "--assign-log",
        default="/tmp/session-router/assignments.jsonl",
        help="Append-only placement record. Empty disables it",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    upstreams = args.upstream or [
        f"http://{args.upstream_host}:{args.base_port + i}"
        for i in range(args.replicas)
    ]
    router = Router(
        upstreams,
        api_key=args.api_key,
        active_ttl=args.active_ttl,
        session_ttl=args.session_ttl,
        sock_read=args.read_timeout,
        health_interval=args.health_interval,
        spread_interval=args.spread_interval,
        require_session=args.require_session,
        assign_log=Path(args.assign_log) if args.assign_log else None,
        reopen_warm=args.reopen_warm,
        load_probe=args.load_probe,
        strip_effort=args.strip_effort,
    )

    # Ships with sglang via uvicorn[standard]; worth taking when it is there
    # and not worth failing over when it is not.
    try:
        import uvloop
    except ImportError:
        pass
    else:
        uvloop.install()

    fds = raise_fd_limit()
    log.info("routing %d engine(s):", len(router.upstreams))
    for upstream in router.upstreams:
        log.info("  %s", upstream.url)
    log.info("listening on http://%s:%d  (nofile=%d)", args.host, args.port, fds)
    log.info("  MODEL_BASE_URL=http://<frps-ip>:<remote-port>/v1  MODEL_ROUTING=session")
    if router.strip_effort:
        log.info("  dropping output_config.effort from POST /v1/messages")
    if router.assign_log:
        log.info("  placements -> %s", router.assign_log)

    run_kwargs: dict[str, object] = {
        "host": args.host,
        "port": args.port,
        # aiohttp's default access logger formats a string per request, and at
        # 48 loops this is the one thing in the hot path that does real work.
        "access_log": None,
        "keepalive_timeout": CLIENT_KEEPALIVE_S,
    }
    # aiohttp >= 3.9. Without it a handler is only cancelled on its next write,
    # which never comes for a non-streaming request — so a client that hung up
    # would leave the engine generating.
    if "handler_cancellation" in inspect.signature(web.run_app).parameters:
        run_kwargs["handler_cancellation"] = True
    web.run_app(build_app(router), **run_kwargs)


if __name__ == "__main__":
    main()
