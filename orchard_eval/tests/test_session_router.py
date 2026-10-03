"""Placement and path handling for scripts/session_router.py.

The router is a standalone script rather than a package module, because it runs
in the sglang venv on the GPU host where orchard_evalkit is not installed. So
it is loaded here by path.

These are unit tests on the session table, which is where the interesting
decisions are, plus the one part of the relay that does not pass its bytes
through untouched. The rest of the relay is exercised against a live fleet by
the curl sequence in the README.
"""

import asyncio
import importlib.util
import json
import logging
import sys
from pathlib import Path

import pytest

pytest.importorskip("aiohttp", reason="the router runs in the sglang venv")

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "session_router.py"
_spec = importlib.util.spec_from_file_location("session_router", _PATH)
session_router = importlib.util.module_from_spec(_spec)
# Registered before exec: @dataclass resolves annotations through
# sys.modules[cls.__module__], which is None for a module loaded by path alone.
sys.modules["session_router"] = session_router
_spec.loader.exec_module(session_router)

Router = session_router.Router
split_session = session_router.split_session
strip_effort = session_router.strip_effort


def fleet(n: int = 8) -> Router:
    return Router([f"http://127.0.0.1:{8000 + i}" for i in range(n)])


class TestSplitSession:
    def test_a_prefixed_path_is_split(self):
        assert split_session("/session/abc/v1/responses") == ("abc", "/v1/responses")

    def test_an_escaped_slash_stays_in_the_id(self):
        # A Harbor task name is `org/task`. Decoding it before the split would
        # move half of it into the upstream path.
        assert split_session("/session/org%2Ftask/v1/models") == (
            "org%2Ftask",
            "/v1/models",
        )

    def test_an_unprefixed_path_is_left_alone(self):
        assert split_session("/v1/models") == ("", "/v1/models")

    def test_an_empty_id_is_not_a_session(self):
        assert split_session("/session//v1/models") == ("", "/session//v1/models")

    def test_a_bare_session_path_still_reaches_the_root(self):
        assert split_session("/session/abc") == ("abc", "/")


class TestPlacement:
    def test_the_first_sessions_land_one_per_engine(self):
        # The whole claim of the design: a hash would have collided by now.
        router = fleet()
        for i in range(8):
            router.session_for(f"task-{i}", now=0.0)
        assert sorted(s.upstream for s in router.sessions.values()) == sorted(
            u.url for u in router.upstreams
        )

    def test_load_stays_even_as_sessions_accumulate(self):
        router = fleet()
        for i in range(48):
            router.session_for(f"task-{i}", now=0.0)
        counts = [
            sum(1 for s in router.sessions.values() if s.upstream == u.url)
            for u in router.upstreams
        ]
        assert counts == [6] * 8

    def test_a_session_keeps_its_engine(self):
        router = fleet()
        first = router.session_for("astropy__astropy-12907", now=0.0).upstream
        for turn in range(1, 40):
            assert (
                router.session_for("astropy__astropy-12907", now=float(turn)).upstream
                == first
            )

    def test_a_finished_session_stops_occupying_its_engine(self):
        # Load is counted over live sessions, so an idle one must free its slot
        # or the fleet fills up and placement stops meaning anything.
        router = fleet(2)
        session = router.session_for("old", now=0.0)
        assert session.live(now=60.0, ttl=router.active_ttl)
        assert not session.live(now=router.active_ttl + 1, ttl=router.active_ttl)

    def test_an_idle_session_frees_its_engine_for_a_burst(self):
        # Three sessions crowded onto one engine, all long finished: the next
        # three must be free to spread rather than be pushed onto the other.
        router = fleet(2)
        router.upstreams[1].healthy = False
        for i in range(3):
            router.session_for(f"old-{i}", now=0.0)
        router.upstreams[1].healthy = True

        later = router.active_ttl + 1
        for i in range(2):
            router.session_for(f"new-{i}", now=later)
        fresh = [router.sessions[f"new-{i}"].upstream for i in range(2)]
        assert set(fresh) == {u.url for u in router.upstreams}

    def test_an_in_flight_session_counts_however_long_it_takes(self):
        # One generation can outlast any reasonable idle window by itself.
        router = fleet(2)
        router.session_for("slow", now=0.0).inflight = 1
        router.session_for("next", now=router.active_ttl * 10)
        assert router.sessions["next"].upstream != router.sessions["slow"].upstream

    def test_an_unhealthy_engine_takes_no_new_sessions(self):
        router = fleet()
        router.upstreams[0].healthy = False
        for i in range(16):
            router.session_for(f"task-{i}", now=0.0)
        assert all(
            s.upstream != router.upstreams[0].url for s in router.sessions.values()
        )

    def test_an_all_unhealthy_fleet_is_still_used(self):
        # Refusing to place is strictly worse than trying an engine that may
        # have recovered since the last probe.
        router = fleet(2)
        for upstream in router.upstreams:
            upstream.healthy = False
        assert router.session_for("task", now=0.0).upstream


class TestHeaderFiltering:
    def test_hop_by_hop_headers_are_dropped(self):
        from multidict import CIMultiDict

        headers = CIMultiDict(
            {
                "Authorization": "Bearer token-abc123",
                "Content-Type": "application/json",
                "Transfer-Encoding": "chunked",
                "Connection": "keep-alive, X-Internal",
                "X-Internal": "drop me",
            }
        )
        drop = session_router._hop_by_hop(headers)
        kept = {k for k in headers if k.lower() not in drop}
        # Authorization survives, which is what keeps sglang's --api-key working.
        assert kept == {"Authorization", "Content-Type"}


class TestClose:
    def test_a_closed_session_frees_its_slot_at_once(self):
        # The point of the close: without it the slot comes back only after
        # ACTIVE_TTL_S, and that wait is what makes a fast engine look busy.
        router = fleet(2)
        router.session_for("done", now=0.0)
        router.close("done", now=1.0)
        assert not router.sessions["done"].live(now=2.0, ttl=router.active_ttl)

    def test_closing_is_idempotent(self):
        router = fleet(2)
        router.session_for("done", now=0.0)
        first = router.close("done", now=1.0)
        again = router.close("done", now=50.0)
        assert again is first
        assert first.closed_at == 1.0
        assert router.closed == 1

    def test_closing_an_unknown_session_is_a_no_op(self):
        # A rollout whose agent never reached the model still tears down.
        router = fleet(2)
        assert router.close("never-ran", now=0.0) is None
        assert router.closed == 0

    def test_a_closed_session_still_streaming_keeps_its_slot(self):
        # Harbor abandons a trial at its timeout while the CLI in the pod goes
        # on generating for another three quarters of an hour. Those tokens are
        # real GPU seconds on this engine whatever the harness thinks.
        router = fleet(2)
        session = router.session_for("orphan", now=0.0)
        session.inflight = 1
        router.close("orphan", now=1.0)
        assert session.live(now=10_000.0, ttl=router.active_ttl)

    def test_a_closed_session_leaves_room_for_the_next_one(self):
        router = fleet(2)
        router.upstreams[1].healthy = False
        for i in range(3):
            router.session_for(f"old-{i}", now=0.0)
            router.close(f"old-{i}", now=1.0)
        router.upstreams[1].healthy = True

        for i in range(2):
            router.session_for(f"new-{i}", now=2.0)
        fresh = {router.sessions[f"new-{i}"].upstream for i in range(2)}
        assert fresh == {u.url for u in router.upstreams}

    def test_a_retry_soon_after_a_close_keeps_its_engine(self):
        # Keying on the task rather than the attempt exists precisely so this
        # lands back on the engine still holding the conversation.
        router = fleet(4)
        first = router.session_for("django__django-11095", now=0.0).upstream
        router.close("django__django-11095", now=10.0)
        back = router.session_for("django__django-11095", now=20.0)
        assert back.upstream == first
        assert back.closed_at == 0.0
        assert router.reopened == 0

    def test_a_session_returning_long_after_its_close_is_placed_again(self):
        # By now the engines have served other rollouts for longer than this
        # one was away, so the pin is no longer a cache hit.
        router = fleet(2)
        stale = router.session_for("slow-retry", now=0.0).upstream
        router.close("slow-retry", now=1.0)
        # Fill the old engine so placement has a reason to move it.
        for i in range(4):
            router.session_for(f"filler-{i}", now=2.0)

        later = 2.0 + router.reopen_warm + 1
        assert router.session_for("slow-retry", now=later).upstream != stale
        assert router.reopened == 1

    def test_reopening_does_not_move_a_session_that_is_still_streaming(self):
        router = fleet(2)
        session = router.session_for("orphan", now=0.0)
        router.close("orphan", now=1.0)
        session.inflight = 1
        pinned = session.upstream
        assert (
            router.session_for("orphan", now=router.reopen_warm + 100).upstream
            == pinned
        )

    def test_the_close_path_splits_ids_the_way_the_proxy_does(self):
        # Both sides have to agree, or a close names an id nobody stored.
        raw = f"{session_router.CLOSE_PREFIX}org%2Ftask"
        assert split_session(raw, session_router.CLOSE_PREFIX)[0] == "org%2Ftask"
        assert split_session("/session/org%2Ftask/v1/responses")[0] == "org%2Ftask"

    def test_a_close_without_an_id_is_not_a_session(self):
        assert split_session(
            session_router.CLOSE_PREFIX, session_router.CLOSE_PREFIX
        ) == ("", session_router.CLOSE_PREFIX)


def sample(router, *, queued, running=0.0, now=0.0):
    """Give every engine a fresh load sample, `queued` per engine in order."""
    for upstream, depth in zip(router.upstreams, queued):
        upstream.queued = float(depth)
        upstream.running = float(running)
        upstream.load_at = now or 1e-9  # 0.0 means "never sampled"


class TestLoadTiebreak:
    def test_an_equal_session_count_is_broken_by_the_shallower_queue(self):
        # Six rollouts over eight engines is the steady state of a run at
        # concurrency 48, so this tiebreak is where most placements are decided.
        router = fleet(2)
        for i in range(2):
            router.session_for(f"seed-{i}", now=0.0)
        sample(router, queued=[5, 0], now=1.0)
        assert router.session_for("next", now=2.0).upstream == router.upstreams[1].url

    def test_a_stale_sample_disables_the_tiebreak_for_everyone(self):
        # Scoring one engine on a real queue and another on a sample that
        # stopped arriving would rank the engine that went quiet as the
        # emptiest in the fleet.
        router = fleet(2)
        sample(router, queued=[5, 0], now=1.0)
        stale = 1.0 + router.health_interval * session_router.LOAD_STALE_FACTOR + 1
        assert not router._load_is_usable(router.upstreams, now=stale)

    def test_an_engine_that_has_never_answered_disables_it_too(self):
        router = fleet(2)
        router.upstreams[0].queued = 5.0
        router.upstreams[0].load_at = 1.0
        assert not router._load_is_usable(router.upstreams, now=2.0)

    def test_probing_can_be_turned_off_entirely(self):
        router = fleet(2)
        router.load_probe = False
        sample(router, queued=[5, 0], now=1.0)
        assert not router._load_is_usable(router.upstreams, now=2.0)

    def test_a_burst_does_not_stampede_onto_one_stale_sample(self):
        # Rollouts that fail in seconds place, close, and place again between
        # two samples. Counting what has been sent since the sample is what
        # stops all of them reading the same number and going to one engine.
        router = fleet(2)
        sample(router, queued=[0, 3], now=1.0)
        for i in range(6):
            router.session_for(f"quick-{i}", now=2.0)
            router.close(f"quick-{i}", now=2.0)
        used = {s.upstream for s in router.sessions.values()}
        assert used == {u.url for u in router.upstreams}

    def test_a_fresh_sample_forgets_what_was_sent_since_the_last_one(self):
        router = fleet(2)
        router.upstreams[0].since_sample = 9
        router.upstreams[0].running = 2.0
        assert router._load_score(router.upstreams[0]) == 11.0


class TestEvacuation:
    def _failed(self, n=2):
        router = fleet(n)
        for i in range(4):
            router.session_for(f"task-{i}", now=0.0)
        return router

    def test_idle_sessions_leave_a_failed_engine(self):
        # The usual argument against moving a session is the warm prefix cache
        # it discards, and that does not survive the engine having gone away.
        router = self._failed()
        dead = router.upstreams[0]
        dead.healthy = False
        moved = router._evacuate(dead, now=100.0)
        assert moved == 2
        assert all(s.upstream != dead.url for s in router.sessions.values())
        assert router.evacuated == 2

    def test_a_streaming_session_is_left_where_it_is(self):
        # Its bytes are on a socket this router cannot reclaim.
        router = self._failed()
        dead = router.upstreams[0]
        dead.healthy = False
        streaming = next(s for s in router.sessions.values() if s.upstream == dead.url)
        streaming.inflight = 1
        router._evacuate(dead, now=100.0)
        assert streaming.upstream == dead.url

    def test_a_closed_session_is_not_worth_moving(self):
        router = self._failed()
        dead = router.upstreams[0]
        for session in list(router.sessions.values()):
            if session.upstream == dead.url:
                router.close(session.sid, now=1.0)
        dead.healthy = False
        assert router._evacuate(dead, now=100.0) == 0

    def test_an_expired_session_is_not_worth_moving(self):
        router = self._failed()
        dead = router.upstreams[0]
        dead.healthy = False
        assert router._evacuate(dead, now=router.active_ttl + 100) == 0

    def test_nothing_moves_when_no_healthy_engine_is_left(self):
        # A fleet-wide stall must not become a fleet-wide shuffle.
        router = self._failed()
        for upstream in router.upstreams:
            upstream.healthy = False
        assert router._evacuate(router.upstreams[0], now=100.0) == 0
        assert router.evacuated == 0


class _Reply:
    """One canned HTTP response, usable as aiohttp uses one."""

    def __init__(self, status: int, payload: bytes = b"") -> None:
        self.status = status
        self._payload = payload

    async def read(self) -> bytes:
        return self._payload

    async def __aenter__(self) -> "_Reply":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False


class _Engine:
    """A fleet member that answers each route with a fixed status."""

    def __init__(self, **by_route: tuple[int, bytes]) -> None:
        self.by_route = {f"/{k}": v for k, v in by_route.items()}

    def get(self, url: str, **_: object) -> _Reply:
        for route, reply in self.by_route.items():
            if url.endswith(route):
                return _Reply(*reply)
        return _Reply(404)


class TestRejectedLoadProbe:
    """sglang guards /get_server_info behind --api-key and lets /v1/models
    answer 401 rather than 5xx, so a router holding the wrong key sees a fleet
    that is unanimously healthy and unanimously silent about its load. That
    combination ran a whole DeepSWE sweep with the tiebreak off and nothing in
    the log to say so.
    """

    def _probe(self, router, upstream, times=1):
        for _ in range(times):
            asyncio.run(router._sample_load(upstream, {}))

    def test_a_401_is_reported_once(self, caplog):
        router = fleet(1)
        router.client = _Engine(get_server_info=(401, b""), metrics=(401, b""))
        upstream = router.upstreams[0]
        with caplog.at_level(logging.WARNING, logger="session-router"):
            self._probe(router, upstream, times=3)
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        # The message has to name the fix, or it is one more line to grep past.
        assert "--api-key" in warnings[0].getMessage()
        assert upstream.load_denied
        assert upstream.load_at == 0.0

    def test_a_403_counts_too(self, caplog):
        router = fleet(1)
        router.client = _Engine(get_server_info=(403, b""), metrics=(403, b""))
        with caplog.at_level(logging.WARNING, logger="session-router"):
            self._probe(router, router.upstreams[0])
        assert [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_a_fleet_offering_neither_route_stays_quiet(self, caplog):
        # Supported configuration, not a fault: an older sglang, or one started
        # without --enable-metrics. Placing on session count alone is what it
        # did before either route was asked for.
        router = fleet(1)
        router.client = _Engine(get_server_info=(404, b""), metrics=(404, b""))
        with caplog.at_level(logging.WARNING, logger="session-router"):
            self._probe(router, router.upstreams[0], times=3)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not router.upstreams[0].load_denied

    def test_one_working_route_is_enough_to_stay_quiet(self, caplog):
        router = fleet(1)
        router.client = _Engine(
            get_server_info=(401, b""),
            metrics=(200, b"sglang:num_running_reqs 3\nsglang:num_queue_reqs 1\n"),
        )
        upstream = router.upstreams[0]
        with caplog.at_level(logging.WARNING, logger="session-router"):
            self._probe(router, upstream)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert upstream.load_source == "/metrics"
        assert upstream.running == 3.0

    def test_a_key_that_starts_working_clears_the_flag(self):
        router = fleet(1)
        upstream = router.upstreams[0]
        router.client = _Engine(get_server_info=(401, b""), metrics=(401, b""))
        self._probe(router, upstream)
        assert upstream.load_denied
        router.client = _Engine(metrics=(200, b"sglang:num_queue_reqs 2\n"))
        self._probe(router, upstream)
        assert not upstream.load_denied
        assert upstream.queued == 2.0


class TestLoadParsing:
    def test_internal_states_is_read(self):
        payload = (
            b'{"model_path": "/m", "internal_states": '
            b'[{"num_running_reqs": 3, "num_queue_reqs": 2, "token_usage": 0.5}]}'
        )
        assert session_router.parse_server_info(payload) == (3.0, 2.0, 0.5)

    def test_the_older_top_level_spelling_is_read(self):
        payload = b'{"num_running_reqs": 1, "num_queued_reqs": 7}'
        assert session_router.parse_server_info(payload) == (1.0, 7.0, 0.0)

    def test_data_parallel_ranks_add_up(self):
        payload = (
            b'{"internal_states": [{"num_running_reqs": 2, "token_usage": 0.1}, '
            b'{"num_running_reqs": 3, "token_usage": 0.4}]}'
        )
        # Counts add; a fraction of the KV pool does not.
        assert session_router.parse_server_info(payload) == (5.0, 0.0, 0.4)

    def test_a_release_that_renames_everything_yields_nothing(self):
        # Not a zero: a zero reads as an idle engine and would attract the
        # whole fleet's next placements.
        assert session_router.parse_server_info(b'{"model_path": "/m"}') is None

    def test_a_404_body_is_not_a_sample(self):
        assert session_router.parse_server_info(b"<html>404</html>") is None
        assert session_router.parse_metrics(b"<html>404</html>") is None

    def test_prometheus_series_are_read(self):
        payload = (
            b"# HELP sglang:num_running_reqs running\n"
            b'sglang:num_running_reqs{model_name="m"} 4.0\n'
            b'sglang:num_queue_reqs{model_name="m"} 1.0\n'
            b'sglang:token_usage{model_name="m"} 0.25\n'
            b"python_gc_objects_collected_total 100.0\n"
        )
        assert session_router.parse_metrics(payload) == (4.0, 1.0, 0.25)

    def test_a_sample_that_is_not_a_number_is_skipped(self):
        # sglang publishes NaN for a gauge it has not computed yet, and an
        # unordered value in a sort key makes placement arbitrary.
        payload = b"sglang:num_running_reqs 1.0\nsglang:token_usage NaN\n"
        assert session_router.parse_metrics(payload) == (1.0, 0.0, 0.0)

    def test_a_trailing_timestamp_does_not_break_a_line(self):
        payload = b"sglang:num_running_reqs 2.0 1700000000000\n"
        assert session_router.parse_metrics(payload) == (2.0, 0.0, 0.0)

    def test_the_route_selects_the_parser(self):
        assert session_router.parse_load("/metrics", b"sglang:num_queue_reqs 3\n") == (
            0.0,
            3.0,
            0.0,
        )
        assert session_router.parse_load(
            "/get_server_info", b'{"num_queue_reqs": 3}'
        ) == (0.0, 3.0, 0.0)


class TestCloseOverHttp:
    """The close arrives over HTTP, so the route is exercised over HTTP.

    Which aiohttp versions decode ``%2F`` before matching has changed more than
    once. An id that survives the proxy's split but not the close route's would
    leave a slot occupied for the rest of the run with nothing in any log to
    say so, so the agreement is checked against the real dispatcher rather than
    against :func:`split_session` alone.
    """

    @staticmethod
    def _delete(router, path: str):
        from aiohttp.test_utils import TestClient, TestServer
        from yarl import URL

        async def go():
            client = TestClient(TestServer(session_router.build_app(router)))
            await client.start_server()
            try:
                # encoded=True, or yarl re-quotes the %2F this is here to test.
                url = URL(f"{client.make_url('/')}".rstrip("/") + path, encoded=True)
                response = await client.session.delete(url)
                return response.status, await response.json()
            finally:
                await client.close()

        return asyncio.run(go())

    def test_an_escaped_slash_closes_the_session_the_proxy_made(self):
        from orchard_evalkit.config import session_url
        from orchard_evalkit.router import close_url

        router = fleet(2)
        # Exactly what the suite hands the agent, and what it derives from it.
        endpoint = session_url("http://h:30021/v1", "org/task")
        sid, _ = split_session("/session/org%2Ftask/v1/responses")
        router.session_for(sid, now=0.0)

        path = close_url(endpoint).split("30021", 1)[1]
        status, body = self._delete(router, path)
        assert status == 200
        assert body["known"] and body["closed"]
        assert body["sid"] == "org/task"
        assert router.sessions[sid].closed_at

    def test_closing_an_unknown_id_is_a_200(self):
        # A rollout whose agent never reached the model still tears down, and a
        # 404 in the harness log would read as a fault.
        status, body = self._delete(fleet(2), "/router/session/never-ran")
        assert status == 200
        assert body == {"sid": "never-ran", "known": False, "closed": False}

    def test_a_close_naming_no_session_is_refused(self):
        status, body = self._delete(fleet(2), "/router/session/")
        assert status == 400
        assert "router/session" in body["error"]["message"]

    def test_the_reserved_routes_still_answer(self):
        from aiohttp.test_utils import TestClient, TestServer

        async def go():
            router = fleet(2)
            client = TestClient(TestServer(session_router.build_app(router)))
            await client.start_server()
            try:
                stats = await (await client.get("/router/stats")).json()
                health = await (await client.get("/router/healthz")).json()
                return stats, health
            finally:
                await client.close()

        stats, health = asyncio.run(go())
        assert health == {"ok": True, "healthy_upstreams": 2}
        assert stats["sessions"]["closed"] == 0
        # Never sampled, so the tiebreak must be off rather than scoring zero.
        assert stats["load_in_use"] is False
        assert stats["upstreams"][0]["load"]["source"] is None


class TestStripEffort:
    """The rewrite itself, on bodies rather than on requests."""

    def test_the_effort_is_removed(self):
        raw = b'{"model": "m", "output_config": {"effort": "medium"}}'
        assert json.loads(strip_effort(raw)) == {"model": "m"}

    def test_a_sibling_field_keeps_its_output_config(self):
        raw = b'{"output_config": {"effort": "high", "max_tokens": 64}}'
        assert json.loads(strip_effort(raw)) == {"output_config": {"max_tokens": 64}}

    def test_a_body_without_an_effort_is_returned_unchanged(self):
        # Identity, not equality: returning the same object is what tells
        # _relay that Content-Length still holds and nothing was stripped.
        raw = b'{"model": "m", "messages": []}'
        assert strip_effort(raw) is raw

    @pytest.mark.parametrize(
        "raw",
        [
            b"",
            b"not json at all",
            b"[1, 2, 3]",
            b'{"output_config": "medium"}',
            b'{"effort": "high"}',
            b"\xff\xfe\x00 gzip, maybe",
        ],
        ids=["empty", "garbage", "array", "scalar-config", "top-level", "binary"],
    )
    def test_anything_unexpected_is_passed_through(self, raw):
        # A proxy that rewrites what it does not understand is worse than one
        # that rewrites nothing.
        assert strip_effort(raw) is raw


class TestEffortOverHttp:
    """What actually reaches the engine.

    Claude Code puts an effort level on every request and offers no way not to,
    and this fleet's chat template accepts three levels of which the CLI's
    default is not one. Dropping the field is what puts a claude-code rollout on
    the template's own xhigh default — so it is checked on the bytes an engine
    receives, through the real dispatcher, rather than on the helper alone.
    """

    @staticmethod
    def _post(payload, *, path: str = "/session/task/v1/messages", strip: bool = True):
        """Relay *payload* to an engine that records what it was handed."""
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        seen: dict[str, object] = {}

        async def echo(request: web.Request) -> web.Response:
            seen["path"] = request.path
            seen["body"] = await request.read()
            seen["length"] = request.headers.get("Content-Length")
            return web.json_response({"ok": True})

        async def go():
            engine = web.Application()
            engine.router.add_route("*", "/{tail:.*}", echo)
            engine_server = TestServer(engine)
            await engine_server.start_server()
            router = Router(
                [str(engine_server.make_url("/")).rstrip("/")],
                # No background probing of a stand-in that serves no stats.
                health_interval=0,
                spread_interval=0,
                strip_effort=strip,
            )
            client = TestClient(TestServer(session_router.build_app(router)))
            await client.start_server()
            try:
                response = await client.post(path, data=payload)
                await response.read()
                seen["status"] = response.status
                return router, seen
            finally:
                await client.close()
                await engine_server.close()

        return asyncio.run(go())

    def test_the_engine_is_handed_a_body_with_no_effort(self):
        router, seen = self._post(b'{"model": "m", "output_config": {"effort": "high"}}')
        assert seen["status"] == 200
        assert json.loads(seen["body"]) == {"model": "m"}
        # The rewrite shortened the body; a stale length would hang the relay
        # or truncate the prompt, and neither would name this line.
        assert seen["length"] == str(len(seen["body"]))
        assert router.effort_stripped == 1

    def test_a_chunked_upload_arrives_with_a_length(self):
        # A body sent without a Content-Length is the case the header line
        # exists for: there is nothing to correct, only something to add.
        async def chunks():
            yield b'{"output_config": '
            yield b'{"effort": "high"}, "model": "m"}'

        _, seen = self._post(chunks())
        assert json.loads(seen["body"]) == {"model": "m"}
        assert seen["length"] == str(len(seen["body"]))

    def test_the_openai_routes_are_never_touched(self):
        # Narrowness is the point: codex and mini-swe-agent bodies must reach
        # the engine byte for byte, and streamed rather than buffered.
        raw = b'{"model": "m", "output_config": {"effort": "high"}}'
        router, seen = self._post(raw, path="/session/task/v1/chat/completions")
        assert seen["body"] == raw
        assert router.effort_stripped == 0

    def test_no_strip_effort_relays_the_body_verbatim(self):
        raw = b'{"model": "m", "output_config": {"effort": "high"}}'
        router, seen = self._post(raw, strip=False)
        assert seen["body"] == raw
        assert router.effort_stripped == 0

    def test_a_stripped_request_still_reaches_its_own_engine(self):
        # Buffering happens before placement is used, so the path the engine
        # sees must be the untouched suffix, not a rebuilt one.
        _, seen = self._post(b'{"output_config": {"effort": "high"}}')
        assert seen["path"] == "/v1/messages"

    def test_the_count_is_reported(self):
        from aiohttp.test_utils import TestClient, TestServer

        async def go():
            router = fleet(1)
            router.effort_stripped = 7
            client = TestClient(TestServer(session_router.build_app(router)))
            await client.start_server()
            try:
                return await (await client.get("/router/stats")).json()
            finally:
                await client.close()

        assert asyncio.run(go())["requests"]["effort_stripped"] == 7
