"""Deriving and sending the close that frees a session's engine.

The URL is derived rather than remembered, so the only thing that can make a
close miss is the derivation disagreeing with what the router stored. That is
what most of this checks; :mod:`tests.test_session_router` checks the other
side of the same agreement.
"""

import asyncio
import logging

import pytest

from orchard_evalkit.config import session_url
from orchard_evalkit.router import (
    CLOSE_SEGMENT,
    CONTROL_URL_ENV,
    close_session,
    close_url,
)

log = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _no_inherited_control_url(monkeypatch):
    """Nothing here derives a URL from the ambient environment by accident.

    The override is read per call, so a shell that exports it — which is what
    running the suite on the GPU host looks like — would otherwise rewrite the
    host in every assertion below.
    """
    monkeypatch.delenv(CONTROL_URL_ENV, raising=False)


class TestCloseUrl:
    def test_a_session_endpoint_becomes_a_close(self):
        assert (
            close_url("http://h:30021/session/django__django-11095/v1")
            == "http://h:30021/router/session/django__django-11095"
        )

    def test_the_id_is_left_percent_encoded(self):
        # Decoding and re-encoding would be two more chances for a task named
        # `org/task` to close an id nobody stored.
        assert close_url(session_url("http://h:30021/v1", "org/task")) == (
            f"http://h:30021{CLOSE_SEGMENT}org%2Ftask"
        )

    def test_the_whole_round_trip_agrees(self):
        for key in ("simple", "org/task", "a b", "astropy__astropy-12907", "100%"):
            endpoint = session_url("http://h:30021/v1", key)
            assert close_url(endpoint).startswith(f"http://h:30021{CLOSE_SEGMENT}")
            # What the router would have stored, split from the proxied path.
            stored = endpoint.split("/session/", 1)[1].rsplit("/v1", 1)[0]
            assert close_url(endpoint).endswith(stored)

    def test_an_unrouted_endpoint_has_nothing_to_close(self):
        # What makes calling this unconditionally safe: a single sglang, a
        # sticky fleet and a hosted API all say so here rather than at the
        # call site.
        assert close_url("http://h:30021/v1") is None
        assert close_url("https://api.openai.com/v1") is None
        assert close_url(None) is None
        assert close_url("") is None

    def test_a_relative_url_is_not_a_close(self):
        assert close_url("/session/a/v1") is None

    def test_an_empty_id_is_not_a_close(self):
        assert close_url("http://h:30021/session//v1") is None


class TestControlUrl:
    """The close goes to the orchestrator's address for the router, not the
    sandboxes'. Those differ whenever the published tunnel does not answer from
    inside the cluster, which is the case this exists for.
    """

    def test_the_override_replaces_the_host(self, monkeypatch):
        monkeypatch.setenv(CONTROL_URL_ENV, "http://127.0.0.1:8100")
        assert (
            close_url("http://203.0.113.10:30021/session/astropy__astropy-12907/v1")
            == "http://127.0.0.1:8100/router/session/astropy__astropy-12907"
        )

    def test_a_bare_host_and_port_is_enough(self, monkeypatch):
        # What someone types when they are not thinking about schemes.
        for raw in ("127.0.0.1:8100", "http://127.0.0.1:8100", " 127.0.0.1:8100 "):
            monkeypatch.setenv(CONTROL_URL_ENV, raw)
            assert close_url("http://h:30021/session/a/v1") == (
                "http://127.0.0.1:8100/router/session/a"
            )

    def test_the_id_survives_the_redirect(self, monkeypatch):
        monkeypatch.setenv(CONTROL_URL_ENV, "http://127.0.0.1:8100")
        endpoint = session_url("http://h:30021/v1", "org/task")
        assert close_url(endpoint) == f"http://127.0.0.1:8100{CLOSE_SEGMENT}org%2Ftask"

    def test_it_cannot_invent_a_session_to_close(self, monkeypatch):
        # Pointing the close somewhere reachable says nothing about whether
        # there is a session at the other end. A sticky fleet still has none.
        monkeypatch.setenv(CONTROL_URL_ENV, "http://127.0.0.1:8100")
        assert close_url("http://h:30021/v1") is None
        assert close_url("https://api.openai.com/v1") is None

    def test_unset_or_unusable_leaves_the_endpoint_alone(self, monkeypatch):
        # A typo here has to cost the balance the close improves, not a trial
        # whose result is already decided.
        for raw in ("", "   ", "://", "http://"):
            monkeypatch.setenv(CONTROL_URL_ENV, raw)
            assert close_url("http://h:30021/session/a/v1") == (
                "http://h:30021/router/session/a"
            )
        monkeypatch.delenv(CONTROL_URL_ENV, raising=False)
        assert close_url("http://h:30021/session/a/v1") == (
            "http://h:30021/router/session/a"
        )

    def test_it_is_read_per_call(self, monkeypatch):
        # Exported by run_all_evals.sh, which may well be read after this
        # module is imported.
        monkeypatch.delenv(CONTROL_URL_ENV, raising=False)
        assert close_url("http://h:30021/session/a/v1").startswith("http://h:30021")
        monkeypatch.setenv(CONTROL_URL_ENV, "http://127.0.0.1:8100")
        assert close_url("http://h:30021/session/a/v1").startswith("http://127.0.0.1")


class TestCloseSession:
    def test_an_unreachable_router_is_not_an_error(self):
        # This runs on the teardown path of a trial whose result is already
        # decided. Raising here would turn a balance optimisation into a way
        # to lose a rollout that had already finished.
        endpoint = "http://127.0.0.1:1/session/a/v1"
        assert asyncio.run(close_session(endpoint, logger=log, timeout=1.0)) is False

    def test_nothing_is_sent_for_an_unrouted_endpoint(self):
        assert asyncio.run(close_session("http://h:30021/v1", logger=log)) is False
