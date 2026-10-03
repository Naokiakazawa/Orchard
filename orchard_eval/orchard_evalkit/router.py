"""Telling the session router that a trial is over.

``scripts/session_router.py`` places each new session on whichever engine is
carrying the least work, and "least work" is counted over the sessions those
engines still hold. It cannot see a rollout end by itself: the last request of
a forty-turn conversation is indistinguishable from the pause before turn
forty-one, so without this the router waits out an idle window — long, because
it has to cover an agent running a test suite between turns — before the slot
comes back.

That wait is not merely late, it is biased. Phantom load accumulates in
proportion to how fast an engine retires rollouts, so the engine that just
finished six short ones looks like the busiest in the fleet and is given
nothing, while the one still grinding through a single long rollout looks free.

One ``DELETE`` at teardown removes the guess. Everything here is best effort in
both directions: a router that is down, slow, or simply not in use must not
disturb a trial whose result is already decided, and a close that never arrives
costs only the balance it exists to improve.

Duplicated as ``harbor_orchard/router.py`` for the reason
:func:`config.session_url` is: Harbor imports that package in its own process,
where this one is not necessarily installed. Keep the two in step.
"""

from __future__ import annotations

import asyncio
import logging
import os
import urllib.error
import urllib.request
from urllib.parse import urlsplit, urlunsplit

from orchard_evalkit.config import SESSION_SEGMENT

#: Where the router listens for it. Matches ``CLOSE_PREFIX`` in
#: ``scripts/session_router.py``.
CLOSE_SEGMENT = "/router/session/"

#: Host to send the close to, when it is not the one in the model endpoint.
#: Those two addresses are written for different callers: ``MODEL_BASE_URL``
#: has to be reachable from inside a sandbox, so it names the frp tunnel, while
#: this call is made by the harness host, which on a single-host setup is the
#: same machine the router runs on. A tunnel that the pods can reach and this
#: process cannot is not an exotic failure — it cost a whole DeepSWE run every
#: one of its closes, leaving the router to reap sessions on the idle window
#: and place new ones on counts that only ever grew.
#:
#: Accepts ``http://127.0.0.1:8100`` or a bare ``127.0.0.1:8100``. Only the
#: scheme and authority are used; unset leaves the close where the endpoint
#: points, which is what it did before this existed.
CONTROL_URL_ENV = "ORCHARD_ROUTER_CONTROL_URL"

#: Seconds one close may take. Short on purpose: it runs on the teardown path
#: of a trial that has already produced its result, so a router which has
#: stopped answering costs a moment rather than a phase.
CLOSE_TIMEOUT_S = 5.0

#: Proxies are for reaching the internet, and the router is reached through an
#: frp tunnel to the GPU host. Plain ``urlopen`` would honour an ``HTTP_PROXY``
#: that happens to be set on the harness host and send the close elsewhere.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def control_origin() -> tuple[str, str] | None:
    """``(scheme, netloc)`` from :data:`CONTROL_URL_ENV`, or ``None``.

    Read per call rather than at import, so a value exported after this module
    is loaded still takes effect and a test can set one without reloading.
    Anything unparseable is ``None``: a typo here must cost the balance the
    close exists to improve, not the trial.
    """
    raw = os.environ.get(CONTROL_URL_ENV, "").strip()
    if not raw:
        return None
    parts = urlsplit(raw if "://" in raw else f"http://{raw}")
    if not parts.scheme or not parts.netloc:
        return None
    return parts.scheme, parts.netloc


def close_url(endpoint: str | None) -> str | None:
    """``http://h:30021/session/a/v1`` -> ``http://h:30021/router/session/a``.

    ``None`` for anything that is not a session-routed endpoint, which is what
    makes calling this unconditionally safe: a run against one sglang, a sticky
    fleet, or a hosted API has nothing to close and says so here instead of at
    every call site.

    The id is copied out of the path still percent-encoded, exactly as the
    router stored it. Decoding and re-encoding would be two more chances for a
    task named ``org/task`` to close an id nobody has.

    :data:`CONTROL_URL_ENV` replaces the host the close is sent to, and only
    that: an endpoint with no session in it still has nothing to close, so an
    override cannot invent one.
    """
    if not endpoint:
        return None
    parts = urlsplit(endpoint)
    if not parts.scheme or not parts.netloc:
        return None
    if not parts.path.startswith(SESSION_SEGMENT):
        return None
    sid = parts.path[len(SESSION_SEGMENT) :].partition("/")[0]
    if not sid:
        return None
    scheme, netloc = control_origin() or (parts.scheme, parts.netloc)
    return urlunsplit((scheme, netloc, f"{CLOSE_SEGMENT}{sid}", "", ""))


async def close_session(
    endpoint: str | None,
    *,
    logger: logging.Logger,
    timeout: float = CLOSE_TIMEOUT_S,
) -> bool:
    """Best-effort ``DELETE`` of *endpoint*'s session. Never raises.

    A blocking ``urlopen`` on a worker thread rather than an async client,
    because this package depends on neither aiohttp nor httpx directly and a
    new dependency to make one five-second request at teardown is a bad trade.
    """
    url = close_url(endpoint)
    if url is None:
        return False
    try:
        # A deadline as well as the socket timeout: urlopen's timeout covers
        # each socket operation rather than the call, so a router answering one
        # byte at a time would outlast it. Abandoning the wait leaves the
        # thread running, which is why the socket timeout has to exist too —
        # it is what actually ends the thread.
        await asyncio.wait_for(asyncio.to_thread(_delete, url, timeout), timeout + 5.0)
    except Exception as exc:  # noqa: BLE001 - teardown is best effort
        logger.debug("could not close %s at the session router: %s", url, exc)
        return False
    logger.debug("closed %s at the session router", url)
    return True


def _delete(url: str, timeout: float) -> None:
    request = urllib.request.Request(url, method="DELETE")
    with _OPENER.open(request, timeout=timeout) as response:
        response.read()
