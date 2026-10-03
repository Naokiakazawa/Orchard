"""Turn the hosts an isolated pod still needs into Kubernetes egress rules.

A task that declares ``network_mode = "no-network"`` wants the *internet* gone,
not every socket. An agent CLI running inside the pod still has to reach the
model server, and DeepSWE is the case that forces the distinction: all 113 of
its tasks are air-gapped, and an in-pod agent with no route to inference fails
every one of them on turn 1.

Kubernetes NetworkPolicy allows destinations by CIDR, not by name, so a
hostname has to be resolved here — in the Harbor process, which has DNS — and
handed to the orchestrator as addresses. Resolution happens once per
environment, at start, because a policy is a fact about the pod rather than
about any one request.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterable, Sequence
from urllib.parse import urlsplit, urlunsplit

#: Ports opened for a destination that does not name one of its own. A model
#: endpoint is reached over plain HTTP or TLS, and nothing else in an isolated
#: pod has anywhere to go.
DEFAULT_PORTS: tuple[int, ...] = (80, 443)


def host_of(value: str) -> str | None:
    """The host part of a URL, ``host:port`` pair, or bare hostname."""
    raw = (value or "").strip()
    if not raw:
        return None
    if "://" not in raw:
        # urlsplit reads a bare `host:port` as scheme `host`, so give it one.
        raw = f"//{raw}"
    try:
        hostname = urlsplit(raw).hostname
    except ValueError:
        return None
    return hostname.rstrip(".").lower() if hostname else None


def _cidr_of_address(address: str) -> str | None:
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return None
    return f"{parsed}/{parsed.max_prefixlen}"


def cidrs_for(
    values: Iterable[str], *, resolve=socket.getaddrinfo
) -> tuple[list[str], list[str]]:
    """Resolve *values* to CIDRs, and say which ones could not be resolved.

    Accepts URLs, ``host:port`` pairs, bare hostnames, literal addresses, and
    CIDRs — a CIDR is passed through untouched, which is the escape hatch for a
    fleet behind a load balancer whose address set is wider than DNS admits.

    Returns ``(cidrs, unresolved)``. Failures are returned rather than raised:
    one unreachable name should be reported and skipped, not turn a whole sweep
    into a configuration error.
    """
    cidrs: list[str] = []
    unresolved: list[str] = []
    seen: set[str] = set()

    def add(cidr: str) -> None:
        if cidr not in seen:
            seen.add(cidr)
            cidrs.append(cidr)

    for value in values:
        raw = (value or "").strip()
        if not raw:
            continue

        if "/" in raw and "://" not in raw:
            try:
                add(str(ipaddress.ip_network(raw, strict=False)))
                continue
            except ValueError:
                pass  # not a CIDR after all; fall through to hostname handling

        host = host_of(raw)
        if not host:
            unresolved.append(raw)
            continue

        literal = _cidr_of_address(host)
        if literal:
            add(literal)
            continue

        # A leading-dot suffix ('.openai.com') names a set of hosts, which has
        # no CIDR: skip it loudly rather than allowing something narrower and
        # pretending the allowlist was honoured.
        if host.startswith("."):
            unresolved.append(raw)
            continue

        try:
            infos = resolve(host, None, type=socket.SOCK_STREAM)
        except OSError:
            unresolved.append(raw)
            continue

        resolved_any = False
        for info in infos:
            address = info[4][0]
            cidr = _cidr_of_address(address)
            if cidr:
                add(cidr)
                resolved_any = True
        if not resolved_any:
            unresolved.append(raw)

    return cidrs, unresolved


def port_of(value: str) -> int | None:
    """The port *value* names, or the one its scheme implies.

    Returns ``None`` when neither is present — a bare hostname or a CIDR says
    nothing about ports, and guessing one would silently narrow the allowlist.
    """
    raw = (value or "").strip()
    if not raw:
        return None
    scheme_given = "://" in raw
    try:
        parsed = urlsplit(raw if scheme_given else f"//{raw}")
        port = parsed.port
    except ValueError:
        return None
    if port is not None:
        return port
    if scheme_given:
        return 443 if parsed.scheme == "https" else 80
    return None


def allow_rules_for(
    values: Iterable[str],
    *,
    default_ports: Sequence[int] = DEFAULT_PORTS,
    protocol: str = "TCP",
    resolve=socket.getaddrinfo,
) -> tuple[list[dict], list[str]]:
    """Resolve *values* into orchestrator egress allowlist rules.

    Each rule is ``{cidr, protocol, port_start, port_end}``. The orchestrator
    normalizes a bare address to ``/32`` and upper-cases the protocol itself, so
    neither is done here — doing it twice is how the two drift apart.

    A destination that names a port (``host:30000``, ``http://host:8000/v1``)
    opens only that one. Everything else opens *default_ports*, because a CIDR
    carries no port and the alternative — opening all of them — gives back most
    of what isolating the pod was for.

    Returns ``(rules, unresolved)``, mirroring :func:`cidrs_for`.
    """
    rules: list[dict] = []
    unresolved: list[str] = []
    seen: set[tuple[str, str, int]] = set()

    for value in values:
        raw = (value or "").strip()
        if not raw:
            continue
        cidrs, missing = cidrs_for([raw], resolve=resolve)
        unresolved.extend(missing)
        named = port_of(raw)
        ports = (named,) if named is not None else tuple(default_ports)
        for cidr in cidrs:
            for port in ports:
                key = (cidr, protocol, port)
                if key in seen:
                    continue
                seen.add(key)
                rules.append(
                    {
                        "cidr": cidr,
                        "protocol": protocol,
                        "port_start": port,
                        "port_end": port,
                    }
                )

    return rules, unresolved


def cidrs_of_rules(rules: Iterable[dict]) -> list[str]:
    """The distinct CIDRs in *rules*, order preserved.

    For the older orchestrator API, which allowed a destination outright rather
    than per port.
    """
    cidrs: list[str] = []
    for rule in rules:
        cidr = rule.get("cidr")
        if cidr and cidr not in cidrs:
            cidrs.append(cidr)
    return cidrs


def with_resolved_host(url: str, *, resolve=socket.getaddrinfo) -> str:
    """*url* with its hostname replaced by a literal address.

    An isolated pod cannot resolve names: the orchestrator's allowlist is
    expressed in CIDRs and carries no DNS exemption, so a pod that is handed a
    hostname fails on the first lookup and the cause reads as a model error.
    Handing it an address instead removes the need for DNS entirely.

    Returns *url* unchanged when there is nothing to do — no scheme, no
    hostname, already a literal, or unresolvable. Note this pins a load-balanced
    name to one backend; that is why it is behind a setting.
    """
    raw = (url or "").strip()
    if "://" not in raw:
        return url
    try:
        parsed = urlsplit(raw)
        host, port = parsed.hostname, parsed.port
    except ValueError:
        return url
    if not host or _cidr_of_address(host):
        return url

    try:
        infos = resolve(host, None, type=socket.SOCK_STREAM)
    except OSError:
        return url
    address = next((i[4][0] for i in infos if _cidr_of_address(i[4][0])), None)
    if address is None:
        return url

    netloc = f"[{address}]" if ":" in address else address
    if port is not None:
        netloc = f"{netloc}:{port}"
    if parsed.username:
        credentials = parsed.username
        if parsed.password:
            credentials = f"{credentials}:{parsed.password}"
        netloc = f"{credentials}@{netloc}"
    return urlunsplit(
        (parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)
    )
