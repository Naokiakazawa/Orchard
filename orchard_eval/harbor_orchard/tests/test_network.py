"""The allowlist that keeps an air-gapped pod useful.

DeepSWE declares every task ``network_mode = "no-network"``, so these are the
rules that decide whether an in-pod agent can reach inference at all. Getting
them wrong does not fail loudly: the agent starts, cannot connect, and reports
a non-zero exit that reads as a model failure across all 113 tasks.
"""

from __future__ import annotations

import socket

from harbor_orchard.network import (
    allow_rules_for,
    cidrs_for,
    cidrs_of_rules,
    host_of,
    port_of,
    with_resolved_host,
)


def fake_resolver(table: dict[str, list[str]]):
    def resolve(host, _port, **_kwargs):
        addresses = table.get(host)
        if addresses is None:
            raise OSError(f"unknown host {host}")
        return [(None, None, None, None, (address, 0)) for address in addresses]

    return resolve


class TestHostOf:
    def test_reads_a_url(self):
        assert host_of("http://sglang.internal:30021/v1") == "sglang.internal"

    def test_reads_a_bare_host_and_port(self):
        # urlsplit reads this as scheme 'sglang.internal' without help.
        assert host_of("sglang.internal:30021") == "sglang.internal"

    def test_reads_a_bare_hostname(self):
        assert host_of("sglang.internal") == "sglang.internal"

    def test_empty_is_nothing(self):
        assert host_of("") is None


class TestCidrsFor:
    def test_resolves_a_url_to_every_address_behind_it(self):
        resolve = fake_resolver({"sglang.internal": ["10.0.0.5", "10.0.0.6"]})
        cidrs, unresolved = cidrs_for(
            ["http://sglang.internal:30021/v1"], resolve=resolve
        )
        assert cidrs == ["10.0.0.5/32", "10.0.0.6/32"]
        assert unresolved == []

    def test_a_literal_address_needs_no_dns(self):
        def explode(*_args, **_kwargs):
            raise AssertionError("a literal address must not be resolved")

        cidrs, unresolved = cidrs_for(["http://10.1.2.3:8000/v1"], resolve=explode)
        assert cidrs == ["10.1.2.3/32"]
        assert unresolved == []

    def test_a_cidr_passes_through(self):
        cidrs, unresolved = cidrs_for(["10.4.0.0/16"], resolve=fake_resolver({}))
        assert cidrs == ["10.4.0.0/16"]
        assert unresolved == []

    def test_the_same_address_is_listed_once(self):
        # A fleet of eight ports on one host is one destination, not eight
        # rules: NetworkPolicy allows all ports to an allowed peer anyway.
        resolve = fake_resolver({"sglang.internal": ["10.0.0.5"]})
        cidrs, _ = cidrs_for(
            [f"http://sglang.internal:{port}/v1" for port in range(30021, 30029)],
            resolve=resolve,
        )
        assert cidrs == ["10.0.0.5/32"]

    def test_an_unresolvable_host_is_reported_not_raised(self):
        # One bad name should be named in a log line, not turn a 113-task sweep
        # into a configuration error before the first pod exists.
        cidrs, unresolved = cidrs_for(
            ["http://good.internal/v1", "http://bad.internal/v1"],
            resolve=fake_resolver({"good.internal": ["10.0.0.5"]}),
        )
        assert cidrs == ["10.0.0.5/32"]
        assert unresolved == ["http://bad.internal/v1"]

    def test_a_suffix_domain_has_no_cidr(self):
        # Harbor's agent allowlists are domains like '.openai.com'. There is no
        # address range for that, and silently allowing nothing would look like
        # the allowlist worked.
        cidrs, unresolved = cidrs_for([".openai.com"], resolve=fake_resolver({}))
        assert cidrs == []
        assert unresolved == [".openai.com"]

    def test_ipv6_gets_a_host_prefix(self):
        cidrs, _ = cidrs_for(["http://[fd00::5]:8000/v1"], resolve=fake_resolver({}))
        assert cidrs == ["fd00::5/128"]

    def test_the_resolver_is_asked_for_tcp_only(self):
        seen = {}

        def resolve(host, port, **kwargs):
            seen.update({"host": host, "port": port, **kwargs})
            return [(None, None, None, None, ("10.0.0.5", 0))]

        cidrs_for(["sglang.internal"], resolve=resolve)
        assert seen["host"] == "sglang.internal"
        assert seen["type"] == socket.SOCK_STREAM


class TestPortOf:
    def test_reads_an_explicit_port(self):
        assert port_of("http://sglang.internal:30021/v1") == 30021

    def test_reads_a_bare_host_and_port(self):
        assert port_of("sglang.internal:30021") == 30021

    def test_a_scheme_implies_its_port(self):
        assert port_of("http://sglang.internal/v1") == 80
        assert port_of("https://sglang.internal/v1") == 443

    def test_a_bare_host_implies_nothing(self):
        # Guessing here would silently narrow the allowlist to one port when
        # the operator named a destination and meant all of them.
        assert port_of("sglang.internal") is None

    def test_a_cidr_implies_nothing(self):
        assert port_of("10.4.0.0/16") is None

    def test_empty_is_nothing(self):
        assert port_of("") is None


class TestAllowRulesFor:
    def test_a_destination_keeps_its_own_port(self):
        rules, unresolved = allow_rules_for(
            ["http://sglang.internal:30021/v1"],
            resolve=fake_resolver({"sglang.internal": ["10.4.7.9"]}),
        )
        assert rules == [
            {
                "cidr": "10.4.7.9/32",
                "protocol": "TCP",
                "port_start": 30021,
                "port_end": 30021,
            }
        ]
        assert unresolved == []

    def test_a_portless_destination_fans_out_over_the_defaults(self):
        rules, _ = allow_rules_for(
            ["10.4.0.0/16"], default_ports=(80, 443), resolve=fake_resolver({})
        )
        assert [(r["cidr"], r["port_start"]) for r in rules] == [
            ("10.4.0.0/16", 80),
            ("10.4.0.0/16", 443),
        ]

    def test_an_unresolvable_name_is_reported_not_raised(self):
        # One bad entry must not cost the whole sweep, matching cidrs_for.
        rules, unresolved = allow_rules_for(
            ["nope.invalid", "fleet.example:8000"],
            resolve=fake_resolver({"fleet.example": ["10.9.9.9"]}),
        )
        assert unresolved == ["nope.invalid"]
        assert [(r["cidr"], r["port_start"]) for r in rules] == [("10.9.9.9/32", 8000)]

    def test_the_same_destination_twice_yields_one_rule(self):
        rules, _ = allow_rules_for(
            ["fleet.example:8000", "10.9.9.9:8000"],
            resolve=fake_resolver({"fleet.example": ["10.9.9.9"]}),
        )
        assert len(rules) == 1

    def test_normalization_is_left_to_the_orchestrator(self):
        # It already turns a bare address into /32 and upper-cases the protocol.
        # Doing it here as well is how the two drift apart.
        rules, _ = allow_rules_for(
            ["fleet.example:8000"], resolve=fake_resolver({"fleet.example": ["10.9.9.9"]})
        )
        assert rules[0]["protocol"] == "TCP"

    def test_nothing_wanted_is_no_rules(self):
        assert allow_rules_for([], resolve=fake_resolver({})) == ([], [])


class TestCidrsOfRules:
    def test_distinct_and_ordered(self):
        # The older orchestrator API allowed a destination outright rather than
        # per port, so several rules collapse to one CIDR.
        assert cidrs_of_rules(
            [
                {"cidr": "10.0.0.1/32", "port_start": 80},
                {"cidr": "10.0.0.1/32", "port_start": 443},
                {"cidr": "10.0.0.2/32", "port_start": 80},
            ]
        ) == ["10.0.0.1/32", "10.0.0.2/32"]


class TestWithResolvedHost:
    def test_substitutes_the_address_and_keeps_the_rest(self):
        assert (
            with_resolved_host(
                "http://sglang.internal:30021/v1",
                resolve=fake_resolver({"sglang.internal": ["10.4.7.9"]}),
            )
            == "http://10.4.7.9:30021/v1"
        )

    def test_a_default_port_is_not_materialised(self):
        assert (
            with_resolved_host(
                "http://sglang.internal/v1",
                resolve=fake_resolver({"sglang.internal": ["10.4.7.9"]}),
            )
            == "http://10.4.7.9/v1"
        )

    def test_a_query_survives(self):
        assert (
            with_resolved_host(
                "http://sglang.internal/v1?a=1",
                resolve=fake_resolver({"sglang.internal": ["10.4.7.9"]}),
            )
            == "http://10.4.7.9/v1?a=1"
        )

    def test_an_ipv6_address_is_bracketed(self):
        assert (
            with_resolved_host(
                "http://fleet.example:8000/v1",
                resolve=fake_resolver({"fleet.example": ["fd00::5"]}),
            )
            == "http://[fd00::5]:8000/v1"
        )

    def test_a_literal_is_left_alone(self):
        url = "http://10.4.7.9:30021/v1"
        assert with_resolved_host(url, resolve=fake_resolver({})) == url

    def test_an_unresolvable_name_is_left_alone(self):
        # Returning the name unchanged keeps the failure where it is legible —
        # the agent's connect error — rather than turning it into a URL bug.
        url = "http://nope.invalid/v1"
        assert with_resolved_host(url, resolve=fake_resolver({})) == url

    def test_something_that_is_not_a_url_is_left_alone(self):
        assert (
            with_resolved_host("sglang.internal:30021", resolve=fake_resolver({}))
            == "sglang.internal:30021"
        )

    def test_the_allowlist_and_the_base_url_agree_on_one_address(self):
        # The point of resolving once: a round-robin name must not allowlist one
        # address while the agent dials another.
        resolve = fake_resolver({"sglang.internal": ["10.4.7.9", "10.4.7.10"]})
        resolved = with_resolved_host("http://sglang.internal:30021/v1", resolve=resolve)
        rules, _ = allow_rules_for([resolved], resolve=resolve)
        assert resolved == "http://10.4.7.9:30021/v1"
        assert rules == [
            {
                "cidr": "10.4.7.9/32",
                "protocol": "TCP",
                "port_start": 30021,
                "port_end": 30021,
            }
        ]
