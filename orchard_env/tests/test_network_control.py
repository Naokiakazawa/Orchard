"""Unit tests for dynamic sandbox network configuration."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kubernetes.client import ApiException
from pydantic import ValidationError

from orchard_env.client.sandbox_client import AsyncSandboxInstance, SandboxInstance
from orchard_env.orchestrator.api import NetworkAllowRule, UpdateNetworkRequest
from orchard_env.orchestrator.k8s_client import K8sClient
from orchard_env.orchestrator.redis_store import RedisSandboxStore
from orchard_env.orchestrator.sandbox_manager import (
    NetworkLockLostError,
    NetworkUpdateConflictError,
    Sandbox,
    SandboxManager,
)

ALLOWLIST = [
    {
        "cidr": "203.0.113.10/32",
        "protocol": "TCP",
        "port_start": 30000,
        "port_end": 31000,
    }
]


class TestNetworkModels:
    def test_normalizes_single_ip_protocol_and_port(self):
        rule = NetworkAllowRule(
            cidr="203.0.113.10",
            protocol="tcp",
            port_start=30000,
        )

        assert rule.cidr == "203.0.113.10/32"
        assert rule.protocol == "TCP"
        assert rule.port_end == 30000

    @pytest.mark.parametrize(
        "data",
        [
            {"cidr": "not-an-ip", "port_start": 30000},
            {"cidr": "0.0.0.0/0", "port_start": 30000},
            {
                "cidr": "203.0.113.10",
                "port_start": 31000,
                "port_end": 30000,
            },
        ],
    )
    def test_rejects_invalid_allowlist_rule(self, data):
        with pytest.raises(ValidationError):
            NetworkAllowRule(**data)

    def test_enabled_mode_rejects_allowlist(self):
        with pytest.raises(ValidationError, match="allowlist must be empty"):
            UpdateNetworkRequest(mode="enabled", allowlist=ALLOWLIST)


class TestNetworkPolicy:
    def test_builds_allow_all_policy(self):
        policy = K8sClient._build_egress_network_policy(
            name="allow-egress-s1",
            pod_labels={"sandbox-id": "s1"},
            allow_all=True,
            allowlist=[],
        )

        assert policy.spec.pod_selector.match_labels == {"sandbox-id": "s1"}
        assert len(policy.spec.egress) == 1
        assert policy.spec.egress[0].to is None
        assert policy.spec.egress[0].ports is None

    def test_builds_cidr_port_range_policy(self):
        policy = K8sClient._build_egress_network_policy(
            name="allow-egress-s1",
            pod_labels={"sandbox-id": "s1"},
            allow_all=False,
            allowlist=ALLOWLIST,
        )

        rule = policy.spec.egress[0]
        assert rule.to[0].ip_block.cidr == "203.0.113.10/32"
        assert rule.ports[0].protocol == "TCP"
        assert rule.ports[0].port == 30000
        assert rule.ports[0].end_port == 31000

    def test_builds_deny_all_for_empty_allowlist(self):
        policy = K8sClient._build_egress_network_policy(
            name="allow-egress-s1",
            pod_labels={"sandbox-id": "s1"},
            allow_all=False,
            allowlist=[],
        )

        assert policy.spec.egress == []

    @pytest.mark.asyncio
    async def test_upsert_creates_missing_policy(self):
        k8s = object.__new__(K8sClient)
        networking = MagicMock()
        k8s._get_networking_v1_api = MagicMock(return_value=networking)
        k8s._k8s_call = AsyncMock(side_effect=[ApiException(status=404), None])

        await k8s.upsert_egress_network_policy(
            name="allow-egress-s1",
            namespace="sandbox-pods",
            pod_labels={"sandbox-id": "s1"},
            allow_all=False,
            allowlist=ALLOWLIST,
        )

        assert k8s._k8s_call.await_count == 2
        create_call = k8s._k8s_call.await_args_list[1]
        assert create_call.args[0] == networking.create_namespaced_network_policy
        assert create_call.kwargs["body"].spec.egress[0].ports[0].end_port == 31000

    @pytest.mark.asyncio
    async def test_upsert_replaces_policy_with_resource_version(self):
        k8s = object.__new__(K8sClient)
        networking = MagicMock()
        existing = SimpleNamespace(metadata=SimpleNamespace(resource_version="42"))
        k8s._get_networking_v1_api = MagicMock(return_value=networking)
        k8s._k8s_call = AsyncMock(side_effect=[existing, None])

        await k8s.upsert_egress_network_policy(
            name="allow-egress-s1",
            namespace="sandbox-pods",
            pod_labels={"sandbox-id": "s1"},
            allow_all=True,
            allowlist=[],
        )

        replace_call = k8s._k8s_call.await_args_list[1]
        assert replace_call.args[0] == networking.replace_namespaced_network_policy
        assert replace_call.kwargs["body"].metadata.resource_version == "42"

    @pytest.mark.asyncio
    async def test_upsert_confirms_revision_after_ambiguous_write_failure(self):
        k8s = object.__new__(K8sClient)
        networking = MagicMock()
        existing = SimpleNamespace(metadata=SimpleNamespace(resource_version="42"))
        current = K8sClient._build_egress_network_policy(
            name="allow-egress-s1",
            pod_labels={"sandbox-id": "s1"},
            allow_all=True,
            allowlist=[],
            revision="request-revision",
        )
        k8s._get_networking_v1_api = MagicMock(return_value=networking)
        k8s._k8s_call = AsyncMock(
            side_effect=[existing, TimeoutError("response lost"), current]
        )

        await k8s.upsert_egress_network_policy(
            name="allow-egress-s1",
            namespace="sandbox-pods",
            pod_labels={"sandbox-id": "s1"},
            allow_all=True,
            allowlist=[],
            revision="request-revision",
        )

        assert k8s._k8s_call.await_count == 3

    @pytest.mark.asyncio
    async def test_create_confirms_revision_after_ambiguous_write_failure(self):
        k8s = object.__new__(K8sClient)
        networking = MagicMock()
        current = K8sClient._build_egress_network_policy(
            name="allow-egress-s1",
            pod_labels={"sandbox-id": "s1"},
            allow_all=True,
            allowlist=[],
            revision="request-revision",
        )
        k8s._get_networking_v1_api = MagicMock(return_value=networking)
        k8s._k8s_call = AsyncMock(side_effect=[TimeoutError("response lost"), current])

        created = await k8s.create_egress_network_policy(
            name="allow-egress-s1",
            namespace="sandbox-pods",
            pod_labels={"sandbox-id": "s1"},
            allow_all=True,
            allowlist=[],
            revision="request-revision",
            owner_pod_name="sandbox-s1",
            owner_pod_uid="pod-uid",
        )

        assert created is True


@pytest.mark.asyncio
class TestSandboxManagerNetwork:
    @staticmethod
    def make_manager(block_network=False, allowlist=None):
        k8s = MagicMock()
        k8s.has_network_policy = AsyncMock(return_value=True)
        k8s.upsert_egress_network_policy = AsyncMock()
        k8s.create_egress_network_policy = AsyncMock(return_value=True)
        k8s.delete_network_policy = AsyncMock()
        k8s.delete_pod = AsyncMock()
        k8s.get_pod = AsyncMock(
            return_value=SimpleNamespace(
                metadata=SimpleNamespace(
                    uid="pod-uid",
                    resource_version="pod-rv",
                )
            )
        )

        async def current_policy(**kwargs):
            revision = k8s.upsert_egress_network_policy.await_args.kwargs["revision"]
            return {
                "mode": "restricted",
                "allowlist": ALLOWLIST,
                "revision": revision,
            }

        k8s.get_egress_network_policy = AsyncMock(side_effect=current_policy)
        manager = SandboxManager(k8s)
        manager._sandboxes["s1"] = Sandbox(
            sandbox_id="s1",
            namespace="sandbox-pods",
            image="ubuntu:22.04",
            pod_name="sandbox-s1",
            block_network=block_network,
            cpu="1",
            memory="1Gi",
        )
        return manager, k8s

    async def test_restricts_and_persists_allowlist(self):
        with patch("orchestrator.sandbox_manager.settings.use_redis", False):
            manager, k8s = self.make_manager()

            result = await manager.update_network_config(
                "s1", mode="restricted", allowlist=ALLOWLIST
            )

        assert result == {
            "sandbox_id": "s1",
            "mode": "restricted",
            "allowlist": ALLOWLIST,
        }
        assert manager._sandboxes["s1"].block_network is True
        update = k8s.upsert_egress_network_policy.await_args.kwargs
        assert update["name"] == "allow-egress-s1"
        assert update["namespace"] == "sandbox-pods"
        assert update["pod_labels"] == {"sandbox-id": "s1"}
        assert update["allow_all"] is False
        assert update["allowlist"] == ALLOWLIST
        assert update["revision"]
        assert update["owner_pod_name"] == "sandbox-s1"
        assert update["owner_pod_uid"] == "pod-uid"

    async def test_enables_all_egress(self):
        with patch("orchestrator.sandbox_manager.settings.use_redis", False):
            manager, k8s = self.make_manager(block_network=True, allowlist=ALLOWLIST)

            result = await manager.update_network_config(
                "s1", mode="enabled", allowlist=[]
            )

        assert result["mode"] == "enabled"
        assert manager._sandboxes["s1"].block_network is False
        assert k8s.upsert_egress_network_policy.await_args.kwargs["allow_all"] is True

    async def test_policy_remains_authoritative_when_cache_update_fails(self):
        with patch("orchestrator.sandbox_manager.settings.use_redis", False):
            manager, k8s = self.make_manager()
            manager._update_sandbox = AsyncMock(
                side_effect=RuntimeError("state unavailable")
            )

            result = await manager.update_network_config(
                "s1", mode="restricted", allowlist=ALLOWLIST
            )

        assert result["mode"] == "restricted"
        assert k8s.upsert_egress_network_policy.await_count == 1
        k8s.delete_network_policy.assert_not_awaited()

    async def test_get_reads_authoritative_policy_without_changing_cache(self):
        with patch("orchestrator.sandbox_manager.settings.use_redis", False):
            manager, k8s = self.make_manager(block_network=False)
            k8s.get_egress_network_policy = AsyncMock(
                return_value={
                    "mode": "restricted",
                    "allowlist": ALLOWLIST,
                    "revision": "policy-revision",
                }
            )

            result = await manager.get_network_config("s1")

        assert result["mode"] == "restricted"
        assert result["allowlist"] == ALLOWLIST
        assert manager._sandboxes["s1"].block_network is False

    async def test_missing_sandbox_returns_error_without_policy_change(self):
        with patch("orchestrator.sandbox_manager.settings.use_redis", False):
            manager, k8s = self.make_manager()
            del manager._sandboxes["s1"]

            with pytest.raises(ValueError, match="not found"):
                await manager.update_network_config(
                    "s1", mode="restricted", allowlist=ALLOWLIST
                )

        k8s.upsert_egress_network_policy.assert_not_awaited()

    async def test_duplicate_creation_does_not_delete_existing_resources(self):
        with patch("orchestrator.sandbox_manager.settings.use_redis", False):
            manager, k8s = self.make_manager()
            k8s.create_pod = AsyncMock()

            with pytest.raises(ValueError, match="already exists"):
                await manager.create_sandbox(
                    sandbox_id="s1",
                    image="ubuntu:22.04",
                    block_network=False,
                    wait_ready=False,
                )

        k8s.create_pod.assert_not_awaited()
        k8s.delete_network_policy.assert_not_awaited()
        assert "s1" in manager._sandboxes

    async def test_restricted_creation_rejects_stale_allow_policy(self):
        with patch("orchestrator.sandbox_manager.settings.use_redis", False):
            manager, k8s = self.make_manager()
            del manager._sandboxes["s1"]
            k8s.get_pod.return_value = None
            k8s.get_egress_network_policy = AsyncMock(
                return_value={
                    "mode": "enabled",
                    "allowlist": [],
                    "revision": "stale",
                }
            )
            k8s.create_pod = AsyncMock()

            with pytest.raises(ValueError, match="Network policy"):
                await manager.create_sandbox(
                    sandbox_id="s1",
                    image="ubuntu:22.04",
                    block_network=True,
                    wait_ready=False,
                )

        k8s.create_pod.assert_not_awaited()
        k8s.delete_network_policy.assert_not_awaited()

    async def test_delete_uses_policy_preconditions(self):
        with patch("orchestrator.sandbox_manager.settings.use_redis", False):
            manager, k8s = self.make_manager()
            k8s.get_egress_network_policy = AsyncMock(
                return_value={
                    "mode": "enabled",
                    "allowlist": [],
                    "revision": "r1",
                    "uid": "policy-uid",
                    "resource_version": "42",
                }
            )

            await manager.delete_sandbox("s1")

        k8s.delete_network_policy.assert_awaited_once_with(
            "allow-egress-s1",
            "sandbox-pods",
            uid="policy-uid",
            resource_version="42",
        )
        k8s.delete_pod.assert_awaited_once_with(
            "sandbox-s1",
            "sandbox-pods",
            grace_period_seconds=0,
            uid="pod-uid",
        )
        assert "s1" not in manager._sandboxes

    async def test_reconciles_state_when_request_still_owns_policy_revision(self):
        with patch("orchestrator.sandbox_manager.settings.use_redis", False):
            manager, k8s = self.make_manager()
            lock_attempt = 0

            def fake_lock(sandbox_id):
                @asynccontextmanager
                async def context():
                    nonlocal lock_attempt
                    lock_attempt += 1
                    ownership_check = 0

                    async def ensure_owned():
                        nonlocal ownership_check
                        ownership_check += 1
                        if lock_attempt == 1 and ownership_check == 2:
                            raise NetworkLockLostError("lost")

                    yield ensure_owned

                return context()

            manager._network_operation_lock = fake_lock

            async def current_policy(**kwargs):
                revision = k8s.upsert_egress_network_policy.await_args.kwargs[
                    "revision"
                ]
                return {
                    "mode": "restricted",
                    "allowlist": ALLOWLIST,
                    "revision": revision,
                }

            k8s.get_egress_network_policy = AsyncMock(side_effect=current_policy)

            result = await manager.update_network_config(
                "s1", mode="restricted", allowlist=ALLOWLIST
            )

        assert result["mode"] == "restricted"
        assert manager._sandboxes["s1"].block_network is True
        assert lock_attempt == 2

    async def test_lost_request_does_not_overwrite_newer_policy(self):
        with patch("orchestrator.sandbox_manager.settings.use_redis", False):
            manager, k8s = self.make_manager()
            lock_attempt = 0

            def fake_lock(sandbox_id):
                @asynccontextmanager
                async def context():
                    nonlocal lock_attempt
                    lock_attempt += 1
                    ownership_check = 0

                    async def ensure_owned():
                        nonlocal ownership_check
                        ownership_check += 1
                        if lock_attempt == 1 and ownership_check == 2:
                            raise NetworkLockLostError("lost")

                    yield ensure_owned

                return context()

            manager._network_operation_lock = fake_lock
            k8s.get_egress_network_policy = AsyncMock(
                return_value={
                    "mode": "enabled",
                    "allowlist": [],
                    "revision": "newer-request",
                }
            )

            with pytest.raises(NetworkUpdateConflictError, match="superseded"):
                await manager.update_network_config(
                    "s1", mode="restricted", allowlist=ALLOWLIST
                )

        assert manager._sandboxes["s1"].block_network is False


@pytest.mark.asyncio
class TestRedisNetworkLock:
    async def test_uses_owner_token_for_release(self):
        store = RedisSandboxStore()
        client = AsyncMock()
        client.set.return_value = True
        store._client = client

        token = await store.acquire_network_lock("s1", timeout=60)
        await store.release_network_lock("s1", token)

        assert token
        client.set.assert_awaited_once_with(
            "sandbox:network-lock:s1", token, nx=True, ex=60
        )
        eval_call = client.eval.await_args
        assert eval_call.args[1:] == (
            1,
            "sandbox:network-lock:s1",
            token,
        )

    async def test_refreshes_only_the_owned_lock(self):
        store = RedisSandboxStore()
        client = AsyncMock()
        client.eval.return_value = 1
        store._client = client

        refreshed = await store.refresh_network_lock("s1", "owner-token", timeout=60)

        assert refreshed is True
        assert client.eval.await_args.args[1:] == (
            1,
            "sandbox:network-lock:s1",
            "owner-token",
            60,
        )

    async def test_partial_state_update_uses_atomic_lua_merge(self):
        store = RedisSandboxStore()
        client = AsyncMock()
        client.eval.return_value = 1
        store._client = client

        updated = await store.update_sandbox("s1", {"block_network": True})

        assert updated is True
        eval_args = client.eval.await_args.args
        assert eval_args[1] == 1
        assert eval_args[2] == "sandbox:s1"
        assert '"block_network": true' in eval_args[3]


class TestSyncNetworkClient:
    def test_disable_and_enable_network(self):
        client = MagicMock()
        client._request.side_effect = [
            {
                "sandbox_id": "s1",
                "mode": "restricted",
                "allowlist": ALLOWLIST,
            },
            {"sandbox_id": "s1", "mode": "enabled", "allowlist": []},
        ]
        sandbox = SandboxInstance(
            client=client,
            sandbox_id="s1",
            data={"block_network": False},
        )

        restricted = sandbox.disable_network(ALLOWLIST)
        assert restricted["mode"] == "restricted"
        assert sandbox.block_network is True

        enabled = sandbox.enable_network()
        assert enabled["mode"] == "enabled"
        assert sandbox.block_network is False


@pytest.mark.asyncio
class TestAsyncNetworkClient:
    async def test_disable_and_enable_network(self):
        client = MagicMock()
        client._request = AsyncMock(
            side_effect=[
                {
                    "sandbox_id": "s1",
                    "mode": "restricted",
                    "allowlist": ALLOWLIST,
                },
                {"sandbox_id": "s1", "mode": "enabled", "allowlist": []},
            ]
        )
        sandbox = AsyncSandboxInstance(
            client=client,
            sandbox_id="s1",
            data={"block_network": False},
        )

        restricted = await sandbox.disable_network(ALLOWLIST)
        assert restricted["mode"] == "restricted"
        assert sandbox.block_network is True

        enabled = await sandbox.enable_network()
        assert enabled["mode"] == "enabled"
        assert sandbox.block_network is False
