"""Redis-based sandbox state store for multi-replica support."""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict
from typing import Dict, Optional, Set

import redis.asyncio as redis

from orchard_env.orchestrator.settings import settings

logger = logging.getLogger(__name__)


class RedisSandboxStore:
    """Redis-based storage for sandbox state.
    
    This allows multiple orchestrator replicas to share sandbox state,
    avoiding race conditions in reconciliation.
    
    Key schema:
    - sandbox:{sandbox_id} -> JSON of Sandbox dataclass
    - sandbox:lock:{sandbox_id} -> distributed lock for exec serialization
    - sandbox:all -> Set of all sandbox IDs
    """
    
    SANDBOX_PREFIX = "sandbox:"
    SANDBOX_SET_KEY = "sandbox:all"
    LOCK_PREFIX = "sandbox:lock:"
    NETWORK_LOCK_PREFIX = "sandbox:network-lock:"
    DEFAULT_TTL = 3600 * 24  # 24 hours
    
    def __init__(self, redis_url: str = None):
        """Initialize Redis store.
        
        Args:
            redis_url: Redis connection URL, e.g., redis://localhost:6379/0
        """
        self.redis_url = redis_url or settings.redis_url
        self._client: Optional[redis.Redis] = None
        self._locks: Dict[str, asyncio.Lock] = {}  # Local locks for async coordination
    
    async def connect(self) -> None:
        """Connect to Redis."""
        if self._client is None:
            self._client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            # Test connection
            await self._client.ping()
            logger.info(f"Connected to Redis at {self.redis_url}")
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("Closed Redis connection")
    
    async def _ensure_connected(self) -> redis.Redis:
        """Ensure we have a Redis connection."""
        if self._client is None:
            await self.connect()
        return self._client
    
    async def store_sandbox(self, sandbox_id: str, sandbox_data: dict) -> None:
        """Store sandbox metadata.
        
        Args:
            sandbox_id: Unique sandbox ID
            sandbox_data: Sandbox data as dict (from dataclass)
        """
        client = await self._ensure_connected()
        key = f"{self.SANDBOX_PREFIX}{sandbox_id}"
        
        # Store sandbox data with TTL
        await client.set(
            key,
            json.dumps(sandbox_data),
            ex=self.DEFAULT_TTL
        )
        
        # Add to set of all sandboxes
        await client.sadd(self.SANDBOX_SET_KEY, sandbox_id)
        
        logger.debug(f"Stored sandbox {sandbox_id} in Redis")
    
    async def get_sandbox(self, sandbox_id: str) -> Optional[dict]:
        """Get sandbox metadata.
        
        Args:
            sandbox_id: Sandbox ID to retrieve
            
        Returns:
            Sandbox data as dict, or None if not found
        """
        client = await self._ensure_connected()
        key = f"{self.SANDBOX_PREFIX}{sandbox_id}"
        
        data = await client.get(key)
        if data:
            return json.loads(data)
        return None
    
    async def update_sandbox(self, sandbox_id: str, updates: dict) -> bool:
        """Update sandbox metadata.
        
        Args:
            sandbox_id: Sandbox ID to update
            updates: Dict of fields to update
            
        Returns:
            True if updated, False if sandbox not found
        """
        client = await self._ensure_connected()
        key = f"{self.SANDBOX_PREFIX}{sandbox_id}"
        
        updated = await client.eval(
            """
            local data = redis.call("get", KEYS[1])
            if not data then
                return 0
            end
            local sandbox = cjson.decode(data)
            local updates = cjson.decode(ARGV[1])
            for key, value in pairs(updates) do
                sandbox[key] = value
            end
            redis.call("set", KEYS[1], cjson.encode(sandbox), "EX", ARGV[2])
            return 1
            """,
            1,
            key,
            json.dumps(updates),
            self.DEFAULT_TTL,
        )
        logger.debug(f"Updated sandbox {sandbox_id}: {updates}")
        return bool(updated)

    async def delete_sandbox(self, sandbox_id: str) -> bool:
        """Delete sandbox metadata.
        
        Args:
            sandbox_id: Sandbox ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        client = await self._ensure_connected()
        key = f"{self.SANDBOX_PREFIX}{sandbox_id}"
        
        # Remove from set
        await client.srem(self.SANDBOX_SET_KEY, sandbox_id)
        
        # Delete key
        deleted = await client.delete(key)
        
        # Clean up lock
        lock_key = f"{self.LOCK_PREFIX}{sandbox_id}"
        await client.delete(lock_key)
        
        if deleted:
            logger.debug(f"Deleted sandbox {sandbox_id} from Redis")
        return deleted > 0
    
    async def get_all_sandbox_ids(self) -> Set[str]:
        """Get all tracked sandbox IDs.
        
        Returns:
            Set of sandbox IDs
        """
        client = await self._ensure_connected()
        return await client.smembers(self.SANDBOX_SET_KEY)
    
    async def get_all_sandboxes(self) -> Dict[str, dict]:
        """Get all sandbox metadata.
        
        Returns:
            Dict mapping sandbox_id to sandbox data
        """
        client = await self._ensure_connected()
        sandbox_ids = await self.get_all_sandbox_ids()
        
        result = {}
        for sandbox_id in sandbox_ids:
            data = await self.get_sandbox(sandbox_id)
            if data:
                result[sandbox_id] = data
            else:
                # Clean up stale entry
                await client.srem(self.SANDBOX_SET_KEY, sandbox_id)
        
        return result
    
    async def acquire_lock(self, sandbox_id: str, timeout: int = 300) -> bool:
        """Acquire a distributed lock for a sandbox.
        
        Used for serializing exec operations on a sandbox.
        
        Args:
            sandbox_id: Sandbox ID to lock
            timeout: Lock timeout in seconds
            
        Returns:
            True if lock acquired, False otherwise
        """
        client = await self._ensure_connected()
        lock_key = f"{self.LOCK_PREFIX}{sandbox_id}"
        
        # Try to acquire lock with NX (only if not exists)
        acquired = await client.set(
            lock_key,
            str(time.time()),
            nx=True,
            ex=timeout
        )
        
        return acquired is not None
    
    async def release_lock(self, sandbox_id: str) -> None:
        """Release a distributed lock for a sandbox.
        
        Args:
            sandbox_id: Sandbox ID to unlock
        """
        client = await self._ensure_connected()
        lock_key = f"{self.LOCK_PREFIX}{sandbox_id}"
        await client.delete(lock_key)

    async def acquire_network_lock(
        self, sandbox_id: str, timeout: int = 300
    ) -> Optional[str]:
        """Acquire an owner-token-protected lock for network mutations."""
        client = await self._ensure_connected()
        lock_key = f"{self.NETWORK_LOCK_PREFIX}{sandbox_id}"
        token = uuid.uuid4().hex
        acquired = await client.set(lock_key, token, nx=True, ex=timeout)
        return token if acquired is not None else None

    async def release_network_lock(self, sandbox_id: str, token: str) -> None:
        """Release a network lock only when the caller still owns it."""
        client = await self._ensure_connected()
        lock_key = f"{self.NETWORK_LOCK_PREFIX}{sandbox_id}"
        await client.eval(
            """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            end
            return 0
            """,
            1,
            lock_key,
            token,
        )

    async def refresh_network_lock(
        self, sandbox_id: str, token: str, timeout: int = 300
    ) -> bool:
        """Extend a network lock only when the caller still owns it."""
        client = await self._ensure_connected()
        lock_key = f"{self.NETWORK_LOCK_PREFIX}{sandbox_id}"
        refreshed = await client.eval(
            """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("expire", KEYS[1], ARGV[2])
            end
            return 0
            """,
            1,
            lock_key,
            token,
            timeout,
        )
        return bool(refreshed)
    
    async def get_ready_sandbox_ids(self) -> Set[str]:
        """Get IDs of sandboxes that are marked as ready.
        
        Returns:
            Set of sandbox IDs that are ready
        """
        sandboxes = await self.get_all_sandboxes()
        return {
            sid for sid, data in sandboxes.items()
            if data.get("ready", False)
        }
    
    async def sandbox_exists(self, sandbox_id: str) -> bool:
        """Check if a sandbox exists in the store.
        
        Args:
            sandbox_id: Sandbox ID to check
            
        Returns:
            True if exists, False otherwise
        """
        client = await self._ensure_connected()
        return await client.sismember(self.SANDBOX_SET_KEY, sandbox_id)

    # ---- Pod IP cache (lightweight, dedicated keys) ----

    POD_IP_PREFIX = "sandbox:ip:"
    POD_IP_TTL = 3600 * 24  # 24 hours — same as sandbox TTL

    async def store_pod_ip(self, sandbox_id: str, pod_ip: str) -> None:
        """Cache a pod IP in Redis for cross-replica lookups.

        Uses a dedicated key (``sandbox:ip:<id>``) instead of embedding
        it inside the sandbox JSON blob so that reads are a single
        O(1) GET rather than GET + JSON parse.
        """
        client = await self._ensure_connected()
        key = f"{self.POD_IP_PREFIX}{sandbox_id}"
        await client.set(key, pod_ip, ex=self.POD_IP_TTL)
        logger.debug(f"Stored pod IP for {sandbox_id} in Redis: {pod_ip}")

    async def get_pod_ip(self, sandbox_id: str) -> Optional[str]:
        """Get cached pod IP from Redis.

        Returns:
            Pod IP string, or None if not cached.
        """
        client = await self._ensure_connected()
        key = f"{self.POD_IP_PREFIX}{sandbox_id}"
        return await client.get(key)

    async def delete_pod_ip(self, sandbox_id: str) -> None:
        """Remove cached pod IP from Redis."""
        client = await self._ensure_connected()
        key = f"{self.POD_IP_PREFIX}{sandbox_id}"
        await client.delete(key)


# Global store instance
_redis_store: Optional[RedisSandboxStore] = None


async def get_redis_store() -> RedisSandboxStore:
    """Get the global Redis store instance."""
    global _redis_store
    if _redis_store is None:
        _redis_store = RedisSandboxStore()
        await _redis_store.connect()
    return _redis_store


async def close_redis_store() -> None:
    """Close the global Redis store."""
    global _redis_store
    if _redis_store:
        await _redis_store.close()
        _redis_store = None
