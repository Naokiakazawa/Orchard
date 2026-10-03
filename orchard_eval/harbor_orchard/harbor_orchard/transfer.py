"""Move bytes and directory trees in and out of a sandbox, without a mount.

The Orchard file API carries a whole file as base64 inside one JSON request, so
a terminal-bench task shipping a 400 MB reference genome would build a ~530 MB
request body and hold several copies of it in memory on both ends. Everything
here is therefore chunked: files move as fixed-size parts that are reassembled
in the sandbox with ``cat``, and come back the same way via ``split``.

Directories move as a single tar stream rather than file-by-file. Per-file
transfer silently drops executable bits, symlinks and empty directories — so a
task whose ``COPY`` brings in a ``solve.sh`` would land it non-executable — and
it turns one failure into a partially-populated tree instead of a loud error.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import tarfile
import tempfile
import uuid
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

#: Raw bytes per part. Base64 inflates this by 4/3 on the wire.
DEFAULT_CHUNK_SIZE = 16 * 1024 * 1024

#: Above this, a single-request transfer is split into parts.
_SINGLE_SHOT_LIMIT = DEFAULT_CHUNK_SIZE

#: Seconds for the trivial control commands below — ``test -d``, ``mkdir``,
#: ``stat``. These run in milliseconds; the budget exists to outwait the
#: per-sandbox exec lock, which an agent job Harbor has already abandoned goes
#: on holding until its own deadline. Sized too tightly, every one of them
#: fails in turn and the verifier reports AddTestsDirError instead of a reward.
CONTROL_TIMEOUT = int(os.environ.get("ORCHARD_HARBOR_CONTROL_TIMEOUT", "600"))


class TransferError(RuntimeError):
    """A file transfer failed or arrived incomplete."""


def scratch_path(prefix: str = "hb") -> str:
    return f"/tmp/.{prefix}-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Tar packing
# ---------------------------------------------------------------------------


def pack_entries(entries: Iterable[tuple[Path, str]], archive_path: Path) -> None:
    """Write *entries* — ``(local path, name inside the archive)`` — to a tarball.

    Symlinks are stored as links rather than followed, matching what a real
    build context does. Following them would both inflate the archive and
    silently change task semantics.
    """
    with tarfile.open(archive_path, "w:gz") as tar:
        for source, arcname in entries:
            tar.add(source, arcname=arcname, recursive=True)


def pack_dir(source_dir: Path, archive_path: Path) -> None:
    """Pack the *contents* of a directory, rooted at ``.``."""
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(source_dir, arcname=".", recursive=True)


def extract_archive(archive_path: Path, target_dir: Path) -> None:
    """Extract into *target_dir* with the ``data`` filter.

    Refused members are skipped one by one instead of aborting: an agent that
    left a symlink to an absolute path inside ``/logs`` must not cost us the
    rest of the logs.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar:
            try:
                tar.extract(member, target_dir, filter="data")
            except tarfile.FilterError:
                continue


# ---------------------------------------------------------------------------
# Sandbox transfers
# ---------------------------------------------------------------------------


async def upload_bytes(
    instance,
    data: bytes,
    remote_path: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    if len(data) <= _SINGLE_SHOT_LIMIT:
        await instance.upload_content(data, remote_path)
        return
    with tempfile.NamedTemporaryFile(delete=False) as handle:
        handle.write(data)
        staged = Path(handle.name)
    try:
        await upload_file(instance, staged, remote_path, chunk_size=chunk_size)
    finally:
        staged.unlink(missing_ok=True)


async def upload_file(
    instance,
    local_path: Path,
    remote_path: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Upload a local file, splitting it into parts when it is large."""
    size = local_path.stat().st_size
    parent = os.path.dirname(remote_path) or "/"
    await _exec_checked(instance, f"mkdir -p {_q(parent)}", "prepare upload directory")

    if size <= _SINGLE_SHOT_LIMIT:
        await instance.upload_file(str(local_path), remote_path)
        await _verify_size(instance, remote_path, size)
        return

    part_dir = scratch_path("up")
    await _exec_checked(instance, f"mkdir -p {_q(part_dir)}", "prepare upload parts")
    try:
        index = 0
        with local_path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                # Zero padding keeps the shell glob's lexical order equal to
                # the byte order; without it part10 would be concatenated
                # between part1 and part2 and the file would be corrupt.
                await instance.upload_content(chunk, f"{part_dir}/part{index:06d}")
                index += 1
        await _exec_checked(
            instance,
            f"cat {_q(part_dir)}/part* > {_q(remote_path)}",
            "reassemble uploaded file",
            timeout=600,
        )
        await _verify_size(instance, remote_path, size)
    finally:
        await _exec_best_effort(instance, f"rm -rf {_q(part_dir)}")


async def download_file(
    instance,
    remote_path: str,
    local_path: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Download a remote file, reassembling parts when it is large."""
    size = await _remote_size(instance, remote_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if size <= _SINGLE_SHOT_LIMIT:
        await instance.download_file(remote_path, str(local_path))
        _verify_local_size(local_path, size)
        return

    part_dir = scratch_path("down")
    await _exec_checked(
        instance,
        f"mkdir -p {_q(part_dir)} && split -b {chunk_size} -d -a 6 "
        f"{_q(remote_path)} {_q(part_dir + '/part')}",
        "split file for download",
        timeout=600,
    )
    try:
        listing = await _exec_checked(
            instance, f"ls -1 {_q(part_dir)}", "list download parts"
        )
        names = sorted(line.strip() for line in listing.splitlines() if line.strip())
        with local_path.open("wb") as handle:
            for name in names:
                handle.write(await instance.download_content(f"{part_dir}/{name}"))
        _verify_local_size(local_path, size)
    finally:
        await _exec_best_effort(instance, f"rm -rf {_q(part_dir)}")


async def upload_dir(
    instance,
    source_dir: Path,
    target_dir: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Upload a directory's contents into *target_dir*."""
    with tempfile.TemporaryDirectory() as staging:
        archive = Path(staging) / "payload.tar.gz"
        pack_dir(source_dir, archive)
        await upload_archive(instance, archive, target_dir, chunk_size=chunk_size)


async def upload_archive(
    instance,
    archive: Path,
    target_dir: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Upload a local tarball and unpack it into *target_dir*."""
    remote_archive = scratch_path("tar") + ".tar.gz"
    await upload_file(instance, archive, remote_archive, chunk_size=chunk_size)
    try:
        await _exec_checked(
            instance,
            f"mkdir -p {_q(target_dir)} && tar -xzf {_q(remote_archive)} "
            f"-C {_q(target_dir)}",
            f"unpack archive into {target_dir}",
            timeout=1800,
        )
    finally:
        await _exec_best_effort(instance, f"rm -f {_q(remote_archive)}")


async def download_dir(
    instance,
    source_dir: str,
    target_dir: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Download a remote directory's contents into *target_dir*.

    A missing source is not an error: Harbor asks for ``/logs/agent`` whether or
    not the agent wrote anything, and failing there would turn a quiet rollout
    into an infrastructure failure. It is logged, though — a missing
    ``/logs/verifier`` means the reward file cannot exist, and silence there
    turns a clear cause into a confusing RewardFileNotFoundError.
    """
    exists = await instance.exec(f"test -d {_q(source_dir)}", timeout=CONTROL_TIMEOUT)
    if exists.exit_code != 0:
        logger.warning(
            "%s does not exist in the sandbox; downloading nothing", source_dir
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        return

    remote_archive = scratch_path("tar") + ".tar.gz"
    await _exec_checked(
        instance,
        f"tar -czf {_q(remote_archive)} -C {_q(source_dir)} .",
        f"archive {source_dir}",
        timeout=1800,
    )
    try:
        with tempfile.TemporaryDirectory() as staging:
            local_archive = Path(staging) / "payload.tar.gz"
            await download_file(
                instance, remote_archive, local_archive, chunk_size=chunk_size
            )
            extract_archive(local_archive, target_dir)
    finally:
        await _exec_best_effort(instance, f"rm -f {_q(remote_archive)}")


async def transfer_between(
    source_instance,
    target_instance,
    remote_paths: list[str],
    target_dir: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> None:
    """Copy paths from one sandbox to another, via a tarball on the host.

    This is what ``COPY --from=builder`` becomes when each build stage is its
    own pod. Globs are expanded by the source pod's shell, because only it can
    see the filesystem they refer to.

    ``--from`` also accepts an *image* rather than a stage, and those images are
    routinely distroless — ``COPY --from=ghcr.io/astral-sh/uv:0.8.14 /uv /bin/``
    appears in terminal-bench and that image contains two binaries and nothing
    else: no shell, no ``tar``, no ``cp``. When the source has no shell, the
    files are fetched through the in-pod agent's HTTP API instead, which is a
    Python service and does not need one.
    """
    if await _has_shell(source_instance):
        await _transfer_via_shell(
            source_instance,
            target_instance,
            remote_paths,
            target_dir,
            chunk_size=chunk_size,
        )
        return
    await _transfer_via_agent(
        source_instance,
        target_instance,
        remote_paths,
        target_dir,
        chunk_size=chunk_size,
    )


async def _has_shell(instance) -> bool:
    try:
        result = await instance.exec("exit 0", timeout=60)
    except Exception:  # noqa: BLE001 - a shell-less image fails in varied ways
        return False
    return result.exit_code == 0


async def _transfer_via_shell(
    source_instance,
    target_instance,
    remote_paths: list[str],
    target_dir: str,
    *,
    chunk_size: int,
) -> None:
    staging = scratch_path("stage")
    globs = " ".join(remote_paths)  # intentionally unquoted: the shell expands
    await _exec_checked(
        source_instance,
        f"set -e; rm -rf {_q(staging)}; mkdir -p {_q(staging)}; "
        f"for candidate in {globs}; do "
        f'[ -e "$candidate" ] || {{ echo "missing: $candidate" >&2; exit 1; }}; '
        f'cp -a "$candidate" {_q(staging)}/; done',
        "stage cross-stage COPY sources",
        timeout=1800,
    )
    archive_remote = scratch_path("tar") + ".tar.gz"
    try:
        await _exec_checked(
            source_instance,
            f"tar -czf {_q(archive_remote)} -C {_q(staging)} .",
            "archive cross-stage COPY sources",
            timeout=1800,
        )
        with tempfile.TemporaryDirectory() as host_staging:
            archive = Path(host_staging) / "stage.tar.gz"
            await download_file(
                source_instance, archive_remote, archive, chunk_size=chunk_size
            )
            await upload_archive(
                target_instance, archive, target_dir, chunk_size=chunk_size
            )
    finally:
        await asyncio.gather(
            _exec_best_effort(source_instance, f"rm -rf {_q(staging)}"),
            _exec_best_effort(source_instance, f"rm -f {_q(archive_remote)}"),
        )


async def _transfer_via_agent(
    source_instance,
    target_instance,
    remote_paths: list[str],
    target_dir: str,
    *,
    chunk_size: int,
) -> None:
    """Pull files out of a shell-less source pod through the agent's file API.

    Permission bits are the one casualty: the agent's listing reports name, type
    and size but not mode, and there is no shell to ask. Everything transferred
    this way is therefore made executable — the path exists for tool images
    whose whole content is binaries, and a data file that gains ``+x`` is
    harmless, whereas a binary that loses it is not.
    """
    for path in remote_paths:
        if any(character in path for character in "*?["):
            raise TransferError(
                f"COPY --from={path!r} uses a glob, but the source image has no "
                "shell to expand it. Name the files explicitly."
            )

    with tempfile.TemporaryDirectory() as host_staging:
        staging_root = Path(host_staging) / "payload"
        staging_root.mkdir()
        for path in remote_paths:
            await _fetch_recursive(
                source_instance, path, staging_root / os.path.basename(path.rstrip("/"))
            )
        archive = Path(host_staging) / "stage.tar.gz"
        pack_dir(staging_root, archive)
        await upload_archive(
            target_instance, archive, target_dir, chunk_size=chunk_size
        )

    await _exec_best_effort(target_instance, f"chmod -R u+rwX,go+rX {_q(target_dir)}")
    await _exec_best_effort(
        target_instance,
        f"find {_q(target_dir)} -type f -exec chmod 0755 {{}} +",
    )


async def _fetch_recursive(instance, remote_path: str, local_path: Path) -> None:
    try:
        data = await instance.download_content(remote_path)
    except Exception:  # noqa: BLE001 - most likely a directory, so try listing
        entries = await instance.list_files(remote_path)
        local_path.mkdir(parents=True, exist_ok=True)
        for entry in entries:
            name = entry.get("name")
            if not name:
                continue
            await _fetch_recursive(
                instance, f"{remote_path.rstrip('/')}/{name}", local_path / name
            )
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _q(value: str) -> str:
    return shlex.quote(value)


async def _exec_checked(
    instance, command: str, what: str, *, timeout: int = CONTROL_TIMEOUT
) -> str:
    result = await instance.exec(command, timeout=timeout)
    if result.exit_code != 0:
        output = ((result.stdout or "") + (result.stderr or ""))[-2000:]
        raise TransferError(f"failed to {what} (exit={result.exit_code}): {output}")
    return result.stdout or ""


async def _exec_best_effort(instance, command: str) -> None:
    try:
        await instance.exec(command, timeout=60)
    except Exception:  # noqa: BLE001 - cleanup must never mask the real error
        pass


async def _remote_size(instance, remote_path: str) -> int:
    # BusyBox `stat` does not implement -c, and several task images are
    # Alpine-based, so fall back to wc rather than assuming GNU coreutils.
    result = await instance.exec(
        f"stat -c %s {_q(remote_path)} 2>/dev/null || wc -c < {_q(remote_path)}",
        timeout=CONTROL_TIMEOUT,
    )
    if result.exit_code != 0:
        raise TransferError(f"cannot stat {remote_path}: {result.stderr}")
    try:
        return int((result.stdout or "").strip().split()[0])
    except (ValueError, IndexError) as exc:
        raise TransferError(f"cannot read size of {remote_path}: {exc}") from exc


async def _verify_size(instance, remote_path: str, expected: int) -> None:
    actual = await _remote_size(instance, remote_path)
    if actual != expected:
        raise TransferError(
            f"{remote_path} is {actual} bytes after upload, expected {expected}"
        )


def _verify_local_size(local_path: Path, expected: int) -> None:
    actual = local_path.stat().st_size
    if actual != expected:
        raise TransferError(
            f"{local_path} is {actual} bytes after download, expected {expected}"
        )
