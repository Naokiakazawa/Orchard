"""Execute a translated Dockerfile against live sandboxes.

One pod plays the role of the final image and receives every operation of the
final stage. When that stage copies from an earlier one, a throwaway pod is
created for the earlier stage, built, harvested, and deleted — which is the
closest a sandbox service without an image builder can get to a multi-stage
build, and costs nothing when a Dockerfile has only one stage.

Two behaviours here are load-bearing:

**The base image's environment is read before translation, not assumed.**
``ENV PATH="/opt/vep:$PATH"`` is the most common Dockerfile idiom there is, and
expanding ``$PATH`` against an empty scope replaces the system PATH with a
single directory. Every command after that point then fails in a way that looks
like a broken task rather than a broken harness.

**A failing step aborts the build.** ``docker build`` stops at the first
non-zero ``RUN``; continuing past one produces a pod that looks healthy, runs
the agent, fails the verifier, and is scored as a model failure.
"""

from __future__ import annotations

import logging
import shlex
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from harbor_orchard import transfer
from harbor_orchard.dockerfile import Dockerfile, Stage
from harbor_orchard.image_config import fetch_image_workdir
from harbor_orchard.images import rewrite as rewrite_image
from harbor_orchard.plan import (
    CopyOp,
    CopySource,
    RunOp,
    StageState,
    WriteFileOp,
    translate_stage,
)
from harbor_orchard.shell import (
    COPY_ROUTINE,
    EXTRACT_ROUTINE,
    export_env,
    script_invocation,
    wrap_user,
)

logger = logging.getLogger(__name__)

#: Written into the pod so that login shells and anything spawning a fresh
#: shell (an agent's own subprocesses, a test script using ``bash -l``) observe
#: the environment the Dockerfile established. Commands we issue also receive it
#: directly, so this is belt and braces rather than the primary mechanism.
PROFILE_PATH = "/etc/profile.d/00-harbor-orchard.sh"


class BuildError(RuntimeError):
    """A replayed Dockerfile instruction failed."""

    def __init__(self, source: str, exit_code: int, output: str):
        self.source = source
        self.exit_code = exit_code
        self.output = output
        super().__init__(f"{source}\n  exit={exit_code}\n{output[-4000:]}")


@dataclass
class BuildResult:
    state: StageState
    base_image: str
    #: Steps actually executed, for the trial log.
    step_count: int = 0
    stage_pods: list[str] = field(default_factory=list)


class SandboxBuilder:
    """Replays a Dockerfile into an already-created sandbox.

    Args:
        client: A connected ``AsyncSandboxClient``, used only to create the
            extra pods multi-stage builds need.
        create_sandbox: Callable returning a ready ``AsyncSandboxInstance`` for
            a given image. Injected so the environment can apply its own
            resource and network settings to stage pods.
    """

    def __init__(
        self,
        *,
        create_sandbox,
        delete_sandbox,
        image_mirror: str | None,
        chunk_size: int,
        step_timeout: int,
        image_remap: Mapping[str, str] | None = None,
        log: logging.Logger | None = None,
    ):
        self._create_sandbox = create_sandbox
        self._delete_sandbox = delete_sandbox
        self._image_mirror = image_mirror
        self._image_remap = image_remap
        self._chunk_size = chunk_size
        self._step_timeout = step_timeout
        self._log = log or logger
        # Stages currently being built, so a Dockerfile whose stages copy from
        # each other in a cycle fails with a message instead of recursing until
        # the process dies.
        self._building: set[str] = set()

    async def build(
        self,
        instance,
        dockerfile: Dockerfile,
        *,
        context_dir: Path,
        stage: Stage | None = None,
        build_args: dict[str, str] | None = None,
    ) -> BuildResult:
        """Replay *stage* (default: the final one) into *instance*."""
        target_stage = stage or dockerfile.final_stage
        base_env = await self.read_environment(instance)
        plan = translate_stage(
            dockerfile,
            target_stage,
            context_dir=context_dir,
            base_env=base_env,
            base_workdir=self.resolve_workdir(self.resolve_image(target_stage.base)),
            build_args=build_args,
        )

        stage_pods: dict[str, object] = {}
        created: list[str] = []
        try:
            for index, operation in enumerate(plan.operations, start=1):
                self._log.debug(
                    "build step %d/%d: %s",
                    index,
                    len(plan.operations),
                    operation.source,
                )
                await self._apply(
                    instance,
                    operation,
                    dockerfile=dockerfile,
                    context_dir=context_dir,
                    build_args=build_args,
                    stage_pods=stage_pods,
                    created=created,
                )
        finally:
            for pod in stage_pods.values():
                await self._delete_sandbox(pod)

        await self._persist_environment(instance, plan.state)
        return BuildResult(
            state=plan.state,
            base_image=plan.base_image,
            step_count=len(plan.operations),
            stage_pods=created,
        )

    def resolve_image(self, reference: str) -> str:
        return rewrite_image(reference, self._image_mirror, self._image_remap)

    def resolve_workdir(self, image: str) -> str:
        """The image's declared ``WorkingDir``, or ``/`` when it declares none.

        Kubernetes overrides the image's working directory with the pod's, so
        this cannot be read from the running container — see
        :mod:`harbor_orchard.image_config`.
        """
        return fetch_image_workdir(image) or "/"

    async def read_environment(self, instance) -> dict[str, str]:
        """Read the container's environment as a fresh process sees it.

        ``env -0`` is used where available because a NUL separator is the only
        way to read a variable whose value contains a newline; several CUDA and
        conda images ship exactly that.
        """
        result = await instance.exec("env -0 2>/dev/null || env", timeout=60)
        text = result.stdout or ""
        separator = "\0" if "\0" in text else "\n"
        environment: dict[str, str] = {}
        for entry in text.split(separator):
            if not entry or "=" not in entry:
                continue
            key, _, value = entry.partition("=")
            environment[key] = value.rstrip("\n") if separator == "\n" else value
        return environment

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    async def _apply(
        self,
        instance,
        operation,
        *,
        dockerfile,
        context_dir,
        build_args,
        stage_pods,
        created,
    ) -> None:
        if isinstance(operation, RunOp):
            await self._run(instance, operation)
        elif isinstance(operation, WriteFileOp):
            await self._write_file(instance, operation)
        elif isinstance(operation, CopyOp):
            await self._copy(
                instance,
                operation,
                dockerfile=dockerfile,
                context_dir=context_dir,
                build_args=build_args,
                stage_pods=stage_pods,
                created=created,
            )
        else:  # pragma: no cover - the union is closed
            raise TypeError(f"unknown build operation {operation!r}")

    async def _run(self, instance, operation: RunOp) -> None:
        command = wrap_user(operation.command(), operation.user)
        result = await instance.exec(
            command,
            timeout=self._step_timeout,
            cwd=operation.cwd,
            env=operation.env,
        )
        if result.exit_code != 0:
            raise BuildError(
                operation.source,
                result.exit_code if result.exit_code is not None else -1,
                (result.stdout or "") + (result.stderr or ""),
            )

    async def _write_file(self, instance, operation: WriteFileOp) -> None:
        await transfer.upload_bytes(
            instance,
            operation.content.encode("utf-8"),
            operation.dest,
            chunk_size=self._chunk_size,
        )
        if operation.chmod:
            await instance.exec(
                f"chmod {shlex.quote(operation.chmod)} {shlex.quote(operation.dest)}",
                timeout=60,
            )
        if operation.chown:
            await instance.exec(
                f"chown {shlex.quote(operation.chown)} {shlex.quote(operation.dest)}",
                timeout=60,
            )

    async def _copy(
        self,
        instance,
        operation: CopyOp,
        *,
        dockerfile,
        context_dir,
        build_args,
        stage_pods,
        created,
    ) -> None:
        staging = transfer.scratch_path("copy")
        await instance.exec(f"mkdir -p {shlex.quote(staging)}", timeout=60)
        try:
            await self._stage_sources(
                instance,
                operation,
                staging,
                dockerfile=dockerfile,
                context_dir=context_dir,
                build_args=build_args,
                stage_pods=stage_pods,
                created=created,
            )

            if operation.extract_archives:
                await self._exec_or_raise(
                    instance,
                    script_invocation(EXTRACT_ROUTINE, staging),
                    operation,
                    env=operation.env,
                )

            environment = dict(operation.env)
            if operation.chown:
                environment["__HB_CHOWN"] = operation.chown
            if operation.chmod:
                environment["__HB_CHMOD"] = operation.chmod

            await self._exec_or_raise(
                instance,
                script_invocation(COPY_ROUTINE, staging, operation.dest),
                operation,
                env=environment,
            )
        finally:
            await instance.exec(f"rm -rf {shlex.quote(staging)}", timeout=120)

    async def _stage_sources(
        self,
        instance,
        operation: CopyOp,
        staging: str,
        *,
        dockerfile,
        context_dir,
        build_args,
        stage_pods,
        created,
    ) -> None:
        local = [source for source in operation.sources if source.local_path]
        remote = [source for source in operation.sources if source.remote_path]
        urls = [source for source in operation.sources if source.url]

        if local:
            await self._stage_local(instance, local, staging)
        if remote:
            source_instance = await self._stage_pod(
                operation.from_stage,
                dockerfile=dockerfile,
                context_dir=context_dir,
                build_args=build_args,
                stage_pods=stage_pods,
                created=created,
            )
            await transfer.transfer_between(
                source_instance,
                instance,
                [source.remote_path for source in remote if source.remote_path],
                staging,
                chunk_size=self._chunk_size,
            )
        for source in urls:
            await self._exec_or_raise(
                instance,
                f"curl -fsSL -o {shlex.quote(staging + '/' + source.name)} "
                f"{shlex.quote(source.url or '')}",
                operation,
                env=operation.env,
            )

    async def _stage_local(
        self, instance, sources: list[CopySource], staging: str
    ) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            archive = Path(workspace) / "context.tar.gz"
            transfer.pack_entries(
                [
                    (source.local_path, f"./{source.name}")
                    for source in sources
                    if source.local_path
                ],
                archive,
            )
            await transfer.upload_archive(
                instance, archive, staging, chunk_size=self._chunk_size
            )

    async def _stage_pod(
        self,
        reference: str | None,
        *,
        dockerfile,
        context_dir,
        build_args,
        stage_pods,
        created,
    ):
        """Return a pod holding the filesystem ``COPY --from`` refers to.

        ``--from`` names either an earlier stage, which has to be built, or an
        external image, which only has to be started. Both are cached for the
        duration of the build so a Dockerfile copying three files out of one
        builder pays for one pod, not three.
        """
        if reference is None:
            raise BuildError("COPY --from", -1, "missing --from target")
        if reference in stage_pods:
            return stage_pods[reference]

        stage = dockerfile.stage_by_name(reference)
        if stage is not None:
            if reference in self._building:
                raise BuildError(
                    f"COPY --from={reference}",
                    -1,
                    f"build stages form a cycle through {reference!r}",
                )
            image = self.resolve_image(stage.base)
            self._log.info("building stage %r from %s", reference, image)
            pod = await self._create_sandbox(image)
            stage_pods[reference] = pod
            created.append(getattr(pod, "sandbox_id", reference))
            self._building.add(reference)
            try:
                await self.build(
                    pod,
                    dockerfile,
                    context_dir=context_dir,
                    stage=stage,
                    build_args=build_args,
                )
            finally:
                self._building.discard(reference)
            return pod

        image = self.resolve_image(reference)
        self._log.info("starting external image %r for COPY --from", image)
        pod = await self._create_sandbox(image)
        stage_pods[reference] = pod
        created.append(getattr(pod, "sandbox_id", reference))
        return pod

    async def _exec_or_raise(self, instance, command: str, operation, *, env) -> None:
        result = await instance.exec(command, timeout=self._step_timeout, env=env)
        if result.exit_code != 0:
            raise BuildError(
                operation.source,
                result.exit_code if result.exit_code is not None else -1,
                (result.stdout or "") + (result.stderr or ""),
            )

    async def _persist_environment(self, instance, state: StageState) -> None:
        """Leave the Dockerfile's environment where a login shell will find it."""
        body = "# Written by harbor-orchard; mirrors the Dockerfile's ENV.\n"
        body += export_env(state.env) + "\n"
        try:
            # Slim base images often have no /etc/profile.d at all.
            await instance.exec("mkdir -p /etc/profile.d", timeout=60)
            await transfer.upload_bytes(
                instance, body.encode("utf-8"), PROFILE_PATH, chunk_size=self._chunk_size
            )
        except Exception as exc:  # noqa: BLE001 - a read-only /etc is survivable
            self._log.debug("could not write %s: %s", PROFILE_PATH, exc)
