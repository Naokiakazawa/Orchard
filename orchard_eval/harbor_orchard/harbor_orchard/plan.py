"""Turn parsed Dockerfile stages into operations a live sandbox can perform.

A Docker build and a replay-in-a-container differ in one structural way: a build
has *state between layers* that the daemon carries for you — the working
directory, the user, and the environment survive from one instruction to the
next. A sandbox ``exec`` is a fresh process every time and remembers none of it.

So this module interprets everything that is state (``ENV``, ``WORKDIR``,
``USER``, ``ARG``, ``SHELL``) while emitting only the instructions that are
*work* (``RUN``, ``COPY``, ``ADD``). Each emitted operation carries the full
accumulated state with it, which is what makes it independently executable. The
final state is returned too, because the agent and the verifier have to run in
the same environment the build left behind — otherwise a task whose Dockerfile
ends with ``ENV PATH=/opt/vep:$PATH`` runs its tests without the tool on PATH.

Translation is pure: no network, no sandbox, no filesystem writes. It takes the
base image's environment as an argument rather than discovering it, so the whole
layer is testable offline — see ``tests/test_plan.py``.
"""

from __future__ import annotations

import posixpath
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from harbor_orchard.dockerfile import (
    Dockerfile,
    DockerfileError,
    Instruction,
    Stage,
    parse_arg,
    parse_env,
)
from harbor_orchard.expand import expand, expand_all
from harbor_orchard.shell import CONTEXT_ROOT_NAME

#: Docker's default when a Dockerfile declares no ``SHELL``.
DEFAULT_SHELL = ("/bin/sh", "-c")

#: Suffixes ``ADD`` unpacks automatically when the source is a local file.
#: Remote URLs are never unpacked, which is a real asymmetry in Docker and not
#: an oversight here. The unpacking itself is done by
#: :data:`harbor_orchard.shell.EXTRACT_ROUTINE`, which must stay in step.
ARCHIVE_SUFFIXES = (
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz2",
    ".tar.xz",
    ".txz",
    ".tar.zst",
)

#: ``RUN`` flags that change what the command can see. Honouring them is
#: impossible without a builder, and ignoring them changes the result, so they
#: are refused: a task that needs a cache mount must be excluded knowingly.
UNSUPPORTED_RUN_FLAGS = ("mount", "security")


class UnsupportedInstruction(DockerfileError):
    """The Dockerfile uses a feature that cannot be replayed in a sandbox."""


@dataclass(frozen=True)
class CopySource:
    """One resolved input to a ``COPY``/``ADD``.

    Exactly one of the three locations is set. ``name`` is the basename the
    builder stages it under, which is what Docker's copy rules operate on.
    """

    name: str = ""
    local_path: Path | None = None
    remote_path: str | None = None
    url: str | None = None


@dataclass
class RunOp:
    """A command to execute, with the build state it must see."""

    env: dict[str, str]
    cwd: str
    user: str | None
    source: str
    script: str | None = None
    shell: tuple[str, ...] = DEFAULT_SHELL
    #: Set instead of ``script`` for the JSON-array form, which bypasses a shell.
    argv: list[str] | None = None

    def command(self) -> str:
        """Render as a single shell string for the sandbox's ``bash -c``.

        The nesting is deliberate. ``RUN`` in shell form is defined as
        ``$SHELL -c "<text>"``, and a Dockerfile that sets
        ``SHELL ["/bin/bash", "-o", "pipefail", "-c"]`` is relying on that
        wrapper for correctness. Flattening it to a bare ``bash -c`` would drop
        ``pipefail`` and let a failing stage of a pipeline pass.
        """
        if self.argv is not None:
            return shlex.join(self.argv)
        assert self.script is not None
        return f"{shlex.join(self.shell)} {shlex.quote(self.script)}"


@dataclass
class CopyOp:
    """A ``COPY``/``ADD`` staged through a directory, then applied."""

    sources: list[CopySource]
    dest: str
    env: dict[str, str]
    cwd: str
    user: str | None
    source: str
    #: ``--from``: another stage's alias/index, or an external image reference.
    from_stage: str | None = None
    chown: str | None = None
    chmod: str | None = None
    #: ``ADD`` unpacks local archives in place of copying them.
    extract_archives: bool = False


@dataclass
class WriteFileOp:
    """``COPY <<EOF /path`` — a heredoc written straight to a file."""

    content: str
    dest: str
    env: dict[str, str]
    cwd: str
    user: str | None
    source: str
    chown: str | None = None
    chmod: str | None = None


Operation = RunOp | CopyOp | WriteFileOp


@dataclass
class StageState:
    """The build state at some point in a stage.

    This doubles as the container configuration the finished environment has:
    ``exec`` in the built sandbox must default to these values, exactly as
    ``docker exec`` inherits them from the image.
    """

    env: dict[str, str] = field(default_factory=dict)
    workdir: str = "/"
    user: str | None = None
    shell: tuple[str, ...] = DEFAULT_SHELL
    entrypoint: list[str] | None = None
    cmd: list[str] | None = None
    labels: dict[str, str] = field(default_factory=dict)
    exposed_ports: list[str] = field(default_factory=list)
    stopsignal: str | None = None
    volumes: list[str] = field(default_factory=list)

    def copy(self) -> StageState:
        return StageState(
            env=dict(self.env),
            workdir=self.workdir,
            user=self.user,
            shell=self.shell,
            entrypoint=list(self.entrypoint) if self.entrypoint else None,
            cmd=list(self.cmd) if self.cmd else None,
            labels=dict(self.labels),
            exposed_ports=list(self.exposed_ports),
            stopsignal=self.stopsignal,
            volumes=list(self.volumes),
        )


@dataclass
class StagePlan:
    """Everything needed to reproduce one ``FROM`` block."""

    stage: Stage
    base_image: str
    operations: list[Operation]
    state: StageState
    #: Stage aliases/indexes this stage copies from, in dependency order.
    depends_on: list[str] = field(default_factory=list)


def translate_stage(
    dockerfile: Dockerfile,
    stage: Stage,
    *,
    context_dir: Path,
    base_env: dict[str, str],
    base_workdir: str = "/",
    build_args: dict[str, str] | None = None,
) -> StagePlan:
    """Translate one stage against the environment its base image provides.

    Args:
        base_env: The base image's own environment, as ``printenv`` reports it
            in a fresh container. Required, not optional: ``ENV PATH="/x:$PATH"``
            is the single most common Dockerfile idiom, and expanding ``$PATH``
            against an empty scope silently deletes the system PATH and breaks
            every command afterwards.
        base_workdir: The base image's ``WorkingDir``. ``WORKDIR`` is inherited
            across ``FROM``, so a stage that never declares one runs where its
            base image left off.
        build_args: Values supplied for ``ARG``s, overriding their defaults.
    """
    supplied = dict(build_args or {})
    state = StageState(env=dict(base_env), workdir=base_workdir or "/")

    # An ARG declared before the first FROM is in scope for FROM lines only. A
    # stage that wants one must redeclare it, at which point it inherits the
    # global default. Getting this wrong makes `ARG UV_VERSION=0.9.7` either
    # invisible to RUN or visible in stages that never asked for it.
    args: dict[str, str] = {}

    operations: list[Operation] = []
    depends_on: list[str] = []

    for instruction in stage.instructions:
        scope = {**args, **state.env}
        handler = _HANDLERS.get(instruction.name)
        if handler is None:
            raise UnsupportedInstruction(
                f"line {instruction.line}: {instruction.name} cannot be replayed "
                "in a sandbox"
            )
        handler(
            instruction,
            state=state,
            scope=scope,
            args=args,
            supplied=supplied,
            global_defaults=dockerfile.global_args,
            operations=operations,
            depends_on=depends_on,
            context_dir=context_dir,
        )

    base_image = expand(stage.base, {**dockerfile.global_args_scope(), **supplied})
    return StagePlan(
        stage=stage,
        base_image=base_image,
        operations=operations,
        state=state,
        depends_on=depends_on,
    )


def translate(
    dockerfile: Dockerfile,
    *,
    context_dir: Path,
    base_env: dict[str, str],
    base_workdir: str = "/",
    build_args: dict[str, str] | None = None,
) -> StagePlan:
    """Translate the final stage — the one that becomes the sandbox.

    Earlier stages are translated lazily by the builder, and only when the final
    stage actually copies from them: a multi-stage Dockerfile whose builder
    output is unused costs nothing.
    """
    return translate_stage(
        dockerfile,
        dockerfile.final_stage,
        context_dir=context_dir,
        base_env=base_env,
        base_workdir=base_workdir,
        build_args=build_args,
    )


# ---------------------------------------------------------------------------
# Per-instruction handlers
# ---------------------------------------------------------------------------


def _source_label(instruction: Instruction) -> str:
    text = instruction.value.replace("\n", " ")
    if len(text) > 120:
        text = text[:117] + "..."
    return f"Dockerfile:{instruction.line} {instruction.name} {text}".rstrip()


def _handle_run(instruction: Instruction, *, state, scope, args, operations, **_):
    for flag in UNSUPPORTED_RUN_FLAGS:
        if flag in instruction.flags:
            raise UnsupportedInstruction(
                f"line {instruction.line}: RUN --{flag} requires an image builder"
            )
    # ARGs are visible to RUN as environment variables; ENV takes precedence.
    run_env = {**args, **state.env}
    if instruction.json_args is not None:
        operations.append(
            RunOp(
                argv=instruction.json_args,
                env=run_env,
                cwd=state.workdir,
                user=state.user,
                shell=state.shell,
                source=_source_label(instruction),
            )
        )
        return
    operations.append(
        RunOp(
            script=instruction.script,
            shell=state.shell,
            env=run_env,
            cwd=state.workdir,
            user=state.user,
            source=_source_label(instruction),
        )
    )


def _handle_env(instruction: Instruction, *, state, scope, **_):
    # Pairs resolve left to right, so `ENV a=1 b=$a` sees a=1 — BuildKit's
    # behaviour, and the one every Dockerfile author expects.
    running = dict(scope)
    for key, value in parse_env(instruction):
        resolved = expand(value, running)
        state.env[key] = resolved
        running[key] = resolved


def _handle_arg(instruction: Instruction, *, state, scope, args, supplied, global_defaults, **_):
    for name, default in parse_arg(instruction).items():
        if name in supplied:
            args[name] = supplied[name]
        elif default is not None:
            args[name] = expand(default, {**args, **state.env})
        else:
            args[name] = global_defaults.get(name) or ""


def _handle_workdir(instruction: Instruction, *, state, scope, operations, **_):
    value = expand(instruction.value.strip(), scope)
    if not value:
        raise DockerfileError(f"line {instruction.line}: WORKDIR is empty")
    state.workdir = _resolve_path(value, state.workdir)
    # Docker creates a missing WORKDIR. It is created as root and handed to the
    # current USER, so a later unprivileged RUN can still write into it.
    script = f"mkdir -p {shlex.quote(state.workdir)}"
    if state.user:
        script += f" && chown {shlex.quote(state.user)} {shlex.quote(state.workdir)}"
    operations.append(
        RunOp(
            script=script,
            shell=DEFAULT_SHELL,
            env=dict(state.env),
            cwd="/",
            user="root",
            source=_source_label(instruction),
        )
    )


def _handle_user(instruction: Instruction, *, state, scope, **_):
    value = expand(instruction.value.strip(), scope)
    state.user = value or None


def _handle_shell(instruction: Instruction, *, state, **_):
    if instruction.json_args is None:
        raise DockerfileError(
            f"line {instruction.line}: SHELL requires the JSON array form"
        )
    state.shell = tuple(instruction.json_args)


def _handle_copy(
    instruction: Instruction, *, state, scope, operations, depends_on, context_dir, **_
):
    _emit_copy(
        instruction,
        state=state,
        scope=scope,
        operations=operations,
        depends_on=depends_on,
        context_dir=context_dir,
        is_add=False,
    )


def _handle_add(
    instruction: Instruction, *, state, scope, operations, depends_on, context_dir, **_
):
    _emit_copy(
        instruction,
        state=state,
        scope=scope,
        operations=operations,
        depends_on=depends_on,
        context_dir=context_dir,
        is_add=True,
    )


def _handle_entrypoint(instruction: Instruction, *, state, **_):
    state.entrypoint = _exec_form(instruction)


def _handle_cmd(instruction: Instruction, *, state, **_):
    state.cmd = _exec_form(instruction)


def _handle_expose(instruction: Instruction, *, state, scope, **_):
    state.exposed_ports.extend(expand_all(instruction.split_args(), scope))


def _handle_label(instruction: Instruction, *, state, scope, **_):
    for key, value in parse_env(instruction):
        state.labels[key] = expand(value, scope)


def _handle_stopsignal(instruction: Instruction, *, state, scope, **_):
    state.stopsignal = expand(instruction.value.strip(), scope)


def _handle_volume(instruction: Instruction, *, state, scope, operations, **_):
    paths = expand_all(instruction.split_args(), scope)
    state.volumes.extend(paths)
    # No volume driver here, but Docker does create the mount points, and tasks
    # do rely on the directory existing.
    resolved = [_resolve_path(path, state.workdir) for path in paths]
    operations.append(
        RunOp(
            script="mkdir -p " + " ".join(shlex.quote(path) for path in resolved),
            shell=DEFAULT_SHELL,
            env=dict(state.env),
            cwd="/",
            user="root",
            source=_source_label(instruction),
        )
    )


def _handle_noop(instruction: Instruction, **_):
    """``MAINTAINER``/``HEALTHCHECK`` carry no build-time effect here."""


def _handle_onbuild(instruction: Instruction, **_):
    raise UnsupportedInstruction(
        f"line {instruction.line}: ONBUILD has no meaning without an image builder"
    )


_HANDLERS = {
    "ADD": _handle_add,
    "ARG": _handle_arg,
    "CMD": _handle_cmd,
    "COPY": _handle_copy,
    "ENTRYPOINT": _handle_entrypoint,
    "ENV": _handle_env,
    "EXPOSE": _handle_expose,
    "HEALTHCHECK": _handle_noop,
    "LABEL": _handle_label,
    "MAINTAINER": _handle_noop,
    "ONBUILD": _handle_onbuild,
    "RUN": _handle_run,
    "SHELL": _handle_shell,
    "STOPSIGNAL": _handle_stopsignal,
    "USER": _handle_user,
    "VOLUME": _handle_volume,
    "WORKDIR": _handle_workdir,
}


# ---------------------------------------------------------------------------
# COPY / ADD resolution
# ---------------------------------------------------------------------------


def _emit_copy(
    instruction: Instruction,
    *,
    state: StageState,
    scope: dict[str, str],
    operations: list[Operation],
    depends_on: list[str],
    context_dir: Path,
    is_add: bool,
) -> None:
    tokens = expand_all(instruction.split_args(), scope)
    if len(tokens) < 2:
        raise DockerfileError(
            f"line {instruction.line}: {instruction.name} needs a source and a destination"
        )

    *patterns, raw_dest = tokens
    dest = _resolve_dest(raw_dest, state.workdir)
    chown = instruction.flags.get("chown")
    chmod = instruction.flags.get("chmod")

    if instruction.heredocs:
        # ``COPY <<EOF /path`` — the body is the file. Multiple heredocs pair
        # positionally with the sources they replace.
        for (word, body), _pattern in zip(instruction.heredocs, patterns, strict=False):
            target = dest if len(instruction.heredocs) == 1 else posixpath.join(dest, word)
            operations.append(
                WriteFileOp(
                    content=body + "\n",
                    dest=target,
                    env=dict(state.env),
                    cwd=state.workdir,
                    user=state.user,
                    chown=chown,
                    chmod=chmod,
                    source=_source_label(instruction),
                )
            )
        return

    from_stage = instruction.flags.get("from")
    if from_stage:
        if from_stage not in depends_on:
            depends_on.append(from_stage)
        sources = [
            CopySource(
                remote_path=_resolve_path(pattern, "/"),
                name=posixpath.basename(pattern.rstrip("/")),
            )
            for pattern in patterns
        ]
    else:
        sources = _resolve_context_sources(
            context_dir, patterns, instruction, allow_urls=is_add
        )

    operations.append(
        CopyOp(
            sources=sources,
            dest=dest,
            from_stage=from_stage,
            chown=chown,
            chmod=chmod,
            extract_archives=is_add,
            env=dict(state.env),
            cwd=state.workdir,
            user=state.user,
            source=_source_label(instruction),
        )
    )


def _resolve_context_sources(
    context_dir: Path,
    patterns: list[str],
    instruction: Instruction,
    *,
    allow_urls: bool,
) -> list[CopySource]:
    """Expand build-context patterns, refusing anything outside the context."""
    sources: list[CopySource] = []
    for pattern in patterns:
        if _is_url(pattern):
            if not allow_urls:
                raise DockerfileError(
                    f"line {instruction.line}: COPY cannot take a URL; use ADD"
                )
            sources.append(
                CopySource(
                    url=pattern,
                    name=posixpath.basename(pattern.split("?")[0]) or "download",
                )
            )
            continue

        normalized = pattern.strip()
        if normalized.startswith("/"):
            normalized = normalized.lstrip("/")
        if normalized in ("", ".", "./"):
            sources.append(CopySource(local_path=context_dir, name=CONTEXT_ROOT_NAME))
            continue

        if any(character in normalized for character in "*?["):
            matches = sorted(context_dir.glob(normalized))
            if not matches:
                raise DockerfileError(
                    f"line {instruction.line}: {pattern!r} matched no files in the build context"
                )
            for match in matches:
                _reject_escape(context_dir, match, instruction)
                sources.append(CopySource(local_path=match, name=match.name))
            continue

        candidate = context_dir / normalized
        _reject_escape(context_dir, candidate, instruction)
        if not candidate.exists():
            raise DockerfileError(
                f"line {instruction.line}: {pattern!r} does not exist in the build context"
            )
        sources.append(CopySource(local_path=candidate, name=candidate.name))
    return sources


def _reject_escape(context_dir: Path, candidate: Path, instruction: Instruction) -> None:
    try:
        candidate.resolve().relative_to(context_dir.resolve())
    except ValueError:
        raise DockerfileError(
            f"line {instruction.line}: {candidate} is outside the build context"
        ) from None


def _resolve_dest(raw: str, workdir: str) -> str:
    """Resolve a destination, keeping the trailing slash that means "directory"."""
    trailing = raw.endswith("/") and raw not in ("/",)
    resolved = _resolve_path(raw, workdir)
    if trailing and not resolved.endswith("/"):
        resolved += "/"
    return resolved


def _resolve_path(value: str, workdir: str) -> str:
    if value.startswith("/"):
        return posixpath.normpath(value)
    return posixpath.normpath(posixpath.join(workdir or "/", value))


def _is_url(value: str) -> bool:
    return value.startswith(("http://", "https://"))


def _exec_form(instruction: Instruction) -> list[str]:
    if instruction.json_args is not None:
        return list(instruction.json_args)
    # Shell form: Docker prepends the configured SHELL. We only record metadata,
    # so keeping the raw text is enough.
    return ["/bin/sh", "-c", instruction.value]
