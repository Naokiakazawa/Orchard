"""Harnesses for agent CLIs that already live inside the sandbox.

Every Orchard Env sandbox ships ``codex``, ``claude``, ``pi``, ``opencode``,
``hermes`` and ``mini`` on ``PATH`` regardless of the base image, so evaluating
one of them is a single ``exec`` — no install step, no image rebuild, no network
fetch inside the pod. That makes the difference between two of these harnesses
purely declarative, which is what :class:`CliSpec` captures.

The agent is told to edit the repository in place; the patch is then recovered
with ``git diff`` rather than trusted from the agent's own report, because most
CLI agents have no submission protocol. The ones that do
(:attr:`CliSpec.patch_from_submission`) hand in their own diff instead.
"""

from __future__ import annotations

import json
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from orchard_evalkit.config import ModelConfig, anthropic_base_url
from orchard_evalkit.harnesses.base import Harness, RolloutContext, register_harness
from orchard_evalkit.harnesses.trajectory import parse_trajectory
from orchard_evalkit.models import (
    EXIT_AGENT_ERROR,
    EXIT_COMPLETED,
    EXIT_INFRA_ERROR,
    EXIT_TIMEOUT,
    RolloutResult,
)
from orchard_evalkit.router import close_session
from orchard_evalkit.sandbox import is_sandbox_failure

#: Where the rendered prompt is staged inside the sandbox.
PROMPT_PATH = "/tmp/orchard_eval_prompt.txt"
#: Where :attr:`CliSpec.mirror_stdout` CLIs duplicate their stdout inside the sandbox.
STDOUT_MIRROR_PATH = "/tmp/orchard_eval_stdout.log"
#: Shell variable holding the prompt when it is passed as an argument.
PROMPT_VAR = "ORCHARD_EVAL_PROMPT"
#: Sentinel used in :attr:`CliSpec.args` for the prompt position.
PROMPT_TOKEN = "{prompt}"
#: Stdout past which a rollout is warned about. The whole stream is buffered in
#: memory — framed over a WebSocket, then copied again for `agent.log` and the
#: raw trajectory — so a CLI that emits hundreds of MB is on its way to losing
#: the exec connection entirely, and silently: the instance comes back as an
#: empty patch. Well above what a healthy rollout produces (mini-swe-agent is
#: under 1MB, a filtered pi run a few MB) and well under where it breaks.
STDOUT_WARN_BYTES = 50 * 1024**2

#: Seconds between the SIGTERM and the SIGKILL of an in-sandbox deadline, for
#: the CLI to flush whatever it was writing.
AGENT_KILL_GRACE_S = 10

#: Binary that enforces :attr:`InstalledCliHarness.params` ``timeout_margin``
#: inside the sandbox. Probed before use: it is coreutils on every Debian-based
#: task image and busybox on Alpine, but "every" is not "all".
TIMEOUT_BINARY = "timeout"

#: Default instructions given to a CLI agent, which — unlike mini-swe-agent —
#: has no benchmark-specific prompt of its own. Deliberately minimal: it states
#: the task and the boundaries, and leaves the strategy to the agent.
DEFAULT_PROMPT_TEMPLATE = """\
You are working in the software repository at {workdir}.

Solve the following issue by editing the source code in that repository.

<issue>
{problem_statement}
</issue>

Requirements:
- Modify only non-test source files needed to fix the issue.
- Do NOT modify existing tests, and do not weaken them.
- Do not commit anything; leave your changes in the working tree.
- Make the fix general and consistent with the surrounding codebase, not a
  special case that only satisfies the example in the issue.
- When you are done, stop. Your working-tree diff is what gets evaluated.
"""

#: How a dynamic loader refuses to start a binary, across the two libcs. glibc's
#: ld.so says "cannot open shared object file"; musl's says "Error relocating";
#: and when the ``PT_INTERP`` loader is missing altogether ``execve`` returns
#: ENOENT and the shell prints "not found" for a file that plainly exists. All
#: three surface as exit 127, which is why they need naming explicitly.
LOADER_FAILURE_MARKERS = (
    "not found",
    "no such file or directory",
    "error relocating",
    "cannot open shared object file",
    "ld-linux",
    "ld-musl",
    "symbol not found",
    "exec format error",
)


def _looks_like_loader_failure(detail: str) -> bool:
    """True when a failed launch reads like an unloadable executable."""
    text = detail.lower()
    return any(marker in text for marker in LOADER_FAILURE_MARKERS)


@dataclass(frozen=True)
class BinaryProbe:
    """The outcome of looking for a runnable CLI inside the sandbox.

    ``path`` is set only when the CLI both resolved *and* started. When it is
    ``None``, ``unrunnable_at`` distinguishes the two failures that used to
    read identically: the CLI was missing, or it was right there and unable to
    run.
    """

    path: str | None
    install_tried: bool
    unrunnable_at: str = ""
    detail: str = ""


@dataclass(frozen=True)
class CliSpec:
    """Everything that differs between two in-sandbox CLI agents.

    Args:
        binary: Executable name, resolved on the sandbox ``PATH``.
        subcommand: Verbs that must follow ``binary`` before any flag (``codex
            exec``, ``opencode run``). Kept out of ``args`` because a
            subcommand's own flags are rejected when they precede it: ``codex
            --json exec`` fails with "unexpected argument '--json' found".
        args: Argument *groups*. A group is dropped entirely when any of its
            placeholders renders empty, so ``("-m", "{model}")`` simply
            disappears when no model is configured instead of emitting a
            dangling flag.
        prompt_delivery: ``"argv"`` passes the prompt as the final argument via
            a shell variable; ``"stdin"`` pipes the prompt file in. Both read
            the same staged file, so no prompt — however long or however full of
            quotes and backticks — is ever interpolated into a command line.
        trajectory_args: Argument groups that switch the CLI into structured
            event output. Inserted after ``subcommand`` but ahead of ``args`` so
            they reach the right parser and cannot land after a positional
            prompt.
        trajectory_format: Key into :data:`orchard_evalkit.harnesses.trajectory.PARSERS`.
        trajectory_path: File the CLI writes its trajectory to inside the
            sandbox. When set, that file — not stdout — is what gets parsed,
            for CLIs that log prose to the terminal and structured data to disk.
        mirror_stdout: Tee stdout to :data:`STDOUT_MIRROR_PATH` in the sandbox.
            For CLIs that stream their trajectory to stdout and nowhere else: a
            command killed at its timeout returns no stdout at all, so without
            this the whole run is unrecoverable. The pod outlives the killed
            command, so the file can still be read back.
        drop_event_types: Event types discarded *inside the sandbox*, before the
            stream ever crosses the exec WebSocket. For a CLI whose incremental
            events re-send the whole accumulated message instead of a delta,
            stdout is quadratic in message length: pi's ``message_update`` took
            one Qwen3.8-27B rollout to 758MB, and past ~500MB the connection
            dies before the exit frame arrives, costing the whole instance. Only
            for events the trajectory parser already discards — see
            :func:`~orchard_evalkit.harnesses.trajectory.parse_pi_events`, which
            reads the complete message off ``message_end``/``agent_end``.
        api_key_env: Environment variable the CLI reads its credential from.
        base_url_env: Environment variable for an OpenAI-compatible endpoint.
        env: Static environment additions.
        setup_command: Shell run in the sandbox before the agent starts. Used
            for state a CLI requires but will not create itself — codex aborts
            outright when ``CODEX_HOME`` does not already exist.
        install_command: Shell run only when :attr:`binary` is missing from
            ``PATH``, to install the CLI into the pod. This is a fallback for
            sandboxes whose tools image predates the CLI, not the normal path:
            it costs network egress and time on every pod.
        config_path: Config file to stage inside the sandbox before the agent
            starts. Needed by CLIs whose endpoint routing cannot be expressed
            through environment variables alone.
        config_template: Contents of that file, formatted with ``base_url``,
            ``api_key_env``, ``api_key_ref``, ``model``, ``model_alias`` and
            ``wire_api``, plus whatever a harness adds in
            :meth:`InstalledCliHarness._config_substitutions`. Each value
            arrives pre-quoted as a literal, so the template must *not* add
            quotes of its own. Only written when a ``base_url`` is configured.
        model_alias: Stable name the staged config gives the model. Used for
            the CLI's model flag in place of the served id, which for a
            self-hosted model is a filesystem path that a CLI's pattern matcher
            can misread as ``provider/model``. Ignored without a config.
        last_message_path: File the CLI is asked to write its final message to,
            recorded as the rollout's ``submission``.
        prompt_template: Replaces :data:`DEFAULT_PROMPT_TEMPLATE` for a CLI that
            brings its own benchmark prompt and only wants the raw task.
        patch_from_submission: Grade the agent's own submitted diff when it
            produced one, for CLIs with a real submission protocol. The
            working-tree diff is still the fallback.
        login_shell: Launch the CLI from a login shell, for agents that shell
            out through ``/bin/sh`` and therefore inherit — rather than build —
            the image's toolchain environment.
        version_args: Cheapest arguments that make the CLI start, do nothing and
            exit zero. Run before every rollout to prove the payload can
            actually execute in *this* image; see
            :meth:`InstalledCliHarness._ensure_binary`.
    """

    binary: str
    args: tuple[tuple[str, ...], ...]
    subcommand: tuple[str, ...] = ()
    version_args: tuple[str, ...] = ("--version",)
    prompt_delivery: str = "argv"
    trajectory_args: tuple[tuple[str, ...], ...] = ()
    trajectory_format: str = ""
    trajectory_path: str = ""
    mirror_stdout: bool = False
    drop_event_types: tuple[str, ...] = ()
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    setup_command: str = ""
    install_command: str = ""
    config_path: str | None = None
    config_template: str = ""
    model_alias: str = ""
    last_message_path: str | None = None
    prompt_template: str = ""
    patch_from_submission: bool = False
    login_shell: bool = False

    def render_command(
        self,
        substitutions: dict[str, str],
        extra_args: list[str] | None = None,
        *,
        capture_trajectory: bool = True,
        binary: str | None = None,
        agent_timeout: int | None = None,
    ) -> str:
        """Build the shell command that launches the agent.

        ``binary`` overrides :attr:`binary` with an already-resolved path, so
        the launch does not depend on ``PATH`` still being right when the
        command finally runs.

        ``extra_args`` are appended verbatim *before* any stdin redirection, so
        a caller-supplied flag lands where the CLI expects an argument rather
        than trailing after a redirect.

        ``agent_timeout`` bounds the CLI *inside* the sandbox, ahead of the
        deadline the orchestrator enforces on the exec itself. See
        :meth:`InstalledCliHarness._agent_deadline` for why that margin is
        worth paying for.
        """
        groups = list(self.args)
        if capture_trajectory and self.trajectory_args:
            # Ahead of `args`: several of these CLIs take the prompt as a
            # trailing positional, and a flag after it would be read as more
            # prompt. Still after `subcommand`, which is emitted below.
            groups = list(self.trajectory_args) + groups

        tokens: list[str] = [shlex.quote(binary or self.binary)]
        tokens.extend(shlex.quote(verb) for verb in self.subcommand)
        for group in groups:
            rendered: list[str] = []
            drop = False
            for token in group:
                if token == PROMPT_TOKEN:
                    rendered.append(f'"${PROMPT_VAR}"')
                    continue
                value = token.format(**substitutions)
                if not value:
                    drop = True
                    break
                rendered.append(shlex.quote(value))
            if not drop:
                tokens.extend(rendered)
        tokens.extend(str(arg) for arg in (extra_args or []))
        command = " ".join(tokens)

        if agent_timeout:
            # Wraps the CLI alone, not the pipeline: when the deadline fires,
            # the CLI dies, its pipe closes, and `awk`/`tee` drain and flush
            # normally instead of being killed mid-write.
            #
            # `-k N` and a positional duration rather than `--kill-after=Ns`:
            # busybox timeout (Alpine task images) has the short options and
            # not the long ones.
            command = (
                f"{TIMEOUT_BINARY} -k {AGENT_KILL_GRACE_S} {agent_timeout} {command}"
            )

        if self.prompt_delivery == "stdin":
            command = f"{command} < {shlex.quote(PROMPT_PATH)}"

        if self.drop_event_types:
            # Ahead of the tee, so the pod's own mirror stays small too.
            #
            # `[{]` rather than `\{`: a brace opens an interval in ERE and only
            # some awks accept the backslash escape. awk rather than `grep -v`
            # because grep exits 1 when it selects nothing, which pipefail would
            # then report as the agent's exit code. The match is anchored so an
            # event whose *content* quotes one of these type strings survives.
            # fflush keeps the mirror complete when the pipeline is killed at
            # its timeout, which is the only time the mirror is read.
            alternation = "|".join(self.drop_event_types)
            command = (
                f'{command} | awk \'!/^[{{]"type":"({alternation})"/ '
                f"{{ print; fflush() }}'"
            )

        if self.mirror_stdout:
            command = f"{command} | tee {shlex.quote(STDOUT_MIRROR_PATH)}"

        if self.drop_event_types or self.mirror_stdout:
            # pipefail so the exit code stays the CLI's rather than the last
            # stage's. Both filters always exit zero, so what survives is the
            # agent's own status — including the non-zero several of these CLIs
            # return on a hit step limit.
            command = f"set -o pipefail; {command}"

        if self.prompt_delivery == "stdin":
            return command
        # Reading through a shell variable keeps arbitrary prompt content —
        # quotes, newlines, `$(...)` — out of the command's parse tree.
        return (
            f"{PROMPT_VAR}=$(cat {shlex.quote(PROMPT_PATH)}); "
            f"export {PROMPT_VAR}; {command}"
        )


class InstalledCliHarness(Harness):
    """Drive a CLI agent that is already present inside the sandbox.

    Recognized ``harness.params``:

    ``prompt_template``
        Override :data:`DEFAULT_PROMPT_TEMPLATE`. Formatted with
        ``problem_statement``, ``workdir``, ``instance_id`` and ``repo``.
    ``extra_args``
        Extra CLI arguments appended verbatim (already-quoted by you).
    ``env``
        Extra environment variables for the agent process.
    ``require_binary``
        Fail fast when the CLI is missing from ``PATH`` — or present and unable
        to run in this image (default ``True``).
    ``auto_install``
        Install the CLI in the sandbox when it is unusable and the spec knows
        how (default ``True``). See :attr:`CliSpec.install_command`.
    ``install_command``
        Override that install script.
    ``install_timeout``
        Seconds allowed for it (default 900).
    ``version_args``
        Override :attr:`CliSpec.version_args`, the arguments used to prove the
        CLI starts.
    ``verify_timeout``
        Seconds allowed for that probe (default 120).
    ``timeout_margin``
        Seconds to stop the CLI *before* ``harness.timeout``, enforced by a
        ``timeout`` inside the sandbox. Unset leaves the deadline entirely to
        the orchestrator, where a rollout that spends its whole budget comes
        back with no stdout, no trajectory and no diff — see
        :meth:`InstalledCliHarness._agent_deadline`.
    ``wire_api``
        Protocol the CLI's staged provider config declares, for CLIs that have
        one — ``"responses"`` or ``"chat"`` (default :data:`DEFAULT_WIRE_API`).
        Also settable as ``orchard-eval run --wire-api chat``.
    """

    spec: ClassVar[CliSpec]

    async def rollout(self, ctx: RolloutContext) -> RolloutResult:
        spec = self.spec
        started = time.monotonic()
        capture = bool(self.params.get("capture_trajectory", True))
        # Validated here rather than at config-render time so a typo costs
        # nothing: this runs before the sandbox does any work.
        wire_api = self._wire_api() if "{wire_api}" in spec.config_template else ""
        model = self._model_for(ctx)
        binary = spec.binary

        if self.params.get("require_binary", True):
            probe = await self._ensure_binary(ctx)
            if probe.path is None:
                # Stays agent_error rather than infra_error: an image whose libc
                # cannot load the payload will do exactly this on every fresh
                # pod, so retrying the rollout only spends more of them.
                return RolloutResult(
                    exit_status=EXIT_AGENT_ERROR,
                    error=self._unusable_binary_error(ctx, probe),
                    metrics={"harness": self.name, "image": ctx.instance.image},
                )
            # The probe runs a plain shell, but the agent may be launched from a
            # login one — and Debian's /etc/profile *assigns* PATH, dropping the
            # tools directory the probe just found the CLI in. Exit 127.
            if probe.path.startswith("/"):
                binary = probe.path

        if spec.setup_command:
            await ctx.sandbox.exec(spec.setup_command, timeout=60)

        config = self._render_config(model)
        if config is not None and spec.config_path:
            ctx.save_artifact(f"agent_config{Path(spec.config_path).suffix}", config)
            await ctx.sandbox.write_file(config, spec.config_path)

        prompt = self._render_prompt(ctx)
        ctx.save_artifact("prompt.txt", prompt)
        await ctx.sandbox.write_file(prompt, PROMPT_PATH)

        agent_deadline = await self._resolve_agent_deadline(ctx)
        command = self._build_command(
            ctx,
            capture_trajectory=capture,
            binary=binary,
            agent_timeout=agent_deadline,
        )
        env = self._build_env(model)

        ctx.logger.info(
            "[%s] running %s (timeout=%ss, endpoint=%s)",
            ctx.instance.instance_id,
            spec.binary,
            self.timeout,
            model.base_url or "<cli default>",
        )

        error: str | None = None
        exit_status = EXIT_COMPLETED
        exec_started = time.monotonic()
        try:
            result = await ctx.sandbox.exec(
                command,
                timeout=self.timeout,
                cwd=ctx.workdir,
                env=env,
                login_shell=bool(self.params.get("login_shell", spec.login_shell)),
            )
            stdout, stderr, exit_code = result.stdout, result.stderr, result.exit_code
        except TimeoutError as exc:
            stdout, stderr, exit_code = "", str(exc), -1
            exit_status = EXIT_TIMEOUT
            error = f"agent timed out after {self.timeout}s"
        except Exception as exc:  # noqa: BLE001 - a crashed agent is a result
            stdout, stderr, exit_code = "", str(exc), -1
            # A dead or unreachable pod is the cluster's failure, and the runner
            # retries it on a fresh one. A crashed agent is a result and is not.
            exit_status = (
                EXIT_INFRA_ERROR if is_sandbox_failure(exc) else EXIT_AGENT_ERROR
            )
            error = f"{type(exc).__name__}: {exc}"
        exec_elapsed = time.monotonic() - exec_started
        # The agent loop is over at this point — everything below reads files
        # and diffs — so the engine holding this conversation gets its slot
        # back now rather than after grading. A no-op unless the model is
        # reached through scripts/session_router.py.
        await close_session(model.base_url, logger=ctx.logger)

        if len(stdout) > STDOUT_WARN_BYTES:
            # Loud, because the damage is otherwise invisible: an exec stream
            # that dies under its own weight is recorded as an agent error with
            # an empty patch, which reads as a weak model rather than a broken
            # harness.
            ctx.logger.warning(
                "[%s] %s produced %.1fMB of stdout — the exec stream buffers "
                "all of it in memory and drops the connection somewhere past "
                "500MB. If its incremental events re-send the whole message "
                "rather than a delta, filter them with CliSpec.drop_event_types.",
                ctx.instance.instance_id,
                spec.binary,
                len(stdout) / 1024**2,
            )

        ctx.save_artifact("agent.log", f"$ {command}\n\n{stdout}\n{stderr}")

        raw_trajectory = stdout
        if capture and spec.trajectory_path:
            raw_trajectory = await self._read_trajectory_file(ctx)
        elif spec.mirror_stdout and not stdout.strip():
            # A command killed at its timeout comes back with empty stdout, so
            # the mirror is the only surviving record of what the agent did.
            raw_trajectory = await self._read_mirrored_stdout(ctx)
            if raw_trajectory:
                stdout = raw_trajectory
                ctx.logger.info(
                    "[%s] recovered %d chars of stdout from %s",
                    ctx.instance.instance_id,
                    len(raw_trajectory),
                    STDOUT_MIRROR_PATH,
                )
                ctx.save_artifact("agent.log", f"$ {command}\n\n{stdout}\n{stderr}")
        self._dump_raw(ctx, raw_trajectory, stderr, capture)

        # A non-zero exit is worth recording but is NOT disqualifying: several
        # CLIs exit non-zero on a hit step limit while still leaving a good diff.
        if exit_status == EXIT_COMPLETED and exit_code != 0:
            exit_status, classified = self._classify_nonzero_exit(
                exit_code, exec_elapsed, len(stdout)
            )
            error = error or classified

        messages, traj_info = self._parse_trajectory(ctx, raw_trajectory, capture)
        # A completion the provider cut off at its output-token cap is not a
        # finished rollout. pi ends its whole session on the first one rather
        # than re-prompting the model to continue, then exits 0 — leaving the
        # working tree wherever the agent had got to, which is usually
        # untouched. Recorded as `completed`, that is indistinguishable from an
        # agent that looked around and decided the code was already correct; on
        # SWE-bench Verified it silently accounted for 89 of 500 instances.
        # Naming it here costs nothing and makes the failure countable.
        if exit_status == EXIT_COMPLETED and traj_info.get("stop_reason") == "length":
            exit_status = EXIT_AGENT_ERROR
            error = error or (
                f"{spec.binary} ended on a truncated completion: the model hit "
                f"its output-token limit mid-message "
                f"({traj_info.get('length_stops') or 1}x this rollout) and the "
                f"CLI stopped instead of continuing"
            )
        submission = str(traj_info.get("submission") or "")
        if not submission:
            submission = await self._read_last_message(ctx)
        patch = submission if spec.patch_from_submission else ""
        if not patch.strip():
            try:
                patch = await ctx.sandbox.extract_patch(ctx.instance.base_commit)
            except Exception as exc:  # noqa: BLE001 - classified, not swallowed
                if not is_sandbox_failure(exc):
                    raise
                # The pod died between the agent finishing and the diff being
                # read. The rollout is unscoreable, but everything already
                # parsed out of the event stream is still worth keeping.
                ctx.logger.error(
                    "[%s] patch extraction lost: %s", ctx.instance.instance_id, exc
                )
                # Unless the rollout already timed out: the sandbox stops
                # answering at the moment the deadline is enforced, so an
                # unreadable diff there is the timeout's consequence. Calling
                # it infra would send the instance back for two more full-length
                # attempts that end the same way.
                if exit_status != EXIT_TIMEOUT:
                    exit_status = EXIT_INFRA_ERROR
                error = error or f"{type(exc).__name__}: {exc}"

        metrics: dict[str, Any] = {
            "harness": self.name,
            "binary": spec.binary,
            # Which image ran is not otherwise recoverable from a finished run,
            # and image-specific faults are exactly what needs it.
            "image": ctx.instance.image,
            "wire_api": wire_api,
            "base_url": model.base_url or "",
            "exit_code": exit_code,
            "agent_run_time": round(time.monotonic() - started, 3),
            # The exec alone, against the deadline it was given — the pair that
            # separates a spent budget from a dropped connection.
            "exec_time": round(exec_elapsed, 3),
            "exec_timeout": self.timeout,
            "agent_deadline": agent_deadline or 0,
            "stdout_chars": len(stdout),
            # Why the last completion ended, and how many of this rollout's
            # completions the output cap truncated. `summarize()` totals the
            # latter, because a run losing rollouts to truncation looks exactly
            # like a weak model until someone counts them.
            "stop_reason": str(traj_info.get("stop_reason") or ""),
            "length_stops": int(traj_info.get("length_stops") or 0),
            "turns": sum(1 for m in messages if m.get("role") == "assistant"),
            "tool_calls": sum(1 for m in messages if m.get("role") == "tool"),
        }
        # Token usage and cost land in metrics so `summarize()` can total them.
        usage = traj_info.get("usage")
        if isinstance(usage, dict):
            metrics["usage"] = usage
        cost = traj_info.get("cost")
        if isinstance(cost, int | float):
            metrics["cost"] = float(cost)

        return RolloutResult(
            submission=submission,
            patch=patch,
            exit_status=exit_status,
            messages=messages,
            trajectory_extra={"format": spec.trajectory_format, **traj_info},
            stdout=stdout,
            stderr=stderr,
            error=error,
            metrics=metrics,
        )

    # ------------------------------------------------------------------

    async def _ensure_binary(self, ctx: RolloutContext) -> BinaryProbe:
        """Resolve a *runnable* CLI on the sandbox ``PATH``, installing if allowed.

        Resolving is not enough. The sandbox tools are dynamically linked ELFs
        mounted into an arbitrary task image, so on a musl (Alpine) sandbox the
        wrapper is found and then cannot start — which the shell reports as a
        bare exit 127 once the rollout is already under way. So every resolved
        path is executed here, and a path that will not run is treated exactly
        like a missing one: the install fallback gets its turn, and what comes
        back names the real fault instead of "not on PATH".
        """
        resolved = await ctx.sandbox.probe_harness(self.spec.binary)
        unrunnable, detail = "", ""
        if resolved is not None:
            usable, detail = await self._is_runnable(ctx, resolved)
            if usable:
                return BinaryProbe(path=resolved, install_tried=False)
            unrunnable = resolved
            ctx.logger.warning(
                "[%s] %s resolves to %s but will not execute: %s",
                ctx.instance.instance_id,
                self.spec.binary,
                resolved,
                detail or "(no output)",
            )

        command = self._install_command()
        if not command or not self.params.get("auto_install", True):
            return BinaryProbe(None, False, unrunnable_at=unrunnable, detail=detail)

        ctx.logger.warning(
            "[%s] %s is not usable in the sandbox image; installing it in the pod",
            ctx.instance.instance_id,
            self.spec.binary,
        )
        try:
            result = await ctx.sandbox.exec(
                command,
                timeout=int(self.params.get("install_timeout", 900)),
                merge_stderr=True,
            )
            log = result.output
        except Exception as exc:  # noqa: BLE001 - reported through the probe below
            log = f"{type(exc).__name__}: {exc}"
        ctx.save_artifact("install.log", f"$ {command}\n\n{log}")

        installed = await ctx.sandbox.probe_harness(self.spec.binary)
        if installed is None:
            return BinaryProbe(None, True, unrunnable_at=unrunnable, detail=detail)
        # An install that lands on top of an unrunnable payload has to clear the
        # same bar, or the rollout dies at launch exactly as before.
        usable, install_detail = await self._is_runnable(ctx, installed)
        if usable:
            return BinaryProbe(path=installed, install_tried=True)
        return BinaryProbe(
            None, True, unrunnable_at=installed, detail=install_detail or detail
        )

    async def _is_runnable(
        self, ctx: RolloutContext, path: str
    ) -> tuple[bool, str]:
        """Return ``(usable, first line of output)`` for a resolved CLI."""
        args = self.params.get("version_args") or self.spec.version_args
        # A YAML `version_args: --version` is a string, and iterating one would
        # send the CLI eleven single-character flags.
        if isinstance(args, str):
            args = [args]
        try:
            probe = await ctx.sandbox.verify_harness(
                path,
                [str(arg) for arg in args],
                timeout=int(self.params.get("verify_timeout", 120)),
            )
        except Exception as exc:  # noqa: BLE001 - a dead pod is classified upstream
            if is_sandbox_failure(exc):
                raise
            return False, f"{type(exc).__name__}: {exc}"
        if probe.succeeded:
            return True, ""
        text = (probe.stderr or probe.stdout or "").strip()
        first = text.splitlines()[0][:300] if text else ""
        return False, f"exit={probe.exit_code} {first}".strip()

    def _unusable_binary_error(self, ctx: RolloutContext, probe: BinaryProbe) -> str:
        """Explain a CLI that is present or absent but in either case unusable."""
        binary = self.spec.binary
        if probe.unrunnable_at:
            image = ctx.instance.image or "<unknown image>"
            hint = (
                " The sandbox-tools payload is a dynamically linked glibc ELF "
                "and this image is musl/Alpine, which has no glibc loader."
                if _looks_like_loader_failure(probe.detail)
                else ""
            )
            extra = (
                " Installing it in the pod did not produce a working one either "
                "— see install.log."
                if probe.install_tried
                else ""
            )
            return (
                f"{binary!r} resolves to {probe.unrunnable_at} but cannot execute "
                f"in this image ({image}).{hint} Probe said: "
                f"{probe.detail or '(no output)'}.{extra}"
            )
        hint = (
            " Installing it in the pod also failed — see install.log; "
            "that needs egress to PyPI (sandbox.block_network: false)."
            if probe.install_tried
            else ""
        )
        return (
            f"{binary!r} is not on PATH in the sandbox. Sandbox tools may be "
            f"disabled on this orchestrator (ENABLE_SANDBOX_TOOLS).{hint}"
        )

    def _install_command(self) -> str:
        return str(self.params.get("install_command") or self.spec.install_command)

    def _dump_raw(
        self, ctx: RolloutContext, raw_trajectory: str, stderr: str, capture: bool
    ) -> None:
        """Persist the untouched streams alongside the normalized trajectory.

        ``agent.log`` interleaves both streams behind the command line, which is
        readable but not machine-readable. These two files are byte-for-byte
        what the CLI emitted, so a rollout can be re-parsed after a parser fix
        and — more often — so a failed exchange with the model server or the
        sandbox can be diagnosed from the artifacts alone: the transport errors
        the CLI reports (401s, a missing ``/v1/responses`` route, a dropped
        connection) surface on stderr and never reach the event stream.
        """
        if not self.params.get("dump_raw_trajectory", True):
            return
        if capture and self.spec.trajectory_format:
            suffix = Path(self.spec.trajectory_path).suffix or ".jsonl"
            ctx.save_artifact(f"trajectory.raw{suffix}", raw_trajectory)
        if stderr:
            ctx.save_artifact("agent.stderr.log", stderr)

    def _classify_nonzero_exit(
        self, exit_code: int, elapsed: float, stdout_len: int
    ) -> tuple[str, str]:
        """Name a non-zero exit, and say whether a fresh pod could do better.

        ``-1`` is not a status any agent returned: the sandbox client reports
        it when the exec stream ended without ever delivering an exit frame.
        Two unrelated things produce it, and telling them apart decides whether
        retrying the instance is worth another half hour.

        A stream that dies early lost its transport, and a fresh pod is the
        obvious next move. A stream that dies having spent the *entire* budget
        hit the deadline instead, and the next pod will spend the same half
        hour reaching the same wall. On one SWE-bench Verified run that was 19
        instances x 3 attempts = 28 pod-hours, all of it filed as
        infrastructure failure. Calling it a timeout is what stops the retry,
        because the runner only retries infra errors.
        """
        if exit_code != -1:
            return EXIT_AGENT_ERROR, f"{self.spec.binary} exited with code {exit_code}"
        if elapsed >= self.timeout:
            return EXIT_TIMEOUT, (
                f"{self.spec.binary} hit its {self.timeout}s deadline and the "
                f"exec stream was dropped without an exit frame, discarding "
                f"{stdout_len} bytes of stdout"
            )
        return EXIT_AGENT_ERROR, (
            f"{self.spec.binary} produced no exit status — the exec stream "
            f"ended after {stdout_len} bytes of stdout without an exit frame"
        )

    async def _read_trajectory_file(self, ctx: RolloutContext) -> str:
        """Fetch a trajectory the CLI wrote to disk inside the sandbox."""
        try:
            return await ctx.sandbox.read_file(self.spec.trajectory_path)
        except Exception as exc:  # noqa: BLE001 - a lost trajectory is not a lost run
            ctx.logger.warning(
                "[%s] could not read %s from the sandbox: %s",
                ctx.instance.instance_id,
                self.spec.trajectory_path,
                exc,
            )
            return ""

    async def _read_mirrored_stdout(self, ctx: RolloutContext) -> str:
        """Fetch the stdout mirror left behind by a killed or crashed CLI."""
        try:
            return await ctx.sandbox.read_file(STDOUT_MIRROR_PATH)
        except Exception:  # noqa: BLE001 - absent whenever the agent never started
            return ""

    def _parse_trajectory(
        self, ctx: RolloutContext, stdout: str, capture: bool
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Normalize the CLI's event stream, without ever failing the rollout."""
        if not capture or not self.spec.trajectory_format:
            return [], {}

        messages, info = parse_trajectory(self.spec.trajectory_format, stdout)
        if not messages and stdout.strip():
            # The agent produced output but none of it parsed. Worth a warning:
            # the run is still valid and the raw stream is retained, but the
            # trajectory is unusable for distillation until the parser is fixed.
            ctx.logger.warning(
                "[%s] %s produced no parseable trajectory events (%d chars of "
                "stdout, %s skipped lines) — raw stream retained",
                ctx.instance.instance_id,
                self.spec.binary,
                len(stdout),
                info.get("skipped_lines", "?"),
            )
        return messages, info

    def _model_for(self, ctx: RolloutContext) -> ModelConfig:
        """The one endpoint this attempt is allowed to use.

        A rollout is a single agent loop, so keeping all of its turns on one
        server is what lets that server's prefix KV cache survive between them.
        """
        return self.model.for_key(ctx.instance.instance_id)

    def _render_config(self, model: ModelConfig | None = None) -> str | None:
        """Render the CLI's own config file, for CLIs that need one.

        Skipped without a ``base_url``, which is the only thing these configs
        exist to express: with no endpoint override the CLI's own defaults are
        already correct.
        """
        spec = self.spec
        model = model or self.model
        if not spec.config_path or not spec.config_template:
            return None
        if not model.base_url:
            return None
        return spec.config_template.format(**self._config_substitutions(model))

    def _config_substitutions(self, model: ModelConfig) -> dict[str, str]:
        """Values :attr:`CliSpec.config_template` is rendered with.

        Every string value is pre-quoted as a literal — JSON string literals
        are valid TOML basic strings, and escaping here keeps a URL containing
        quotes from breaking the file — so a template must not add quotes of
        its own. Subclasses extend this with settings only their CLI has.
        """
        spec = self.spec
        return {
            "base_url": json.dumps(model.base_url),
            "api_key_env": json.dumps(spec.api_key_env),
            # `$VAR` indirection, for configs that interpolate the environment
            # themselves rather than naming the variable to read.
            "api_key_ref": json.dumps(f"${spec.api_key_env}"),
            "model": json.dumps(model.name or ""),
            "model_alias": json.dumps(spec.model_alias),
            "wire_api": json.dumps(self._wire_api()),
        }

    def _wire_api(self) -> str:
        """Protocol the staged provider config declares.

        Rejected up front rather than passed through: the CLI would otherwise
        fail at startup on every instance of the run with an error that names
        the config file instead of the setting the user actually typed.
        """
        wire_api = str(self.params.get("wire_api") or DEFAULT_WIRE_API)
        if wire_api not in WIRE_APIS:
            raise ValueError(
                f"harness.params.wire_api must be one of {', '.join(WIRE_APIS)}; "
                f"got {wire_api!r}"
            )
        return wire_api

    def _model_ref(self, model: ModelConfig | None = None) -> str:
        """What to pass to the CLI's model flag."""
        model = model or self.model
        if self.spec.model_alias and self._render_config(model) is not None:
            return self.spec.model_alias
        return model.name or ""

    def _render_prompt(self, ctx: RolloutContext) -> str:
        template = (
            self.params.get("prompt_template")
            or self.spec.prompt_template
            or DEFAULT_PROMPT_TEMPLATE
        )
        return template.format(
            problem_statement=ctx.task,
            workdir=ctx.workdir,
            instance_id=ctx.instance.instance_id,
            repo=ctx.instance.repo,
        )

    def _substitutions(self, ctx: RolloutContext) -> dict[str, str]:
        """Values available to ``{placeholder}`` tokens in :attr:`CliSpec.args`."""
        spec = self.spec
        model = self._model_for(ctx)
        return {
            "model": model.name or "",
            "model_ref": self._model_ref(model),
            "workdir": ctx.workdir,
            "instance_id": ctx.instance.instance_id,
            "last_message": spec.last_message_path or "",
            "trajectory": spec.trajectory_path,
            "timeout": str(self.timeout),
        }

    def _extra_args(self) -> list[str]:
        """Arguments appended verbatim, already shell-quoted by the caller."""
        return [str(arg) for arg in (self.params.get("extra_args") or [])]

    def _build_command(
        self,
        ctx: RolloutContext,
        *,
        capture_trajectory: bool = True,
        binary: str | None = None,
        agent_timeout: int | None = None,
    ) -> str:
        return self.spec.render_command(
            self._substitutions(ctx),
            extra_args=self._extra_args(),
            capture_trajectory=capture_trajectory,
            binary=binary,
            agent_timeout=agent_timeout,
        )

    def _agent_deadline(self) -> int | None:
        """Seconds the CLI gets, or ``None`` to leave it to the exec deadline.

        The orchestrator enforces ``harness.timeout`` on the exec by running
        the command under its own ``timeout`` and racing that with a Python
        deadline ten seconds later. Ten seconds is not always enough for a
        killed pipeline to tear down, and when the Python deadline wins the
        exec stream is abandoned rather than drained: the rollout comes back
        with a null exit code and *zero* bytes of stdout, having thrown away
        everything the agent wrote. The stdout mirror cannot rescue it either,
        because the sandbox stops answering at the same moment.

        On SWE-bench Verified that lost 18 of 500 pi rollouts outright — and 17
        of 500 mini-swe-agent ones, so it is the deadline that is fragile, not
        any one CLI. Rollouts that ended in the 100 seconds *before* the
        deadline came back intact, which is the whole idea here: stopping the
        agent early enough that the exec returns through its ordinary path
        leaves the transcript and the diff readable.
        """
        margin = self.params.get("timeout_margin")
        if margin is None:
            return None
        try:
            margin = int(margin)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"harness.params.timeout_margin must be an integer; got {margin!r}"
            ) from exc
        if margin <= 0:
            raise ValueError(
                f"harness.params.timeout_margin must be positive; got {margin}"
            )
        if margin >= self.timeout:
            raise ValueError(
                f"harness.params.timeout_margin ({margin}s) must leave the agent "
                f"some of harness.timeout ({self.timeout}s)"
            )
        return self.timeout - margin

    async def _resolve_agent_deadline(self, ctx: RolloutContext) -> int | None:
        """Validate the configured margin and confirm the sandbox can enforce it.

        A missing ``timeout`` binary would otherwise turn every rollout of the
        run into an immediate exit 127, which is a far worse failure than the
        one the margin exists to prevent — so it degrades to the orchestrator's
        deadline instead, loudly.
        """
        deadline = self._agent_deadline()
        if deadline is None:
            return None
        if await ctx.sandbox.probe_harness(TIMEOUT_BINARY) is None:
            ctx.logger.warning(
                "[%s] harness.params.timeout_margin is set but %r is not on the "
                "sandbox PATH; falling back to the exec deadline, where a "
                "rollout that runs the full %ss loses its stdout and its diff",
                ctx.instance.instance_id,
                TIMEOUT_BINARY,
                self.timeout,
            )
            return None
        return deadline

    def _build_env(self, model: ModelConfig | None = None) -> dict[str, str]:
        spec = self.spec
        model = model or self.model
        env: dict[str, str] = dict(spec.env)
        api_key = model.resolve_api_key()
        if api_key:
            env[spec.api_key_env] = api_key
        if model.base_url and spec.base_url_env:
            env[spec.base_url_env] = model.base_url
        env.update({k: str(v) for k, v in (self.params.get("env") or {}).items()})
        return env

    async def _read_last_message(self, ctx: RolloutContext) -> str:
        if not self.spec.last_message_path:
            return ""
        try:
            return await ctx.sandbox.read_file(self.spec.last_message_path)
        except Exception:  # noqa: BLE001 - the file is optional by design
            return ""


#: Codex declines to create its PATH helper binaries when CODEX_HOME sits under
#: the system temp dir ("Refusing to create helper binaries under temporary
#: dir"), so this deliberately is not /tmp.
CODEX_HOME = "/var/tmp/orchard-codex"

#: Protocols a staged provider config may declare.
WIRE_APIS = ("responses", "chat")

#: Protocol declared in a staged provider config. "responses" is what current
#: codex expects — codex#10157 turned "chat" into a startup error, so anything
#: but a pinned older build needs this. Fall back with `--wire-api chat` (or
#: `harness.params.wire_api=chat`) when the server has no working /v1/responses
#: route; older vLLM and SGLang builds serve only /chat/completions.
DEFAULT_WIRE_API = "responses"

#: Codex reads OPENAI_BASE_URL only for its built-in provider, and that provider
#: talks to api.openai.com regardless. Reaching a self-hosted OpenAI-compatible
#: server therefore requires a declared provider, which exists only in the
#: config file — without it every rollout dies on a 401 from api.openai.com.
#:
#: The feature switches below all exist for one reason: codex groups some of its
#: built-in tools into Responses API `{"type": "namespace"}` entries, which only
#: OpenAI's own endpoint accepts. vLLM and SGLang reject the whole request body
#: ("unknown variant `namespace`, expected one of `function`, ..."), so the very
#: first turn of every rollout fails. Each of these is on by default and each
#: contributes a namespace tool: multi-agent (`multi_agent_v1`), web search —
#: which a full-access sandbox promotes to "live" unless disabled — apps,
#: memories and the remote plugin catalog. With them off, codex sends only
#: plain `function` tools.
#:
#: The reasoning-summary switches are the same class of problem one turn later.
#: Codex replays its own reasoning items as `summary: [{"type":
#: "summary_text"}]`, and a server that models reasoning content as
#: `reasoning_text` only rejects the entire input array on turn 2 — after the
#: model has already done a turn's worth of work. They are rendered from
#: `harness.params` rather than hardcoded here; see
#: :meth:`CodexHarness._reasoning_block`.
CODEX_CONFIG_TEMPLATE = """\
model_provider = "orchard"
web_search = "disabled"
{reasoning}

[model_providers.orchard]
name = "orchard"
base_url = {base_url}
env_key = {api_key_env}
wire_api = {wire_api}

[features]
multi_agent = false
apps = false
memories = false
remote_plugin = false

[agents]
enabled = false

[tools]
web_search = false
"""


#: Values codex accepts for `model_reasoning_effort`.
REASONING_EFFORTS = ("minimal", "low", "medium", "high")


@register_harness
class CodexHarness(InstalledCliHarness):
    """OpenAI Codex CLI, run non-interactively inside the sandbox.

    Recognized ``harness.params``, on top of :class:`InstalledCliHarness`'s:

    ``reasoning_effort``
        Written to the staged config as ``model_reasoning_effort``. Empty
        (the default) leaves codex at its own, which is what every run before
        this parameter existed measured. Empty is also the *highest* setting
        against a Qwen3.8-27B fleet, where unset means codex sends
        ``"reasoning": {}`` and the chat template applies its own
        ``default('xhigh')``; ``high`` is not one of the three levels that
        template accepts and 4xxs every request.

        Harbor's codex agent passes ``-c model_reasoning_effort=high`` by
        default, and on SWE-bench Pro that was once the largest difference
        between the two paths: the same model, same fleet and same pods scored
        34.47% through ``orchard-eval run`` and 43.09% through Harbor, with
        2,394 reasoning tokens per rollout here against Harbor's 6,050. pi and
        mini-swe-agent, which have no such asymmetry, agree to within 3 points.
        ``scripts/run_all_evals.sh`` now closes it from the other side, with
        ``--ak reasoning_effort=null`` on the Harbor stages, which drops the
        flag and puts both arms on the template default.

    ``reasoning_summaries``
        Value of ``model_supports_reasoning_summaries`` (default ``False``).
        It is off because codex replaying its own summaries as ``summary:
        [{"type": "summary_text"}]`` breaks turn 2 against a server that
        models reasoning as ``reasoning_text``. Off does *not* suppress
        ``reasoning_effort``: the payload's codex build sends ``"reasoning":
        {"effort": ...}`` either way, captured 2026-09-14 by
        ``scripts/capture_codex_request.py --reasoning-effort high``. Leave it
        off unless that capture says otherwise on a newer build.

    ``reasoning_summary``
        Value of ``model_reasoning_summary`` (default ``"none"``): what codex
        asks the server to send back, as opposed to whether it asks at all.
    """

    name = "codex"
    description = "OpenAI Codex CLI (`codex exec`), running inside the sandbox"
    spec = CliSpec(
        binary="codex",
        subcommand=("exec",),
        args=(
            # The pod IS the sandbox; Codex's own seccomp layer would only add
            # a second, redundant jail around an already-isolated container.
            ("--dangerously-bypass-approvals-and-sandbox",),
            ("--skip-git-repo-check",),
            ("-C", "{workdir}"),
            ("-m", "{model}"),
            ("-o", "{last_message}"),
            # `-` makes Codex read the prompt from stdin.
            ("-",),
        ),
        prompt_delivery="stdin",
        # Without this, stdout is only the final message and progress goes to
        # stderr as prose — there is no trajectory to recover.
        trajectory_args=(("--json",),),
        trajectory_format="codex",
        api_key_env="OPENAI_API_KEY",
        base_url_env="OPENAI_BASE_URL",
        env={
            "CODEX_HOME": CODEX_HOME,
            "RUST_LOG": "error",
            # Overridden by the real credential when there is one. Codex aborts
            # when `env_key` resolves to nothing, and a self-hosted server
            # started without --api-key accepts any value.
            "OPENAI_API_KEY": "EMPTY",
        },
        # Codex refuses to start when CODEX_HOME does not already exist
        # ("Error finding codex home"), and will not create it itself.
        setup_command=f"mkdir -p {CODEX_HOME}",
        config_path=f"{CODEX_HOME}/config.toml",
        config_template=CODEX_CONFIG_TEMPLATE,
        last_message_path="/tmp/orchard_eval_last_message.txt",
    )

    def _config_substitutions(self, model: ModelConfig) -> dict[str, str]:
        subs = super()._config_substitutions(model)
        subs["reasoning"] = self._reasoning_block()
        return subs

    def _reasoning_block(self) -> str:
        """The staged config's reasoning settings, as TOML lines.

        Written into the config rather than passed as ``-c key=value``, which
        is the same thing to codex but not to this harness: codex reads its
        prompt from a trailing ``-``, and :meth:`CliSpec.render_command`
        appends extra arguments after it.

        Rejected here rather than passed through, because codex validates the
        value at startup and would fail every instance of the run with an error
        that names the config file instead of the setting the caller typed.
        """
        effort = str(self.params.get("reasoning_effort") or "").strip().lower()
        if effort and effort not in REASONING_EFFORTS:
            raise ValueError(
                "harness.params.reasoning_effort must be one of "
                f"{', '.join(REASONING_EFFORTS)}; got {effort!r}"
            )
        summaries = bool(self.params.get("reasoning_summaries", False))
        summary = str(self.params.get("reasoning_summary") or "none")
        lines = [
            f"model_supports_reasoning_summaries = {str(summaries).lower()}",
            f"model_reasoning_summary = {json.dumps(summary)}",
        ]
        if effort:
            lines.insert(0, f"model_reasoning_effort = {json.dumps(effort)}")
        return "\n".join(lines)


#: Config directory for pi. Overrides the default ~/.pi/agent so the harness
#: never depends on the sandbox image having a writable HOME.
PI_HOME = "/var/tmp/orchard-pi"

#: Name pi's model picker matches on. The served id is a filesystem path, and
#: pi reads `--model` patterns as an optional `provider/id`, so the raw path is
#: not a safe thing to pass on the command line.
PI_MODEL_ALIAS = "orchard-model"

#: pi validates `--model` against its own catalog and exits with "Model ... not
#: found" for anything absent from it, so a self-hosted endpoint has to be
#: declared as a custom provider. The compat flags are the ones pi documents for
#: vLLM/SGLang-class servers, which reject the `developer` role and
#: `reasoning_effort` that pi otherwise sends to reasoning-capable models.
#:
#: ``{model_limits}`` is the optional per-model limit fragment rendered by
#: :meth:`PiHarness._model_limits`, including its own leading comma.
PI_CONFIG_TEMPLATE = """\
{{
  "providers": {{
    "orchard": {{
      "baseUrl": {base_url},
      "api": "openai-completions",
      "apiKey": {api_key_ref},
      "compat": {{
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false
      }},
      "models": [
        {{
          "id": {model},
          "name": {model_alias}{model_limits}
        }}
      ]
    }}
  }}
}}
"""

#: ``harness.params`` name -> models.json field, for the per-model limits pi
#: otherwise fills in from its own defaults. Left unset, pi caps output at
#: 16384 tokens, which a reasoning model reaches mid-message — and a truncated
#: completion ends a pi run outright, so the cap decides how many rollouts
#: survive. It belongs in the run config where it can be read off afterwards.
PI_MODEL_LIMITS = {
    "max_tokens": "maxTokens",
    "context_window": "contextWindow",
}


@register_harness
class PiHarness(InstalledCliHarness):
    """earendil-works ``pi``, run in print mode inside the sandbox.

    Recognized ``harness.params`` beyond
    :class:`~orchard_evalkit.harnesses.installed_cli.InstalledCliHarness`:

    ``max_tokens``
        Output-token cap written into pi's ``models.json``. Unset leaves pi's
        own default (16384).
    ``context_window``
        Context size pi assumes when deciding to compact. Unset leaves pi's
        own default; set it to the serving engine's real limit when that is
        what binds.
    """

    name = "pi"
    description = "pi coding agent (`pi -p`), running inside the sandbox"
    spec = CliSpec(
        binary="pi",
        args=(
            ("--print",),
            # Sessions would persist to disk for no benefit in a one-shot
            # rollout, and the pod is discarded immediately afterwards. The
            # trajectory comes from the event stream instead.
            ("--no-session",),
            ("--approve",),
            ("--model", "{model_ref}"),
            (PROMPT_TOKEN,),
        ),
        prompt_delivery="argv",
        trajectory_args=(("--mode", "json"),),
        trajectory_format="pi",
        # pi streams its events to stdout only, and a timed-out exec returns
        # none of it.
        mirror_stdout=True,
        # `message_update` re-sends the whole accumulated message on every
        # streamed token, so stdout grows with the square of the message
        # length. A reasoning model makes that fatal: Qwen3.8-27B produced a
        # median of 205MB per rollout and up to 1.3GB, and every rollout past
        # ~500MB lost its exec stream before the exit frame arrived — 116 of
        # 500 SWE-bench Verified instances, all scored as empty patches. The
        # parser already drops these events and reads the finished message off
        # `message_end`/`agent_end`, so nothing is lost by never sending them.
        drop_event_types=("message_update",),
        api_key_env="OPENAI_API_KEY",
        base_url_env="OPENAI_BASE_URL",
        env={"PI_OFFLINE": "1", "PI_CODING_AGENT_DIR": PI_HOME},
        setup_command=f"mkdir -p {PI_HOME}",
        config_path=f"{PI_HOME}/models.json",
        config_template=PI_CONFIG_TEMPLATE,
        model_alias=PI_MODEL_ALIAS,
    )

    def _config_substitutions(self, model: ModelConfig) -> dict[str, str]:
        return {**super()._config_substitutions(model), **self._model_limits()}

    def _model_limits(self) -> dict[str, str]:
        """Render :data:`PI_MODEL_LIMITS` as an optional models.json fragment.

        An unset knob emits nothing rather than a default of this harness's
        choosing: the right context window is the serving engine's, and
        guessing it high makes pi hold a context the server will refuse, while
        guessing it low throws away usable context on every rollout.
        """
        fields = []
        for param, key in PI_MODEL_LIMITS.items():
            value = self.params.get(param)
            if value is None:
                continue
            try:
                number = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"harness.params.{param} must be an integer; got {value!r}"
                ) from exc
            if number <= 0:
                raise ValueError(
                    f"harness.params.{param} must be positive; got {number}"
                )
            fields.append(f'"{key}": {number}')
        return {"model_limits": "".join(f",\n          {f}" for f in fields)}


#: Config and state directory for Claude Code. Under /var/tmp for the same
#: reason CODEX_HOME is: a task image's HOME is not reliably writable, and
#: Claude Code writes onboarding state, project history and shell snapshots
#: before it sends its first request.
CLAUDE_HOME = "/var/tmp/orchard-claude"

#: Model aliases Claude Code resolves independently of ``--model``. Background
#: work (conversation titles, compaction) goes to the haiku alias and subagents
#: to their own, so leaving these unset sends part of every rollout to a model
#: id the fleet does not serve — and the resulting failure surfaces mid-session
#: with nothing in the trajectory to explain it. Harbor's own Claude Code agent
#: mirrors the same set for the same reason.
CLAUDE_MODEL_ALIAS_VARS = (
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_FABLE_MODEL",
    "CLAUDE_CODE_SUBAGENT_MODEL",
)

#: Output-token cap for Claude Code, matching pi's `max_tokens` default so the
#: two harnesses are comparable on the same benchmark.
DEFAULT_CLAUDE_MAX_TOKENS = 16384

#: Extended-thinking budget. Zero by design — see :meth:`ClaudeCodeHarness._build_env`.
DEFAULT_CLAUDE_THINKING_TOKENS = 0

#: Levels `claude --effort` accepts. Anything else is rejected by the CLI at
#: startup, before it has sent a request.
CLAUDE_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")

#: Effort level used on the paths where the session router is *not* in front of
#: the endpoint. Under the router the flag never reaches the server at all, so
#: this is a fallback and is named like one — see :meth:`ClaudeCodeHarness._effort`
#: for why leaving it to the CLI fails every request against this fleet, and why
#: "medium" rather than something higher.
DEFAULT_CLAUDE_EFFORT_FALLBACK = "medium"


@register_harness
class ClaudeCodeHarness(InstalledCliHarness):
    """Anthropic Claude Code, run non-interactively inside the sandbox.

    The one harness here that does not speak OpenAI chat-completions: it posts
    to ``{ANTHROPIC_BASE_URL}/v1/messages``, the Anthropic Messages API. No
    translation layer is needed, because the sglang build this suite pins
    (``lmsysorg/sglang:v0.5.18``) serves that API natively alongside the OpenAI
    one, and ``scripts/session_router.py`` relays any path it is given. What the
    protocol difference does change is how the endpoint and the credential have
    to be spelled — see :attr:`spec` and :meth:`_build_env` — and which
    request-shaping fields reach the server at all, which is what
    :meth:`_effort` exists for.
    """

    name = "claude-code"
    description = "Anthropic Claude Code (`claude -p`), running inside the sandbox"
    spec = CliSpec(
        binary="claude",
        args=(
            ("-p",),
            ("--permission-mode", "bypassPermissions"),
            ("--model", "{model}"),
            # Dropped when `harness.params.effort_fallback` is "" — see `_effort`.
            ("--effort", "{effort}"),
            (PROMPT_TOKEN,),
        ),
        prompt_delivery="argv",
        # `--verbose` is required alongside stream-json under `-p`, otherwise
        # Claude Code refuses the combination.
        trajectory_args=(("--output-format", "stream-json"), ("--verbose",)),
        trajectory_format="claude",
        # Claude Code streams its events to stdout only, and a timed-out exec
        # returns none of it — same failure the pi mirror exists to prevent.
        mirror_stdout=True,
        # Not ANTHROPIC_API_KEY, which Claude Code sends as the `X-Api-Key`
        # header. sglang's auth middleware reads `Authorization: Bearer` and
        # nothing else, so that spelling 401s on every request. ANTHROPIC_AUTH_
        # TOKEN is the one that becomes a bearer token. It also avoids the
        # interactive "approve this API key?" prompt ANTHROPIC_API_KEY triggers.
        api_key_env="ANTHROPIC_AUTH_TOKEN",
        base_url_env="ANTHROPIC_BASE_URL",
        env={
            # Keep config, history and shell snapshots off an unwritable HOME.
            "CLAUDE_CONFIG_DIR": CLAUDE_HOME,
            # Overridden by the real credential when there is one. Without a
            # value Claude Code falls through to looking for stored OAuth
            # credentials, finds none, and exits asking to be logged in — under
            # `-p` there is nobody to ask. A self-hosted server started without
            # --api-key accepts any value, same as codex's OPENAI_API_KEY.
            "ANTHROPIC_AUTH_TOKEN": "EMPTY",
            # Claude Code refuses `--permission-mode bypassPermissions` when it
            # is running as root and nothing tells it that is deliberate. Every
            # task image here runs the agent as root.
            "IS_SANDBOX": "1",
            # The pod may or may not have egress, and none of this traffic is
            # worth a rollout stalling on a DNS timeout.
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "DISABLE_TELEMETRY": "1",
            "DISABLE_ERROR_REPORTING": "1",
            "DISABLE_AUTOUPDATER": "1",
        },
        setup_command=f"mkdir -p {CLAUDE_HOME}",
    )

    def _effort(self) -> str:
        """Level for ``--effort``, or ``""`` to leave the flag off entirely.

        Claude Code puts ``output_config: {"effort": ...}`` on every request and
        defaults it to ``high``. sglang's Anthropic adapter forwards that
        straight through as chat-completions ``reasoning_effort`` (rewriting
        ``xhigh`` to ``max`` on the way), and a chat template that does not
        recognise the level it is handed raises inside jinja — which sglang does
        not catch, so the request comes back 500. Qwen3.8-27B's template accepts
        ``xhigh``, ``medium`` and ``low`` and nothing else, so left to itself
        Claude Code 500s on its *first* request, burns its ten retries, and ends
        the rollout with two messages, an empty patch and `agent_error`.

        Verified against the live fleet: ``high``, ``xhigh``, ``max`` and
        ``minimal`` all 500; ``low``, ``medium`` and omitting the field
        succeed. ``medium`` is therefore the highest level that survives — the
        CLI's ``xhigh`` does not, because the adapter renames it to ``max``.

        Which is why the session router drops ``output_config.effort`` outright
        (``strip_effort`` in ``scripts/session_router.py``): the template's own
        default is ``xhigh``, and no value of this flag can reach it. Under
        ``--routing session`` the level below is therefore discarded and the
        rollout runs at ``xhigh`` whatever this says, which is what the
        ``_fallback`` in the parameter name is there to stop anyone reading the
        config from getting wrong. What it pins is the paths where the router is
        not in front: a sticky endpoint, an engine addressed directly, or
        ``--no-strip-effort``.

        Set ``harness.params.effort_fallback`` to ``""`` against an endpoint
        whose template takes the CLI's own default.
        """
        if "effort" in self.params:
            raise ValueError(
                "harness.params.effort is now harness.params.effort_fallback. "
                "The old name read as the level the rollout runs at, which it "
                "is not under the session router — the router strips the field "
                "and the model runs at its template default. Rename the key; "
                "the accepted values have not changed."
            )
        effort = self.params.get("effort_fallback", DEFAULT_CLAUDE_EFFORT_FALLBACK)
        effort = "" if effort is None else str(effort)
        if effort and effort not in CLAUDE_EFFORT_LEVELS:
            raise ValueError(
                "harness.params.effort_fallback must be one of "
                f'{", ".join(CLAUDE_EFFORT_LEVELS)} (or "" to leave --effort '
                f"off); got {effort!r}"
            )
        return effort

    def _substitutions(self, ctx: RolloutContext) -> dict[str, str]:
        return {**super()._substitutions(ctx), "effort": self._effort()}

    def _build_env(self, model: ModelConfig | None = None) -> dict[str, str]:
        """Add the env that depends on the model or on run params.

        Everything here is out of reach of :attr:`CliSpec.env`, which is a flat
        ``dict[str, str]`` with no interpolation.
        """
        model = model or self.model
        env = super()._build_env(model)
        if model.base_url:
            # The base URL the rest of the suite carries ends in /v1 so codex
            # and pi can append their own protocol paths. Claude Code appends
            # `/v1/messages`, so it gets the root instead — while model.base_url
            # keeps its /v1 for router.close_url() at teardown.
            env[self.spec.base_url_env] = anthropic_base_url(model.base_url)
        for var in CLAUDE_MODEL_ALIAS_VARS:
            env[var] = model.name
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(
            self.params.get("max_tokens", DEFAULT_CLAUDE_MAX_TOKENS)
        )
        # Default 0: sglang's Anthropic adapter rejects `redacted_thinking`
        # outright, and with `--reasoning-parser qwen3` it emits thinking blocks
        # with no signature — which Claude Code replays verbatim on the next
        # turn. Not requesting extended thinking sidesteps the whole class, and
        # matches pi, which disables `reasoning_effort` for these same servers.
        # The model still reasons; only the request-side parameter goes away.
        env["MAX_THINKING_TOKENS"] = str(
            self.params.get("max_thinking_tokens", DEFAULT_CLAUDE_THINKING_TOKENS)
        )
        # Re-apply harness.params.env last so an operator override still wins.
        env.update({k: str(v) for k, v in (self.params.get("env") or {}).items()})
        return env



@register_harness
class OpencodeHarness(InstalledCliHarness):
    """OpenCode, run non-interactively inside the sandbox."""

    name = "opencode"
    description = "OpenCode (`opencode run`), running inside the sandbox"
    spec = CliSpec(
        binary="opencode",
        subcommand=("run",),
        args=(
            ("--model", "{model}"),
            (PROMPT_TOKEN,),
        ),
        prompt_delivery="argv",
        trajectory_args=(("--format", "json"),),
        trajectory_format="opencode",
        api_key_env="OPENAI_API_KEY",
        base_url_env="OPENAI_BASE_URL",
    )


__all__ = [
    "CLAUDE_HOME",
    "CLAUDE_MODEL_ALIAS_VARS",
    "CODEX_CONFIG_TEMPLATE",
    "CODEX_HOME",
    "LOADER_FAILURE_MARKERS",
    "BinaryProbe",
    "CliSpec",
    "ClaudeCodeHarness",
    "CodexHarness",
    "DEFAULT_PROMPT_TEMPLATE",
    "InstalledCliHarness",
    "OpencodeHarness",
    "PiHarness",
    "PROMPT_PATH",
    "PROMPT_TOKEN",
    "PROMPT_VAR",
]
