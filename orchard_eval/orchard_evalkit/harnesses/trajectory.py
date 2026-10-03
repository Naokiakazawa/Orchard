"""Normalized agent trajectories, across every harness.

A trajectory is the point of most of this: distillation and RL need the
*sequence* of what the agent thought and did, not just the final patch. Every
harness must therefore produce one, and the runner writes it unconditionally —
see :func:`orchard_evalkit.runner.EvalRunner._drive_sandbox`.

Harnesses disagree completely about format: mini-swe-agent saves a JSON
trajectory to disk, and the CLI agents each emit their own JSONL event stream on
stdout. This module turns all of them into one shape:

    {"role": ..., "content": ..., "extra": {...}}

which is deliberately mini-swe-agent's shape, so anything already consuming
Orchard SWE trajectories keeps working.

**Normalization must not silently lose events.** These parsers track upstream
CLIs that change their event vocabulary without notice; a parser that dropped an
event it does not recognize would quietly corrupt a distillation set. So an
unrecognized event is preserved as a message with ``extra.kind == "unknown"``
rather than discarded.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Normalized roles. These mirror mini-swe-agent's vocabulary.
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"
ROLE_EXIT = "exit"


def parse_jsonl(text: str) -> tuple[list[dict[str, Any]], int]:
    """Read a JSONL stream tolerantly.

    Returns ``(events, skipped)``. CLIs interleave human-readable banners and
    warning lines into the stream (codex prints a PATH-alias warning before its
    first event), so a non-JSON line is skipped rather than treated as a
    failure. The skip count is surfaced so a stream that is *mostly* unparseable
    can be spotted instead of silently yielding an empty trajectory.
    """
    events: list[dict[str, Any]] = []
    skipped = 0
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.startswith("{"):
            skipped += 1
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
        else:
            skipped += 1
    return events, skipped


def _message(role: str, content: str, **extra: Any) -> dict[str, Any]:
    return {"role": role, "content": content, "extra": extra}


def _unknown(event: dict[str, Any]) -> dict[str, Any]:
    """Preserve an event this parser does not understand."""
    return _message(
        ROLE_TOOL,
        "",
        kind="unknown",
        event_type=str(event.get("type", "")),
        event=event,
    )


# ---------------------------------------------------------------------------
# codex
# ---------------------------------------------------------------------------


#: Event and item vocabulary documented at
#: https://learn.chatgpt.com/docs/non-interactive-mode
def parse_codex_events(text: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse ``codex exec --json`` output.

    Stream shape::

        {"type":"thread.started","thread_id":"..."}
        {"type":"turn.started"}
        {"type":"item.started","item":{"type":"command_execution","command":"..."}}
        {"type":"item.completed","item":{"type":"agent_message","text":"..."}}
        {"type":"turn.completed","usage":{"input_tokens":...}}

    Only ``item.completed`` is turned into a message: ``item.started`` and
    ``item.updated`` are progress notifications for the *same* item, so
    consuming all three would triplicate every step of the trajectory.
    """
    events, skipped = parse_jsonl(text)
    messages: list[dict[str, Any]] = []
    info: dict[str, Any] = {"skipped_lines": skipped, "events": len(events)}

    for event in events:
        etype = event.get("type", "")

        if etype == "thread.started":
            info["thread_id"] = event.get("thread_id", "")
            continue

        if etype == "turn.completed":
            usage = event.get("usage") or {}
            if isinstance(usage, dict):
                info["usage"] = usage
            continue

        if etype == "turn.failed":
            error = event.get("error") or {}
            message = (
                error.get("message", "") if isinstance(error, dict) else str(error)
            )
            info["error"] = message
            messages.append(_message(ROLE_EXIT, message, kind="turn_failed"))
            continue

        if etype == "error":
            messages.append(
                _message(ROLE_TOOL, str(event.get("message", "")), kind="error")
            )
            continue

        if etype in ("turn.started", "item.started", "item.updated"):
            continue

        if etype == "item.completed":
            item = event.get("item") or {}
            if isinstance(item, dict):
                messages.append(_codex_item(item))
            continue

        messages.append(_unknown(event))

    return messages, info


def _codex_item(item: dict[str, Any]) -> dict[str, Any]:
    """Map one completed codex item onto a normalized message."""
    itype = str(item.get("type", ""))

    if itype == "agent_message":
        return _message(ROLE_ASSISTANT, str(item.get("text", "")), kind="agent_message")

    if itype == "reasoning":
        # Reasoning is the model's own thinking, not an action — keep it on the
        # assistant turn but flagged, so a distillation set can include or drop
        # it deliberately rather than by accident.
        text = item.get("text") or item.get("summary") or ""
        return _message(ROLE_ASSISTANT, str(text), kind="reasoning", reasoning=True)

    if itype == "command_execution":
        command = str(item.get("command", ""))
        return _message(
            ROLE_TOOL,
            str(item.get("aggregated_output") or item.get("output") or ""),
            kind="command_execution",
            actions=[{"command": command}],
            command=command,
            exit_code=item.get("exit_code"),
            status=item.get("status"),
        )

    if itype == "file_change":
        return _message(
            ROLE_TOOL,
            json.dumps(item.get("changes", []), default=str),
            kind="file_change",
            changes=item.get("changes"),
        )

    if itype == "mcp_tool_call":
        return _message(
            ROLE_TOOL,
            json.dumps(item.get("result", ""), default=str),
            kind="mcp_tool_call",
            server=item.get("server"),
            tool=item.get("tool"),
        )

    if itype == "web_search":
        return _message(ROLE_TOOL, str(item.get("query", "")), kind="web_search")

    if itype == "todo_list":
        return _message(
            ROLE_ASSISTANT,
            json.dumps(item.get("items", []), default=str),
            kind="todo_list",
        )

    if itype == "error":
        return _message(ROLE_TOOL, str(item.get("message", "")), kind="error")

    return _message(ROLE_TOOL, "", kind="unknown", item_type=itype, item=item)


# ---------------------------------------------------------------------------
# pi
# ---------------------------------------------------------------------------


def parse_pi_events(text: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse ``pi --mode json`` output.

    pi emits a session header, then ``AgentSessionEvent`` records. Its
    ``agent_end`` event carries the *complete* final message list, which is a
    better trajectory than anything reconstructed from the incremental
    ``message_end`` events — so it is preferred when present, and the
    incremental path is the fallback for a run that never reached ``agent_end``
    (a crash, a timeout, a kill).

    ``info`` carries ``stop_reason`` and ``length_stops`` from
    :func:`_pi_stop_reasons` on top of the usual session fields.
    """
    events, skipped = parse_jsonl(text)
    info: dict[str, Any] = {"skipped_lines": skipped, "events": len(events)}

    final_messages: list[Any] | None = None
    incremental: list[dict[str, Any]] = []

    for event in events:
        etype = event.get("type", "")

        if etype == "session":
            info["session_id"] = event.get("id", "")
            info["session_version"] = event.get("version")
            continue

        if etype == "agent_end":
            candidate = event.get("messages")
            if isinstance(candidate, list):
                final_messages = candidate
            continue

        if etype == "message_end":
            message = event.get("message")
            if isinstance(message, dict):
                incremental.append(_pi_message(message))
            continue

        if etype == "tool_execution_end":
            incremental.append(
                _message(
                    ROLE_TOOL,
                    _stringify(event.get("result")),
                    kind="tool_result",
                    tool=event.get("toolName"),
                    tool_call_id=event.get("toolCallId"),
                    is_error=bool(event.get("isError")),
                )
            )
            continue

        # Lifecycle chatter carries no trajectory content.
        if etype in (
            "agent_start",
            "turn_start",
            "turn_end",
            "message_start",
            "message_update",
            "tool_execution_start",
            "tool_execution_update",
            "queue_update",
        ):
            continue

        if etype in ("compaction_start", "compaction_end"):
            # Compaction rewrites the context the agent sees; a trajectory that
            # hides it is not reproducible.
            incremental.append(_message(ROLE_TOOL, "", kind=etype, event=event))
            continue

        incremental.append(_unknown(event))

    if final_messages is not None:
        info["source"] = "agent_end"
        messages = [
            _pi_message(m) if isinstance(m, dict) else _message(ROLE_TOOL, str(m))
            for m in final_messages
        ]
    else:
        info["source"] = "incremental"
        messages = incremental

    info["stop_reason"], info["length_stops"] = _pi_stop_reasons(messages)
    return messages, info


def _pi_stop_reasons(messages: list[dict[str, Any]]) -> tuple[str, int]:
    """Why the last completion ended, and how many the output cap cut off.

    pi records a ``stopReason`` on every assistant message. ``"length"`` means
    the provider truncated that completion at its output-token cap *mid
    message*, which for pi is terminal: it ends the session (``willRetry:
    false``) instead of re-prompting the model to continue, so the rollout
    stops wherever the agent happened to be. Nothing in the exit code
    distinguishes that from a clean finish — pi still exits 0 — so it is
    surfaced here for the harness to classify on.
    """
    last = ""
    truncated = 0
    for message in messages:
        if message.get("role") != ROLE_ASSISTANT:
            continue
        raw = (message.get("extra") or {}).get("message")
        if not isinstance(raw, dict):
            continue
        reason = str(raw.get("stopReason") or "")
        if not reason:
            continue
        last = reason
        truncated += reason == "length"
    return last, truncated


def _pi_message(message: dict[str, Any]) -> dict[str, Any]:
    """Normalize one pi ``AgentMessage``.

    pi content is a list of typed blocks (text / thinking / image / tool calls),
    so the text is flattened for ``content`` while the blocks are kept intact in
    ``extra``.
    """
    role = str(message.get("role", ROLE_ASSISTANT))
    content = message.get("content")
    text_parts: list[str] = []
    blocks: list[Any] = []

    if isinstance(content, str):
        text_parts.append(content)
    elif isinstance(content, list):
        blocks = content
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
            elif block.get("type") == "thinking":
                text_parts.append(str(block.get("thinking", "")))

    normalized_role = {
        "assistant": ROLE_ASSISTANT,
        "user": ROLE_USER,
        "system": ROLE_SYSTEM,
        "toolResult": ROLE_TOOL,
        "tool": ROLE_TOOL,
    }.get(role, role)

    return _message(
        normalized_role,
        "\n".join(p for p in text_parts if p),
        kind=str(message.get("type", "") or role),
        blocks=blocks,
        message=message,
    )


# ---------------------------------------------------------------------------
# claude code
# ---------------------------------------------------------------------------


def parse_claude_events(text: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse ``claude -p --output-format stream-json`` output.

    Records are ``{"type": "system"|"assistant"|"user"|"result", ...}``, with
    Anthropic-shaped message bodies nested under ``message``.
    """
    events, skipped = parse_jsonl(text)
    messages: list[dict[str, Any]] = []
    info: dict[str, Any] = {"skipped_lines": skipped, "events": len(events)}

    for event in events:
        etype = event.get("type", "")

        if etype == "system":
            info.setdefault("session_id", event.get("session_id", ""))
            continue

        if etype == "result":
            info["result"] = {
                k: event.get(k)
                for k in ("subtype", "is_error", "num_turns", "total_cost_usd", "usage")
                if k in event
            }
            messages.append(
                _message(ROLE_EXIT, str(event.get("result", "")), kind="result")
            )
            continue

        if etype in ("assistant", "user"):
            body = event.get("message")
            if isinstance(body, dict):
                messages.append(_anthropic_message(etype, body))
            continue

        messages.append(_unknown(event))

    return messages, info


def _anthropic_message(role: str, body: dict[str, Any]) -> dict[str, Any]:
    content = body.get("content")
    text_parts: list[str] = []
    if isinstance(content, str):
        text_parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text_parts.append(str(block.get("text", "")))
            elif btype == "thinking":
                text_parts.append(str(block.get("thinking", "")))
            elif btype == "tool_result":
                text_parts.append(_stringify(block.get("content")))

    return _message(
        ROLE_ASSISTANT if role == "assistant" else ROLE_USER,
        "\n".join(p for p in text_parts if p),
        kind=role,
        blocks=content if isinstance(content, list) else [],
        usage=body.get("usage"),
    )


# ---------------------------------------------------------------------------
# opencode
# ---------------------------------------------------------------------------


def parse_opencode_events(text: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse ``opencode run --format json`` output.

    opencode's stream is the least specified of the four, so this parser is
    deliberately shallow: it keeps every event, mapping the ones it recognizes
    and preserving the rest verbatim. The raw stream remains the source of
    truth.
    """
    events, skipped = parse_jsonl(text)
    messages: list[dict[str, Any]] = []
    info: dict[str, Any] = {"skipped_lines": skipped, "events": len(events)}

    for event in events:
        etype = str(event.get("type", ""))
        part = event.get("part") if isinstance(event.get("part"), dict) else None

        if part and part.get("type") == "text":
            messages.append(
                _message(ROLE_ASSISTANT, str(part.get("text", "")), kind="text")
            )
            continue
        if part and part.get("type") == "tool":
            messages.append(
                _message(
                    ROLE_TOOL,
                    _stringify(part.get("state")),
                    kind="tool",
                    tool=part.get("tool"),
                )
            )
            continue
        if etype in ("message.updated", "message.part.updated", "step.started"):
            continue

        messages.append(_unknown(event))

    return messages, info


# ---------------------------------------------------------------------------
# mini-swe-agent
# ---------------------------------------------------------------------------


def parse_mini_trajectory(text: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse the trajectory JSON ``mini -o <path>`` writes inside the sandbox.

    No normalization is needed: this module's message shape *is*
    mini-swe-agent's, so its messages pass through untouched. What the parser
    adds is the run summary the harness cannot get any other way — the agent's
    own ``submission`` (the diff it chose to hand in), its exit status, and its
    cost — all of which live in ``info`` rather than in the message list.
    """
    if not (text or "").strip():
        return [], {"error": "trajectory file was empty or missing"}

    data = json.loads(text)
    messages = [m for m in (data.get("messages") or []) if isinstance(m, dict)]
    info = data.get("info") or {}
    stats = info.get("model_stats") or {}
    return messages, {
        "trajectory_format": data.get("trajectory_format", ""),
        "mini_version": info.get("mini_version", ""),
        "exit_status": info.get("exit_status", ""),
        "submission": info.get("submission", ""),
        "cost": stats.get("instance_cost", 0.0),
        "api_calls": stats.get("api_calls", 0),
    }


# ---------------------------------------------------------------------------


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return str(value)


#: Parser registry, keyed by ``CliSpec.trajectory_format``.
PARSERS = {
    "codex": parse_codex_events,
    "mini": parse_mini_trajectory,
    "pi": parse_pi_events,
    "claude": parse_claude_events,
    "opencode": parse_opencode_events,
}


def parse_trajectory(
    fmt: str, text: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse ``text`` with the named parser.

    Never raises: a parser that fails on an unexpected stream returns no
    messages plus an ``error`` in the info dict, because losing the trajectory
    is bad but failing a paid rollout over it is worse. The raw stream is
    retained regardless.
    """
    parser = PARSERS.get(fmt)
    if parser is None:
        return [], {"error": f"no parser for format {fmt!r}"}
    try:
        return parser(text)
    except Exception as exc:  # noqa: BLE001 - never fail a rollout over parsing
        logger.exception("trajectory parsing failed for format %r", fmt)
        return [], {"error": f"{type(exc).__name__}: {exc}"}


__all__ = [
    "PARSERS",
    "ROLE_ASSISTANT",
    "ROLE_EXIT",
    "ROLE_SYSTEM",
    "ROLE_TOOL",
    "ROLE_USER",
    "parse_claude_events",
    "parse_codex_events",
    "parse_jsonl",
    "parse_mini_trajectory",
    "parse_opencode_events",
    "parse_pi_events",
    "parse_trajectory",
]
