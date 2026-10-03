"""Trajectory capture and normalization, for every harness.

The fixtures here are **real** event lines, captured from the actual CLIs
(codex 0.149.1 against a deliberately-rejected key) or taken from the vendors'
published schemas, rather than invented. A parser tested only against its
author's guess at a format proves nothing about the format.
"""

import json

import pytest

from orchard_evalkit.harnesses.trajectory import (
    ROLE_ASSISTANT,
    ROLE_EXIT,
    ROLE_TOOL,
    parse_claude_events,
    parse_codex_events,
    parse_jsonl,
    parse_opencode_events,
    parse_pi_events,
    parse_trajectory,
)


class TestParseJsonl:
    def test_reads_json_objects(self):
        events, skipped = parse_jsonl('{"a":1}\n{"b":2}')
        assert events == [{"a": 1}, {"b": 2}]
        assert skipped == 0

    def test_skips_interleaved_banner_lines(self):
        # codex really does print this warning before its first event.
        text = (
            "WARNING: proceeding, even though we could not create PATH aliases\n"
            '{"type":"thread.started","thread_id":"abc"}\n'
        )
        events, skipped = parse_jsonl(text)
        assert len(events) == 1
        assert skipped == 1

    def test_skips_malformed_json(self):
        events, skipped = parse_jsonl('{"ok":1}\n{"broken":')
        assert len(events) == 1
        assert skipped == 1

    def test_non_object_json_is_skipped(self):
        events, skipped = parse_jsonl("[1,2,3]\n")
        assert events == []
        assert skipped == 1


# Captured verbatim from `codex exec --json` (codex-cli 0.149.1). The error
# path is real output from a run with a rejected API key.
CODEX_REAL_FAILURE = "\n".join(
    [
        "WARNING: proceeding, even though we could not create PATH aliases",
        '{"type":"thread.started","thread_id":"01a039be-08b5-7be1-a512-dbe1152e652e"}',
        '{"type":"turn.started"}',
        '{"type":"error","message":"Reconnecting... 2/5 (unexpected status 401)"}',
        '{"type":"item.completed","item":{"id":"item_0","type":"error",'
        '"message":"Falling back from WebSockets to HTTPS transport."}}',
        '{"type":"turn.failed","error":{"message":"unexpected status 401 Unauthorized"}}',
    ]
)

# Shape documented at https://learn.chatgpt.com/docs/non-interactive-mode
CODEX_SUCCESS = "\n".join(
    [
        '{"type":"thread.started","thread_id":"0199a213"}',
        '{"type":"turn.started"}',
        '{"type":"item.started","item":{"id":"item_1","type":"command_execution",'
        '"command":"bash -lc ls","status":"in_progress"}}',
        '{"type":"item.completed","item":{"id":"item_1","type":"command_execution",'
        '"command":"bash -lc ls","aggregated_output":"setup.py\\n","exit_code":0,'
        '"status":"completed"}}',
        '{"type":"item.completed","item":{"id":"item_2","type":"reasoning",'
        '"text":"I should read setup.py first."}}',
        '{"type":"item.completed","item":{"id":"item_3","type":"agent_message",'
        '"text":"Repo contains docs, sdk, and examples."}}',
        '{"type":"turn.completed","usage":{"input_tokens":24763,'
        '"cached_input_tokens":24448,"output_tokens":122}}',
    ]
)


class TestCodexParser:
    def test_real_failure_stream_parses(self):
        messages, info = parse_codex_events(CODEX_REAL_FAILURE)
        assert info["thread_id"] == "01a039be-08b5-7be1-a512-dbe1152e652e"
        assert info["skipped_lines"] == 1  # the WARNING banner
        assert info["error"] == "unexpected status 401 Unauthorized"
        assert messages[-1]["role"] == ROLE_EXIT

    def test_success_stream_yields_the_agent_turns(self):
        messages, info = parse_codex_events(CODEX_SUCCESS)
        kinds = [m["extra"]["kind"] for m in messages]
        assert kinds == ["command_execution", "reasoning", "agent_message"]
        assert messages[-1]["content"] == "Repo contains docs, sdk, and examples."

    def test_command_execution_keeps_command_and_output(self):
        messages, _ = parse_codex_events(CODEX_SUCCESS)
        cmd = messages[0]
        assert cmd["role"] == ROLE_TOOL
        assert cmd["extra"]["command"] == "bash -lc ls"
        assert cmd["extra"]["exit_code"] == 0
        assert cmd["content"] == "setup.py\n"
        # `actions` mirrors mini-swe-agent's shape so downstream code is uniform.
        assert cmd["extra"]["actions"] == [{"command": "bash -lc ls"}]

    def test_reasoning_is_flagged_not_silently_merged(self):
        # Distillation sets need to include or drop reasoning deliberately.
        messages, _ = parse_codex_events(CODEX_SUCCESS)
        reasoning = messages[1]
        assert reasoning["role"] == ROLE_ASSISTANT
        assert reasoning["extra"]["reasoning"] is True

    def test_item_started_does_not_duplicate_item_completed(self):
        # started/updated/completed describe the SAME item; consuming all three
        # would triplicate every step of the trajectory.
        messages, _ = parse_codex_events(CODEX_SUCCESS)
        commands = [m for m in messages if m["extra"]["kind"] == "command_execution"]
        assert len(commands) == 1

    def test_usage_is_captured_for_metrics(self):
        _, info = parse_codex_events(CODEX_SUCCESS)
        assert info["usage"]["input_tokens"] == 24763

    def test_an_unknown_event_is_preserved_not_dropped(self):
        # Silently dropping an unrecognized event would quietly corrupt a
        # distillation set when the CLI adds a new event type.
        messages, _ = parse_codex_events('{"type":"brand.new.event","payload":42}')
        assert len(messages) == 1
        assert messages[0]["extra"]["kind"] == "unknown"
        assert messages[0]["extra"]["event"]["payload"] == 42

    def test_an_unknown_item_type_is_preserved(self):
        messages, _ = parse_codex_events(
            '{"type":"item.completed","item":{"type":"future_thing","x":1}}'
        )
        assert messages[0]["extra"]["kind"] == "unknown"
        assert messages[0]["extra"]["item_type"] == "future_thing"


PI_STREAM = "\n".join(
    [
        '{"type":"session","version":3,"id":"uuid-1","cwd":"/testbed"}',
        '{"type":"agent_start"}',
        '{"type":"turn_start"}',
        '{"type":"message_end","message":{"role":"assistant","content":'
        '[{"type":"text","text":"Reading the file."}]}}',
        '{"type":"tool_execution_end","toolCallId":"c1","toolName":"bash",'
        '"result":"setup.py","isError":false}',
        '{"type":"agent_end","messages":['
        '{"role":"user","content":[{"type":"text","text":"fix it"}]},'
        '{"role":"assistant","content":[{"type":"thinking","thinking":"hmm"},'
        '{"type":"text","text":"Done."}]}]}',
    ]
)

#: The same run as it arrives from a crash, a timeout or a kill: no agent_end,
#: so the parser has to fall back to the incremental events.
PI_STREAM_TRUNCATED = "\n".join(PI_STREAM.splitlines()[:-1])

#: A run pi ended because the provider cut a completion off at its output-token
#: cap. pi exits 0 and reports a full `agent_end`, so only `stopReason` tells
#: the difference between this and an agent that finished.
PI_STREAM_LENGTH_CAPPED = "\n".join(
    [
        '{"type":"session","version":3,"id":"uuid-2","cwd":"/testbed"}',
        '{"type":"message_end","message":{"role":"assistant","content":'
        '[{"type":"text","text":"Looking."}],"stopReason":"toolUse"}}',
        '{"type":"agent_end","messages":['
        '{"role":"user","content":[{"type":"text","text":"fix it"}]},'
        '{"role":"assistant","content":[{"type":"text","text":"Looking."}],'
        '"stopReason":"toolUse"},'
        '{"role":"assistant","content":[{"type":"thinking","thinking":"wait"}],'
        '"stopReason":"length"}]}',
    ]
)


class TestPiParser:
    def test_agent_end_is_preferred_as_the_complete_record(self):
        messages, info = parse_pi_events(PI_STREAM)
        assert info["source"] == "agent_end"
        assert info["session_id"] == "uuid-1"
        assert [m["role"] for m in messages] == ["user", ROLE_ASSISTANT]

    def test_content_blocks_are_flattened_but_retained(self):
        messages, _ = parse_pi_events(PI_STREAM)
        assistant = messages[-1]
        # Thinking and text both surface in content...
        assert "hmm" in assistant["content"]
        assert "Done." in assistant["content"]
        # ...and the structured blocks survive for anyone who needs them.
        assert len(assistant["extra"]["blocks"]) == 2

    def test_incremental_fallback_when_the_run_never_finished(self):
        # A crashed or killed run has no agent_end; the trajectory must still
        # be recoverable from the incremental events.
        messages, info = parse_pi_events(PI_STREAM_TRUNCATED)
        assert info["source"] == "incremental"
        assert any(m["extra"].get("kind") == "tool_result" for m in messages)

    def test_a_finished_run_reports_no_truncation(self):
        _, info = parse_pi_events(PI_STREAM)
        assert info["length_stops"] == 0
        assert info["stop_reason"] != "length"

    def test_a_length_capped_run_is_reported_as_such(self):
        # pi exits 0 and emits a complete agent_end here, so without the stop
        # reason this is indistinguishable from an agent that chose to stop.
        _, info = parse_pi_events(PI_STREAM_LENGTH_CAPPED)
        assert info["stop_reason"] == "length"
        assert info["length_stops"] == 1

    def test_truncation_is_counted_from_the_incremental_fallback_too(self):
        # A run killed before agent_end still has to report why it stopped.
        incremental = "\n".join(PI_STREAM_LENGTH_CAPPED.splitlines()[:-1])
        messages, info = parse_pi_events(incremental)
        assert info["source"] == "incremental"
        assert info["stop_reason"] == "toolUse"
        assert info["length_stops"] == 0

    def test_compaction_is_recorded(self):
        # Compaction rewrites the context the agent saw; hiding it makes the
        # trajectory non-reproducible.
        messages, _ = parse_pi_events(
            '{"type":"compaction_start","reason":"threshold"}'
        )
        assert messages[0]["extra"]["kind"] == "compaction_start"

    @pytest.mark.parametrize("stream", [PI_STREAM, PI_STREAM_TRUNCATED])
    def test_message_update_events_change_nothing(self, stream):
        # pi re-sends the whole accumulated message on every streamed token
        # rather than a delta, so these events are quadratic in message length
        # and a long-reasoning model buries the exec stream under them. They are
        # filtered away inside the sandbox now (CliSpec.drop_event_types), which
        # is only safe because the parser reconstructs the identical trajectory
        # without them — on both the agent_end path and the incremental
        # fallback. Pinned here so a future parser that starts reading
        # message_update fails loudly instead of silently losing a trajectory.
        partials = "\n".join(
            json.dumps(
                {
                    "type": "message_update",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": prefix}],
                    },
                }
            )
            for prefix in ("R", "Reading", "Reading the file.")
        )
        noisy = stream.replace(
            '{"type":"message_end"', f'{partials}\n{{"type":"message_end"'
        )
        # The substitution landed, and every added line is a real event rather
        # than one parse_jsonl quietly skips — otherwise this proves nothing.
        assert parse_pi_events(noisy)[1]["events"] == parse_pi_events(stream)[1][
            "events"
        ] + len(partials.splitlines())

        assert parse_pi_events(noisy)[0] == parse_pi_events(stream)[0]


CLAUDE_STREAM = "\n".join(
    [
        '{"type":"system","subtype":"init","session_id":"s-1"}',
        '{"type":"assistant","message":{"role":"assistant","content":'
        '[{"type":"text","text":"Let me look."}],"usage":{"input_tokens":10}}}',
        '{"type":"user","message":{"role":"user","content":'
        '[{"type":"tool_result","content":"setup.py"}]}}',
        '{"type":"result","subtype":"success","is_error":false,"num_turns":2,'
        '"total_cost_usd":0.031,"result":"done"}',
    ]
)


class TestClaudeParser:
    def test_stream_json_is_normalized(self):
        messages, info = parse_claude_events(CLAUDE_STREAM)
        assert info["session_id"] == "s-1"
        assert messages[0]["content"] == "Let me look."
        assert messages[-1]["role"] == ROLE_EXIT

    def test_result_metadata_is_captured(self):
        _, info = parse_claude_events(CLAUDE_STREAM)
        assert info["result"]["total_cost_usd"] == 0.031
        assert info["result"]["num_turns"] == 2

    def test_tool_results_surface_as_content(self):
        messages, _ = parse_claude_events(CLAUDE_STREAM)
        assert any("setup.py" in m["content"] for m in messages)


class TestOpencodeParser:
    def test_text_and_tool_parts(self):
        stream = "\n".join(
            [
                '{"type":"message.part.updated","part":{"type":"text","text":"hi"}}',
                '{"type":"message.part.updated","part":{"type":"tool",'
                '"tool":"bash","state":{"output":"ok"}}}',
            ]
        )
        messages, _ = parse_opencode_events(stream)
        assert messages[0]["content"] == "hi"
        assert messages[1]["extra"]["tool"] == "bash"

    def test_unrecognized_events_are_kept(self):
        messages, _ = parse_opencode_events('{"type":"something.else","v":1}')
        assert messages[0]["extra"]["kind"] == "unknown"


class TestParseTrajectoryDispatch:
    @pytest.mark.parametrize("fmt", ["codex", "pi", "claude", "opencode"])
    def test_every_registered_format_dispatches(self, fmt):
        messages, info = parse_trajectory(fmt, "")
        assert isinstance(messages, list)
        assert "error" not in info

    def test_an_unknown_format_is_reported_not_raised(self):
        messages, info = parse_trajectory("nope", '{"a":1}')
        assert messages == []
        assert "no parser" in info["error"]

    def test_a_parser_crash_never_fails_the_rollout(self, monkeypatch):
        # Losing a trajectory is bad; failing a paid rollout over a parser bug
        # is worse. The raw stream is retained either way.
        def boom(_text):
            raise ValueError("parser bug")

        monkeypatch.setitem(
            __import__(
                "orchard_evalkit.harnesses.trajectory", fromlist=["PARSERS"]
            ).PARSERS,
            "codex",
            boom,
        )
        messages, info = parse_trajectory("codex", "{}")
        assert messages == []
        assert "parser bug" in info["error"]

    def test_output_is_json_serializable(self):
        # The runner writes these straight to disk.
        messages, info = parse_codex_events(CODEX_SUCCESS)
        json.dumps({"messages": messages, "info": info})
