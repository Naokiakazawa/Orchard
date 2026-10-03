"""Parser tests, concentrated on the cases that silently corrupt a build.

Every test here corresponds to a real construct in the terminal-bench task set.
A parser bug in any of them does not raise — it produces a plausible-looking
command that does the wrong thing, which is why they are pinned.
"""

from __future__ import annotations

import pytest

from harbor_orchard.dockerfile import DockerfileError, parse, parse_arg, parse_env


def test_continuation_joins_without_inserting_whitespace():
    # A pinned dependency split across lines must not gain a space inside the
    # version specifier, which would turn `foo==1.0` into an invalid requirement.
    dockerfile = parse(
        "FROM python:3.11-slim\n"
        "RUN pip install foo==1.\\\n"
        "0 bar\n"
    )
    assert dockerfile.stages[0].instructions[0].value == "pip install foo==1.0 bar"


def test_continuation_preserves_surrounding_spaces():
    dockerfile = parse(
        "FROM ubuntu:24.04\n"
        "RUN apt-get update && \\\n"
        "    apt-get install -y curl\n"
    )
    assert (
        dockerfile.stages[0].instructions[0].value
        == "apt-get update &&     apt-get install -y curl"
    )


def test_comments_inside_a_continuation_are_dropped():
    dockerfile = parse(
        "FROM ubuntu:24.04\n"
        "RUN echo one && \\\n"
        "# an explanatory comment\n"
        "    echo two\n"
    )
    assert "explanatory" not in dockerfile.stages[0].instructions[0].value


def test_a_doubled_trailing_backslash_does_not_continue():
    # The line ends with an escaped backslash, not a continuation marker, so the
    # WORKDIR after it is a separate instruction rather than part of the RUN.
    dockerfile = parse(
        "FROM ubuntu:24.04\n"
        "RUN echo a\\\\\n"
        "WORKDIR /app\n"
    )
    names = [i.name for i in dockerfile.stages[0].instructions]
    assert names == ["RUN", "WORKDIR"]


def test_heredoc_body_is_captured_and_restored():
    dockerfile = parse(
        "FROM python:3.13-slim\n"
        "RUN python3 - <<'PY'\n"
        "from pathlib import Path\n"
        "print(Path('/'))\n"
        "PY\n"
        "WORKDIR /app\n"
    )
    instructions = dockerfile.stages[0].instructions
    assert [i.name for i in instructions] == ["RUN", "WORKDIR"]

    run = instructions[0]
    assert run.heredocs == (("PY", "from pathlib import Path\nprint(Path('/'))"),)
    # The reassembled script is what a shell would need to run it verbatim.
    assert run.script == (
        "python3 - <<'PY'\nfrom pathlib import Path\nprint(Path('/'))\nPY"
    )


def test_double_left_angle_without_a_terminator_is_not_a_heredoc():
    dockerfile = parse(
        "FROM ubuntu:24.04\n"
        "RUN echo \"shift is a << b\"\n"
        "WORKDIR /app\n"
    )
    assert [i.name for i in dockerfile.stages[0].instructions] == ["RUN", "WORKDIR"]


def test_herestring_is_not_treated_as_a_heredoc():
    dockerfile = parse("FROM ubuntu:24.04\nRUN cat <<<hello\nWORKDIR /app\n")
    assert [i.name for i in dockerfile.stages[0].instructions] == ["RUN", "WORKDIR"]


def test_multi_stage_names_and_lookup():
    dockerfile = parse(
        "FROM golang:1.22 AS builder\n"
        "RUN go build\n"
        "FROM ubuntu:24.04\n"
        "COPY --from=builder /out /out\n"
    )
    assert len(dockerfile.stages) == 2
    assert dockerfile.stages[0].name == "builder"
    assert dockerfile.final_stage.base == "ubuntu:24.04"
    assert dockerfile.stage_by_name("builder") is dockerfile.stages[0]
    assert dockerfile.stage_by_name("0") is dockerfile.stages[0]
    assert dockerfile.stage_by_name("nope") is None


def test_copy_flags_are_separated_from_arguments():
    dockerfile = parse(
        "FROM ubuntu:24.04\nCOPY --from=builder --chown=1000:1000 /a /b\n"
    )
    copy = dockerfile.stages[0].instructions[0]
    assert copy.flags == {"from": "builder", "chown": "1000:1000"}
    assert copy.split_args() == ["/a", "/b"]


def test_json_array_form_is_decoded():
    dockerfile = parse('FROM ubuntu:24.04\nSHELL ["/bin/bash", "-o", "pipefail", "-c"]\n')
    shell = dockerfile.stages[0].instructions[0]
    assert shell.json_args == ["/bin/bash", "-o", "pipefail", "-c"]


def test_escape_directive_switches_the_continuation_character():
    dockerfile = parse(
        "# escape=`\n"
        "FROM ubuntu:24.04\n"
        "RUN echo one `\n"
        " && echo two\n"
    )
    assert dockerfile.escape == "`"
    assert dockerfile.stages[0].instructions[0].value == "echo one  && echo two"


def test_a_leading_comment_does_not_stop_parsing():
    # Every terminal-bench Dockerfile opens with a canary comment.
    dockerfile = parse("# harbor-canary GUID abc\nFROM ubuntu:24.04\n")
    assert dockerfile.stages[0].base == "ubuntu:24.04"


def test_unknown_instruction_names_the_line():
    with pytest.raises(DockerfileError, match="line 2"):
        parse("FROM ubuntu:24.04\nRUNN echo hi\n")


def test_missing_from_is_rejected():
    with pytest.raises(DockerfileError, match="before any FROM"):
        parse("RUN echo hi\n")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # Space form: the whole remainder is the value, spaces included.
        ("ENV PATH /a /b", [("PATH", "/a /b")]),
        ("ENV GREETING hello world", [("GREETING", "hello world")]),
        # Equals form: shell-like splitting, quotes removed.
        ("ENV A=1 B=2", [("A", "1"), ("B", "2")]),
        ('ENV MSG="hello world" N=3', [("MSG", "hello world"), ("N", "3")]),
    ],
)
def test_env_parses_both_forms(source, expected):
    dockerfile = parse(f"FROM ubuntu:24.04\n{source}\n")
    assert parse_env(dockerfile.stages[0].instructions[0]) == expected


def test_arg_defaults():
    dockerfile = parse("FROM ubuntu:24.04\nARG UV_VERSION=0.9.7\nARG BARE\n")
    instructions = dockerfile.stages[0].instructions
    assert parse_arg(instructions[0]) == {"UV_VERSION": "0.9.7"}
    assert parse_arg(instructions[1]) == {"BARE": None}


def test_global_args_are_collected_before_the_first_from():
    dockerfile = parse("ARG TAG=24.04\nFROM ubuntu:${TAG}\nRUN echo hi\n")
    assert dockerfile.global_args == {"TAG": "24.04"}
    assert dockerfile.stages[0].base == "ubuntu:${TAG}"
