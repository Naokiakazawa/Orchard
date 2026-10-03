"""Parse a Dockerfile into the instruction stream needed to replay it as shell.

This is deliberately not a general Dockerfile implementation. It covers exactly
what a build has to reproduce when there is no image builder — when the "build"
is a sequence of commands run inside a live container. Layer caching, build
secrets, mounts and BuildKit frontends have no meaning in that world and are
rejected rather than silently ignored, because a build that quietly skips
``RUN --mount=type=cache`` produces an environment the task author never tested.

The three things that are easy to get wrong, and that this module exists to get
right:

**Line continuations.** ``\\`` at end of line joins the next line *without*
inserting whitespace, and comment lines inside a continuation are dropped. A
parser that joins with ``" "`` corrupts ``RUN pip install foo==1.\\``+``0``.

**Heredocs.** Three terminal-bench Dockerfiles open a shell heredoc
(``RUN python - <<'PY'``) whose body is plain following lines with no
continuation markers. A line-oriented parser reads the body as separate
instructions and produces garbage. Openers are only treated as heredocs when a
matching terminator line actually exists, so ``RUN echo "a << b"`` is left
alone.

**Variable expansion.** Docker expands ``$VAR`` in ``COPY``/``WORKDIR``/``ENV``
and friends at build time, but *not* in ``RUN`` — there the shell does it at run
time. Expanding a ``RUN`` here would resolve variables against the wrong scope;
see :mod:`harbor_orchard.plan`, which expands the former and exports the
environment for the latter.
"""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field

DEFAULT_ESCAPE = "\\"


class DockerfileError(ValueError):
    """The Dockerfile could not be parsed."""


#: ``# key=value`` ahead of the first instruction. Docker honours only ``escape``
#: and ``syntax``, and stops looking at the first line that is not itself a
#: directive — including an ordinary comment.
_DIRECTIVE_RE = re.compile(r"^#\s*(escape|syntax)\s*=\s*(\S+)\s*$", re.IGNORECASE)

_INSTRUCTION_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)(?:[ \t]+(.*))?$", re.DOTALL)

#: ``--flag`` or ``--flag=value`` at the head of an instruction's arguments.
_FLAG_RE = re.compile(r"^--([A-Za-z][A-Za-z0-9-]*)(?:=(\S*))?(?:[ \t]+|$)")

#: A heredoc opener: ``<<WORD``, ``<<-WORD``, ``<<'WORD'``, ``<<"WORD"``. The
#: negative lookahead keeps ``<<<`` (a herestring) out.
_HEREDOC_RE = re.compile(r"<<(-?)(?!<)(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\2")

#: Instructions whose arguments may start with ``--flag``.
_FLAG_INSTRUCTIONS = frozenset({"ADD", "COPY", "FROM", "RUN", "HEALTHCHECK"})

#: Instructions that accept the JSON-array ("exec") form.
_JSON_INSTRUCTIONS = frozenset({"ADD", "CMD", "COPY", "ENTRYPOINT", "RUN", "SHELL", "VOLUME"})

#: Everything Docker defines. An instruction outside this set is a typo, and a
#: typo that reaches the sandbox becomes a confusing runtime failure instead of
#: a parse error pointing at a line number.
KNOWN_INSTRUCTIONS = frozenset(
    {
        "ADD",
        "ARG",
        "CMD",
        "COPY",
        "ENTRYPOINT",
        "ENV",
        "EXPOSE",
        "FROM",
        "HEALTHCHECK",
        "LABEL",
        "MAINTAINER",
        "ONBUILD",
        "RUN",
        "SHELL",
        "STOPSIGNAL",
        "USER",
        "VOLUME",
        "WORKDIR",
    }
)


@dataclass(frozen=True)
class Instruction:
    """One parsed instruction, with continuations and heredocs already folded in."""

    name: str
    value: str
    line: int
    flags: dict[str, str] = field(default_factory=dict)
    #: Populated only for the JSON-array form, which suppresses shell parsing.
    json_args: list[str] | None = None
    #: ``(word, body)`` per heredoc opened on this instruction, in source order.
    heredocs: tuple[tuple[str, str], ...] = ()

    def split_args(self) -> list[str]:
        """Shell-split ``value``, or return the JSON array when there is one.

        Only meaningful for instructions whose arguments are paths or words
        (``COPY``, ``FROM``, ``WORKDIR``…). Calling it on a ``RUN`` is a bug:
        a shell script is not a token list.
        """
        if self.json_args is not None:
            return list(self.json_args)
        try:
            return shlex.split(self.value, posix=True)
        except ValueError as exc:
            raise DockerfileError(
                f"line {self.line}: cannot split {self.name} arguments: {exc}"
            ) from exc

    @property
    def script(self) -> str:
        """The shell text of a ``RUN``, heredoc bodies restored.

        BuildKit hands the whole thing — command line, heredoc bodies and
        terminators — to the shell as one script, so reassembling it verbatim is
        both the simplest and the most faithful representation.
        """
        if not self.heredocs:
            return self.value
        parts = [self.value]
        for word, body in self.heredocs:
            parts.append(body)
            parts.append(word)
        return "\n".join(parts)


@dataclass
class Stage:
    """One ``FROM`` and everything after it, up to the next ``FROM``."""

    index: int
    base: str
    name: str | None
    platform: str | None
    line: int
    instructions: list[Instruction] = field(default_factory=list)


@dataclass
class Dockerfile:
    stages: list[Stage]
    #: ``ARG``s declared before the first ``FROM``. They are in scope for
    #: ``FROM`` image references and, unlike ordinary ARGs, for no ``RUN`` until
    #: a stage redeclares them.
    global_args: dict[str, str | None]
    escape: str = DEFAULT_ESCAPE
    syntax: str | None = None

    @property
    def final_stage(self) -> Stage:
        return self.stages[-1]

    def global_args_scope(self) -> dict[str, str]:
        """Pre-``FROM`` ARGs as an expansion scope, undeclared defaults empty."""
        return {name: value or "" for name, value in self.global_args.items()}

    def stage_by_name(self, name: str) -> Stage | None:
        """Look up a stage by ``AS`` alias or by numeric index, as ``--from`` does."""
        lowered = name.lower()
        for stage in self.stages:
            if stage.name and stage.name.lower() == lowered:
                return stage
        if name.isdigit():
            index = int(name)
            if 0 <= index < len(self.stages):
                return self.stages[index]
        return None


def parse(text: str) -> Dockerfile:
    """Parse Dockerfile *text*.

    Raises :class:`DockerfileError` with a line number for anything malformed.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    escape, syntax, cursor = _parse_directives(lines)

    stages: list[Stage] = []
    global_args: dict[str, str | None] = {}

    for instruction in _iter_instructions(lines, cursor, escape):
        if instruction.name == "FROM":
            stages.append(_make_stage(instruction, len(stages)))
            continue
        if not stages:
            if instruction.name == "ARG":
                global_args.update(parse_arg(instruction))
                continue
            if instruction.name in ("MAINTAINER", "LABEL"):
                continue
            raise DockerfileError(
                f"line {instruction.line}: {instruction.name} appears before any FROM"
            )
        stages[-1].instructions.append(instruction)

    if not stages:
        raise DockerfileError("Dockerfile contains no FROM instruction")

    return Dockerfile(
        stages=stages, global_args=global_args, escape=escape, syntax=syntax
    )


def parse_file(path) -> Dockerfile:
    from pathlib import Path

    return parse(Path(path).read_text(encoding="utf-8"))


def _parse_directives(lines: list[str]) -> tuple[str, str | None, int]:
    escape = DEFAULT_ESCAPE
    syntax: str | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        match = _DIRECTIVE_RE.match(line)
        if match is None:
            break
        key, value = match.group(1).lower(), match.group(2)
        if key == "escape":
            if value not in ("\\", "`"):
                raise DockerfileError(f"line {index + 1}: invalid escape {value!r}")
            escape = value
        else:
            syntax = value
        index += 1
    return escape, syntax, index


def _make_stage(instruction: Instruction, index: int) -> Stage:
    args = instruction.split_args()
    if not args:
        raise DockerfileError(f"line {instruction.line}: FROM requires an image")
    base = args[0]
    name: str | None = None
    if len(args) >= 3 and args[1].upper() == "AS":
        name = args[2]
    elif len(args) != 1:
        raise DockerfileError(
            f"line {instruction.line}: cannot parse FROM {instruction.value!r}"
        )
    return Stage(
        index=index,
        base=base,
        name=name,
        platform=instruction.flags.get("platform"),
        line=instruction.line,
    )


def _iter_instructions(lines: list[str], cursor: int, escape: str):
    index = cursor
    while index < len(lines):
        line = lines[index]
        if not line.strip() or line.lstrip().startswith("#"):
            index += 1
            continue

        start_line = index + 1
        logical, index = _join_continuations(lines, index, escape)
        heredocs, index = _consume_heredocs(lines, index, logical)

        match = _INSTRUCTION_RE.match(logical)
        if match is None:
            raise DockerfileError(f"line {start_line}: cannot parse {logical!r}")

        name = match.group(1).upper()
        if name not in KNOWN_INSTRUCTIONS:
            raise DockerfileError(f"line {start_line}: unknown instruction {name}")

        rest = (match.group(2) or "").strip()
        flags: dict[str, str] = {}
        if name in _FLAG_INSTRUCTIONS:
            while True:
                flag_match = _FLAG_RE.match(rest)
                if flag_match is None:
                    break
                value = flag_match.group(2)
                flags[flag_match.group(1)] = "true" if value is None else value
                rest = rest[flag_match.end() :].lstrip()

        json_args: list[str] | None = None
        if name in _JSON_INSTRUCTIONS and rest.startswith("["):
            try:
                decoded = json.loads(rest)
            except ValueError:
                decoded = None
            if isinstance(decoded, list) and all(isinstance(x, str) for x in decoded):
                json_args = decoded

        yield Instruction(
            name=name,
            value=rest,
            line=start_line,
            flags=flags,
            json_args=json_args,
            heredocs=heredocs,
        )


def _join_continuations(lines: list[str], index: int, escape: str) -> tuple[str, int]:
    """Fold ``\\``-continued lines into one logical line.

    Joins with no separator, because the escape character replaces only itself
    and the newline — the whitespace either side of it is part of the command.
    """
    parts: list[str] = []
    while index < len(lines):
        stripped = lines[index].rstrip()
        if _continues(stripped, escape):
            parts.append(stripped[: -len(escape)])
            index += 1
            # Docker drops comment lines that interrupt a continuation, but a
            # blank line ends the instruction.
            while index < len(lines) and lines[index].lstrip().startswith("#"):
                index += 1
            continue
        parts.append(stripped)
        index += 1
        break
    return "".join(parts), index


def _continues(stripped: str, escape: str) -> bool:
    """True when a line ends in an *unescaped* escape character.

    ``RUN printf 'a\\\\'`` ends with two backslashes: the first escapes the
    second, so the instruction ends there. Counting the trailing run and
    checking for an odd length is what distinguishes the two cases.
    """
    if not stripped.endswith(escape):
        return False
    count = 0
    position = len(stripped)
    while position >= len(escape) and stripped[position - len(escape) : position] == escape:
        count += 1
        position -= len(escape)
    return count % 2 == 1


def _consume_heredocs(
    lines: list[str], index: int, logical: str
) -> tuple[tuple[tuple[str, str], ...], int]:
    """Collect heredoc bodies opened by *logical*, starting at *index*.

    An opener without a matching terminator is not a heredoc — it is ``<<``
    inside a quoted string — so the whole instruction is left untouched rather
    than swallowing the rest of the file.
    """
    openers = [
        (match.group(1) == "-", match.group(3))
        for match in _HEREDOC_RE.finditer(logical)
    ]
    if not openers:
        return (), index

    collected: list[tuple[str, str]] = []
    cursor = index
    for dash, word in openers:
        body: list[str] = []
        probe = cursor
        found = False
        while probe < len(lines):
            candidate = lines[probe]
            terminator = candidate.lstrip("\t") if dash else candidate
            if terminator.rstrip() == word:
                found = True
                break
            body.append(candidate)
            probe += 1
        if not found:
            return (), index
        collected.append((word, "\n".join(body)))
        cursor = probe + 1
    return tuple(collected), cursor


# ---------------------------------------------------------------------------
# Instruction-specific argument parsing
# ---------------------------------------------------------------------------


def parse_env(instruction: Instruction) -> list[tuple[str, str]]:
    """Parse ``ENV``/``LABEL`` in both of Docker's forms.

    ``ENV key value`` takes the entire rest of the line as the value, spaces and
    all; ``ENV key=value key2=value2`` splits like a shell. Choosing the wrong
    one turns ``ENV PATH /a /b`` into a truncated PATH.
    """
    value = instruction.value
    if not value.strip():
        raise DockerfileError(f"line {instruction.line}: empty {instruction.name}")

    first, _, remainder = value.partition(" ")
    if "=" not in first:
        key = first.strip()
        if not key:
            raise DockerfileError(f"line {instruction.line}: malformed {instruction.name}")
        return [(key, _strip_quotes(remainder.strip()))]

    pairs: list[tuple[str, str]] = []
    for token in _split_preserving_quotes(value, instruction.line):
        if "=" not in token:
            raise DockerfileError(
                f"line {instruction.line}: {instruction.name} token {token!r} has no '='"
            )
        key, _, item = token.partition("=")
        pairs.append((key.strip(), item))
    return pairs


def parse_arg(instruction: Instruction) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for token in _split_preserving_quotes(instruction.value, instruction.line):
        if "=" in token:
            key, _, value = token.partition("=")
            result[key.strip()] = value
        else:
            result[token.strip()] = None
    return result


def _split_preserving_quotes(value: str, line: int) -> list[str]:
    lexer = shlex.shlex(value, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError as exc:
        raise DockerfileError(f"line {line}: {exc}") from exc


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value
