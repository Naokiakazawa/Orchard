"""Build-time variable expansion, as Docker performs it.

Docker expands ``$VAR`` in the arguments of ``ADD``, ``COPY``, ``ENV``,
``EXPOSE``, ``LABEL``, ``STOPSIGNAL``, ``USER``, ``VOLUME`` and ``WORKDIR``
against the accumulated ``ARG`` + ``ENV`` scope. It does **not** expand ``RUN``:
that text goes to a shell, which does its own expansion at run time against the
process environment.

Reproducing this distinction is what makes ``ENV PATH="/opt/bin:${PATH}"``
resolve against the base image's real PATH while ``RUN echo $HOME`` still sees
whatever the shell says.
"""

from __future__ import annotations

import re

#: ``$name``, ``${name}``, and the four modifiers Docker supports.
_VAR_RE = re.compile(
    r"""
    (?P<escaped>\\\$)
    | \$(?:
        \{ (?P<braced>[A-Za-z_][A-Za-z0-9_]*)
           (?: (?P<op>:\-|:\+|-|\+) (?P<alt>[^}]*) )?
        \}
        | (?P<simple>[A-Za-z_][A-Za-z0-9_]*)
      )
    """,
    re.VERBOSE,
)


def expand(value: str, scope: dict[str, str]) -> str:
    """Substitute ``$VAR`` references in *value* using *scope*.

    An undefined variable expands to the empty string, matching Docker. ``\\$``
    is a literal dollar sign.
    """

    def replace(match: re.Match[str]) -> str:
        if match.group("escaped"):
            return "$"
        simple = match.group("simple")
        if simple is not None:
            return scope.get(simple, "")

        name = match.group("braced")
        operator = match.group("op")
        alternative = match.group("alt") or ""
        current = scope.get(name)

        if operator is None:
            return current or ""
        if operator == ":-":
            return current if current else alternative
        if operator == "-":
            return current if current is not None else alternative
        if operator == ":+":
            return alternative if current else ""
        # operator == "+"
        return alternative if current is not None else ""

    return _VAR_RE.sub(replace, value)


def expand_all(values: list[str], scope: dict[str, str]) -> list[str]:
    return [expand(value, scope) for value in values]
