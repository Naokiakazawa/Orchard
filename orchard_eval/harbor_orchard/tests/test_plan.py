"""Translation tests: build state, path resolution, and COPY semantics."""

from __future__ import annotations

import pytest

from harbor_orchard.dockerfile import parse
from harbor_orchard.plan import (
    CopyOp,
    RunOp,
    UnsupportedInstruction,
    translate,
)
from harbor_orchard.shell import CONTEXT_ROOT_NAME, wrap_user

BASE_ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/root"}


def plan_for(tmp_path, source, **kwargs):
    return translate(parse(source), context_dir=tmp_path, base_env=dict(BASE_ENV), **kwargs)


def runs(plan):
    return [op for op in plan.operations if isinstance(op, RunOp)]


def copies(plan):
    return [op for op in plan.operations if isinstance(op, CopyOp)]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def test_env_expands_against_the_base_image_path(tmp_path):
    # The idiom this exists for. Expanding $PATH against an empty scope would
    # leave PATH=/opt/vep and break every command after this line.
    plan = plan_for(
        tmp_path,
        'FROM python:3.11-slim\nENV PATH="/opt/vep:${PATH}"\nRUN vep --help\n',
    )
    assert plan.state.env["PATH"] == "/opt/vep:/usr/local/bin:/usr/bin:/bin"
    assert runs(plan)[0].env["PATH"] == "/opt/vep:/usr/local/bin:/usr/bin:/bin"


def test_env_pairs_resolve_left_to_right(tmp_path):
    plan = plan_for(tmp_path, "FROM ubuntu:24.04\nENV A=1 B=${A}2\n")
    assert plan.state.env["B"] == "12"


def test_undefined_variables_expand_to_empty(tmp_path):
    plan = plan_for(tmp_path, "FROM ubuntu:24.04\nENV OUT=x${NOPE}y\n")
    assert plan.state.env["OUT"] == "xy"


def test_default_value_syntax(tmp_path):
    plan = plan_for(tmp_path, "FROM ubuntu:24.04\nENV OUT=${NOPE:-fallback}\n")
    assert plan.state.env["OUT"] == "fallback"


def test_run_is_not_expanded_at_build_time(tmp_path):
    # The shell must do this at run time; expanding here would resolve $HOME
    # against the build scope rather than the executing user's environment.
    plan = plan_for(tmp_path, "FROM ubuntu:24.04\nRUN echo $HOME > /tmp/x\n")
    assert "$HOME" in runs(plan)[0].script


# ---------------------------------------------------------------------------
# WORKDIR / USER / SHELL
# ---------------------------------------------------------------------------


def test_workdir_is_created_and_resolves_relatively(tmp_path):
    plan = plan_for(tmp_path, "FROM ubuntu:24.04\nWORKDIR /app\nWORKDIR src\n")
    assert plan.state.workdir == "/app/src"
    mkdirs = [op.script for op in runs(plan)]
    assert mkdirs == ["mkdir -p /app", "mkdir -p /app/src"]


def test_workdir_applies_to_later_runs(tmp_path):
    plan = plan_for(tmp_path, "FROM ubuntu:24.04\nWORKDIR /app\nRUN make\n")
    assert runs(plan)[-1].cwd == "/app"


def test_user_switches_subsequent_steps(tmp_path):
    plan = plan_for(tmp_path, "FROM ubuntu:24.04\nUSER agent\nRUN whoami\n")
    step = runs(plan)[-1]
    assert step.user == "agent"
    assert wrap_user(step.command(), step.user) == (
        "su -m -s /bin/sh -c '/bin/sh -c whoami' agent"
    )


def test_root_needs_no_su_wrapper(tmp_path):
    plan = plan_for(tmp_path, "FROM ubuntu:24.04\nUSER root\nRUN whoami\n")
    step = runs(plan)[-1]
    assert wrap_user(step.command(), step.user) == step.command()


def test_shell_directive_is_honoured(tmp_path):
    # `pipefail` is the reason this matters: flattening the SHELL would let a
    # failing stage of a pipeline report success.
    plan = plan_for(
        tmp_path,
        'FROM ubuntu:24.04\nSHELL ["/bin/bash", "-o", "pipefail", "-c"]\n'
        "RUN false | true\n",
    )
    assert runs(plan)[0].command() == "/bin/bash -o pipefail -c 'false | true'"


def test_default_shell_is_sh(tmp_path):
    plan = plan_for(tmp_path, "FROM ubuntu:24.04\nRUN echo hi\n")
    assert runs(plan)[0].command() == "/bin/sh -c 'echo hi'"


def test_exec_form_run_bypasses_the_shell(tmp_path):
    plan = plan_for(tmp_path, 'FROM ubuntu:24.04\nRUN ["/bin/echo", "a b"]\n')
    assert runs(plan)[0].command() == "/bin/echo 'a b'"


# ---------------------------------------------------------------------------
# ARG
# ---------------------------------------------------------------------------


def test_arg_default_reaches_run_as_an_environment_variable(tmp_path):
    plan = plan_for(
        tmp_path, "FROM ubuntu:24.04\nARG UV_VERSION=0.9.7\nRUN echo $UV_VERSION\n"
    )
    assert runs(plan)[0].env["UV_VERSION"] == "0.9.7"


def test_build_arg_overrides_the_default(tmp_path):
    plan = plan_for(
        tmp_path,
        "FROM ubuntu:24.04\nARG UV_VERSION=0.9.7\nRUN true\n",
        build_args={"UV_VERSION": "1.0.0"},
    )
    assert runs(plan)[0].env["UV_VERSION"] == "1.0.0"


def test_a_global_arg_is_not_in_stage_scope_until_redeclared(tmp_path):
    plan = plan_for(tmp_path, "ARG TAG=24.04\nFROM ubuntu:${TAG}\nRUN true\n")
    assert "TAG" not in runs(plan)[0].env
    assert plan.base_image == "ubuntu:24.04"


def test_redeclaring_a_global_arg_inherits_its_default(tmp_path):
    plan = plan_for(tmp_path, "ARG TAG=24.04\nFROM ubuntu:${TAG}\nARG TAG\nRUN true\n")
    assert runs(plan)[0].env["TAG"] == "24.04"


def test_env_shadows_arg(tmp_path):
    plan = plan_for(tmp_path, "FROM ubuntu:24.04\nARG V=1\nENV V=2\nRUN true\n")
    assert runs(plan)[0].env["V"] == "2"


# ---------------------------------------------------------------------------
# COPY
# ---------------------------------------------------------------------------


def test_copy_resolves_destination_against_workdir(tmp_path):
    (tmp_path / "app.py").write_text("x")
    plan = plan_for(tmp_path, "FROM ubuntu:24.04\nWORKDIR /srv\nCOPY app.py ./bin/\n")
    assert copies(plan)[0].dest == "/srv/bin/"


def test_copy_keeps_the_trailing_slash_that_means_directory(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "a.txt").write_text("x")
    plan = plan_for(tmp_path, "FROM ubuntu:24.04\nCOPY data /app/data/\n")
    operation = copies(plan)[0]
    assert operation.dest == "/app/data/"
    assert [source.name for source in operation.sources] == ["data"]


def test_copy_of_the_whole_context_is_staged_as_a_directory(tmp_path):
    # `COPY . /app` copies the context's *contents*, which is the same rule as
    # any directory source — so it is staged as one rather than special-cased.
    (tmp_path / "pkg").mkdir()
    plan = plan_for(tmp_path, "FROM ubuntu:24.04\nCOPY . /app\n")
    assert [source.name for source in copies(plan)[0].sources] == [CONTEXT_ROOT_NAME]


def test_copy_expands_globs_in_the_context(tmp_path):
    for name in ("a.txt", "b.txt", "c.md"):
        (tmp_path / name).write_text("x")
    plan = plan_for(tmp_path, "FROM ubuntu:24.04\nCOPY *.txt /app/\n")
    assert sorted(source.name for source in copies(plan)[0].sources) == ["a.txt", "b.txt"]


def test_copy_of_a_missing_path_fails_at_translation(tmp_path):
    with pytest.raises(Exception, match="does not exist"):
        plan_for(tmp_path, "FROM ubuntu:24.04\nCOPY nope.txt /app/\n")


def test_copy_cannot_escape_the_build_context(tmp_path):
    with pytest.raises(Exception, match="does not exist|outside the build context"):
        plan_for(tmp_path, "FROM ubuntu:24.04\nCOPY ../secret /app/\n")


def test_copy_from_records_a_stage_dependency(tmp_path):
    plan = plan_for(
        tmp_path,
        "FROM golang:1.22 AS builder\nRUN go build\n"
        "FROM ubuntu:24.04\nCOPY --from=builder /out/app /usr/bin/app\n",
    )
    operation = copies(plan)[0]
    assert operation.from_stage == "builder"
    assert plan.depends_on == ["builder"]
    assert [source.remote_path for source in operation.sources] == ["/out/app"]


def test_copy_url_is_rejected_but_add_url_is_not(tmp_path):
    with pytest.raises(Exception, match="use ADD"):
        plan_for(tmp_path, "FROM ubuntu:24.04\nCOPY https://x/y.tgz /tmp/\n")
    plan = plan_for(tmp_path, "FROM ubuntu:24.04\nADD https://x/y.tgz /tmp/\n")
    assert copies(plan)[0].sources[0].url == "https://x/y.tgz"


# ---------------------------------------------------------------------------
# Metadata and refusals
# ---------------------------------------------------------------------------


def test_metadata_instructions_do_not_emit_steps(tmp_path):
    plan = plan_for(
        tmp_path,
        "FROM ubuntu:24.04\n"
        'CMD ["sleep", "infinity"]\n'
        'ENTRYPOINT []\n'
        "EXPOSE 8080\n"
        "LABEL org.opencontainers.image.title=demo\n",
    )
    assert plan.operations == []
    assert plan.state.cmd == ["sleep", "infinity"]
    assert plan.state.entrypoint == []
    assert plan.state.exposed_ports == ["8080"]
    assert plan.state.labels == {"org.opencontainers.image.title": "demo"}


def test_run_mount_is_refused_rather_than_ignored(tmp_path):
    # Silently dropping the mount would change what the command can see, and the
    # failure would surface much later as a task that mysteriously scores zero.
    with pytest.raises(UnsupportedInstruction, match="image builder"):
        plan_for(
            tmp_path,
            "FROM ubuntu:24.04\nRUN --mount=type=cache,target=/root/.cache pip install x\n",
        )


def test_onbuild_is_refused(tmp_path):
    with pytest.raises(UnsupportedInstruction, match="ONBUILD"):
        plan_for(tmp_path, "FROM ubuntu:24.04\nONBUILD RUN echo hi\n")
