"""The shell fragments the sandbox runs on our behalf.

``shell.py`` exists so this text can be asserted without a cluster, and the
deadline shim is the fragment where that matters most: it is the only thing
standing between an agent Harbor has stopped waiting for and a pod that runs it
for another hour.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from harbor_orchard.settings import AGENT_KILL_GRACE_S, DEADLINE_ENV_VAR
from harbor_orchard.shell import (
    CLEAN_TREE_PREFIX,
    PREFER_IPV4_ROUTINE,
    clean_tree_routine,
    deadline_shim,
    model_patch_routine,
)


def _shim(binary: str = "/opt/sandbox-tools/bin/mini") -> str:
    return deadline_shim(
        binary, deadline_var=DEADLINE_ENV_VAR, grace=AGENT_KILL_GRACE_S
    )


class TestDeadlineShim:
    def test_it_wraps_the_cli_in_timeout(self):
        assert f'exec timeout -k {AGENT_KILL_GRACE_S} "${DEADLINE_ENV_VAR}"' in _shim()

    def test_it_runs_the_cli_unwrapped_when_no_deadline_is_set(self):
        # The fallback has to be the old behaviour, not a failure: a task image
        # is entitled to have no `timeout`, and an environment is entitled to
        # set no deadline.
        assert _shim().rstrip().endswith('exec /opt/sandbox-tools/bin/mini "$@"')

    def test_it_checks_for_timeout_before_using_it(self):
        # Without the guard, an image with no `timeout` fails every rollout
        # with exit 127 — much worse than the orphan this is preventing.
        assert "command -v timeout >/dev/null 2>&1" in _shim()

    def test_it_uses_short_options_only(self):
        # busybox `timeout`, which is what Alpine task images carry, has `-k`
        # and not `--kill-after`.
        assert "--kill-after" not in _shim()

    def test_it_quotes_the_binary(self):
        assert "'/opt/sandbox tools/mini'" in _shim("/opt/sandbox tools/mini")

    def test_it_passes_arguments_through(self):
        assert _shim().count('"$@"') == 2

    def test_it_is_valid_posix_sh(self, tmp_path):
        script = tmp_path / "shim"
        script.write_text(_shim())
        subprocess.run(["sh", "-n", str(script)], check=True)


@pytest.mark.skipif(
    shutil.which("timeout") is None,
    reason="no GNU/busybox timeout here; the behaviour it guards is the image's",
)
class TestDeadlineShimBehaviour:
    """What the rendered script actually does, run for real.

    Asserting on the text cannot catch a wrong flag order, and a wrong flag
    order is silent: `timeout` would treat the CLI as its duration argument.
    """

    def _run(self, tmp_path, deadline, *args):
        script = tmp_path / "shim"
        script.write_text(_shim(shutil.which("sleep") or "/bin/sleep"))
        script.chmod(0o755)
        env = {"PATH": "/usr/bin:/bin"}
        if deadline is not None:
            env[DEADLINE_ENV_VAR] = str(deadline)
        return subprocess.run([str(script), *args], env=env)

    def test_it_exits_124_when_the_deadline_fires(self, tmp_path):
        assert self._run(tmp_path, 1, "5").returncode == 124

    def test_it_leaves_a_shorter_run_alone(self, tmp_path):
        assert self._run(tmp_path, 5, "0.2").returncode == 0

    def test_an_unset_deadline_does_not_bound_the_cli(self, tmp_path):
        assert self._run(tmp_path, None, "0.2").returncode == 0


class TestModelPatchRoutine:
    """The snapshot that lets a run be re-graded in a container of its own.

    SWE-bench Pro V2 reports the score a *fresh* sandbox gives the agent's
    diff. Getting the diff out is this routine's whole job, and every way it
    can quietly produce an empty file is a trial that re-grades as zero with
    nothing in the log saying why — so it is run for real here rather than
    only matched against.
    """

    def test_it_is_valid_posix_sh(self):
        subprocess.run(
            ["sh", "-n"], input=model_patch_routine(), text=True, check=True
        )

    def _repo(self, tmp_path):
        repo = tmp_path / "app"
        repo.mkdir()
        run = lambda *a: subprocess.run(  # noqa: E731
            a, cwd=repo, check=True, capture_output=True
        )
        run("git", "init", "-q", "-b", "main")
        run("git", "config", "user.email", "t@t")
        run("git", "config", "user.name", "t")
        (repo / "kept.py").write_text("print('base')\n")
        (repo / "doomed.py").write_text("print('gone')\n")
        run("git", "add", "-A")
        run("git", "commit", "-qm", "base")
        return repo, run

    def _capture(self, tmp_path, repo):
        """Run the routine against *repo*, returning the patch text."""
        patch = tmp_path / "model.patch"
        # The routine looks under /app and /testbed, which no test may create;
        # point it at the fixture by running it from inside the checkout with
        # the search loop satisfied by `git rev-parse` there instead.
        routine = model_patch_routine(str(patch)).replace(
            "for __hb_candidate in /app /testbed; do",
            f"for __hb_candidate in {repo}; do",
        )
        done = subprocess.run(
            ["sh", "-c", routine], capture_output=True, text=True, cwd=repo
        )
        assert done.returncode == 0, done.stderr
        return patch.read_text(encoding="utf-8"), done.stdout

    def test_it_captures_edits_additions_and_deletions(self, tmp_path):
        # A bare `git diff` misses the last two, and an agent that adds a file
        # is the common case, not the exotic one.
        repo, run = self._repo(tmp_path)
        (repo / "kept.py").write_text("print('fixed')\n")
        (repo / "added.py").write_text("print('new')\n")
        (repo / "doomed.py").unlink()

        patch, stdout = self._capture(tmp_path, repo)

        assert "print('fixed')" in patch
        assert "added.py" in patch
        assert "doomed.py" in patch
        assert "model.patch bytes:" in stdout

    def test_it_leaves_the_index_as_it_found_it(self, tmp_path):
        # On a shared-verifier task the grader runs next in this same
        # container; staging the agent's work and walking away would hand it a
        # repository state the agent never created.
        repo, run = self._repo(tmp_path)
        (repo / "kept.py").write_text("print('fixed')\n")

        self._capture(tmp_path, repo)

        staged = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            cwd=repo, capture_output=True, text=True, check=True,
        )
        assert staged.stdout.strip() == ""

    def test_an_untouched_checkout_yields_an_empty_patch(self, tmp_path):
        # Not an error: an agent that changed nothing is a real outcome, and
        # the replay agent is the thing that turns it into a zero.
        repo, _ = self._repo(tmp_path)

        patch, _ = self._capture(tmp_path, repo)

        assert patch == ""

    def test_it_exits_zero_with_no_checkout_to_capture(self, tmp_path):
        # The capture runs after the rollout is over. Failing here would turn a
        # finished trial into a failed one.
        empty = tmp_path / "nowhere"
        empty.mkdir()
        routine = model_patch_routine(str(tmp_path / "model.patch")).replace(
            "for __hb_candidate in /app /testbed; do",
            f"for __hb_candidate in {empty}; do",
        )

        done = subprocess.run(
            ["sh", "-c", routine], capture_output=True, text=True, cwd=empty
        )

        assert done.returncode == 0
        assert "no patch captured" in done.stderr


class TestCleanTreeRoutine:
    """The pre-flight that says whether this trial is a measurement at all.

    An agent is entitled to the checkout the image shipped. When 96 trials of
    the 2026-09-24 V2 sweep did not get one — a re-submitted exec had started a
    second agent on top of a first one's finished work — nothing in the run
    said so, and the scores were reported. This routine is what notices, so
    what it calls clean has to be exactly what a pristine image looks like.
    """

    def test_it_is_valid_posix_sh(self):
        subprocess.run(["sh", "-n"], input=clean_tree_routine(), text=True, check=True)

    def _repo(self, tmp_path):
        repo = tmp_path / "app"
        repo.mkdir()
        run = lambda *a: subprocess.run(  # noqa: E731
            a, cwd=repo, check=True, capture_output=True
        )
        run("git", "init", "-q", "-b", "main")
        run("git", "config", "user.email", "t@t")
        run("git", "config", "user.name", "t")
        (repo / "src.py").write_text("print('base')\n")
        run("git", "add", "-A")
        run("git", "commit", "-qm", "base")
        return repo

    def _check(self, cwd, repo=None):
        """Run the routine, returning its first report line."""
        routine = clean_tree_routine()
        if repo is not None:
            routine = routine.replace(
                "for __hb_candidate in /app /testbed; do",
                f"for __hb_candidate in {repo}; do",
            )
        done = subprocess.run(
            ["sh", "-c", routine], capture_output=True, text=True, cwd=cwd
        )
        assert done.returncode == 0, done.stderr
        report = next(
            (
                line
                for line in done.stdout.splitlines()
                if line.startswith(CLEAN_TREE_PREFIX)
            ),
            "",
        )
        return report, done.stdout

    def test_a_pristine_checkout_reads_clean(self, tmp_path):
        repo = self._repo(tmp_path)

        report, _ = self._check(repo, repo)

        assert report.startswith(f"{CLEAN_TREE_PREFIX}clean")

    def test_a_modified_tracked_file_reads_dirty_and_names_it(self, tmp_path):
        repo = self._repo(tmp_path)
        (repo / "src.py").write_text("print('someone was here')\n")

        report, stdout = self._check(repo, repo)

        assert report.startswith(f"{CLEAN_TREE_PREFIX}dirty")
        assert "1 tracked file(s)" in report
        # The paths are what turn the failure into a diagnosis: on the V2 sweep
        # they were the task's own gold-patch targets every time.
        assert "src.py" in stdout

    def test_untracked_files_alone_read_clean(self, tmp_path):
        # This is the whole reason for `--untracked-files=no`. A pristine V2
        # image carries untracked files — upstream's leak probe reports
        # UNTRACKED=3 with MODIFIED=0 on a clean NodeBB image — so counting
        # them would fail every trial on those images and teach the operator
        # to switch the check off.
        repo = self._repo(tmp_path)
        (repo / "scratch.log").write_text("build output\n")

        report, _ = self._check(repo, repo)

        assert report.startswith(f"{CLEAN_TREE_PREFIX}clean")

    def test_an_image_with_no_checkout_reads_unknown(self, tmp_path):
        # Not every dataset is a git repository, and a check that cannot run is
        # not evidence of anything. Reporting `unknown` is what keeps this from
        # failing rollouts it knows nothing about.
        report, _ = self._check(tmp_path)

        assert report.startswith(f"{CLEAN_TREE_PREFIX}unknown")

    def test_it_never_fails_on_its_own_terms(self, tmp_path):
        # The caller decides what a dirty tree costs. If the routine exited
        # non-zero the exec would raise first, and a dirty tree would be
        # indistinguishable from a broken pod.
        repo = self._repo(tmp_path)
        (repo / "src.py").write_text("changed\n")

        done = subprocess.run(
            ["sh", "-c", clean_tree_routine().replace(
                "for __hb_candidate in /app /testbed; do",
                f"for __hb_candidate in {repo}; do",
            )],
            capture_output=True,
            text=True,
            cwd=repo,
        )

        assert done.returncode == 0


class TestPreferIPv4:
    """``localhost`` must mean 127.0.0.1 in a dual-stack pod.

    This one is executable here: POSIX shell and awk over two files. Each test
    points it at a fake ``/etc`` and checks what it leaves behind; the only edit
    is the path prefix.

    What it is defending: on the 2026-09-24 SWE-bench Pro V2 oracle run both of
    the two zeros were ``connect ECONNREFUSED ::1:4568`` against a NodeBB that
    had just logged "listening on 0.0.0.0:4568", and all 23 passing trials had
    none.
    """

    @staticmethod
    def _routine(etc) -> str:
        return PREFER_IPV4_ROUTINE.replace("/etc/", f"{etc}/")

    @classmethod
    def _run(cls, tmp_path, hosts: str | None, gai: str | None = None):
        etc = tmp_path / "etc"
        etc.mkdir()
        if hosts is not None:
            (etc / "hosts").write_text(hosts)
        if gai is not None:
            (etc / "gai.conf").write_text(gai)
        done = subprocess.run(
            ["sh", "-c", cls._routine(etc)],
            capture_output=True,
            text=True,
            check=True,
        )

        def read(name: str) -> str | None:
            path = etc / name
            return path.read_text() if path.exists() else None

        return done.stdout.strip(), read("hosts"), read("gai.conf")

    def test_it_drops_only_the_localhost_alias(self, tmp_path):
        # `ip6-localhost` still has to resolve; it is the alias that lies.
        _, hosts, _ = self._run(
            tmp_path,
            "127.0.0.1\tlocalhost\n::1\tlocalhost ip6-localhost ip6-loopback\n",
        )
        assert hosts == "127.0.0.1\tlocalhost\n::1 ip6-localhost ip6-loopback\n"

    def test_it_removes_a_line_that_was_only_the_alias(self, tmp_path):
        _, hosts, _ = self._run(tmp_path, "127.0.0.1 localhost\n::1 localhost\n")
        assert hosts == "127.0.0.1 localhost\n"

    def test_it_leaves_other_entries_alone(self, tmp_path):
        # A pod's own name and the cluster's records are not ours to touch.
        _, hosts, _ = self._run(
            tmp_path,
            "127.0.0.1\tlocalhost\n"
            "::1\tlocalhost ip6-localhost\n"
            "fe00::0\tip6-localnet\n"
            "10.0.1.5\torchard-trial-7\n",
        )
        spaced = hosts.replace("\t", " ")
        assert "fe00::0 ip6-localnet" in spaced
        assert "10.0.1.5 orchard-trial-7" in spaced

    def test_it_writes_the_glibc_precedence_rule(self, tmp_path):
        # What musl ignores and glibc reads, for names not in `hosts`.
        _, _, gai = self._run(tmp_path, "127.0.0.1 localhost\n::1 localhost\n")
        assert gai is not None and "precedence ::ffff:0:0/96" in gai

    def test_running_it_twice_changes_nothing(self, tmp_path):
        # Harbor may restart an environment, and a doubled precedence rule is a
        # broken gai.conf.
        out, hosts, gai = self._run(
            tmp_path, "127.0.0.1\tlocalhost\n::1\tlocalhost ip6-localhost\n"
        )
        assert out == "hosts gai.conf"

        etc = tmp_path / "etc"
        again = subprocess.run(
            ["sh", "-c", self._routine(etc)],
            capture_output=True,
            text=True,
            check=True,
        )

        assert again.stdout.strip() == "nothing to change"
        assert (etc / "hosts").read_text() == hosts
        assert (etc / "gai.conf").read_text() == gai

    def test_it_succeeds_when_there_is_nothing_to_fix(self, tmp_path):
        # A single-stack image is already what this is trying to produce.
        out, hosts, _ = self._run(tmp_path, "127.0.0.1 localhost\n")
        assert hosts == "127.0.0.1 localhost\n"
        assert out == "gai.conf"

    def test_a_read_only_etc_is_not_an_error(self, tmp_path):
        # Best-effort: an unwritable pod keeps the resolver it shipped with,
        # rather than losing the trial over a file it could not open.
        etc = tmp_path / "etc"
        etc.mkdir()
        (etc / "hosts").write_text("::1 localhost\n")
        (etc / "hosts").chmod(0o444)
        etc.chmod(0o555)
        try:
            done = subprocess.run(
                ["sh", "-c", self._routine(etc)], capture_output=True, text=True
            )
        finally:
            etc.chmod(0o755)

        assert done.returncode == 0
        assert (etc / "hosts").read_text() == "::1 localhost\n"
