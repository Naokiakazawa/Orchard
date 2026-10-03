"""What the live progress bar says a run is scoring.

The number on that bar is the only score anyone sees for the nine hours a
DeepSWE job takes, and Harbor's own choice of it — first reward key
alphabetically — reads ``f2p``, roughly three times the score. These check that
the line says ``reward`` instead, and that every shape this cannot read hands
the line back to Harbor rather than inventing a number.
"""

from __future__ import annotations

import pytest

from harbor_orchard import progress

#: One DeepSWE trial, verbatim from a run's ``result.json``. Every F2P test
#: failed, so the task is unsolved, and yet two of the keys read above 0.7.
DEEPSWE_UNSOLVED = {
    "reward": 0,
    "f2p_total": 55,
    "f2p_passed": 0,
    "p2p_total": 145,
    "p2p_passed": 145,
    "f2p": 0.0,
    "p2p": 1.0,
    "partial": 0.725,
}

#: The same task solved: all fail-to-pass green, nothing regressed.
DEEPSWE_SOLVED = {
    "reward": 1,
    "f2p_total": 55,
    "f2p_passed": 55,
    "p2p_total": 145,
    "p2p_passed": 145,
    "f2p": 1.0,
    "p2p": 1.0,
    "partial": 1.0,
}

#: Half solved, and with an f2p far above that — the gap this module exists for.
DEEPSWE_PARTIAL = {**DEEPSWE_UNSOLVED, "f2p_passed": 43, "f2p": 0.78, "partial": 0.94}


class TestTrialScore:
    def test_the_reward_key_wins(self):
        assert progress.trial_score(DEEPSWE_UNSOLVED) == 0.0
        assert progress.trial_score(DEEPSWE_SOLVED) == 1.0
        # Not `partial`, not `f2p`, however much higher they read.
        assert progress.trial_score(DEEPSWE_PARTIAL) == 0.0

    def test_a_lone_metric_is_the_score(self):
        # terminal-bench and SWE-bench Pro report exactly this.
        assert progress.trial_score({"reward": 1.0}) == 1.0
        assert progress.trial_score({"accuracy": 0.0}) == 0.0

    def test_a_trial_with_no_rewards_is_a_zero(self):
        # A verifier that crashed is a task that was not solved, which is also
        # how Harbor's own metrics count it.
        assert progress.trial_score(None) == 0.0

    def test_an_unreadable_shape_says_so(self):
        # Several metrics and no `reward` among them: guessing which one is the
        # score is worse than leaving Harbor's line alone.
        assert progress.trial_score({"f2p": 1.0, "p2p": 1.0}) is None


class TestSolveRate:
    def test_it_counts_solved_over_finished(self):
        assert progress.solve_rate([DEEPSWE_SOLVED, DEEPSWE_UNSOLVED]) == (1, 2)

    def test_partial_credit_counts_as_solved(self):
        # `> 0`, not `== 1`, so a partial-credit verifier is counted the way
        # harbor_bridge.summarize counts it.
        assert progress.solve_rate([{"reward": 0.4}]) == (1, 1)

    def test_a_failed_trial_is_in_the_denominator(self):
        assert progress.solve_rate([DEEPSWE_SOLVED, None]) == (1, 2)

    def test_nothing_finished_yet_is_not_a_zero_score(self):
        assert progress.solve_rate([]) is None

    def test_one_unreadable_trial_gives_up_on_all_of_them(self):
        assert progress.solve_rate([DEEPSWE_SOLVED, {"f2p": 1.0, "p2p": 1.0}]) is None


class TestDescribe:
    def test_it_reads_as_a_solve_rate(self):
        assert progress.describe([DEEPSWE_SOLVED] + [DEEPSWE_UNSOLVED] * 3) == (
            "Solved: 0.250 (1/4)"
        )

    def test_it_does_not_report_f2p(self):
        # The whole point: 113 trials that each passed 78% of their F2P tests
        # and solved nothing score zero, and the bar has to say zero.
        assert progress.describe([DEEPSWE_PARTIAL] * 113) == "Solved: 0.000 (0/113)"

    def test_an_unreadable_shape_leaves_harbor_alone(self):
        assert progress.describe([{"f2p": 1.0, "p2p": 1.0}]) is None
        assert progress.describe([]) is None


class FakeProgress:
    """Stands in for the ``rich`` Progress the real method writes to."""

    def __init__(self):
        self.description = None

    def update(self, task, description):
        self.description = description


class FakeEvent:
    def __init__(self, result="result"):
        self.result = result


class FakeJob:
    """A Harbor ``Job`` reduced to what the patched method reads off one."""

    def __init__(self, rewards, *, metrics=None):
        self._live_rewards = {"agent__model__dataset": dict(enumerate(rewards))}
        self._metrics = metrics or {}

    def _evals_key_for_result(self, result):
        return "agent__model__dataset", "dataset"


try:  # Everything above runs on a laptop with nothing but this package.
    from harbor.job import Job
    from harbor.metrics.mean import Mean
except ImportError:  # pragma: no cover - exercised by not having Harbor
    Job = Mean = None

requires_harbor = pytest.mark.skipif(Job is None, reason="Harbor is not installed")


@pytest.fixture
def patched():
    """Install the patch and put Harbor back afterwards.

    Only if *this* call installed it: importing the provider installs it for
    the whole session, and undoing that would leave later tests — and any test
    file that imports `environment` — looking at an unpatched Harbor.
    """
    original = Job.__dict__.get("_update_metric_display")
    installed = progress.install()
    yield Job
    if installed and original is not None:
        Job._update_metric_display = original


@requires_harbor
class TestInstall:
    def test_harbor_still_has_the_method_to_patch(self):
        # The one thing that silently un-does all of this: an upgrade that
        # renames or drops the method. Failing here is the point.
        assert callable(getattr(Job, "_update_metric_display", None))
        assert callable(getattr(Job, "_evals_key_for_result", None))

    def test_harbor_would_otherwise_show_f2p(self):
        # The premise, checked against the installed Harbor rather than
        # asserted in a docstring: its metric aggregate sorts the keys, so the
        # first one — what the bar prints — is `f2p` and not `reward`.
        aggregate = Mean().compute([DEEPSWE_PARTIAL] * 4)
        name, value = next(iter(aggregate.items()))
        assert name == "f2p"
        assert value == pytest.approx(0.78)
        assert aggregate["reward"] == 0

    def test_the_bar_reports_the_solve_rate(self, patched):
        bar = FakeProgress()
        job = FakeJob([DEEPSWE_SOLVED, DEEPSWE_PARTIAL, DEEPSWE_UNSOLVED])
        patched._update_metric_display(job, FakeEvent(), bar, "task-id")
        assert bar.description == "Solved: 0.333 (1/3)"

    def test_installing_twice_does_not_wrap_twice(self, patched):
        assert progress.install() is False
        bar = FakeProgress()
        patched._update_metric_display(
            FakeJob([DEEPSWE_SOLVED]), FakeEvent(), bar, "task-id"
        )
        assert bar.description == "Solved: 1.000 (1/1)"

    def test_an_unreadable_shape_falls_through_to_harbor(self, patched):
        # `_metrics` empty is Harbor's own early return, so reaching it without
        # raising is what proves the original method ran.
        bar = FakeProgress()
        patched._update_metric_display(
            FakeJob([{"f2p": 1.0, "p2p": 1.0}]), FakeEvent(), bar, "task-id"
        )
        assert bar.description is None

    def test_a_job_it_cannot_read_falls_through_to_harbor(self, patched):
        class Unreadable(FakeJob):
            def _evals_key_for_result(self, result):
                raise AttributeError("Harbor moved this")

        bar = FakeProgress()
        patched._update_metric_display(
            Unreadable([DEEPSWE_SOLVED]), FakeEvent(), bar, "task-id"
        )
        assert bar.description is None

    def test_loading_the_provider_installs_it(self):
        # The wiring, not the patch: `environment` is imported exactly when
        # Harbor loads `harbor_orchard:OrchardEnvironment`, which is what puts
        # the patch in place before any trial can finish. Nothing else calls
        # install(), so losing that line loses the whole thing silently.
        import harbor_orchard.environment  # noqa: F401

        assert getattr(Job._update_metric_display, "_orchard_solve_rate", False)
