"""Putting the score on Harbor's live progress line.

Harbor's running bar carries exactly one number, and it is not the benchmark's
score. ``Job._update_metric_display`` computes the job's first metric over the
trials that have finished and prints the *first key* of the result; the keys
come out of ``harbor.metrics.base.aggregate_reward_dicts``, which sorts them.
So the number on screen is whichever reward key sorts first alphabetically.

On a verifier that reports one reward — terminal-bench, SWE-bench Pro — that is
the score, and this module changes nothing but the label. On DeepSWE it is not.
That verifier reports ``f2p``, ``f2p_passed``, ``f2p_total``, ``p2p``,
``p2p_passed``, ``p2p_total``, ``partial`` and ``reward``, so the line reads
``F2P: 0.710``: the mean share of fail-to-pass tests left green. The benchmark's
own score is ``reward``, which is all-or-nothing — every F2P test green *and* no
P2P regression — and on the same run it was 0.21. Three times the score, sitting
where the score belongs, for the nine hours the run takes: that is what this
module exists to prevent, not a matter of taste about labels.

The line becomes ``Solved: 0.213 (24/113)`` — the same count
``orchard-eval``'s own summary will report when the job finishes, so watching
the run predicts the scoreboard instead of contradicting it.

Everything here is display-only and best effort. A reward shape this cannot
read, a Harbor that has moved the method, a Harbor that is not installed at all:
each one leaves Harbor's own line exactly as it was, because a progress bar is
not worth a trial.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

logger = logging.getLogger(__name__)

#: The key a Harbor verifier writes its score to. Harbor scores on this when it
#: is present; ``orchard_evalkit.harbor_bridge._primary_reward`` picks the same
#: one for the final summary, and the two must agree or the live line stops
#: predicting the scoreboard.
SCORE_KEY = "reward"

#: Marks the patched method, so installing twice is a no-op rather than a
#: wrapper around a wrapper.
_MARKER = "_orchard_solve_rate"


def trial_score(reward: dict[str, Any] | None) -> float | None:
    """The scalar one trial is scored on, or ``None`` if the shape is unknown.

    A trial whose verifier never produced rewards is a zero, not an unknown:
    Harbor's own metrics count it as one (``aggregate_reward_dicts`` substitutes
    0 for ``None``), and a crashed verifier is a trial that did not solve its
    task. ``None`` here means something else — a reward dict with several keys
    and no ``reward`` among them — where guessing which of them is the score
    would be worse than leaving Harbor's line alone.
    """
    if reward is None:
        return 0.0
    if SCORE_KEY in reward:
        return float(reward[SCORE_KEY])
    if len(reward) == 1:
        return float(next(iter(reward.values())))
    return None


def solve_rate(rewards: list[dict[str, Any] | None]) -> tuple[int, int] | None:
    """``(solved, finished)`` over the trials Harbor has results for.

    Solved is ``score > 0`` rather than ``score == 1`` so a partial-credit
    verifier still counts, which is also how ``harbor_bridge.summarize`` counts
    it. The denominator is the trials that have *finished*, matching the left
    number in Harbor's own ``97/113`` column.
    """
    if not rewards:
        return None
    solved = 0
    for reward in rewards:
        score = trial_score(reward)
        if score is None:
            return None
        solved += score > 0
    return solved, len(rewards)


def describe(rewards: list[dict[str, Any] | None]) -> str | None:
    """The progress line, or ``None`` to leave Harbor's own line in place."""
    counted = solve_rate(rewards)
    if counted is None:
        return None
    solved, finished = counted
    return f"Solved: {solved / finished:.3f} ({solved}/{finished})"


def _live_rewards(job: Any, event: Any) -> list[dict[str, Any] | None]:
    """Every reward Harbor holds for the eval group this trial belongs to.

    A job may run several agents, models and datasets at once, and Harbor keys
    its live rewards by the three together. Reading the whole map instead would
    average a model's score with another model's.
    """
    evals_key, _ = job._evals_key_for_result(event.result)
    return list(job._live_rewards.get(evals_key, {}).values())


def install() -> bool:
    """Replace Harbor's live metric line with the solve rate. Never raises.

    Returns whether this call was the one that installed it, so a caller can
    say so in a log line; ``False`` covers both "already installed" and "no
    Harbor to install into".

    Called for its effect on a process Harbor owns, which is why it is a patch
    and not a subclass: the progress bar belongs to ``Job``, and Harbor gives a
    provider no say in it. The patch is confined to the description string —
    the metrics Harbor records, and every number in its final table, are
    untouched.
    """
    try:
        from harbor.job import Job
    except Exception as exc:  # pragma: no cover - Harbor is optional here
        logger.debug("leaving Harbor's progress line alone: %s", exc)
        return False

    original = getattr(Job, "_update_metric_display", None)
    if original is None:
        # A Harbor that has renamed or dropped the method. tests/test_progress.py
        # fails loudly on this so an upgrade is noticed here, not by someone
        # reading a stale number off a nine-hour run.
        logger.debug("Harbor has no _update_metric_display; progress unchanged")
        return False
    if getattr(original, _MARKER, False):
        return False

    @functools.wraps(original)
    def _update_metric_display(
        self, event, loading_progress, loading_progress_task
    ) -> None:
        line = None
        try:
            line = describe(_live_rewards(self, event))
        except Exception as exc:
            logger.debug("falling back to Harbor's progress line: %s", exc)
        if line is None:
            original(self, event, loading_progress, loading_progress_task)
            return
        loading_progress.update(loading_progress_task, description=line)

    setattr(_update_metric_display, _MARKER, True)
    Job._update_metric_display = _update_metric_display
    return True
