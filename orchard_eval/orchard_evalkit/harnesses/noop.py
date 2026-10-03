"""Do nothing, then grade — the suite's *lower* bound.

:mod:`~orchard_evalkit.harnesses.gold` establishes the ceiling: replaying the
human fix must resolve ~100%. This establishes the floor, and the two together
are what make a resolve rate meaningful. Running the benchmark's own tests
against the untouched repository must resolve **~0%**, because every instance's
``FAIL_TO_PASS`` tests are, by definition, failing at ``base_commit``.

A materially *higher* number is not a lucky agent — it is the grading path
handing out points for nothing:

* the eval script is selecting tests that already pass before the fix (wrong
  test spec, wrong ``base_commit``, an image whose working tree already carries
  the fix);
* the log parser is scoring an unparsed or empty log as success;
* ``FAIL_TO_PASS`` is empty for some rows, which grades as vacuously resolved.

None of those is visible in a gold run — they inflate that number too, and it
is supposed to be 100% — so this is the only check that catches them.

Grading normally short-circuits an empty patch to ``EMPTY_PATCH`` without
spending a pod, which would make this harness measure nothing. Pair it with
``grading.grade_empty_patch=true`` (``configs/noop.yaml`` does) so the eval pod
is created and the tests actually run.
"""

from __future__ import annotations

from orchard_evalkit.harnesses.base import Harness, RolloutContext, register_harness
from orchard_evalkit.models import EXIT_COMPLETED, RolloutResult


@register_harness
class NoPatchHarness(Harness):
    name = "noop"
    description = "Submits nothing — lower bound / sanity check"

    async def rollout(self, ctx: RolloutContext) -> RolloutResult:
        metrics: dict[str, object] = {}

        if self.params.get("verify_clean", True):
            # The rollout pod has only been reset to base_commit, so its diff
            # must be empty. Anything here is the image or `prepare_repo`
            # leaving state behind, which would silently contaminate every
            # patch this suite extracts — not just this run's.
            residual = await ctx.sandbox.extract_patch(ctx.instance.base_commit)
            metrics["residual_patch_bytes"] = len(residual)
            if residual.strip():
                ctx.save_artifact("residual.diff", residual)
                ctx.logger.warning(
                    "[%s] pristine tree is not clean: %d bytes of diff at %s",
                    ctx.instance.instance_id,
                    len(residual),
                    ctx.instance.base_commit,
                )

        return RolloutResult(
            patch="",
            submission="",
            exit_status=EXIT_COMPLETED,
            messages=[
                {
                    "role": "assistant",
                    "content": "",
                    "extra": {"kind": "no_patch"},
                }
            ],
            metrics=metrics,
        )


__all__ = ["NoPatchHarness"]
