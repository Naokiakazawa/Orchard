"""The dataset's own fix, replayed as a harness — the suite's sanity check.

A resolve rate only means something if the machinery around it is sound.
SWE-bench ships the human fix for every instance in the ``patch`` column, so
replaying it must resolve **~100%**. Anything materially below that is a bug in
this suite or in the environment — the wrong image, a stale eval script, a
broken ``git apply`` chain, a log parser that no longer matches the test runner
— and every point it costs is a point silently subtracted from every model you
measure afterwards.

Running this as a *harness* rather than a standalone script is the whole point:
it goes through the identical code path a real run takes — same two pods, same
repo preparation, same patch extraction, same eval script, same parser, same
summary. A check that used a shortcut would not be checking the thing that
matters.

By default the gold patch is applied in the rollout pod and then read back out
with :meth:`EvalSandbox.extract_patch`, exactly as a CLI agent's work would be.
That makes the run exercise patch *extraction* too, which is otherwise
invisible: if the excludes or the ``git diff --cached`` are wrong, real
harnesses quietly lose work and only the resolve rate ever shows it. Set
``harness.params.extract=false`` to grade the dataset column verbatim instead,
which isolates grading from extraction when one of them is misbehaving.
"""

from __future__ import annotations

from orchard_evalkit.harnesses.base import Harness, RolloutContext, register_harness
from orchard_evalkit.models import EXIT_AGENT_ERROR, EXIT_COMPLETED, RolloutResult

#: Dataset column holding the human fix. Distinct from ``test_patch``, which the
#: eval script applies itself and which must never reach a prediction.
GOLD_PATCH_COLUMN = "patch"


@register_harness
class GoldPatchHarness(Harness):
    name = "gold"
    description = "Replays the dataset's own patch — upper bound / sanity check"

    async def rollout(self, ctx: RolloutContext) -> RolloutResult:
        gold = str(ctx.instance.raw.get(GOLD_PATCH_COLUMN) or "")
        if not gold.strip():
            return RolloutResult(
                exit_status=EXIT_AGENT_ERROR,
                error=f"dataset row has no `{GOLD_PATCH_COLUMN}` column",
            )

        ctx.save_artifact("gold_patch.diff", gold)
        messages = [
            {
                "role": "assistant",
                "content": gold,
                "extra": {"kind": "gold_patch", "source": GOLD_PATCH_COLUMN},
            }
        ]
        metrics = {"gold_patch_bytes": len(gold)}

        if not self.params.get("extract", True):
            return RolloutResult(
                patch=gold, messages=messages, metrics=metrics, submission=gold
            )

        applied = await ctx.sandbox.apply_patch(gold, timeout=self.timeout)
        if not applied.succeeded:
            # The gold patch failing to apply to its own base commit means the
            # image and the dataset row disagree; grading the row anyway would
            # report this as an ordinary unresolved instance.
            ctx.save_artifact("gold_apply.log", applied.output)
            return RolloutResult(
                patch=gold,
                exit_status=EXIT_AGENT_ERROR,
                error="gold patch did not apply to the pristine tree",
                messages=messages,
                metrics=metrics,
                stdout=applied.output,
            )

        extracted = await ctx.sandbox.extract_patch(ctx.instance.base_commit)
        metrics["extracted_patch_bytes"] = len(extracted)
        if not extracted.strip():
            return RolloutResult(
                patch="",
                exit_status=EXIT_AGENT_ERROR,
                error="patch extraction returned nothing for an applied gold patch",
                messages=messages,
                metrics=metrics,
            )

        return RolloutResult(
            patch=extracted,
            submission=extracted,
            exit_status=EXIT_COMPLETED,
            messages=messages,
            metrics=metrics,
        )


__all__ = ["GOLD_PATCH_COLUMN", "GoldPatchHarness"]
