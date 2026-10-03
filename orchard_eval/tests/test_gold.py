"""The gold-patch harness — the sanity check that validates the suite itself.

If this harness is wrong, the sanity check reports a bad number for a good
pipeline (or, far worse, a good number for a bad one), so its own behaviour is
pinned here.
"""

import pytest

from orchard_evalkit.config import RunConfig
from orchard_evalkit.harnesses import GoldPatchHarness, RolloutContext, get_harness_class
from orchard_evalkit.models import EXIT_AGENT_ERROR, EXIT_COMPLETED, TaskInstance
from orchard_evalkit.sandbox import EvalSandbox
from tests.fakes import FakeJobResult, FakeSandboxInstance

GOLD = "diff --git a/f b/f\n--- a/f\n+++ b/f\n@@\n-broken\n+fixed\n"


def _context(responder=None, *, patch=GOLD, **params):
    sandbox_instance = FakeSandboxInstance(responder=responder)
    task = TaskInstance(
        instance_id="astropy__astropy-12907",
        problem_statement="Fix the thing",
        base_commit="abc123",
        image="img",
        raw={"instance_id": "astropy__astropy-12907", "patch": patch},
    )
    ctx = RolloutContext(
        sandbox=EvalSandbox(sandbox_instance, workdir="/testbed"),
        instance=task,
        config=RunConfig(),
    )
    harness = GoldPatchHarness(model=RunConfig().model, params=params, timeout=300)
    return harness, ctx, sandbox_instance


class TestGoldPatchHarness:
    def test_it_is_registered(self):
        assert get_harness_class("gold") is GoldPatchHarness

    @pytest.mark.asyncio
    async def test_round_trips_the_patch_through_the_sandbox(self):
        # The default path applies the gold patch and reads it back with the
        # same `git diff` a CLI harness goes through, so a broken extractor
        # shows up as a gold-run failure instead of as a weak model.
        def responder(command: str):
            if "git diff" in command:
                return FakeJobResult(stdout=GOLD)
            return FakeJobResult()

        harness, ctx, _ = _context(responder)
        result = await harness.rollout(ctx)

        assert result.patch == GOLD
        assert result.exit_status == EXIT_COMPLETED
        assert result.metrics["extracted_patch_bytes"] == len(GOLD)
        assert result.messages[0]["extra"]["kind"] == "gold_patch"

    @pytest.mark.asyncio
    async def test_extract_false_grades_the_dataset_column_verbatim(self):
        harness, ctx, sandbox_instance = _context(extract=False)
        result = await harness.rollout(ctx)

        assert result.patch == GOLD
        assert sandbox_instance.commands == []

    @pytest.mark.asyncio
    async def test_a_patch_that_will_not_apply_is_not_a_silent_zero(self):
        # The gold patch failing against its own base commit means the image
        # and the dataset row disagree — a cluster defect, not a test failure.
        def responder(command: str):
            if "apply" in command or "patch --batch" in command:
                return FakeJobResult(stderr="does not apply", exit_code=1)
            return FakeJobResult()

        harness, ctx, _ = _context(responder)
        result = await harness.rollout(ctx)

        assert result.exit_status == EXIT_AGENT_ERROR
        assert "did not apply" in (result.error or "")

    @pytest.mark.asyncio
    async def test_lost_extraction_is_reported_rather_than_graded(self):
        # An empty patch after a successful apply means extraction dropped it.
        harness, ctx, _ = _context()
        result = await harness.rollout(ctx)

        assert result.patch == ""
        assert result.exit_status == EXIT_AGENT_ERROR
        assert "extraction" in (result.error or "")

    @pytest.mark.asyncio
    async def test_a_row_without_a_patch_column_is_an_error(self):
        harness, ctx, _ = _context(patch="")
        result = await harness.rollout(ctx)

        assert result.exit_status == EXIT_AGENT_ERROR
        assert "`patch` column" in (result.error or "")
