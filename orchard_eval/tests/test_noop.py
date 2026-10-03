"""The no-patch harness — the floor the gold check is the ceiling for.

A resolve rate only means something when both ends are pinned. If this harness
ever submits something, the floor it measures stops being a floor.
"""

import pytest

from orchard_evalkit.config import RunConfig
from orchard_evalkit.harnesses import NoPatchHarness, RolloutContext, get_harness_class
from orchard_evalkit.models import EXIT_COMPLETED, TaskInstance
from orchard_evalkit.sandbox import EvalSandbox
from tests.fakes import FakeJobResult, FakeSandboxInstance


def _context(responder=None, **params):
    sandbox_instance = FakeSandboxInstance(responder=responder)
    task = TaskInstance(
        instance_id="astropy__astropy-12907",
        problem_statement="Fix the thing",
        base_commit="abc123",
        image="img",
        raw={"instance_id": "astropy__astropy-12907"},
    )
    ctx = RolloutContext(
        sandbox=EvalSandbox(sandbox_instance, workdir="/testbed"),
        instance=task,
        config=RunConfig(),
    )
    harness = NoPatchHarness(model=RunConfig().model, params=params, timeout=300)
    return harness, ctx, sandbox_instance


class TestNoPatchHarness:
    def test_it_is_registered(self):
        assert get_harness_class("noop") is NoPatchHarness

    @pytest.mark.asyncio
    async def test_it_submits_nothing(self):
        harness, ctx, _ = _context(verify_clean=False)
        result = await harness.rollout(ctx)

        assert result.patch == ""
        assert result.submission == ""
        assert result.exit_status == EXIT_COMPLETED

    @pytest.mark.asyncio
    async def test_verify_clean_reports_a_dirty_pristine_tree(self):
        # State the image leaves behind at base_commit would contaminate every
        # patch this suite extracts, not just this run's.
        residual = "diff --git a/f b/f\n+leftover\n"

        def responder(command: str):
            if "git diff" in command:
                return FakeJobResult(stdout=residual)
            return FakeJobResult()

        harness, ctx, _ = _context(responder)
        result = await harness.rollout(ctx)

        assert result.patch == ""
        assert result.metrics["residual_patch_bytes"] == len(residual)
