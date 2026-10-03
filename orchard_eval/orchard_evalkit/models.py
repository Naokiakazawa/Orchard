"""Data models shared by datasets, harnesses, grading, and reporting."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# Exit statuses produced by the suite itself (harnesses may report their own).
EXIT_SUBMITTED = "submitted"
EXIT_COMPLETED = "completed"
EXIT_AGENT_ERROR = "agent_error"
EXIT_INFRA_ERROR = "infra_error"
EXIT_TIMEOUT = "timeout"


class TaskInstance(BaseModel):
    """A single benchmark task, normalized across datasets."""

    instance_id: str
    problem_statement: str
    repo: str = ""
    base_commit: str = ""
    image: str = ""
    workdir: str = "/testbed"
    raw: dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return f"TaskInstance({self.instance_id} @ {self.image})"


class RolloutResult(BaseModel):
    """What a harness produced for one instance.

    ``patch`` is the artifact that gets graded. ``submission`` is whatever the
    harness reported as its final answer — for harnesses that produce the diff
    themselves (mini-swe-agent) the two are the same; for CLI harnesses the
    patch is recovered from the repository with ``git diff``.

    The trajectory is carried in two complementary fields, and every harness
    is expected to populate as many as it can:

    ``messages``
        Normalized turn-by-turn view, in mini-swe-agent's shape. This is what
        distillation and analysis read.
    ``trajectory_extra``
        Harness-native structure that does not fit the normalized shape —
        mini-swe-agent's full serialized agent, a CLI's token usage, session
        ids.
    """

    submission: str = ""
    patch: str = ""
    exit_status: str = EXIT_COMPLETED
    messages: list[dict[str, Any]] = Field(default_factory=list)
    trajectory_extra: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    error: str | None = None


class GradeResult(BaseModel):
    """Outcome of running the benchmark's own tests against a patch."""

    resolved: bool = False
    resolved_status: str = "RESOLVED_NO"
    reward: float = 0.0
    tests_status: dict[str, Any] = Field(default_factory=dict)
    status_map: dict[str, Any] = Field(default_factory=dict)
    eval_exit_code: int | None = None
    error: str | None = None

    @classmethod
    def unresolved(cls, error: str, *, status: str = "RESOLVED_NO") -> GradeResult:
        return cls(resolved=False, resolved_status=status, reward=0.0, error=error)


class Timings(BaseModel):
    """Wall-clock seconds spent in each phase of one instance."""

    create_s: float = 0.0
    setup_s: float = 0.0
    rollout_s: float = 0.0
    #: Creating the second, agent-free pod that grading runs in.
    grade_create_s: float = 0.0
    grade_s: float = 0.0
    total_s: float = 0.0


class InstanceRecord(BaseModel):
    """One line of ``results.jsonl`` — the unit of resume and reporting."""

    instance_id: str
    harness: str
    model: str = ""
    run_name: str = ""
    resolved: bool = False
    reward: float = 0.0
    exit_status: str = EXIT_COMPLETED
    resolved_status: str = "RESOLVED_NO"
    patch: str = ""
    sandbox_id: str = ""
    #: The separate pod the patch was graded in, empty when grading was skipped.
    eval_sandbox_id: str = ""
    attempts: int = 1
    #: Whole-rollout attempts spent, >1 when a sandbox was lost mid-instance.
    rollout_attempts: int = 1
    #: Number of normalized trajectory messages captured, so a run that
    #: silently stopped recording trajectories is visible in the results file
    #: rather than only on disk.
    n_messages: int = 0
    error: str | None = None
    timings: Timings = Field(default_factory=Timings)
    metrics: dict[str, Any] = Field(default_factory=dict)
    tests_status: dict[str, Any] = Field(default_factory=dict)


class RunSummary(BaseModel):
    """Aggregate view of a completed (or partial) run."""

    run_name: str = ""
    harness: str = ""
    model: str = ""
    dataset: str = ""
    #: ``swebench`` or ``swebench-pro`` — the rules the resolve rate was
    #: produced under, which two runs must share before they are comparable.
    benchmark: str = ""
    total: int = 0
    resolved: int = 0
    resolve_rate: float = 0.0
    empty_patches: int = 0
    errors: int = 0
    #: Instances whose trajectory came back empty. Trajectories are the point
    #: of collecting rollouts at all, so a run silently producing none should
    #: be visible in the summary rather than only discoverable on disk.
    missing_trajectories: int = 0
    #: Rollouts in which the model's output-token cap cut a completion off
    #: mid-message. A CLI that ends its session on a truncated completion
    #: instead of re-prompting records those as ordinary finished runs that
    #: happened to produce no diff, so the count belongs next to the resolve
    #: rate rather than buried in per-instance metrics.
    length_truncated: int = 0
    mean_messages: float = 0.0
    exit_statuses: dict[str, int] = Field(default_factory=dict)
    total_cost: float = 0.0
    mean_rollout_s: float = 0.0
    #: Starting the second, agent-free pod that grading runs in.
    mean_grade_create_s: float = 0.0
    mean_grade_s: float = 0.0
    mean_total_s: float = 0.0
    #: Wall-clock seconds for the whole run, which is far less than the sum of
    #: the per-instance totals because instances run concurrently.
    wall_clock_s: float = 0.0
