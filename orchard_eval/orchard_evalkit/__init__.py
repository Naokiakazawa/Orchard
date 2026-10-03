"""Orchard Eval — run agentic benchmarks on Orchard Env sandboxes.

The suite separates three concerns that are usually tangled together:

* **the benchmark** — task instances and their grading rules (SWE-bench Verified)
* **the environment** — an Orchard Env sandbox, one per instance
* **the harness** — the agent loop that actually attempts the task

Only the harness changes when you want to evaluate a different agent, which is
the whole point: every sandbox already ships ``codex``, ``claude``, ``pi``,
``opencode``, ``hermes`` and ``mini`` on ``PATH``, so a harness is a
declarative description of one command rather than a new runner.
"""

from orchard_evalkit.config import (
    DatasetConfig,
    GradingConfig,
    HarnessConfig,
    ModelConfig,
    RunConfig,
    SandboxConfig,
)
from orchard_evalkit.models import (
    GradeResult,
    InstanceRecord,
    RolloutResult,
    RunSummary,
    TaskInstance,
)

__version__ = "0.1.0"

__all__ = [
    "DatasetConfig",
    "GradeResult",
    "GradingConfig",
    "HarnessConfig",
    "InstanceRecord",
    "ModelConfig",
    "RolloutResult",
    "RunConfig",
    "RunSummary",
    "SandboxConfig",
    "TaskInstance",
    "__version__",
]
