"""The harness abstraction: what it means to attempt one task.

A harness receives a prepared sandbox and a task, does whatever it does, and
returns a patch. Everything else — sandbox lifecycle, grading, retries,
reporting — belongs to the runner, so adding a new agent means implementing one
method (or, for a CLI agent already present in the sandbox, filling in a
declarative spec).

Every built-in harness is of the second kind: ``codex``, ``pi``, ``claude``,
``opencode`` and ``mini`` all run entirely inside the pod, and the suite issues
one command and reads back the result.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from orchard_evalkit.config import HarnessConfig, ModelConfig, RunConfig
from orchard_evalkit.models import RolloutResult, TaskInstance
from orchard_evalkit.sandbox import EvalSandbox


@dataclass
class RolloutContext:
    """Everything a harness is given for a single attempt."""

    sandbox: EvalSandbox
    instance: TaskInstance
    config: RunConfig
    #: Directory for this instance's artifacts, or ``None`` when disabled.
    artifacts_dir: Path | None = None
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger("orchard_evalkit.harness")
    )

    @property
    def workdir(self) -> str:
        return self.instance.workdir

    @property
    def task(self) -> str:
        return self.instance.problem_statement

    def save_artifact(self, name: str, content: str) -> None:
        """Best-effort artifact write; never fails a rollout."""
        if self.artifacts_dir is None:
            return
        try:
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            (self.artifacts_dir / name).write_text(content, encoding="utf-8")
        except OSError as exc:  # pragma: no cover - disk-full etc.
            self.logger.warning("could not write artifact %s: %s", name, exc)


class Harness(ABC):
    """Base class for every agent harness.

    Subclasses declare :attr:`name` and implement :meth:`rollout`. Construction
    goes through :meth:`from_config` so a harness can validate its own settings
    up front rather than failing on instance 300 of 500.
    """

    #: Registry key, as used by ``harness.name`` in the run config.
    name: ClassVar[str] = ""
    #: Human-readable one-liner shown by ``orchard-eval list-harnesses``.
    description: ClassVar[str] = ""

    def __init__(self, *, model: ModelConfig, params: dict[str, Any], timeout: int):
        self.model = model
        self.params = params
        self.timeout = timeout

    @classmethod
    def from_config(cls, harness_cfg: HarnessConfig, model_cfg: ModelConfig) -> Harness:
        return cls(
            model=model_cfg,
            params=dict(harness_cfg.params),
            timeout=harness_cfg.timeout,
        )

    @abstractmethod
    async def rollout(self, ctx: RolloutContext) -> RolloutResult:
        """Attempt the task and return the resulting patch and trajectory.

        Implementations **must** populate the trajectory fields on
        :class:`~orchard_evalkit.models.RolloutResult` as fully as the underlying
        agent allows:

        * ``messages`` — normalized turn-by-turn record (see
          :mod:`orchard_evalkit.harnesses.trajectory` for the shape and for
          ready-made parsers for CLI event streams);
        * ``trajectory_extra`` — anything native that does not fit the above.

        The runner persists both for every harness, so a harness that
        leaves them empty produces a run with no usable rollout data — which is
        the main thing these runs exist to collect. An empty trajectory is
        reported as ``missing_trajectories`` in the run summary.
        """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(model={self.model.name!r})"


_REGISTRY: dict[str, type[Harness]] = {}


def register_harness(harness_cls: type[Harness]) -> type[Harness]:
    """Register a harness class under its :attr:`Harness.name`.

    Usable as a decorator. Re-registering the same name is rejected: silently
    shadowing a harness would make a run's results impossible to attribute.
    """
    key = harness_cls.name
    if not key:
        raise ValueError(f"{harness_cls.__name__} must declare a non-empty `name`")
    existing = _REGISTRY.get(key)
    if existing is not None and existing is not harness_cls:
        raise ValueError(
            f"Harness {key!r} is already registered by {existing.__name__}"
        )
    _REGISTRY[key] = harness_cls
    return harness_cls


def get_harness_class(name: str) -> type[Harness]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown harness {name!r}. Available: {', '.join(available_harnesses())}"
        ) from None


def build_harness(harness_cfg: HarnessConfig, model_cfg: ModelConfig) -> Harness:
    return get_harness_class(harness_cfg.name).from_config(harness_cfg, model_cfg)


def available_harnesses() -> list[str]:
    return sorted(_REGISTRY)


def harness_descriptions() -> dict[str, str]:
    return {name: _REGISTRY[name].description for name in available_harnesses()}
