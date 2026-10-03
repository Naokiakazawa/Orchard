"""Run Harbor tasks — terminal-bench and anything else in the Harbor format — on
Orchard Env sandboxes.

The main surface is one class, :class:`~harbor_orchard.environment.OrchardEnvironment`,
which Harbor loads by import path::

    harbor run -d terminal-bench/terminal-bench@4.0.0 \\
        --environment-import-path harbor_orchard:OrchardEnvironment

It is imported lazily so that the translation layer and its command-line tools
stay usable without Harbor installed — auditing a Dockerfile should not require
a cluster, a job, or the framework.

:mod:`harbor_orchard.agents` is the other half, and is loaded the same way
(``--agent harbor_orchard.agents:Pi``). It holds agents that use the CLI already
mounted in the pod instead of fetching one into the task's image at trial time.
It is not re-exported here because, unlike the environment, it cannot be
imported without Harbor at all.
"""

from __future__ import annotations

__all__ = ["OrchardEnvironment", "__version__"]

__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "OrchardEnvironment":
        from harbor_orchard.environment import OrchardEnvironment

        return OrchardEnvironment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
