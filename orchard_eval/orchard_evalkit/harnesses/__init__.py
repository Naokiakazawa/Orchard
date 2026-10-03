"""Agent harnesses and the registry that resolves them by name.

Importing this package registers every built-in harness, which is what makes
``harness.name: codex`` in a YAML config resolvable.
"""

from orchard_evalkit.harnesses.base import (
    Harness,
    RolloutContext,
    available_harnesses,
    build_harness,
    get_harness_class,
    harness_descriptions,
    register_harness,
)
from orchard_evalkit.harnesses.gold import GoldPatchHarness
from orchard_evalkit.harnesses.installed_cli import (
    ClaudeCodeHarness,
    CliSpec,
    CodexHarness,
    InstalledCliHarness,
    OpencodeHarness,
    PiHarness,
)
from orchard_evalkit.harnesses.mini_swe import MiniSweAgentHarness
from orchard_evalkit.harnesses.noop import NoPatchHarness
from orchard_evalkit.harnesses.trajectory import PARSERS, parse_trajectory

__all__ = [
    "PARSERS",
    "ClaudeCodeHarness",
    "CliSpec",
    "CodexHarness",
    "GoldPatchHarness",
    "Harness",
    "InstalledCliHarness",
    "MiniSweAgentHarness",
    "NoPatchHarness",
    "OpencodeHarness",
    "PiHarness",
    "RolloutContext",
    "available_harnesses",
    "build_harness",
    "get_harness_class",
    "harness_descriptions",
    "parse_trajectory",
    "register_harness",
]
