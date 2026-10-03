"""Benchmark dataset loaders, and the choice between them.

Two benchmarks are supported, and they differ in more than their rows: the
repository sits at a different path inside the image, the images come from a
different registry under a different naming convention, and grading is a
different program. Everything downstream of :func:`load_instances` therefore
needs to know *which* benchmark a run is, so this module resolves that once and
returns it alongside the instances.

Resolution order for ``dataset.benchmark``:

1. an explicit value (``swebench`` / ``swebench-pro``) — always wins;
2. the dataset name (``pro``, ``ScaleAI/SWE-bench_Pro``, ...);
3. the columns on the first row, which is what identifies a local ``.jsonl``
   dump carrying no recognizable name.
"""

from __future__ import annotations

import logging
from typing import Any

from orchard_evalkit.config import DatasetConfig, SandboxConfig
from orchard_evalkit.datasets import swebench_pro
from orchard_evalkit.datasets.swebench import (
    SWEBENCH_DATASETS,
    instances_from_records,
    load_records,
    load_swebench_instances,
    normalize_name,
    select_instances,
    swebench_image_name,
    swebench_image_tail,
)
from orchard_evalkit.datasets.swebench import (
    resolve_dataset_name as _resolve_swebench_name,
)
from orchard_evalkit.datasets.swebench_pro import (
    SWEBENCH_PRO_DATASET,
    SWEBENCH_PRO_DATASETS,
    load_swebench_pro_instances,
)
from orchard_evalkit.datasets.swebench_pro import (
    resolve_dataset_name as _resolve_pro_name,
)
from orchard_evalkit.models import TaskInstance

logger = logging.getLogger(__name__)

BENCHMARK_AUTO = "auto"
BENCHMARK_SWEBENCH = "swe-bench"
BENCHMARK_SWEBENCH_PRO = "swe-bench-pro"

#: Spellings accepted for ``dataset.benchmark``, keyed by their normalized
#: form — so ``"SWE-bench Pro"`` and ``"swebench_pro"`` land here as well.
BENCHMARK_ALIASES = {
    "": BENCHMARK_AUTO,
    "auto": BENCHMARK_AUTO,
    "swe-bench": BENCHMARK_SWEBENCH,
    "swe-bench-verified": BENCHMARK_SWEBENCH,
    "swe-bench-lite": BENCHMARK_SWEBENCH,
    "swe-bench-full": BENCHMARK_SWEBENCH,
    "verified": BENCHMARK_SWEBENCH,
    "swe-bench-pro": BENCHMARK_SWEBENCH_PRO,
    "pro": BENCHMARK_SWEBENCH_PRO,
}

#: Defaults on :class:`~orchard_evalkit.config.SandboxConfig`, which describe a
#: SWE-bench image. A Pro run left on them is *unset* rather than deliberately
#: configured, so the Pro defaults take over.
SWEBENCH_DEFAULT_IMAGE_PREFIX = "docker.io/swebench"
SWEBENCH_DEFAULT_WORKDIR = "/testbed"

#: Substrings marking an image prefix as pointing at SWE-bench's images —
#: including mirrors such as ``mirror.gcr.io/swebench``. None of them can serve
#: a Pro image, so keeping one on a Pro run fails every pod creation.
_SWEBENCH_PREFIX_MARKERS = ("swebench", "swe-bench", "swe_bench")


def normalize_benchmark(value: str) -> str:
    """Map a user-supplied benchmark name onto a canonical one."""
    try:
        return BENCHMARK_ALIASES[normalize_name(value)]
    except KeyError:
        raise ValueError(
            f"Unknown benchmark {value!r}. Use one of: "
            f"{BENCHMARK_SWEBENCH}, {BENCHMARK_SWEBENCH_PRO}, {BENCHMARK_AUTO}."
        ) from None


def resolve_dataset_name(name: str) -> str:
    """Expand a dataset name into the HuggingFace id its loader needs.

    Both benchmarks' maps are consulted, so callers never have to know which
    one a name belongs to. Anything unrecognized — a full HuggingFace id, a
    local path — is returned untouched.
    """
    return _resolve_pro_name(_resolve_swebench_name(name))


def detect_benchmark(
    cfg: DatasetConfig, records: list[dict[str, Any]] | None = None
) -> str:
    """Resolve which benchmark ``cfg`` describes."""
    explicit = normalize_benchmark(cfg.benchmark)
    if explicit != BENCHMARK_AUTO:
        return explicit
    if swebench_pro.is_swebench_pro_name(cfg.name):
        return BENCHMARK_SWEBENCH_PRO
    if records and swebench_pro.looks_like_swebench_pro(records[0]):
        return BENCHMARK_SWEBENCH_PRO
    return BENCHMARK_SWEBENCH


def resolve_image_prefix(benchmark: str, configured: str) -> str:
    """Pick the registry to pull instance images from."""
    if benchmark != BENCHMARK_SWEBENCH_PRO:
        return configured
    prefix = (configured or "").strip()
    if not prefix or any(m in prefix.lower() for m in _SWEBENCH_PREFIX_MARKERS):
        if prefix and prefix != SWEBENCH_DEFAULT_IMAGE_PREFIX:
            logger.warning(
                "sandbox.image_prefix=%r holds SWE-bench images, which do not "
                "exist for SWE-bench Pro; using %r instead",
                prefix,
                swebench_pro.DEFAULT_IMAGE_PREFIX,
            )
        return swebench_pro.DEFAULT_IMAGE_PREFIX
    return prefix


def resolve_workdir(benchmark: str, configured: str) -> str:
    """Pick where the repository lives inside the image."""
    if benchmark != BENCHMARK_SWEBENCH_PRO:
        return configured
    if not configured or configured == SWEBENCH_DEFAULT_WORKDIR:
        return swebench_pro.DEFAULT_WORKDIR
    return configured


def load_instances(
    dataset: DatasetConfig, sandbox: SandboxConfig
) -> tuple[str, list[TaskInstance]]:
    """Load a run's instances, and say which benchmark they belong to.

    Rows are fetched once and the benchmark decided from them, so a local
    ``.jsonl`` needs no configuration beyond its path.
    """
    resolved = dataset.model_copy(update={"name": resolve_dataset_name(dataset.name)})
    records = load_records(resolved)
    benchmark = detect_benchmark(dataset, records)
    image_prefix = resolve_image_prefix(benchmark, sandbox.image_prefix)
    workdir = resolve_workdir(benchmark, sandbox.workdir)

    if benchmark == BENCHMARK_SWEBENCH_PRO:
        instances = swebench_pro.instances_from_records(
            records, image_prefix=image_prefix, workdir=workdir
        )
    else:
        instances = instances_from_records(
            records, image_prefix=image_prefix, workdir=workdir
        )

    selected = select_instances(instances, dataset)
    logger.info(
        "Selected %d/%d instances (benchmark=%s, workdir=%s)",
        len(selected),
        len(instances),
        benchmark,
        workdir,
    )
    return benchmark, selected


def benchmark_of(instances: list[TaskInstance], dataset: DatasetConfig) -> str:
    """Resolve the benchmark for instances that were built elsewhere."""
    return detect_benchmark(dataset, [instances[0].raw] if instances else [])


__all__ = [
    "BENCHMARK_ALIASES",
    "BENCHMARK_AUTO",
    "BENCHMARK_SWEBENCH",
    "BENCHMARK_SWEBENCH_PRO",
    "SWEBENCH_DATASETS",
    "SWEBENCH_PRO_DATASET",
    "SWEBENCH_PRO_DATASETS",
    "benchmark_of",
    "detect_benchmark",
    "load_instances",
    "load_swebench_instances",
    "load_swebench_pro_instances",
    "normalize_benchmark",
    "normalize_name",
    "resolve_dataset_name",
    "resolve_image_prefix",
    "resolve_workdir",
    "swebench_image_name",
    "swebench_image_tail",
]
