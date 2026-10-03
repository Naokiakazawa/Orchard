"""Loading SWE-bench (and SWE-bench-shaped) task instances.

Instances can come from three places, in this order of preference:

1. a local ``.jsonl`` / ``.json`` file — reproducible, offline, no HF token;
2. a HuggingFace dataset id (or one of the shorthands in
   :data:`SWEBENCH_DATASETS`), which needs the optional ``datasets`` package;
3. anything already normalized into dicts, via :func:`instances_from_records`.

Whichever the source, the result is a list of
:class:`~orchard_evalkit.models.TaskInstance` carrying the *sandbox image* for the
instance, so the runner never has to know anything about SWE-bench itself.
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from orchard_evalkit.config import DatasetConfig
from orchard_evalkit.models import TaskInstance

logger = logging.getLogger(__name__)

#: Shorthands accepted by ``dataset.name``.
#:
#: The full names are canonical; the terse ones are kept because they read
#: better on a command line and predate this map. Lookups go through
#: :func:`resolve_dataset_name`, which folds spaces and underscores to hyphens,
#: so ``"SWE-bench Verified"`` resolves too.
#:
#: The ``SWE-bench/*`` datasets are the modern ones: each row carries its own
#: ``image``, ``eval_script``, ``log_parser`` and ``eval_type``, which is what
#: ``swebench >= 5.0`` grades from. The ``princeton-nlp/*`` datasets predate
#: that and only work with ``swebench < 5``.
SWEBENCH_DATASETS = {
    "swe-bench-verified": "SWE-bench/SWE-bench_Verified",
    "swe-bench-full": "SWE-bench/SWE-bench",
    "swe-bench-lite": "SWE-bench/SWE-bench_Lite",
    "swe-bench-multimodal": "SWE-bench/SWE-bench_Multimodal",
    "swe-bench-multilingual": "SWE-bench/SWE-bench_Multilingual",
    "swe-bench-verified-legacy": "princeton-nlp/SWE-bench_Verified",
    "swe-bench-full-legacy": "princeton-nlp/SWE-bench",
    "swe-bench-lite-legacy": "princeton-nlp/SWE-bench_Lite",
    # Terse aliases.
    "verified": "SWE-bench/SWE-bench_Verified",
    "full": "SWE-bench/SWE-bench",
    "lite": "SWE-bench/SWE-bench_Lite",
    "multimodal": "SWE-bench/SWE-bench_Multimodal",
    "multilingual": "SWE-bench/SWE-bench_Multilingual",
    "verified-legacy": "princeton-nlp/SWE-bench_Verified",
    "full-legacy": "princeton-nlp/SWE-bench",
    "lite-legacy": "princeton-nlp/SWE-bench_Lite",
}


def normalize_name(name: str) -> str:
    """Fold a dataset or benchmark name into the spelling the maps are keyed by.

    ``"SWE-bench Verified"``, ``"swebench_verified"`` and
    ``"swe-bench-verified"`` all name the same thing, and a run that silently
    fell through to "treat it as a HuggingFace id" would fail hundreds of
    instances later rather than immediately.
    """
    folded = "-".join((name or "").strip().lower().replace("_", "-").split())
    return folded.replace("swebench", "swe-bench")


def resolve_dataset_name(name: str) -> str:
    """Expand a shorthand into a HuggingFace id; pass anything else through."""
    return SWEBENCH_DATASETS.get(normalize_name(name), name)

#: Docker forbids ``__`` in tags, so SWE-bench substitutes this magic token.
_DOUBLE_UNDERSCORE_TOKEN = "_1776_"


def swebench_image_tail(instance_id: str, *, arch: str = "x86_64") -> str:
    """Return the registry-independent image name for a SWE-bench instance.

    The naming convention is fixed by the upstream harness:
    ``astropy__astropy-12907`` becomes
    ``sweb.eval.x86_64.astropy_1776_astropy-12907:latest``, lowercased.
    """
    tag = instance_id.replace("__", _DOUBLE_UNDERSCORE_TOKEN)
    return f"sweb.eval.{arch}.{tag}:latest".lower()


def swebench_image_name(
    instance_id: str,
    *,
    prefix: str = "docker.io/swebench",
    arch: str = "x86_64",
) -> str:
    """Return the prebuilt evaluation image for a SWE-bench instance.

    ``prefix`` selects the registry — point it at a mirror or your own ACR to
    avoid Docker Hub rate limits when launching hundreds of sandboxes at once.
    """
    tail = swebench_image_tail(instance_id, arch=arch)
    prefix = prefix.rstrip("/")
    return f"{prefix}/{tail}" if prefix else tail


def _resolve_image(record: dict[str, Any], prefix: str) -> str:
    """Pick the sandbox image for a row.

    A row that names a *non-standard* image is honoured verbatim — that is
    someone deliberately pointing at their own build. A row naming the standard
    SWE-bench image is re-pointed at ``prefix``, because the modern datasets
    hardcode ``swebench/...`` on Docker Hub and the whole reason to configure a
    registry is to avoid pulling 500 images from there.
    """
    instance_id = str(record["instance_id"])
    tail = swebench_image_tail(instance_id)

    explicit = ""
    for key in ("image", "image_name", "docker_image"):
        if record.get(key):
            explicit = str(record[key])
            break

    if explicit and not explicit.lower().endswith(tail):
        return explicit
    if prefix:
        return f"{prefix.rstrip('/')}/{tail}"
    return explicit or tail


def instances_from_records(
    records: Iterable[dict[str, Any]],
    *,
    image_prefix: str = "docker.io/swebench",
    workdir: str = "/testbed",
) -> list[TaskInstance]:
    """Normalize raw dataset rows into :class:`TaskInstance` objects."""
    instances: list[TaskInstance] = []
    for record in records:
        if "instance_id" not in record:
            raise ValueError(f"Record is missing 'instance_id': {record!r}")
        instances.append(
            TaskInstance(
                instance_id=str(record["instance_id"]),
                problem_statement=str(record.get("problem_statement", "")),
                repo=str(record.get("repo", "")),
                base_commit=str(record.get("base_commit", "")),
                image=_resolve_image(record, image_prefix),
                workdir=str(record.get("workdir", workdir)),
                raw=dict(record),
            )
        )
    return instances


def _load_local(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, dict):
        # Accept ``{instance_id: record}`` as well as a bare list.
        return list(data.values())
    return list(data)


def _load_huggingface(
    name: str, split: str, revision: str | None = None
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            f"Loading '{name}' needs the `datasets` package. Either install the "
            "dataset extra (`pip install -e 'orchard_eval[swebench]'`) or pass a "
            "local .jsonl file via dataset.name."
        ) from exc
    dataset = load_dataset(name, split=split, revision=revision)
    return [dict(row) for row in dataset]


def load_records(cfg: DatasetConfig) -> list[dict[str, Any]]:
    """Fetch raw rows for ``cfg`` without filtering."""
    name = cfg.name
    local = Path(name)
    if local.suffix in {".json", ".jsonl"} or local.exists():
        logger.info("Loading instances from local file %s", local)
        return _load_local(local)
    resolved = resolve_dataset_name(name)
    revision = cfg.revision or None
    logger.info(
        "Loading instances from HuggingFace dataset %s (%s, revision=%s)",
        resolved,
        cfg.split,
        revision or "default branch",
    )
    return _load_huggingface(resolved, cfg.split, revision)


def select_instances(
    instances: list[TaskInstance], cfg: DatasetConfig
) -> list[TaskInstance]:
    """Apply the allow-list / regex / slice / limit / shuffle selection.

    ``instance_ids`` short-circuits everything else, because an explicit list is
    always a deliberate choice; the returned order follows the list so a rerun
    of a handful of failures is predictable.
    """
    if cfg.instance_ids:
        by_id = {inst.instance_id: inst for inst in instances}
        missing = [iid for iid in cfg.instance_ids if iid not in by_id]
        if missing:
            raise KeyError(f"Instance ids not present in dataset: {missing}")
        return [by_id[iid] for iid in cfg.instance_ids]

    selected = list(instances)
    if cfg.shuffle:
        selected.sort(key=lambda inst: inst.instance_id)
        random.Random(cfg.seed).shuffle(selected)

    if cfg.filter:
        pattern = re.compile(cfg.filter)
        before = len(selected)
        selected = [inst for inst in selected if pattern.match(inst.instance_id)]
        logger.info("Instance filter %r: %d -> %d", cfg.filter, before, len(selected))

    if cfg.slice:
        bounds = [int(part) if part else None for part in cfg.slice.split(":")]
        selected = selected[slice(*bounds)]

    if cfg.limit is not None:
        selected = selected[: cfg.limit]

    return selected


def load_swebench_instances(
    cfg: DatasetConfig,
    *,
    image_prefix: str = "docker.io/swebench",
    workdir: str = "/testbed",
) -> list[TaskInstance]:
    """Load and select the instances described by ``cfg``."""
    records = load_records(cfg)
    instances = instances_from_records(
        records, image_prefix=image_prefix, workdir=workdir
    )
    selected = select_instances(instances, cfg)
    logger.info("Selected %d/%d instances", len(selected), len(instances))
    return selected
