"""Loading SWE-bench Pro task instances.

SWE-bench Pro (``ScaleAI/SWE-bench_Pro``) keeps SWE-bench's row shape —
``instance_id``, ``repo``, ``base_commit``, ``patch``, ``problem_statement`` —
and adds the columns its own harness needs. Four of those drive everything in
this module and in :mod:`orchard_evalkit.grading.swebench_pro`:

``dockerhub_tag``
    Tag of the prebuilt image on Docker Hub. Pro does **not** use SWE-bench's
    ``sweb.eval.<arch>.<instance>`` convention; the images live in a single
    ``sweap-images`` repository and the tag is derived from the repo name plus
    the instance hash. See :func:`dockerhub_tag`.
``before_repo_set_cmd`` / ``selected_test_files_to_run``
    The build step and the test-file selection the eval script runs. Grading
    reads them; the loader only carries them through on ``TaskInstance.raw``.
``requirements`` / ``interface``
    Extra task specification. Upstream's scaffolds concatenate them onto the
    issue text before handing it to the model, so :func:`build_problem_statement`
    does the same here — a Pro instance is not solvable from the issue alone,
    and an agent given only ``problem_statement`` is being scored on a
    different, harder task than the leaderboard's.

Two environment facts differ from SWE-bench and are the reason the benchmark
needs its own loader rather than a flag: the repository lives at ``/app``
instead of ``/testbed``, and the images come from ``jefzda/sweap-images``
instead of ``swebench/*``.
"""

from __future__ import annotations

import ast
import json
import logging
from collections.abc import Iterable
from typing import Any

from orchard_evalkit.config import DatasetConfig
from orchard_evalkit.datasets.swebench import (
    load_records,
    normalize_name,
    select_instances,
)
from orchard_evalkit.models import TaskInstance

logger = logging.getLogger(__name__)

#: The dataset on HuggingFace.
SWEBENCH_PRO_DATASET = "ScaleAI/SWE-bench_Pro"

#: Shorthands accepted by ``dataset.name``, keyed by their normalized spelling
#: — so ``"SWE-bench Pro"`` and ``"swebench_pro"`` resolve here too.
SWEBENCH_PRO_DATASETS = {
    "swe-bench-pro": SWEBENCH_PRO_DATASET,
    "pro": SWEBENCH_PRO_DATASET,
}

#: Docker Hub account holding the prebuilt instance images.
DEFAULT_IMAGE_PREFIX = "docker.io/jefzda"

#: Single repository every instance image lives in, tagged per instance.
IMAGE_REPOSITORY = "sweap-images"

#: Where the repository is checked out inside a Pro image.
DEFAULT_WORKDIR = "/app"

# Pro-specific dataset columns.
COL_DOCKERHUB_TAG = "dockerhub_tag"
COL_FAIL_TO_PASS = "fail_to_pass"
COL_PASS_TO_PASS = "pass_to_pass"
COL_BEFORE_REPO_SET_CMD = "before_repo_set_cmd"
COL_SELECTED_TEST_FILES = "selected_test_files_to_run"
COL_REQUIREMENTS = "requirements"
COL_INTERFACE = "interface"

#: Columns that only a Pro row carries, used to recognize a local ``.jsonl``
#: dump of the dataset when ``dataset.benchmark`` was left on ``auto``.
PRO_MARKER_COLUMNS = (COL_DOCKERHUB_TAG, COL_BEFORE_REPO_SET_CMD)

#: The one instance whose image tag does not follow the ``element-web`` ->
#: ``element`` shortening. Verbatim from upstream ``helper_code/image_uri.py``;
#: without it that instance's image cannot be found.
ELEMENT_WEB_EXCEPTION = (
    "instance_element-hq__element-web-"
    "ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan"
)

#: Docker Hub rejects tags longer than 128 characters, so upstream truncates.
MAX_TAG_LENGTH = 128


def parse_string_list(value: Any) -> list[str]:
    """Normalize a dataset field that holds a list into an actual list.

    Pro stores ``fail_to_pass``, ``pass_to_pass`` and
    ``selected_test_files_to_run`` as the *repr* of a Python list, which
    upstream reads back with ``eval()``. ``ast.literal_eval`` is the same thing
    without the arbitrary-code-execution: these strings come from a downloaded
    dataset, and a benchmark row must never be able to run code on the host.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]

    text = str(value).strip()
    if not text:
        return []
    for loads in (ast.literal_eval, json.loads):
        try:
            parsed = loads(text)
        except (ValueError, SyntaxError, TypeError):
            continue
        if isinstance(parsed, (list, tuple, set)):
            return [str(item) for item in parsed]
        return [str(parsed)]
    # Not a literal at all — treat it as one entry per line.
    return [line.strip() for line in text.splitlines() if line.strip()]


def dockerhub_tag(instance_id: str, repo: str) -> str:
    """Derive the image tag for an instance, mirroring upstream exactly.

    ``NodeBB/NodeBB`` + ``instance_NodeBB__NodeBB-7b8bff...-vf2cf3c...`` becomes
    ``nodebb.nodebb-NodeBB__NodeBB-7b8bff...-vf2cf3c...``. The instance's own
    case is preserved; only the repo prefix is lowercased.

    Prefer the row's ``dockerhub_tag`` column when it has one — this exists for
    rows that predate it and for locally assembled datasets.
    """
    if "/" not in repo:
        raise ValueError(
            f"Cannot derive an image tag for {instance_id!r}: the row has no "
            f"usable `repo` (got {repo!r}) and no `{COL_DOCKERHUB_TAG}` column."
        )
    repo_base, repo_name = repo.lower().split("/", 1)
    digest = instance_id.replace("instance_", "")

    if instance_id == ELEMENT_WEB_EXCEPTION:
        repo_name = "element-web"
    elif "element-hq" in repo.lower() and "element-web" in repo.lower():
        repo_name = "element"
        digest = digest.removesuffix("-vnan")
    else:
        # `-vnan` is a missing version, not part of the image tag.
        digest = digest.removesuffix("-vnan")

    return f"{repo_base}.{repo_name}-{digest}"[:MAX_TAG_LENGTH]


def pro_image_name(record: dict[str, Any], prefix: str = DEFAULT_IMAGE_PREFIX) -> str:
    """Return the sandbox image for a Pro row.

    An explicit ``image`` on the row wins: that is someone deliberately pointing
    at their own build or a mirror they populated per instance.
    """
    for key in ("image", "image_name", "docker_image"):
        if record.get(key):
            return str(record[key])

    instance_id = str(record["instance_id"])
    tag = str(record.get(COL_DOCKERHUB_TAG) or "").strip()
    if not tag:
        tag = dockerhub_tag(instance_id, str(record.get("repo") or ""))
    prefix = (prefix or DEFAULT_IMAGE_PREFIX).rstrip("/")
    return f"{prefix}/{IMAGE_REPOSITORY}:{tag}"


def build_problem_statement(record: dict[str, Any]) -> str:
    """Concatenate the issue with the ``requirements`` and ``interface`` columns.

    This is upstream's ``helper_code/create_problem_statement.py``, minus the
    empty sections: a good third of the rows have a null ``interface``, and
    pasting ``New interfaces introduced:\\nNone`` into the prompt only invites
    the model to invent one.
    """
    sections = [str(record.get("problem_statement") or "").strip()]
    requirements = str(record.get(COL_REQUIREMENTS) or "").strip()
    interface = str(record.get(COL_INTERFACE) or "").strip()
    if requirements:
        sections.append(f"Requirements:\n{requirements}")
    if interface:
        sections.append(f"New interfaces introduced:\n{interface}")
    return "\n\n".join(section for section in sections if section)


def resolve_dataset_name(name: str) -> str:
    """Expand a Pro shorthand into the HuggingFace id; pass anything else on."""
    return SWEBENCH_PRO_DATASETS.get(normalize_name(name), name)


def is_swebench_pro_name(name: str) -> bool:
    """True when ``name`` names the Pro dataset, by shorthand or by id."""
    key = normalize_name(name)
    return key in SWEBENCH_PRO_DATASETS or key == normalize_name(SWEBENCH_PRO_DATASET)


def looks_like_swebench_pro(record: dict[str, Any]) -> bool:
    """True when a raw row carries columns only SWE-bench Pro has."""
    return any(column in record for column in PRO_MARKER_COLUMNS)


def instances_from_records(
    records: Iterable[dict[str, Any]],
    *,
    image_prefix: str = DEFAULT_IMAGE_PREFIX,
    workdir: str = DEFAULT_WORKDIR,
) -> list[TaskInstance]:
    """Normalize raw Pro rows into :class:`TaskInstance` objects."""
    instances: list[TaskInstance] = []
    for record in records:
        if "instance_id" not in record:
            raise ValueError(f"Record is missing 'instance_id': {record!r}")
        instances.append(
            TaskInstance(
                instance_id=str(record["instance_id"]),
                problem_statement=build_problem_statement(record),
                repo=str(record.get("repo", "")),
                base_commit=str(record.get("base_commit", "")),
                image=pro_image_name(record, image_prefix),
                workdir=str(record.get("workdir", workdir)),
                raw=dict(record),
            )
        )
    return instances


def load_swebench_pro_instances(
    cfg: DatasetConfig,
    *,
    image_prefix: str = DEFAULT_IMAGE_PREFIX,
    workdir: str = DEFAULT_WORKDIR,
) -> list[TaskInstance]:
    """Load and select the SWE-bench Pro instances described by ``cfg``."""
    resolved = cfg.model_copy(update={"name": resolve_dataset_name(cfg.name)})
    records = load_records(resolved)
    instances = instances_from_records(
        records, image_prefix=image_prefix, workdir=workdir
    )
    selected = select_instances(instances, cfg)
    logger.info("Selected %d/%d SWE-bench Pro instances", len(selected), len(instances))
    return selected


__all__ = [
    "COL_BEFORE_REPO_SET_CMD",
    "COL_DOCKERHUB_TAG",
    "COL_FAIL_TO_PASS",
    "COL_INTERFACE",
    "COL_PASS_TO_PASS",
    "COL_REQUIREMENTS",
    "COL_SELECTED_TEST_FILES",
    "DEFAULT_IMAGE_PREFIX",
    "DEFAULT_WORKDIR",
    "ELEMENT_WEB_EXCEPTION",
    "IMAGE_REPOSITORY",
    "SWEBENCH_PRO_DATASET",
    "SWEBENCH_PRO_DATASETS",
    "build_problem_statement",
    "dockerhub_tag",
    "instances_from_records",
    "is_swebench_pro_name",
    "load_swebench_pro_instances",
    "looks_like_swebench_pro",
    "parse_string_list",
    "pro_image_name",
    "resolve_dataset_name",
]
