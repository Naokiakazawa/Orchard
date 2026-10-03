"""Rewrite image references onto a pull-through mirror.

Sandbox pods pull through ``mirror.gcr.io``, which mirrors Docker Hub and only
Docker Hub. Rewriting a ``ghcr.io`` or ``mcr.microsoft.com`` reference onto it
produces a 404 at pod-create time, several seconds into a trial, so the registry
of a reference has to be identified before anything is rewritten rather than
after.

The subtlety is Docker's implicit naming: ``python:3.11-slim`` means
``docker.io/library/python:3.11-slim``, and the ``library/`` namespace only
applies to references with a single path component. ``coqorg/coq:8.18`` already
has two and must not gain it.

A reference that names a registry the mirror cannot serve is handled one step
earlier, by :func:`remap`: whole repositories can be substituted for copies you
pushed yourself, which is how a benchmark escapes a registry that throttles it.
"""

from __future__ import annotations

from collections.abc import Mapping

DEFAULT_MIRROR = "mirror.gcr.io"

#: Every spelling of Docker Hub. All of them normalize to the same repository.
DOCKER_HUB_REGISTRIES = frozenset(
    {
        "docker.io",
        "index.docker.io",
        "registry-1.docker.io",
        "registry.hub.docker.com",
    }
)

#: Repositories served from a copy rather than from where the dataset points,
#: as ``source repository -> destination repository``.
#:
#: DeepSWE 1.1 names its 113 task images in ``public.ecr.aws``, which throttles
#: anonymous pulls hard. Each task image is pulled at least twice per trial —
#: once for the agent pod, once to build the verifier — so at any real
#: concurrency the pull outlives ``create_timeout`` and the trial is reported as
#: ``EnvironmentStartTimeoutError`` rather than as the rate limit it is.
#:
#: ``orchard_eval/scripts/mirror_deep_swe_images.sh`` populates the destination and
#: reuses the source tag verbatim, which is what lets this be a repository-level
#: substitution instead of a 113-entry table.
DEFAULT_REMAP: Mapping[str, str] = {
    "public.ecr.aws/d3j8x8q7/swe-bench-202605": "wenlinyao/deep-swe",
}

def split_reference(reference: str) -> tuple[str | None, str, str]:
    """Split into ``(registry, repository, suffix)``.

    ``suffix`` carries the ``:tag`` and/or ``@digest``. ``registry`` is ``None``
    when the reference is an implicit Docker Hub one.
    """
    remainder = reference
    digest = ""
    if "@" in remainder:
        remainder, _, digest_part = remainder.partition("@")
        digest = f"@{digest_part}"

    registry: str | None = None
    head, slash, tail = remainder.partition("/")
    if slash and _looks_like_registry(head):
        registry = head
        remainder = tail

    tag = ""
    if ":" in remainder.rsplit("/", 1)[-1]:
        remainder, _, tag_part = remainder.rpartition(":")
        tag = f":{tag_part}"

    return registry, remainder, tag + digest


def _looks_like_registry(head: str) -> bool:
    """True when the first path component names a host rather than a namespace.

    Docker's own rule: a host has a dot, a port, or is exactly ``localhost``.
    Without it, ``coqorg/coq`` would be read as the registry ``coqorg``.
    """
    return "." in head or ":" in head or head == "localhost"


def normalize(reference: str) -> str:
    """Return the fully qualified form of *reference*."""
    registry, repository, suffix = split_reference(reference)
    if registry is None:
        registry = "docker.io"
    if registry in DOCKER_HUB_REGISTRIES and "/" not in repository:
        repository = f"library/{repository}"
    return f"{registry}/{repository}{suffix}"


def is_docker_hub(reference: str) -> bool:
    registry, _, _ = split_reference(reference)
    return registry is None or registry in DOCKER_HUB_REGISTRIES


def repository_key(reference: str) -> str:
    """The ``registry/repository`` of *reference*, with no tag or digest.

    Every spelling of Docker Hub collapses to ``docker.io`` so that a remap
    entry written as ``coqorg/coq`` also matches ``index.docker.io/coqorg/coq``.
    """
    registry, repository, _ = split_reference(reference)
    if registry is None or registry in DOCKER_HUB_REGISTRIES:
        if "/" not in repository:
            repository = f"library/{repository}"
        registry = "docker.io"
    return f"{registry}/{repository}"


def parse_remap(spec: str) -> dict[str, str]:
    """Parse ``src=dst,src2=dst2`` into a remap table keyed by repository."""
    mapping: dict[str, str] = {}
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        source, separator, destination = entry.partition("=")
        source, destination = source.strip(), destination.strip()
        if not separator or not source or not destination:
            raise ValueError(
                f"expected 'source=destination' repository pairs, got {entry!r}"
            )
        _, _, suffix = split_reference(destination)
        if suffix:
            # A tagged destination would concatenate with the source's own tag.
            raise ValueError(
                f"the destination {destination!r} must be a repository, with no "
                "tag or digest: the source's tag is carried over"
            )
        mapping[repository_key(source)] = destination
    return mapping


def remap(reference: str, mapping: Mapping[str, str] | None = None) -> str:
    """Substitute the repository of *reference* using *mapping*.

    The tag is carried over unchanged, because that is what the mirroring script
    pushes and the tag is what carries the task's identity. A digest reference
    has no tag to carry, so it becomes the ``sha256-<digest>`` tag the script
    writes instead — a digest is content-addressed to one repository's manifest
    and does not survive a copy.

    ``mapping=None`` means :data:`DEFAULT_REMAP`; an empty mapping disables
    substitution, which is what an operator with the original registry's
    credentials wants.
    """
    if mapping is None:
        mapping = DEFAULT_REMAP
    if not mapping or reference == "scratch":
        return reference
    destination = mapping.get(repository_key(reference))
    if destination is None:
        return reference

    _, _, suffix = split_reference(reference)
    if "@" in suffix:
        digest = suffix.partition("@")[2]
        suffix = f":{digest.replace(':', '-')}"
    elif not suffix:
        suffix = ":latest"
    return f"{destination}{suffix}"


def rewrite(
    reference: str,
    mirror: str | None = DEFAULT_MIRROR,
    remap_repositories: Mapping[str, str] | None = None,
) -> str:
    """Resolve *reference* to what a pod should actually pull.

    Repository substitution happens first, so an image moved onto Docker Hub is
    then eligible for the Hub mirror as well. Non-Hub references and
    scratch/stage names are returned unchanged. Passing ``mirror=None`` disables
    mirror rewriting, which is what a cluster with a node-level registry mirror
    wants.
    """
    reference = remap(reference, remap_repositories)
    if not mirror:
        return reference
    if reference == "scratch":
        return reference
    if not is_docker_hub(reference):
        return reference
    _, repository, suffix = split_reference(reference)
    if "/" not in repository:
        repository = f"library/{repository}"
    return f"{mirror.rstrip('/')}/{repository}{suffix}"
