"""Read an image's configured ``WORKDIR`` from its registry.

Docker gives this away for free: ``docker exec`` with no working directory lands
in the image's ``WorkingDir``. Kubernetes does not, because Orchard sets the
pod's ``workingDir`` explicitly (``settings.default_working_dir``, ``/workspace``),
which overrides whatever the image declared. So a prebuilt image's working
directory cannot be discovered by running ``pwd`` inside the pod — the answer is
always ``/workspace``.

It matters more than it sounds. A terminal-bench solution is typically
``cat data.txt``-style code that assumes the image's working directory, so
getting it wrong does not error at setup — it runs the reference solution in the
wrong place, fails, and reports as a model failure.

Guessing is not good enough either. ``/app`` covers 151 of the ~166
terminal-bench Dockerfiles, but ``/workspace``, ``/task``, ``/root``, ``/build``
and several ``/app/<subdir>`` values also appear, and a wrong guess is silent.

So this reads the answer from the registry: manifest → config blob →
``config.WorkingDir``. Stdlib only, anonymous where possible, Bearer-token where
the registry demands it, and cached per reference because a job starts one pod
per trial from the same handful of images.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

#: Manifests and indexes, in both the Docker and OCI spellings. A registry that
#: is handed only one of these may answer a multi-arch image with a 404.
_ACCEPT = ", ".join(
    (
        "application/vnd.docker.distribution.manifest.v2+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.oci.image.index.v1+json",
    )
)

#: Preference order when a reference resolves to a multi-arch index. Sandboxes
#: are x86 today; arm64 is here so this keeps working if that changes.
_PLATFORMS = (("linux", "amd64"), ("linux", "arm64"))

_CHALLENGE = re.compile(r'(\w+)="([^"]*)"')

_cache: dict[str, str | None] = {}


class RegistryError(RuntimeError):
    """The image config could not be read."""


def fetch_image_workdir(reference: str, *, timeout: int = 30) -> str | None:
    """Return the image's ``WorkingDir``, or ``None`` if it cannot be read.

    Never raises: a private registry, a rate limit, or an offline node must
    degrade to the caller's fallback rather than failing a whole job.
    """
    if reference in _cache:
        return _cache[reference]
    try:
        workdir = _fetch(reference, timeout=timeout) or None
    except Exception as exc:  # noqa: BLE001 - advisory lookup, never fatal
        logger.debug("could not read image config for %s: %s", reference, exc)
        workdir = None
    _cache[reference] = workdir
    return workdir


def _fetch(reference: str, *, timeout: int) -> str | None:
    registry, repository, tag = _split(reference)
    manifest = _get_json(registry, f"/v2/{repository}/manifests/{tag}", timeout)

    if "manifests" in manifest:
        digest = _select_platform(manifest["manifests"])
        if digest is None:
            raise RegistryError(f"no linux image in the index for {reference}")
        manifest = _get_json(registry, f"/v2/{repository}/manifests/{digest}", timeout)

    config_digest = (manifest.get("config") or {}).get("digest")
    if not config_digest:
        raise RegistryError(f"manifest for {reference} has no config descriptor")

    config = _get_json(registry, f"/v2/{repository}/blobs/{config_digest}", timeout)
    # `config` is the runtime config; `container_config` is the older build-time
    # spelling that some images still carry.
    for key in ("config", "container_config"):
        section = config.get(key) or {}
        workdir = section.get("WorkingDir")
        if workdir:
            return str(workdir)
    return None


def _split(reference: str) -> tuple[str, str, str]:
    """Split a *resolved* reference into ``(registry_host, repository, tag)``."""
    from harbor_orchard.images import split_reference

    registry, repository, suffix = split_reference(reference)
    if registry is None:
        registry = "registry-1.docker.io"
        if "/" not in repository:
            repository = f"library/{repository}"

    if suffix.startswith("@"):
        tag = suffix[1:]
    elif "@" in suffix:
        tag = suffix.split("@", 1)[1]
    elif suffix.startswith(":"):
        tag = suffix[1:]
    else:
        tag = "latest"
    return registry, repository, tag


def _select_platform(entries: list[dict]) -> str | None:
    for want_os, want_arch in _PLATFORMS:
        for entry in entries:
            platform = entry.get("platform") or {}
            if platform.get("os") == want_os and platform.get("architecture") == want_arch:
                return entry.get("digest")
    for entry in entries:
        if (entry.get("platform") or {}).get("os") == "linux":
            return entry.get("digest")
    return None


def _get_json(registry: str, path: str, timeout: int) -> dict:
    url = f"https://{registry}{path}"
    try:
        return _request_json(url, timeout, token=None)
    except urllib.error.HTTPError as exc:
        if exc.code != 401:
            raise
        token = _authorize(exc, timeout)
        if token is None:
            raise
    return _request_json(url, timeout, token=token)


def _request_json(url: str, timeout: int, *, token: str | None) -> dict:
    request = urllib.request.Request(url, headers={"Accept": _ACCEPT})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _authorize(exc: urllib.error.HTTPError, timeout: int) -> str | None:
    """Follow a registry's Bearer challenge to fetch an anonymous pull token."""
    challenge = exc.headers.get("WWW-Authenticate", "") if exc.headers else ""
    if not challenge.lower().startswith("bearer"):
        return None
    fields = dict(_CHALLENGE.findall(challenge))
    realm = fields.pop("realm", None)
    if not realm:
        return None
    query = urllib.parse.urlencode(fields)
    with urllib.request.urlopen(f"{realm}?{query}", timeout=timeout) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    return payload.get("token") or payload.get("access_token")
