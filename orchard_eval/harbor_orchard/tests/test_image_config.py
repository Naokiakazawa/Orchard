"""Reference parsing for the registry image-config lookup.

Only the offline half is tested here — turning a reference into a registry host,
repository and tag. The network half degrades to ``None`` by design, so its
failure modes are the caller's fallback path rather than something to assert on.
"""

from __future__ import annotations

import pytest

from harbor_orchard.image_config import _select_platform, _split


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        # A mirrored reference is queried on the mirror, so the config read
        # describes the exact image the pod will run.
        (
            "mirror.gcr.io/alexgshaw/write-compressor:20251031",
            ("mirror.gcr.io", "alexgshaw/write-compressor", "20251031"),
        ),
        (
            "mirror.gcr.io/library/python:3.13-slim",
            ("mirror.gcr.io", "library/python", "3.13-slim"),
        ),
        ("ghcr.io/astral-sh/uv:0.8.14", ("ghcr.io", "astral-sh/uv", "0.8.14")),
        # Implicit Docker Hub gains both the default host and `library/`.
        ("python:3.11-slim", ("registry-1.docker.io", "library/python", "3.11-slim")),
        ("coqorg/coq:8.18", ("registry-1.docker.io", "coqorg/coq", "8.18")),
        # No tag means latest.
        ("ubuntu", ("registry-1.docker.io", "library/ubuntu", "latest")),
    ],
)
def test_split_reference_for_registry_api(reference, expected):
    assert _split(reference) == expected


def test_a_digest_is_used_as_the_manifest_reference():
    _, _, tag = _split("mirror.gcr.io/library/python@sha256:abc123")
    assert tag == "sha256:abc123"


def test_a_digest_wins_over_a_tag():
    # `repo:tag@sha256:...` pins the digest; the tag is decorative.
    _, _, tag = _split("mirror.gcr.io/library/python:3.11-slim@sha256:abc123")
    assert tag == "sha256:abc123"


def test_platform_selection_prefers_linux_amd64():
    entries = [
        {"digest": "sha256:arm", "platform": {"os": "linux", "architecture": "arm64"}},
        {"digest": "sha256:amd", "platform": {"os": "linux", "architecture": "amd64"}},
        {"digest": "sha256:win", "platform": {"os": "windows", "architecture": "amd64"}},
    ]
    assert _select_platform(entries) == "sha256:amd"


def test_platform_selection_falls_back_to_any_linux():
    entries = [
        {"digest": "sha256:win", "platform": {"os": "windows", "architecture": "amd64"}},
        {"digest": "sha256:s390", "platform": {"os": "linux", "architecture": "s390x"}},
    ]
    assert _select_platform(entries) == "sha256:s390"


def test_platform_selection_gives_up_on_a_windows_only_index():
    entries = [
        {"digest": "sha256:win", "platform": {"os": "windows", "architecture": "amd64"}}
    ]
    assert _select_platform(entries) is None
