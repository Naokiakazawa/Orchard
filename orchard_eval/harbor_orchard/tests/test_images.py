"""Mirror-rewriting tests.

The failure this guards against is asymmetric: rewriting a Docker Hub reference
wrongly costs a slow pull, while rewriting a non-Hub reference produces a 404 at
pod-create time that reads as an infrastructure fault.
"""

from __future__ import annotations

import pytest

from harbor_orchard.images import (
    normalize,
    parse_remap,
    remap,
    rewrite,
    split_reference,
)


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("python:3.11-slim", (None, "python", ":3.11-slim")),
        ("ubuntu", (None, "ubuntu", "")),
        ("coqorg/coq:8.18", (None, "coqorg/coq", ":8.18")),
        ("docker.io/nvidia/cuda:13.2", ("docker.io", "nvidia/cuda", ":13.2")),
        ("mcr.microsoft.com/playwright/python:v1.52.0-noble",
         ("mcr.microsoft.com", "playwright/python", ":v1.52.0-noble")),
        ("localhost:5000/thing:v1", ("localhost:5000", "thing", ":v1")),
        ("python@sha256:abc123", (None, "python", "@sha256:abc123")),
        ("python:3.11-slim@sha256:abc123",
         (None, "python", ":3.11-slim@sha256:abc123")),
    ],
)
def test_split_reference(reference, expected):
    assert split_reference(reference) == expected


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        # Official images gain the implicit `library/` namespace...
        ("python:3.11-slim", "mirror.gcr.io/library/python:3.11-slim"),
        ("ubuntu:24.04", "mirror.gcr.io/library/ubuntu:24.04"),
        # ...but two-component names already have one.
        ("coqorg/coq:8.18", "mirror.gcr.io/coqorg/coq:8.18"),
        ("mambaorg/micromamba:1.5", "mirror.gcr.io/mambaorg/micromamba:1.5"),
        # An explicit docker.io is still Docker Hub.
        ("docker.io/nvidia/cuda:13.2", "mirror.gcr.io/nvidia/cuda:13.2"),
        # Digests survive the rewrite; dropping one would change the image.
        (
            "python:3.11-slim@sha256:9a7765",
            "mirror.gcr.io/library/python:3.11-slim@sha256:9a7765",
        ),
    ],
)
def test_docker_hub_references_are_mirrored(reference, expected):
    assert rewrite(reference) == expected


@pytest.mark.parametrize(
    "reference",
    [
        "mcr.microsoft.com/playwright/python:v1.52.0-noble",
        "ghcr.io/example/tool:v1",
        "nvcr.io/nvidia/pytorch:24.01-py3",
        "localhost:5000/thing:v1",
        "scratch",
    ],
)
def test_non_hub_references_are_left_alone(reference):
    assert rewrite(reference) == reference


def test_mirroring_can_be_disabled():
    assert rewrite("python:3.11-slim", None) == "python:3.11-slim"


def test_normalize_fills_in_the_implicit_parts():
    assert normalize("python:3.11-slim") == "docker.io/library/python:3.11-slim"
    assert normalize("coqorg/coq:8.18") == "docker.io/coqorg/coq:8.18"
    assert normalize("ghcr.io/a/b:1") == "ghcr.io/a/b:1"


class TestRemap:
    """Whole repositories served from a copy, so a throttled registry is skipped.

    DeepSWE's 113 images live in public.ecr.aws, whose anonymous pull quota is
    reached long before 113 tasks are; the symptom is an environment start
    timeout, not a rate-limit error, so the substitution has to be the default.
    """

    def test_the_deep_swe_repository_is_served_from_docker_hub(self):
        assert (
            remap("public.ecr.aws/d3j8x8q7/swe-bench-202605:abc-v1.1")
            == "wenlinyao/deep-swe:abc-v1.1"
        )

    def test_the_tag_is_what_carries_the_task_identity(self):
        # The mirroring script pushes the source tag verbatim; inventing a new
        # one here would point every task at an image that was never pushed.
        tag = "kh72w47mwm321wh9xj47vskc8d822e9t-v1.1"
        assert (
            remap(f"public.ecr.aws/d3j8x8q7/swe-bench-202605:{tag}")
            == f"wenlinyao/deep-swe:{tag}"
        )

    def test_a_digest_becomes_the_tag_the_mirror_script_writes(self):
        assert (
            remap("public.ecr.aws/d3j8x8q7/swe-bench-202605@sha256:abc123")
            == "wenlinyao/deep-swe:sha256-abc123"
        )

    def test_other_repositories_are_untouched(self):
        assert remap("ghcr.io/example/tool:v1") == "ghcr.io/example/tool:v1"
        assert remap("python:3.11-slim") == "python:3.11-slim"

    def test_an_empty_table_disables_substitution(self):
        reference = "public.ecr.aws/d3j8x8q7/swe-bench-202605:abc-v1.1"
        assert remap(reference, {}) == reference

    def test_a_remapped_hub_image_still_goes_through_the_mirror(self):
        assert (
            rewrite("public.ecr.aws/d3j8x8q7/swe-bench-202605:abc-v1.1")
            == "mirror.gcr.io/wenlinyao/deep-swe:abc-v1.1"
        )


class TestParseRemap:
    def test_pairs_are_keyed_by_normalized_repository(self):
        mapping = parse_remap("quay.io/a/b=me/b, coqorg/coq=me/coq")
        assert mapping == {"quay.io/a/b": "me/b", "docker.io/coqorg/coq": "me/coq"}
        assert remap("index.docker.io/coqorg/coq:8.18", mapping) == "me/coq:8.18"

    def test_a_malformed_entry_is_rejected(self):
        with pytest.raises(ValueError):
            parse_remap("quay.io/a/b")

    def test_a_tagged_destination_is_rejected(self):
        # Silently appending the source's tag to it yields `me/b:v1:abc-v1.1`,
        # which fails at pod creation on every task rather than at startup.
        with pytest.raises(ValueError):
            parse_remap("quay.io/a/b=me/b:v1")
