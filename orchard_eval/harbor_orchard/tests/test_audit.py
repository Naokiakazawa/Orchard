"""What ``harbor-orchard audit`` says about a dataset before it costs anything.

The regression these guard is specific: schema 1.3 states network policy under
``[agent]`` and ``[verifier]``, not ``[environment]``. Reading only the old
place reports DeepSWE — 113 air-gapped tasks — as an unrestricted dataset, and
the operator finds out by watching every trial fail on its first model call.
"""

from __future__ import annotations

from pathlib import Path

from harbor_orchard.cli import audit_task

# Trimmed from datacurve/deep-swe-1-1, keeping the shape that matters.
DEEP_SWE_TASK_TOML = """\
schema_version = "1.3"
artifacts = ["/logs/artifacts/model.patch"]

[task]
name = "datacurve/abs-module-cache-flags"

[verifier]
network_mode = "no-network"
environment_mode = "separate"
timeout_sec = 1800.0

[verifier.environment]
build_timeout_sec = 1800.0
cpus = 2
memory_mb = 8192

[[verifier.collect]]
command = "cd /app && git diff --binary HEAD > /logs/artifacts/model.patch"
timeout_sec = 300.0

[agent]
network_mode = "no-network"
timeout_sec = 10800.0

[environment]
build_timeout_sec = 1800.0
docker_image = "public.ecr.aws/d3j8x8q7/swe-bench-202605:abc-v1.1"
os = "linux"
cpus = 2
memory_mb = 8192
gpus = 0
"""

VERIFIER_DOCKERFILE = """\
FROM public.ecr.aws/d3j8x8q7/swe-bench-202605:abc-v1.1
COPY test.sh /tests/test.sh
RUN chmod +x /tests/test.sh
"""


def write_task(root: Path, task_toml: str = DEEP_SWE_TASK_TOML) -> Path:
    task_dir = root / "abs-module-cache-flags"
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "task.toml").write_text(task_toml, encoding="utf-8")
    (task_dir / "tests" / "Dockerfile").write_text(
        VERIFIER_DOCKERFILE, encoding="utf-8"
    )
    (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    return task_dir


class TestDeepSweAudit:
    def test_an_air_gapped_task_is_runnable(self, tmp_path: Path):
        # The provider isolates the pod instead of refusing the task.
        report = audit_task(write_task(tmp_path))
        assert report.supported, report.reasons

    def test_phase_scoped_no_network_is_seen(self, tmp_path: Path):
        report = audit_task(write_task(tmp_path))
        assert report.isolated is True

    def test_the_verifier_dockerfile_is_translated(self, tmp_path: Path):
        # tests/Dockerfile is the separate verifier image; a task whose verifier
        # cannot be built scores zero for reasons that look like a model error.
        # The reported base is what a pod would actually pull: the ECR repository
        # is remapped onto the Docker Hub copy, then onto the Hub mirror.
        report = audit_task(write_task(tmp_path))
        assert report.steps > 0
        assert report.base_images == ["mirror.gcr.io/wenlinyao/deep-swe:abc-v1.1"]

    def test_a_public_task_is_not_reported_as_isolated(self, tmp_path: Path):
        report = audit_task(
            write_task(
                tmp_path,
                DEEP_SWE_TASK_TOML.replace('network_mode = "no-network"', 'network_mode = "public"'),
            )
        )
        assert report.isolated is False

    def test_the_legacy_boolean_still_counts(self, tmp_path: Path):
        report = audit_task(
            write_task(tmp_path, "[environment]\nallow_internet = false\n")
        )
        assert report.isolated is True

    def test_a_gpu_task_is_still_refused(self, tmp_path: Path):
        report = audit_task(
            write_task(tmp_path, DEEP_SWE_TASK_TOML.replace("gpus = 0", "gpus = 1"))
        )
        assert not report.supported
        assert any("gpus" in reason for reason in report.reasons)
