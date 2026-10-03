"""Dataset loading, image naming, and instance selection."""

import json
import sys
import types

import pytest

from orchard_evalkit.config import DatasetConfig
from orchard_evalkit.datasets.swebench import (
    instances_from_records,
    load_records,
    load_swebench_instances,
    select_instances,
    swebench_image_name,
)
from orchard_evalkit.datasets.swebench_pro import load_swebench_pro_instances


class TestImageNaming:
    def test_double_underscore_becomes_the_magic_token(self):
        # Docker tags forbid `__`; SWE-bench substitutes `_1776_`.
        assert swebench_image_name("astropy__astropy-12907") == (
            "docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"
        )

    def test_name_is_lowercased(self):
        name = swebench_image_name("pytest-dev__pytest-5227")
        assert name == name.lower()
        assert "pytest-dev_1776_pytest-5227" in name

    def test_prefix_selects_the_registry(self):
        assert swebench_image_name(
            "django__django-11095", prefix="mirror.gcr.io/swebench"
        ).startswith("mirror.gcr.io/swebench/")

    def test_trailing_slash_in_prefix_is_tolerated(self):
        assert "//" not in swebench_image_name("a__b-1", prefix="reg.io/swebench/")

    def test_empty_prefix_yields_a_bare_name(self):
        assert (
            swebench_image_name("a__b-1", prefix="")
            == "sweb.eval.x86_64.a_1776_b-1:latest"
        )


class TestInstancesFromRecords:
    def test_fields_are_normalized(self):
        [instance] = instances_from_records(
            [
                {
                    "instance_id": "django__django-11095",
                    "problem_statement": "fix it",
                    "repo": "django/django",
                    "base_commit": "abc123",
                }
            ]
        )
        assert instance.instance_id == "django__django-11095"
        assert instance.base_commit == "abc123"
        assert instance.workdir == "/testbed"
        # The raw row is preserved because grading needs FAIL_TO_PASS et al.
        assert instance.raw["repo"] == "django/django"

    @pytest.mark.parametrize("key", ["image", "image_name", "docker_image"])
    def test_a_custom_image_wins_over_the_convention(self, key):
        [instance] = instances_from_records(
            [{"instance_id": "a__b-1", key: "myacr.io/custom:tag"}]
        )
        assert instance.image == "myacr.io/custom:tag"

    def test_a_standard_image_is_repointed_at_the_configured_registry(self):
        # The modern datasets hardcode `swebench/...` on Docker Hub. Honouring
        # that verbatim would defeat the whole point of configuring a mirror.
        [instance] = instances_from_records(
            [
                {
                    "instance_id": "astropy__astropy-12907",
                    "image": "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest",
                }
            ],
            image_prefix="myacr.azurecr.io/swebench",
        )
        assert instance.image == (
            "myacr.azurecr.io/swebench/"
            "sweb.eval.x86_64.astropy_1776_astropy-12907:latest"
        )

    def test_an_empty_prefix_keeps_the_row_image(self):
        [instance] = instances_from_records(
            [
                {
                    "instance_id": "astropy__astropy-12907",
                    "image": "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest",
                }
            ],
            image_prefix="",
        )
        assert instance.image.startswith("swebench/")

    def test_missing_instance_id_is_rejected(self):
        with pytest.raises(ValueError, match="instance_id"):
            instances_from_records([{"problem_statement": "x"}])


def _instances(n: int):
    return instances_from_records(
        [{"instance_id": f"repo__proj-{i}", "problem_statement": "s"} for i in range(n)]
    )


class TestSelectInstances:
    def test_limit_truncates(self):
        assert len(select_instances(_instances(10), DatasetConfig(limit=3))) == 3

    def test_slice_applies(self):
        selected = select_instances(_instances(10), DatasetConfig(slice="2:5"))
        assert [i.instance_id for i in selected] == [
            "repo__proj-2",
            "repo__proj-3",
            "repo__proj-4",
        ]

    def test_filter_is_a_regex_on_instance_id(self):
        selected = select_instances(_instances(20), DatasetConfig(filter=r".*-1\d$"))
        assert len(selected) == 10

    def test_explicit_ids_preserve_the_requested_order(self):
        selected = select_instances(
            _instances(5),
            DatasetConfig(instance_ids=["repo__proj-3", "repo__proj-1"]),
        )
        assert [i.instance_id for i in selected] == ["repo__proj-3", "repo__proj-1"]

    def test_explicit_ids_override_limit(self):
        selected = select_instances(
            _instances(5), DatasetConfig(instance_ids=["repo__proj-0"], limit=99)
        )
        assert len(selected) == 1

    def test_unknown_explicit_id_is_an_error(self):
        # Silently dropping it would quietly shrink the benchmark.
        with pytest.raises(KeyError):
            select_instances(_instances(3), DatasetConfig(instance_ids=["nope"]))

    def test_shuffle_is_deterministic_for_a_seed(self):
        cfg = DatasetConfig(shuffle=True, seed=7)
        first = [i.instance_id for i in select_instances(_instances(20), cfg)]
        second = [i.instance_id for i in select_instances(_instances(20), cfg)]
        assert first == second
        assert first != [f"repo__proj-{i}" for i in range(20)]


class TestLocalFileLoading:
    def test_jsonl(self, tmp_path):
        path = tmp_path / "d.jsonl"
        path.write_text(
            "\n".join(
                json.dumps({"instance_id": f"a__b-{i}", "problem_statement": "s"})
                for i in range(3)
            )
        )
        instances = load_swebench_instances(DatasetConfig(name=str(path)))
        assert len(instances) == 3

    def test_json_list(self, tmp_path):
        path = tmp_path / "d.json"
        path.write_text(json.dumps([{"instance_id": "a__b-1"}]))
        assert len(load_swebench_instances(DatasetConfig(name=str(path)))) == 1

    def test_json_keyed_by_instance_id(self, tmp_path):
        path = tmp_path / "d.json"
        path.write_text(json.dumps({"a__b-1": {"instance_id": "a__b-1"}}))
        assert len(load_swebench_instances(DatasetConfig(name=str(path)))) == 1


class TestRevisionPinning:
    """``dataset.revision`` has to reach ``load_dataset``.

    Unpinned, a benchmark can be replaced underneath a run: ScaleAI swapped
    ``SWE-bench_Pro``'s default config for V2 on 2026-09-22, which is 642 tasks
    instead of 731 and rewrites the ones it keeps. ``configs/swebench-pro.yaml``
    pins ``v1.0``, and that pin is only worth anything if it is passed on.
    """

    def _capture(self, monkeypatch):
        seen = {}

        def fake_load_dataset(name, split=None, revision=None, **kwargs):
            seen.update(name=name, split=split, revision=revision)
            return [
                {
                    "instance_id": "a__b-1",
                    "problem_statement": "s",
                    "repo": "a/b",
                    "dockerhub_tag": "a.b-1",
                }
            ]

        module = types.ModuleType("datasets")
        module.load_dataset = fake_load_dataset
        monkeypatch.setitem(sys.modules, "datasets", module)
        return seen

    def test_revision_is_passed_through(self, monkeypatch):
        seen = self._capture(monkeypatch)
        # The Pro shorthand is expanded by datasets.swebench_pro before it gets
        # here, so this layer sees the HuggingFace id it will actually fetch.
        load_records(DatasetConfig(name="ScaleAI/SWE-bench_Pro", revision="v1.0"))
        assert seen["name"] == "ScaleAI/SWE-bench_Pro"
        assert seen["revision"] == "v1.0"

    def test_empty_revision_means_the_default_branch(self, monkeypatch):
        # None, not "" — an empty string is not a ref and `datasets` would
        # rather be told nothing than be told to resolve "".
        seen = self._capture(monkeypatch)
        load_records(DatasetConfig(name="swe-bench-verified"))
        assert seen["revision"] is None

    def test_pro_loader_keeps_the_pin_while_expanding_the_shorthand(
        self, monkeypatch
    ):
        # The Pro path rewrites cfg.name on the way in; the pin has to survive
        # that copy, or configs/swebench-pro.yaml's `revision: v1.0` is inert
        # for exactly the benchmark it was written for.
        seen = self._capture(monkeypatch)
        load_swebench_pro_instances(
            DatasetConfig(name="swe-bench-pro", revision="v1.0")
        )
        assert seen["name"] == "ScaleAI/SWE-bench_Pro"
        assert seen["revision"] == "v1.0"
