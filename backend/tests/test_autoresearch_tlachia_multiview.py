import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/autoresearch_tlachia_multiview.py"
spec = importlib.util.spec_from_file_location("tlachia_multiview", SCRIPT)
runner = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def hashed_artifact(schema, payload):
    return {
        "schema_version": schema,
        "canonicalization_version": "canonical-json-nfc-v1",
        "hash_algorithm": "sha256",
        "payload": payload,
        "payload_sha256": runner.semantic_hash(payload),
    }


def synthetic_feature_cache():
    groups = [f"group-{index}" for index in range(7)]
    splits = ["structural_dev"] * 6 + ["structural_train"]
    return {
        "schema_version": "tlachia-multiview-p4a-features.v1",
        "features": torch.zeros((7, 5, 384), dtype=torch.float16),
        "row_ids": [f"row-{index}" for index in range(7)],
        "image_paths": [f"/safe/image-{index}.jpg" for index in range(7)],
        "byte_sha256": [f"byte-{index}" for index in range(7)],
        "pixel_sha256": [f"pixel-{index}" for index in range(7)],
        "roles": [["glyph_occurrence"]] * 6 + [["zone"]],
        "source_groups": groups,
        "descendant_class_names": [["x"]] * 7,
        "descendant_class_labels": [[index] for index in range(7)],
        "splits": splits,
        "view_names": list(runner.VIEW_NAMES),
        "feature_dim": 384,
        "promotion_eligible": False,
    }


def test_exactly_sixteen_frozen_non_chained_specs():
    runner.assert_specs()
    assert len(runner.SPECS) == 16
    assert [item.name for item in runner.SPECS[:2]] == ["historical-control", "b1-control"]
    assert all(item.initializer == "historical" for item in runner.SPECS[2:])
    assert sum(item.kind == "hybrid" for item in runner.SPECS) == 3


def test_frozen_protocol_is_json_round_trip_stable():
    protocol = runner.frozen_protocol()
    assert json.loads(json.dumps(protocol)) == protocol


def test_feature_cache_and_sidecar_publish_as_one_immutable_bundle(tmp_path, monkeypatch):
    report_dir = tmp_path / "report"
    cache_path = report_dir / "features/multiview.features.pt"
    sidecar_path = cache_path.with_suffix(cache_path.suffix + ".prov.json")
    monkeypatch.setattr(runner, "REPORT_DIR", report_dir)
    monkeypatch.setattr(runner, "FEATURE_CACHE", cache_path)
    monkeypatch.setattr(runner, "FEATURE_PROVENANCE", sidecar_path)
    runner.publish_feature_bundle(
        {"features": torch.zeros(1)},
        {"schema_version": "test.provenance.v1", "promotion_eligible": False},
    )
    assert cache_path.is_file()
    assert sidecar_path.is_file()
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["cache_sha256"] == runner.sha256_file(cache_path)
    with pytest.raises(RuntimeError, match="immutable feature bundle"):
        runner.publish_feature_bundle(
            {"features": torch.ones(1)},
            {"schema_version": "test.provenance.v1", "promotion_eligible": False},
        )


def test_reviewed_artifact_hash_guard_is_fail_closed():
    value = hashed_artifact("example.v1", {"count": 3, "name": "évidence"})
    runner.verify_hashed_artifact(value, "example.v1", value["payload_sha256"])
    value["payload"]["count"] = 4
    with pytest.raises(ValueError, match="semantic hash"):
        runner.verify_hashed_artifact(value, "example.v1", value["payload_sha256"])


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/annotations/pages/x.jpg",
        "/tmp/business-final/features.pt",
        "/tmp/locked-test/cache.pt",
        "/tmp/locked_test/cache.pt",
    ],
)
def test_forbidden_input_paths(path):
    with pytest.raises(ValueError, match="forbidden research input"):
        runner.assert_allowed_input(Path(path))


def test_deterministic_whole_group_split_maximizes_new_class_evidence():
    rows = []
    evidence = {
        "a": ["0", "1", "2", "3"],
        "b": ["0", "1", "2"],
        "c": ["4", "5", "6"],
        "d": ["7", "8"],
    }
    for group, labels in evidence.items():
        rows.append({"source_group": group, "candidate_runtime_classes": labels})
    first = runner.choose_dev_groups(rows, count=2)
    second = runner.choose_dev_groups(list(reversed(rows)), count=2)
    assert first == second
    assert first[0] == "a"
    assert first[1] == "c"


def test_feature_cache_schema_carries_views_roles_groups_labels_and_split():
    cache = synthetic_feature_cache()
    runner.validate_feature_cache(
        cache, expected_rows=7, expected_groups=7, require_reviewed_paths=False
    )
    assert cache["features"].dtype == torch.float16
    assert cache["view_names"] == ["base", "mild-0", "mild-1", "strong-0", "strong-1"]


def test_feature_cache_rejects_train_dev_group_overlap():
    cache = synthetic_feature_cache()
    cache["source_groups"][-1] = cache["source_groups"][0]
    with pytest.raises(ValueError, match="overlap"):
        runner.validate_feature_cache(
            cache, expected_rows=7, expected_groups=None, require_reviewed_paths=False
        )


def test_role_and_view_filters_never_select_structural_dev():
    cache = {
        "roles": [["glyph_occurrence"], ["zone"], ["planche"], ["glyph_occurrence"]],
        "splits": ["structural_train", "structural_train", "structural_train", "structural_dev"],
        "row_ids": ["g", "z", "p", "held-out"],
    }
    glyph = runner.Spec("glyph", "vicreg", ("glyph_occurrence",), "mild", 1e-5, 1)
    context = runner.Spec("context", "vicreg", ("zone", "planche"), "strong", 1e-5, 1)
    assert runner.eligible_indices(cache, glyph) == [0]
    assert runner.eligible_indices(cache, context) == [1, 2]
    assert runner.view_indices("mild") == (1, 2)
    assert runner.view_indices("strong") == (3, 4)
    assert runner.view_indices("both") == (1, 2, 3, 4)


def test_ssl_plan_is_deterministic_and_uses_distinct_safe_views():
    cache = {
        "roles": [["zone"], ["planche"], ["zone"]],
        "splits": ["structural_train"] * 3,
        "row_ids": ["a", "b", "c"],
    }
    frozen = runner.Spec("test", "vicreg", ("zone", "planche"), "both", 1e-5, 5, True)
    first = runner.build_ssl_plan(cache, frozen, 42)
    second = runner.build_ssl_plan(cache, frozen, 42)
    assert first == second
    assert len(first) == 15
    assert all(row["first_view"] != row["second_view"] for row in first)
    assert all(row["row_id"] != "held-out" for row in first)


def test_sample_ap_and_recall_at_five_cover_multilabel_evidence():
    scores = torch.full((2, runner.CLASS_COUNT), -10.0)
    scores[0, 3] = 5.0
    scores[0, 7] = 4.0
    scores[1, 9] = 5.0
    result = runner.structural_metrics(scores, [[3, 7], [9]], ["p1", "p2"])
    assert result["sample_ap"] == pytest.approx(1.0)
    assert result["recall_at_5"] == pytest.approx(1.0)
    assert result["macro_planche_ap"] == pytest.approx(1.0)
    assert result["groups"] == 2


def _metrics(primary, macro, per_class, group_values):
    return {
        "element_dev": {
            "macro": macro,
            "per_class": {str(label): {"accuracy": value} for label, value in per_class.items()},
        },
        "structural_dev": {
            "macro_planche_ap": primary,
            "per_group": {
                group: {"sample_ap": value, "recall_at_5": 1.0, "samples": 1}
                for group, value in group_values.items()
            },
        },
    }


def test_final_gate_requires_all_preregistered_fresh_confirmation_conditions():
    controls = [
        _metrics(0.40, 0.60, {0: 0.50, 1: 0.60}, {f"p{i}": 0.40 for i in range(6)})
        for _ in range(3)
    ]
    candidates = [
        _metrics(0.44, 0.598, {0: 0.51, 1: 0.61}, {f"p{i}": 0.44 for i in range(6)}),
        _metrics(0.43, 0.599, {0: 0.51, 1: 0.61}, {f"p{i}": 0.43 for i in range(6)}),
        _metrics(0.42, 0.600, {0: 0.51, 1: 0.61}, {f"p{i}": 0.42 for i in range(6)}),
    ]
    outcome = runner.final_gate(candidates, controls, runner.CONFIRMATION_SEEDS)
    assert outcome["pass"] is True
    assert outcome["passes"]["at_least_two_of_three_fresh_seed_gains_positive"] is True
    assert outcome["promotion_eligible"] is False


def test_final_gate_rejects_single_element_class_drop_over_point_two():
    controls = [
        _metrics(0.40, 0.60, {0: 0.80, 1: 0.60}, {f"p{i}": 0.40 for i in range(6)})
        for _ in range(3)
    ]
    candidates = [
        _metrics(0.44, 0.60, {0: 0.59, 1: 0.61}, {f"p{i}": 0.44 for i in range(6)})
        for _ in range(3)
    ]
    outcome = runner.final_gate(candidates, controls, runner.CONFIRMATION_SEEDS)
    assert outcome["pass"] is False
    assert outcome["class_drops_over_0_20"] == ["0"]


def test_strong_seed42_screen_cannot_rescue_tiny_fresh_confirmation_gains():
    # Seed 42 selected this candidate at +0.25, but it is intentionally absent
    # from the final-gate API.  All final evidence below is fresh and +0.001.
    seed42_screening_gain = 0.25
    assert seed42_screening_gain > 0.01
    controls = [
        _metrics(0.40, 0.60, {0: 0.50}, {f"p{i}": 0.40 for i in range(6)})
        for _ in range(3)
    ]
    candidates = [
        _metrics(0.401, 0.60, {0: 0.50}, {f"p{i}": 0.401 for i in range(6)})
        for _ in range(3)
    ]
    outcome = runner.final_gate(candidates, controls, runner.CONFIRMATION_SEEDS)
    assert outcome["primary_gain"] == pytest.approx(0.001)
    assert outcome["pass"] is False


def test_final_gate_rejects_any_decision_seed_set_containing_screen_seed42():
    controls = [
        _metrics(0.40, 0.60, {0: 0.50}, {f"p{i}": 0.40 for i in range(6)})
        for _ in range(3)
    ]
    candidates = [
        _metrics(0.44, 0.60, {0: 0.50}, {f"p{i}": 0.44 for i in range(6)})
        for _ in range(3)
    ]
    with pytest.raises(ValueError, match="preregistered seeds"):
        runner.final_gate(candidates, controls, (42, 73, 101))


def test_final_gain_and_interval_are_computed_from_fresh_seed_group_pairs():
    controls = [
        _metrics(0.30 + seed / 100, 0.60, {0: 0.50}, {f"p{i}": 0.30 + seed / 100 for i in range(6)})
        for seed in range(3)
    ]
    candidates = [
        _metrics(0.32 + seed / 100, 0.60, {0: 0.50}, {f"p{i}": 0.32 + seed / 100 for i in range(6)})
        for seed in range(3)
    ]
    outcome = runner.final_gate(candidates, controls, runner.CONFIRMATION_SEEDS)
    assert outcome["primary_gain"] == pytest.approx(0.02)
    assert outcome["paired_planche_bootstrap"]["mean"] == pytest.approx(0.02)
    assert outcome["paired_planche_bootstrap"]["lower_95"] == pytest.approx(0.02)
    assert outcome["decision_seeds"] == list(runner.CONFIRMATION_SEEDS)


def test_safe_view_is_pixel_deterministic_and_preserves_shape():
    base = Image.new("RGB", (224, 224), "white")
    first = runner.safe_ssl_view(base, "row-1", "strong", 1)
    second = runner.safe_ssl_view(base, "row-1", "strong", 1)
    assert first.size == (224, 224)
    assert first.tobytes() == second.tobytes()
