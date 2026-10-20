from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "autoresearch_legacy_fusion.py"
SPEC = importlib.util.spec_from_file_location("autoresearch_legacy_fusion", SCRIPT)
assert SPEC and SPEC.loader
fusion = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fusion
SPEC.loader.exec_module(fusion)


def _taxonomy():
    shared = [f"shared-{index}" for index in range(31)]
    runtime = shared + [f"b1-only-{index}" for index in range(255)]
    legacy = ["?", *shared[:15], "barra", *shared[15:]]
    return fusion.validate_taxonomy(runtime, legacy)


def _rows(group_count: int = 24):
    return [
        {
            "row_id": f"row-{index}",
            "page_group": f"page-{index}",
            "pixel_sha256": f"pixel-{index}",
            "bmp_sha256": f"bmp-{index}",
            "truth_labels": [index % 6],
        }
        for index in range(group_count)
    ]


def test_taxonomy_maps_exactly_31_shared_and_protects_255_b1_only_classes():
    taxonomy = _taxonomy()
    assert len(taxonomy.shared_names) == 31
    assert len(taxonomy.b1_only_indices) == 255
    assert taxonomy.old_only_names == ("?", "barra")
    assert taxonomy.legacy_to_runtime[0] is None
    assert taxonomy.legacy_to_runtime[16] is None
    assert set(taxonomy.shared_names).isdisjoint(taxonomy.b1_only_names)


def test_taxonomy_fails_closed_on_any_other_overlap_shape():
    shared = [f"shared-{index}" for index in range(31)]
    runtime = shared + [f"b1-only-{index}" for index in range(255)]
    legacy = ["?", *shared[:15], "barra", *shared[15:30], "another-old-only"]
    with pytest.raises(ValueError, match="31 shared"):
        fusion.validate_taxonomy(runtime, legacy)


def test_page_split_is_deterministic_and_exactly_18_train_6_dev():
    rows = _rows()
    first = fusion.assign_page_split(rows)
    second = fusion.assign_page_split(list(reversed(rows)))
    assert {row["row_id"]: row["split"] for row in first} == {
        row["row_id"]: row["split"] for row in second
    }
    train = {row["page_group"] for row in first if row["split"] == "train"}
    dev = {row["page_group"] for row in first if row["split"] == "dev"}
    assert len(train) == 18
    assert len(dev) == 6
    assert train.isdisjoint(dev)


def test_split_keeps_every_page_whole():
    rows = _rows()
    rows.append({**rows[0], "row_id": "same-page-second-glyph", "pixel_sha256": "other"})
    assigned = fusion.assign_page_split(rows)
    assert len({row["split"] for row in assigned if row["page_group"] == "page-0"}) == 1


def test_leakage_filter_excludes_run50_seen_pixels_and_bmp_crossing_splits():
    rows = fusion.assign_page_split(_rows())
    train = next(row for row in rows if row["split"] == "train")
    dev = next(row for row in rows if row["split"] == "dev")
    train["bmp_sha256"] = "crossing-bmp"
    dev["bmp_sha256"] = "crossing-bmp"
    seen = next(row for row in rows if row["row_id"] not in {train["row_id"], dev["row_id"]})
    kept, audit = fusion.exclude_leakage(rows, {seen["pixel_sha256"]})
    excluded_ids = {row["row_id"] for row in audit["excluded_rows"]}
    assert {train["row_id"], dev["row_id"], seen["row_id"]} <= excluded_ids
    assert audit["reasons"]["run50_pixel_identical"] == 1
    assert audit["reasons"]["bmp_crosses_train_dev"] == 2
    assert all(row["row_id"] not in excluded_ids for row in kept)


def test_leakage_filter_rejects_near_duplicate_dhash_and_reports_threshold():
    rows = fusion.assign_page_split(_rows())
    rows[0]["dhash64"] = "000000000000000f"
    kept, audit = fusion.exclude_leakage(
        rows,
        set(),
        {int("0000000000000000", 16)},
    )
    assert rows[0]["row_id"] not in {row["row_id"] for row in kept}
    assert audit["reasons"]["run50_dhash_hamming_le_4"] == 1
    assert audit["dhash_hamming_threshold"] == 4


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/business-final/results.json",
        "/tmp/annotations/export.json",
        "/tmp/locked-test/cache.pt",
        "/tmp/locked_test/cache.pt",
    ],
)
def test_forbidden_inputs_are_rejected(path):
    with pytest.raises(ValueError, match="forbidden research input"):
        fusion.assert_allowed_input(Path(path))


def test_fusion_variants_leave_all_255_b1_only_logits_bit_identical():
    taxonomy = _taxonomy()
    rng = np.random.default_rng(7)
    b1 = rng.normal(size=(4, 286))
    legacy = rng.normal(size=(4, 33))
    variants = fusion.build_fusion_variants(
        b1, legacy, taxonomy, prior_weight=0.5, cascade_threshold=0.7
    )
    assert tuple(variants) == ("b1", "legacy", "prior_global", "cascade")
    for name in ("prior_global", "cascade"):
        np.testing.assert_array_equal(
            variants[name][:, taxonomy.b1_only_indices],
            b1[:, taxonomy.b1_only_indices],
        )
    assert np.all(variants["legacy"][:, taxonomy.b1_only_indices] < -1e8)


def test_cascade_is_identity_when_legacy_confidence_is_below_threshold():
    taxonomy = _taxonomy()
    b1 = np.zeros((2, 286), dtype=np.float64)
    legacy = np.full((2, 33), -20.0, dtype=np.float64)
    variants = fusion.build_fusion_variants(
        b1, legacy, taxonomy, prior_weight=1.0, cascade_threshold=0.9
    )
    np.testing.assert_array_equal(variants["cascade"], b1)


def test_calibration_is_explicit_train_only_and_raw_b1_baseline_is_cache_identical(
    monkeypatch,
):
    taxonomy = _taxonomy()
    rng = np.random.default_rng(23)
    b1 = rng.normal(size=(4, 286))
    legacy = rng.normal(size=(4, 33))
    splits = ["train", "train", "train", "dev"]
    truths = [[0], [1], [2], [3]]
    monkeypatch.setattr(fusion, "TEMPERATURE_GRID", (2.0,))
    first, first_report = fusion.build_train_selected_variants(
        b1, legacy, splits, truths, taxonomy
    )

    changed_b1 = b1.copy()
    changed_legacy = legacy.copy()
    changed_b1[-1] = rng.normal(loc=100.0, size=286)
    changed_legacy[-1] = rng.normal(loc=-100.0, size=33)
    second, second_report = fusion.build_train_selected_variants(
        changed_b1, changed_legacy, splits, truths, taxonomy
    )

    assert first_report == second_report
    assert first_report["selection_split"] == "train"
    assert first_report["train_glyphs"] == 3
    assert first_report["parameters"]["b1_temperature"] == 2.0
    np.testing.assert_array_equal(first["b1"], b1)
    assert first["b1"].dtype == b1.dtype
    assert not np.array_equal(
        first["b1_calibrated"][:, taxonomy.shared_runtime_indices],
        b1[:, taxonomy.shared_runtime_indices],
    )
    np.testing.assert_array_equal(
        first["b1_calibrated"][:, taxonomy.b1_only_indices],
        b1[:, taxonomy.b1_only_indices],
    )
    for name in ("prior_global", "cascade"):
        np.testing.assert_array_equal(
            first[name][:, taxonomy.b1_only_indices],
            b1[:, taxonomy.b1_only_indices],
        )
    np.testing.assert_array_equal(first["prior_global"][:3], second["prior_global"][:3])


def test_oracle_selects_best_available_variant_per_glyph_but_is_marked_non_candidate():
    taxonomy = _taxonomy()
    b1 = np.zeros((2, 286), dtype=np.float64)
    prior = b1.copy()
    cascade = b1.copy()
    legacy = np.full_like(b1, -1e9)
    b1[0, 0] = 4.0
    prior[0, 1] = 5.0
    b1[1, 1] = 4.0
    cascade[1, 2] = 5.0
    scores, sources = fusion.build_oracle_scores(
        {"b1": b1, "legacy": legacy, "prior_global": prior, "cascade": cascade},
        [[1], [2]],
    )
    assert scores.argmax(axis=1).tolist() == [1, 2]
    assert sources == ["prior_global", "cascade"]
    assert fusion.VARIANTS["oracle"]["candidate"] is False


def test_metrics_cover_glyph_macro_sample_ap_recall_set_f1_and_calibration():
    scores = np.full((3, 6), -10.0, dtype=np.float64)
    scores[0, [1, 2]] = [5.0, 4.0]
    scores[1, [3, 4]] = [5.0, 4.0]
    scores[2, [5, 0]] = [5.0, 4.0]
    metrics = fusion.evaluate_scores(
        scores,
        truth_labels=[[1, 2], [4], [5]],
        page_groups=["a", "a", "b"],
    )
    assert metrics["glyphs"]["sample_ap"] == pytest.approx((1.0 + 0.5 + 1.0) / 3)
    assert 0.0 <= metrics["glyphs"]["macro_ap"] <= 1.0
    assert metrics["glyphs"]["recall_at_5"] == pytest.approx(1.0)
    assert metrics["glyphs"]["set_f1"] == pytest.approx((1.0 + 2 / 3 + 2 / 3) / 3)
    assert 0.0 <= metrics["calibration"]["ece_10"] <= 1.0
    assert 0.0 <= metrics["calibration"]["multilabel_brier"] <= 1.0
    assert set(metrics["per_planche"]) == {"a", "b"}


def _gate_metrics(ap: float):
    return {"glyphs": {"sample_ap": ap}}


def test_sanity_gate_requires_plus_one_glyph_ap_positive_ci_and_latency_but_never_element_gate():
    baseline = _gate_metrics(0.50)
    candidate = _gate_metrics(0.52)
    outcome = fusion.final_gate(
        "cascade",
        candidate,
        baseline,
        {"mean": 0.02, "lower_95": 0.005, "upper_95": 0.04},
        latency_ms_per_glyph=120.0,
        candidate_b1_only={"sample_ap": 0.7, "recall_at_5": 0.8, "top1_hit_rate": 0.6},
        baseline_b1_only={"sample_ap": 0.7, "recall_at_5": 0.8, "top1_hit_rate": 0.6},
        b1_only_logits_identical=True,
    )
    assert outcome["sanity_pass"] is True
    assert all(outcome["passes"].values())
    assert outcome["verdict"] == "INSUFFICIENT_FOR_ELEMENT_GATE"
    assert outcome["promotion_eligible"] is False
    assert outcome["automatic_global_training_allowed"] is False


@pytest.mark.parametrize(
    ("mean", "lower", "latency", "failed_gate"),
    [
        (0.009, 0.005, 120.0, "mean_planche_sample_ap_gain_at_least_0_01"),
        (0.02, 0.0, 120.0, "paired_planche_ci_lower_positive"),
        (0.02, 0.005, 251.0, "latency_within_250ms_per_glyph"),
    ],
)
def test_final_gate_fails_each_preregistered_condition(
    mean, lower, latency, failed_gate
):
    outcome = fusion.final_gate(
        "prior_global",
        _gate_metrics(0.52),
        _gate_metrics(0.50),
        {"mean": mean, "lower_95": lower, "upper_95": 0.04},
        latency_ms_per_glyph=latency,
        candidate_b1_only={"sample_ap": 1.0, "recall_at_5": 1.0, "top1_hit_rate": 1.0},
        baseline_b1_only={"sample_ap": 1.0, "recall_at_5": 1.0, "top1_hit_rate": 1.0},
        b1_only_logits_identical=True,
    )
    assert outcome["sanity_pass"] is False
    assert outcome["passes"][failed_gate] is False
    assert outcome["verdict"] == "INSUFFICIENT_FOR_ELEMENT_GATE"


def test_gate_uses_mean_planche_estimand_not_glyph_weighted_sample_ap():
    outcome = fusion.final_gate(
        "cascade",
        _gate_metrics(0.70),
        _gate_metrics(0.50),
        {"mean": 0.005, "lower_95": 0.001, "upper_95": 0.02},
        latency_ms_per_glyph=10.0,
        candidate_b1_only={"sample_ap": 1.0, "recall_at_5": 1.0, "top1_hit_rate": 1.0},
        baseline_b1_only={"sample_ap": 1.0, "recall_at_5": 1.0, "top1_hit_rate": 1.0},
        b1_only_logits_identical=True,
    )
    assert outcome["glyph_weighted_sample_ap_gain_descriptive"] == pytest.approx(0.20)
    assert outcome["mean_planche_sample_ap_gain"] == pytest.approx(0.005)
    assert outcome["sanity_pass"] is False


def test_b1_only_ranking_metrics_and_gate_detect_any_candidate_degradation():
    taxonomy = _taxonomy()
    b1 = np.zeros((2, 286), dtype=np.float64)
    b1[0, 31] = 3.0
    b1[1, 32] = 3.0
    candidate = b1.copy()
    candidate[0, 31] = -3.0
    baseline_metrics = fusion.evaluate_b1_only_ranking(
        b1, [[31], [32]], taxonomy.b1_only_indices
    )
    candidate_metrics = fusion.evaluate_b1_only_ranking(
        candidate, [[31], [32]], taxonomy.b1_only_indices
    )
    outcome = fusion.final_gate(
        "prior_global",
        _gate_metrics(0.52),
        _gate_metrics(0.50),
        {"mean": 0.02, "lower_95": 0.01, "upper_95": 0.03},
        latency_ms_per_glyph=10.0,
        candidate_b1_only=candidate_metrics,
        baseline_b1_only=baseline_metrics,
        b1_only_logits_identical=False,
    )
    assert outcome["passes"]["b1_only_logits_bit_identical"] is False
    assert any(
        outcome["passes"][name] is False
        for name in (
            "b1_only_sample_ap_no_degradation",
            "b1_only_recall_at_5_no_degradation",
            "b1_only_top1_hit_rate_no_degradation",
        )
    )
    assert outcome["sanity_pass"] is False


def test_b1_only_global_ranking_gate_rejects_boosted_shared_distractor():
    taxonomy = _taxonomy()
    protected_truth = taxonomy.b1_only_indices[0]
    shared_distractor = taxonomy.shared_runtime_indices[0]
    raw_b1 = np.full((1, 286), -5.0, dtype=np.float64)
    raw_b1[0, protected_truth] = 4.0
    candidate = raw_b1.copy()
    candidate[0, shared_distractor] = 5.0

    baseline_metrics = fusion.evaluate_b1_only_ranking(
        raw_b1, [[protected_truth]], taxonomy.b1_only_indices
    )
    candidate_metrics = fusion.evaluate_b1_only_ranking(
        candidate, [[protected_truth]], taxonomy.b1_only_indices
    )
    assert baseline_metrics["ranking_universe"] == 286
    assert baseline_metrics["top1_hit_rate"] == 1.0
    assert candidate_metrics["sample_ap"] == pytest.approx(0.5)
    assert candidate_metrics["top1_hit_rate"] == 0.0

    outcome = fusion.final_gate(
        "cascade",
        _gate_metrics(0.52),
        _gate_metrics(0.50),
        {"mean": 0.02, "lower_95": 0.01, "upper_95": 0.03},
        latency_ms_per_glyph=10.0,
        candidate_b1_only=candidate_metrics,
        baseline_b1_only=baseline_metrics,
        b1_only_logits_identical=True,
    )
    assert outcome["passes"]["b1_only_logits_bit_identical"] is True
    assert outcome["passes"]["b1_only_sample_ap_no_degradation"] is False
    assert outcome["passes"]["b1_only_top1_hit_rate_no_degradation"] is False
    assert outcome["sanity_pass"] is False


def test_paired_planche_bootstrap_is_deterministic_and_positive_for_uniform_gain():
    baseline = {f"page-{i}": 0.40 + i * 0.01 for i in range(6)}
    candidate = {page: value + 0.02 for page, value in baseline.items()}
    first = fusion.paired_planche_bootstrap(candidate, baseline, iterations=500, seed=19)
    second = fusion.paired_planche_bootstrap(candidate, baseline, iterations=500, seed=19)
    assert first == second
    assert first["lower_95"] == pytest.approx(0.02)
    assert first["upper_95"] == pytest.approx(0.02)


def test_write_once_json_accepts_idempotent_replay_and_refuses_change(tmp_path):
    path = tmp_path / "artifact.json"
    fusion.write_once_json(path, {"value": 1})
    fusion.write_once_json(path, {"value": 1})
    with pytest.raises(RuntimeError, match="immutable artifact"):
        fusion.write_once_json(path, {"value": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1}


def test_prepared_cache_validation_requires_all_real_model_outputs():
    cache = {
        "schema_version": fusion.CACHE_SCHEMA,
        "scoring_profile": fusion.DEFAULT_SCORING_PROFILE,
        "b1_source_semantics": (
            "CodexClassifier.classify_batch raw cosine scores on whole glyph JPEG; "
            "exact runtime preprocessing and backbone included"
        ),
        "row_ids": ["a", "b"],
        "image_paths": ["a.jpg", "b.jpg"],
        "page_groups": ["p1", "p2"],
        "splits": ["train", "dev"],
        "truth_labels": [[0], [1]],
        "b1_glyph_scores": np.zeros((2, 286), dtype=np.float32),
        "legacy_logits": np.zeros((2, 33), dtype=np.float32),
        "segment_counts": [0, 0],
        "fallback_modes": [fusion.B1_DIRECT_MODE, fusion.B1_DIRECT_MODE],
        "latency": {
            "b1_ms_per_glyph": 1.0,
            "legacy_ms_per_glyph": 1.0,
            "serial_fusion_ms_per_glyph": 2.0,
            "whole_glyph_b1_fallback_count": 0,
            "whole_glyph_b1_fallback_rate": 0.0,
            "whole_glyph_b1_fallback_classify_ms_total": 0.0,
            "whole_glyph_b1_fallback_classify_ms_mean": 0.0,
            "b1_direct_batch_size": fusion.DIRECT_B1_BATCH_SIZE,
            "b1_direct_batches": 1,
        },
        "pixel_sha256": ["x", "y"],
        "protocol_sha256": fusion.semantic_hash(fusion.frozen_protocol()),
        "promotion_eligible": False,
        "max_segments_per_glyph": 0,
    }
    fusion.validate_prepared_cache(cache)
    del cache["legacy_logits"]
    with pytest.raises(ValueError, match="missing fields"):
        fusion.validate_prepared_cache(cache)


def _bound_rows():
    return [
        {
            "row_id": "glyph-train",
            "pixel_sha256": "a" * 64,
            "page_group": "page-train",
            "split": "train",
            "truth_labels": [1, 2],
        },
        {
            "row_id": "glyph-dev",
            "pixel_sha256": "b" * 64,
            "page_group": "page-dev",
            "split": "dev",
            "truth_labels": [3],
        },
    ]


def test_cache_manifest_binds_all_models_p4_and_benchmark_identity(tmp_path):
    cache_path = tmp_path / "cache.npz"
    cache_path.write_bytes(b"safe-npz-placeholder")
    rows = _bound_rows()
    manifest = fusion.build_prepared_manifest(
        cache_path,
        rows,
        fusion.expected_cache_input_binding(),
    )
    assert set(manifest["input_binding"]) >= {
        "run50_checkpoint_sha256",
        "b1_projection_sha256",
        "b1_prototypes_sha256",
        "p4_plan_file_sha256",
        "p4_manifest_file_sha256",
        "p4_plan_payload_sha256",
        "p4_manifest_payload_sha256",
    }
    assert set(manifest["benchmark_binding"]) >= {
        "row_ids_sha256",
        "pixel_sha256_sha256",
        "splits_sha256",
    }
    assert manifest["inference_binding"]["max_segments_per_glyph"] is None
    assert manifest["inference_binding"]["direct_b1_batch_size"] == 16
    fusion.validate_prepared_manifest_before_tensor_load(
        manifest,
        cache_path,
        rows,
        fusion.expected_cache_input_binding(),
    )


def test_safe_npz_cache_round_trip_preserves_raw_b1_scores(tmp_path):
    rows = _bound_rows()
    cache_path = tmp_path / "cache.npz"
    manifest_path = tmp_path / "manifest.json"
    raw_b1 = np.arange(2 * 286, dtype=np.float32).reshape(2, 286)
    cache = {
        "schema_version": fusion.CACHE_SCHEMA,
        "scoring_profile": fusion.DEFAULT_SCORING_PROFILE,
        "b1_source_semantics": (
            "CodexClassifier.classify_batch raw cosine scores on whole glyph JPEG; "
            "exact runtime preprocessing and backbone included"
        ),
        "protocol_sha256": fusion.semantic_hash(fusion.frozen_protocol()),
        "row_ids": [row["row_id"] for row in rows],
        "image_paths": ["train.jpg", "dev.jpg"],
        "pixel_sha256": [row["pixel_sha256"] for row in rows],
        "page_groups": [row["page_group"] for row in rows],
        "splits": [row["split"] for row in rows],
        "truth_labels": [row["truth_labels"] for row in rows],
        "b1_glyph_scores": raw_b1,
        "legacy_logits": np.zeros((2, 33), dtype=np.float32),
        "segment_counts": [0, 0],
        "fallback_modes": [fusion.B1_DIRECT_MODE, fusion.B1_DIRECT_MODE],
        "latency": {
            "b1_ms_per_glyph": 1.0,
            "legacy_ms_per_glyph": 2.0,
            "serial_fusion_ms_per_glyph": 3.0,
            "whole_glyph_b1_fallback_count": 0,
            "whole_glyph_b1_fallback_rate": 0.0,
            "whole_glyph_b1_fallback_classify_ms_total": 0.0,
            "whole_glyph_b1_fallback_classify_ms_mean": 0.0,
            "b1_direct_batch_size": fusion.DIRECT_B1_BATCH_SIZE,
            "b1_direct_batches": 1,
        },
        "promotion_eligible": False,
        "max_segments_per_glyph": 0,
    }
    fusion._write_npz_cache(cache_path, cache)
    manifest = fusion.build_prepared_manifest(
        cache_path, rows, fusion.expected_cache_input_binding()
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = fusion.load_prepared_cache(
        rows,
        fusion.expected_cache_input_binding(),
        cache_path=cache_path,
        manifest_path=manifest_path,
    )
    np.testing.assert_array_equal(loaded["b1_glyph_scores"], raw_b1)
    assert loaded["b1_glyph_scores"].dtype == raw_b1.dtype


@pytest.mark.parametrize("tamper", ["cache", "row_ids", "pixels", "split", "input"])
def test_cache_tampering_is_rejected_before_any_tensor_load(tmp_path, monkeypatch, tamper):
    cache_path = tmp_path / "cache.npz"
    manifest_path = tmp_path / "manifest.json"
    cache_path.write_bytes(b"safe-npz-placeholder")
    rows = _bound_rows()
    observed = fusion.expected_cache_input_binding()
    manifest = fusion.build_prepared_manifest(cache_path, rows, observed)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    expected_rows = [dict(row) for row in rows]
    current_inputs = dict(observed)
    if tamper == "cache":
        cache_path.write_bytes(b"tampered")
    elif tamper == "row_ids":
        expected_rows[0]["row_id"] = "tampered-id"
    elif tamper == "pixels":
        expected_rows[0]["pixel_sha256"] = "c" * 64
    elif tamper == "split":
        expected_rows[0]["split"] = "dev"
    else:
        current_inputs["b1_projection_sha256"] = "0" * 64

    loaded = False

    def forbidden_loader(_path):
        nonlocal loaded
        loaded = True
        raise AssertionError("tensor loader must not run after binding failure")

    monkeypatch.setattr(fusion, "_load_npz_cache", forbidden_loader)
    with pytest.raises(ValueError, match="cache|benchmark|input"):
        fusion.load_prepared_cache(
            expected_rows,
            current_inputs,
            cache_path=cache_path,
            manifest_path=manifest_path,
        )
    assert loaded is False


def test_manifest_rejects_changed_mobile_sam_segment_budget(tmp_path):
    cache_path = tmp_path / "cache.npz"
    cache_path.write_bytes(b"safe-npz-placeholder")
    rows = _bound_rows()
    profile = fusion.PROFILE_MOBILE_SAM_B1
    observed = fusion.expected_cache_input_binding_for_profile(profile)
    manifest = fusion.build_prepared_manifest(
        cache_path, rows, observed, profile=profile
    )
    manifest["inference_binding"]["max_segments_per_glyph"] = 31
    with pytest.raises(ValueError, match="inference binding"):
        fusion.validate_prepared_manifest_before_tensor_load(
            manifest, cache_path, rows, observed, profile
        )


def test_inference_source_hashes_and_import_semantics_are_pinned():
    binding = fusion.expected_cache_input_binding_for_profile(
        fusion.PROFILE_MOBILE_SAM_B1
    )
    assert binding["b1_classifier_source_sha256"] == fusion.sha256_file(
        fusion.B1_CLASSIFIER_SOURCE
    )
    assert binding["mobile_sam_source_sha256"] == fusion.sha256_file(
        fusion.MOBILE_SAM_SOURCE
    )
    assert fusion.frozen_protocol(fusion.PROFILE_MOBILE_SAM_B1)[
        "inference_imports"
    ] == {
        "classifier": "from codex_model import CodexClassifier (codex_model 1.0.0)",
        "segmenter": "from codex_pipeline.segmentation import MobileSAMSegmenter",
    }


def test_mobile_sam_checkpoint_preflight_rejects_corrupt_file(tmp_path):
    corrupt = tmp_path / "mobile_sam.pt"
    corrupt.write_bytes(b"PK\x03\x04truncated-torch-archive")
    with pytest.raises(ValueError, match="not loadable.*weights_only"):
        fusion.validate_mobile_sam_checkpoint(
            corrupt,
            expected_size=corrupt.stat().st_size,
            expected_sha256=fusion.sha256_file(corrupt),
            expected_keys=439,
            expected_key_inventory_sha256="unused-after-load-failure",
        )


def test_mobile_sam_checkpoint_preflight_fails_closed_on_size_before_load(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "mobile_sam.pt"
    checkpoint.write_bytes(b"invalid")
    loaded = False

    def forbidden_torch():
        nonlocal loaded
        loaded = True
        raise AssertionError("must reject size before torch.load")

    monkeypatch.setattr(fusion, "_torch", forbidden_torch)
    with pytest.raises(ValueError, match="size mismatch"):
        fusion.validate_mobile_sam_checkpoint(
            checkpoint,
            expected_size=999,
            expected_sha256="0" * 64,
            expected_keys=439,
            expected_key_inventory_sha256="0" * 64,
        )
    assert loaded is False


def test_zero_mobile_sam_crops_falls_back_once_to_whole_glyph_b1(tmp_path):
    from PIL import Image

    image_path = tmp_path / "glyph.jpg"
    Image.new("RGB", (12, 8), (20, 40, 60)).save(image_path)

    class EmptySegmenter:
        def segment_page(self, image):
            assert image.shape == (8, 12, 3)
            return []

        def extract_crops(self, image, proposals):
            assert proposals == []
            return []

    class RecordingClassifier:
        class_names = [f"class-{index}" for index in range(286)]

        def __init__(self):
            self.batch_sizes = []

        def classify_batch(self, images, top_k):
            self.batch_sizes.append(len(images))
            assert top_k == 286
            assert len(images) == 1
            assert images[0].shape == (8, 12, 3)
            top = [
                {"class_name": name, "confidence": index / 1000.0}
                for index, name in enumerate(self.class_names)
            ]
            return [{"top_k": list(reversed(top))}]

    classifier = RecordingClassifier()
    scores, segment_counts, fallback_modes, latency = (
        fusion.score_b1_glyph_aggregate(
            [{"row_id": "glyph", "image_path": str(image_path)}],
            classifier,
            EmptySegmenter(),
            fusion.MAX_SEGMENTS_PER_GLYPH,
        )
    )
    assert classifier.batch_sizes == [1]
    assert segment_counts == [0]
    assert fallback_modes == [fusion.B1_FALLBACK_MODE]
    assert latency["whole_glyph_b1_fallback_count"] == 1
    assert latency["whole_glyph_b1_fallback_classify_ms_total"] >= 0.0
    expected = (np.arange(286) / 1000.0 - fusion.B1_SIMILARITY_THRESHOLD) / (
        fusion.B1_CALIBRATION_TEMPERATURE
    )
    np.testing.assert_allclose(scores[0], expected, rtol=1e-6)


def test_fallback_manifest_tamper_is_rejected_before_cache_load(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.npz"
    manifest_path = tmp_path / "manifest.json"
    cache_path.write_bytes(b"safe-npz-placeholder")
    rows = _bound_rows()
    observed = fusion.expected_cache_input_binding()
    manifest = fusion.build_prepared_manifest(
        cache_path,
        rows,
        observed,
        fallback_modes=[fusion.B1_FALLBACK_MODE, fusion.B1_PRIMARY_MODE],
    )
    manifest["fallback_binding"]["fallback_count"] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = False

    def forbidden_loader(_path):
        nonlocal loaded
        loaded = True
        raise AssertionError("fallback tamper must fail before cache load")

    monkeypatch.setattr(fusion, "_load_npz_cache", forbidden_loader)
    with pytest.raises(ValueError, match="fallback binding"):
        fusion.load_prepared_cache(
            rows,
            observed,
            cache_path=cache_path,
            manifest_path=manifest_path,
        )
    assert loaded is False


def test_direct_b1_is_default_and_cached_profile_is_diagnostic_only():
    assert fusion.DEFAULT_SCORING_PROFILE == fusion.PROFILE_WHOLE_GLYPH_DIRECT_B1
    direct = fusion.frozen_protocol(fusion.PROFILE_WHOLE_GLYPH_DIRECT_B1)
    cached = fusion.frozen_protocol(fusion.PROFILE_WHOLE_GLYPH_CACHED_B1)
    mobile = fusion.frozen_protocol(fusion.PROFILE_MOBILE_SAM_B1)
    assert direct["schema_version"] == "autoresearch-legacy-fusion-protocol.v3"
    assert direct["scoring_profile"] == "whole_glyph_direct_b1"
    assert direct["b1_glyph_aggregate"]["runtime_preprocess_exact"] is True
    assert direct["b1_glyph_aggregate"]["direct_batch_size"] == 16
    assert direct["baseline_contract"]["production_runtime_equivalent"] is True
    assert cached["scoring_profile"] == "whole_glyph_cached_b1"
    assert cached["b1_glyph_aggregate"]["feature_cache"]["view"] == "base"
    assert cached["b1_glyph_aggregate"]["backbone_extraction_in_timed_run"] is False
    assert cached["b1_glyph_aggregate"]["runtime_preprocess_exact"] is False
    assert cached["baseline_contract"]["production_runtime_equivalent"] is False
    assert cached["baseline_contract"]["status"] == "research_diagnostic_only"
    assert mobile["scoring_profile"] == "mobile_sam_b1"
    assert mobile["b1_glyph_aggregate"]["feature_cache"] is None
    assert fusion.semantic_hash(direct) != fusion.semantic_hash(cached)
    assert fusion.semantic_hash(cached) != fusion.semantic_hash(mobile)
    direct_binding = fusion.expected_cache_input_binding_for_profile(
        fusion.PROFILE_WHOLE_GLYPH_DIRECT_B1
    )
    cached_binding = fusion.expected_cache_input_binding_for_profile(
        fusion.PROFILE_WHOLE_GLYPH_CACHED_B1
    )
    mobile_binding = fusion.expected_cache_input_binding_for_profile(
        fusion.PROFILE_MOBILE_SAM_B1
    )
    assert "p4_feature_cache_sha256" in cached_binding
    assert "mobile_sam_checkpoint_sha256" not in cached_binding
    assert "mobile_sam_checkpoint_sha256" in mobile_binding
    assert "p4_feature_cache_sha256" not in mobile_binding
    assert "mobile_sam_checkpoint_sha256" not in direct_binding
    assert "p4_feature_cache_sha256" not in direct_binding


def test_direct_b1_scores_whole_jpegs_with_frozen_runtime_batches(tmp_path):
    from PIL import Image

    class RecordingClassifier:
        class_names = [f"class-{index}" for index in range(286)]

        def __init__(self):
            self.batch_sizes = []

        def classify_batch(self, images, top_k):
            self.batch_sizes.append(len(images))
            assert top_k == 286
            assert all(image.size == (9, 5) for image in images)
            ranked = [
                {
                    "class_name": name,
                    "confidence": index / 1000.0,
                }
                for index, name in enumerate(self.class_names)
            ]
            return [{"top_k": list(reversed(ranked))} for _ in images]

    image_path = tmp_path / "glyph.jpg"
    Image.new("RGB", (9, 5), (10, 20, 30)).save(image_path, format="JPEG")
    rows = [
        {"row_id": f"glyph-{index}", "image_path": str(image_path)}
        for index in range(fusion.DIRECT_B1_BATCH_SIZE + 1)
    ]
    classifier = RecordingClassifier()

    scores, latency = fusion.score_direct_b1_whole_glyph(rows, classifier)

    assert classifier.batch_sizes == [fusion.DIRECT_B1_BATCH_SIZE, 1]
    expected = np.arange(286, dtype=np.float32) / 1000.0
    np.testing.assert_array_equal(
        scores,
        np.repeat(expected[None, :], len(rows), axis=0),
    )
    assert latency["b1_direct_batch_size"] == fusion.DIRECT_B1_BATCH_SIZE
    assert latency["b1_direct_batches"] == 2
    assert latency["b1_ms_per_glyph"] >= 0.0


def test_direct_b1_matches_real_classifier_on_one_p4_jpeg():
    from PIL import Image

    image_path = next(iter(sorted(fusion.P4_IMAGES.rglob("*.jpg"))))
    classifier = fusion._load_b1_classifier("cpu")
    with Image.open(image_path) as image:
        direct_result = classifier.classify_batch([image.copy()], top_k=286)[0]
    expected_by_name = {
        str(item["class_name"]): np.float32(item["confidence"])
        for item in direct_result["top_k"]
    }
    expected = np.asarray(
        [expected_by_name[name] for name in classifier.class_names],
        dtype=np.float32,
    )

    observed, latency = fusion.score_direct_b1_whole_glyph(
        [{"row_id": "real-p4", "image_path": str(image_path)}], classifier
    )

    np.testing.assert_array_equal(observed[0], expected)
    assert latency["b1_direct_batches"] == 1


def test_manifest_binds_direct_runtime_profile_and_batch_size(tmp_path):
    cache_path = tmp_path / "cache.npz"
    cache_path.write_bytes(b"safe-npz-placeholder")
    rows = _bound_rows()
    direct_binding = fusion.expected_cache_input_binding_for_profile(
        fusion.PROFILE_WHOLE_GLYPH_DIRECT_B1
    )
    direct = fusion.build_prepared_manifest(
        cache_path,
        rows,
        direct_binding,
        profile=fusion.PROFILE_WHOLE_GLYPH_DIRECT_B1,
    )
    assert direct["inference_binding"]["direct_b1_batch_size"] == 16
    assert direct["inference_binding"]["production_runtime_equivalent"] is True
    assert direct["inference_binding"]["segmenter_import"] is None
    assert direct["inference_binding"]["p4_feature_cache_sha256"] is None

    cached_binding = fusion.expected_cache_input_binding_for_profile(
        fusion.PROFILE_WHOLE_GLYPH_CACHED_B1
    )
    cached = fusion.build_prepared_manifest(
        cache_path,
        rows,
        cached_binding,
        profile=fusion.PROFILE_WHOLE_GLYPH_CACHED_B1,
    )
    assert cached["inference_binding"]["direct_b1_batch_size"] is None
    assert cached["inference_binding"]["production_runtime_equivalent"] is False


def test_cached_b1_projection_scores_base_features_without_backbone(monkeypatch):
    taxonomy = fusion.load_taxonomy_from_pins()
    features = fusion._torch().zeros((2, 384), dtype=fusion._torch().float32)
    scores, latency = fusion.score_cached_b1_base_features(
        features, taxonomy, "cpu"
    )
    assert scores.shape == (2, 286)
    assert np.isfinite(scores).all()
    assert latency["b1_backbone_extraction_included"] is False
    assert latency["b1_cached_forward_ms_per_glyph"] >= 0.0


def test_cached_feature_provenance_tamper_rejected_before_tensor_load(
    tmp_path, monkeypatch
):
    cache_path = tmp_path / "multiview.features.pt"
    provenance_path = tmp_path / "multiview.features.pt.prov.json"
    cache_path.write_bytes(b"not-loaded")
    provenance_path.write_text("{}", encoding="utf-8")
    monkeypatch.setitem(
        fusion.EXPECTED_FILE_SHA256, cache_path, fusion.sha256_file(cache_path)
    )
    monkeypatch.setitem(
        fusion.EXPECTED_FILE_SHA256,
        provenance_path,
        fusion.sha256_file(provenance_path),
    )
    loaded = False

    class ForbiddenTorch:
        def load(self, *args, **kwargs):
            nonlocal loaded
            loaded = True
            raise AssertionError("provenance tamper must fail before torch.load")

    monkeypatch.setattr(fusion, "_torch", lambda: ForbiddenTorch())
    with pytest.raises(ValueError, match="provenance binding changed"):
        fusion.load_pinned_base_features(
            _bound_rows(), cache_path=cache_path, provenance_path=provenance_path
        )
    assert loaded is False
