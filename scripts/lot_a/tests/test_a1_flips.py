"""Synthetic-fixture tests for scripts/lot_a/a1_flips.py.

No test in this module reads the real .omc/ caches, the real
prediction_rows.jsonl, or the real backend/codex_model weights — every
fixture (rows and the deployed-runtime directory) is built from scratch
in a tmp_path fixture.
"""

from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from scripts.lot_a import a1_flips, common

SEEDS = (17, 42, 73)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _row(
    row_id: str,
    seed: int,
    label: int,
    b0_topk: list[int],
    c1_topk: list[int],
    candidate_topk: list[int],
    outer_fold: int = 1,
    provenance_component: str = "component:default",
    class_name: str = "cacahuatl",
) -> dict:
    return {
        "row_id": row_id,
        "seed": seed,
        "label": label,
        "b0_topk": b0_topk,
        "c1_topk": c1_topk,
        "candidate_topk": candidate_topk,
        "outer_fold": outer_fold,
        "provenance_component": provenance_component,
        "class_name": class_name,
        "decoded_pixel_sha256": "0" * 64,
    }


CORRECT = [1, 2, 3]  # topk[0] == label(1) -> correct
WRONG = [2, 1, 3]  # topk[0] == 2 != label(1) -> wrong


def _build_synthetic_runtime(tmp_path) -> "object":
    """Minimal valid deployed runtime; a1's runtime guard only needs a
    loadable, hash-stable DeployedRuntime — it never predicts with it."""
    runtime_dir = tmp_path / "runtime"
    weights_dir = runtime_dir / "weights"
    weights_dir.mkdir(parents=True)

    hidden_dim = embedding_dim = 4
    config = {"hidden_dim": hidden_dim, "embedding_dim": embedding_dim, "backbone": "dinov2_vits14"}
    (runtime_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    identity = torch.eye(hidden_dim)
    zero_bias = torch.zeros(hidden_dim)
    projection_state = {
        "net.0.weight": identity.clone(),
        "net.0.bias": zero_bias.clone(),
        "net.3.weight": identity.clone(),
        "net.3.bias": zero_bias.clone(),
    }
    torch.save(projection_state, weights_dir / "projection.pt")

    prototypes = F.normalize(torch.eye(3, hidden_dim), p=2, dim=1)
    proto_data = {
        "prototypes": prototypes,
        "class_names": {5: "p0", 6: "p1", 7: "p2"},
        "class_labels": torch.tensor([5, 6, 7], dtype=torch.int64),
        "embedding_dim": embedding_dim,
    }
    torch.save(proto_data, weights_dir / "prototypes.pt")
    return runtime_dir


def _run(rows: list[dict], runtime_dir, bootstrap_draws: int = 200, bootstrap_seed: int = 1234) -> dict:
    return a1_flips.build_core_report(rows, runtime_dir, bootstrap_seed, bootstrap_draws)


def _rows_for_all_seeds(row_specs: list[dict]) -> list[dict]:
    """Duplicate a list of per-row (row_id, b0/c1/r2 topk) specs across all
    3 expected seeds, identically, so every seed's report is well-formed."""
    rows = []
    for seed in SEEDS:
        for spec in row_specs:
            rows.append(_row(seed=seed, **spec))
    return rows


# ---------------------------------------------------------------------------
# all-flip / no-flip / mixed
# ---------------------------------------------------------------------------


def test_all_flip_every_row_churns_both_transitions(tmp_path):
    runtime_dir = _build_synthetic_runtime(tmp_path)
    specs = [
        dict(row_id="r1", label=1, b0_topk=WRONG, c1_topk=CORRECT, candidate_topk=WRONG),
        dict(row_id="r2", label=1, b0_topk=CORRECT, c1_topk=WRONG, candidate_topk=CORRECT),
    ]
    rows = _rows_for_all_seeds(specs)
    report = _run(rows, runtime_dir)

    for seed_key in ("17", "42", "73"):
        seed_report = report["per_seed"][seed_key]
        assert seed_report["churn_overlap"]["churn_t1_size"] == 2
        assert seed_report["churn_overlap"]["churn_t2_size"] == 2


def test_no_flip_all_arms_agree(tmp_path):
    runtime_dir = _build_synthetic_runtime(tmp_path)
    specs = [
        dict(row_id="r1", label=1, b0_topk=CORRECT, c1_topk=CORRECT, candidate_topk=CORRECT),
        dict(row_id="r2", label=1, b0_topk=WRONG, c1_topk=WRONG, candidate_topk=WRONG),
    ]
    rows = _rows_for_all_seeds(specs)
    report = _run(rows, runtime_dir)

    for seed_key in ("17", "42", "73"):
        seed_report = report["per_seed"][seed_key]
        assert seed_report["churn_overlap"]["churn_t1_size"] == 0
        assert seed_report["churn_overlap"]["churn_t2_size"] == 0
        assert seed_report["churn_overlap"]["p_churn_t1_given_churn_t2"] is None
        assert seed_report["transitions"]["T1_B0_to_C1"]["net"] == 0
        assert seed_report["transitions"]["D_B0_to_R2_direct"]["net"] == 0


def test_mixed_flip_pattern_counts(tmp_path):
    runtime_dir = _build_synthetic_runtime(tmp_path)
    specs = [
        # r1: improves in T1, no change T2 -> churn_T1 only
        dict(row_id="r1", label=1, b0_topk=WRONG, c1_topk=CORRECT, candidate_topk=CORRECT),
        # r2: no change T1, degrades in T2 -> churn_T2 only
        dict(row_id="r2", label=1, b0_topk=CORRECT, c1_topk=CORRECT, candidate_topk=WRONG),
        # r3: improves T1, degrades T2 (both churn, opposite directions)
        dict(row_id="r3", label=1, b0_topk=WRONG, c1_topk=CORRECT, candidate_topk=WRONG),
        # r4: never churns
        dict(row_id="r4", label=1, b0_topk=CORRECT, c1_topk=CORRECT, candidate_topk=CORRECT),
    ]
    rows = _rows_for_all_seeds(specs)
    report = _run(rows, runtime_dir)

    seed_report = report["per_seed"]["17"]
    assert seed_report["churn_overlap"]["churn_t1_size"] == 2  # r1, r3
    assert seed_report["churn_overlap"]["churn_t2_size"] == 2  # r2, r3
    assert seed_report["churn_overlap"]["intersection_size"] == 1  # r3
    assert seed_report["transitions"]["T1_B0_to_C1"]["improvements"] == 2  # r1, r3
    assert seed_report["transitions"]["T2_C1_to_R2"]["degradations"] == 2  # r2, r3


# ---------------------------------------------------------------------------
# B6 fix: strict-subset churn must yield "un phenomene", even though the
# old Jaccard-only rule would have failed it.
# ---------------------------------------------------------------------------


def test_strict_subset_churn_yields_un_phenomene(tmp_path):
    runtime_dir = _build_synthetic_runtime(tmp_path)
    # churn_T2 = {r1, r2} is a strict subset of churn_T1 = {r1, r2, r3, r4, r5}.
    # intersection = 2, |churn_T1| = 5, |churn_T2| = 2, union = 5.
    # P(churn_T1 | churn_T2) = 2/2 = 1.0 (>= 0.5 -> "un phenomene")
    # Jaccard = 2/5 = 0.4 (< 0.5 -> old rule would have said "deux phenomenes")
    specs = [
        dict(row_id="r1", label=1, b0_topk=WRONG, c1_topk=CORRECT, candidate_topk=WRONG),
        dict(row_id="r2", label=1, b0_topk=WRONG, c1_topk=CORRECT, candidate_topk=WRONG),
        dict(row_id="r3", label=1, b0_topk=WRONG, c1_topk=CORRECT, candidate_topk=CORRECT),
        dict(row_id="r4", label=1, b0_topk=WRONG, c1_topk=CORRECT, candidate_topk=CORRECT),
        dict(row_id="r5", label=1, b0_topk=WRONG, c1_topk=CORRECT, candidate_topk=CORRECT),
        dict(row_id="r6", label=1, b0_topk=CORRECT, c1_topk=CORRECT, candidate_topk=CORRECT),
    ]
    rows = _rows_for_all_seeds(specs)
    report = _run(rows, runtime_dir)

    seed_report = report["per_seed"]["17"]
    overlap = seed_report["churn_overlap"]
    assert overlap["churn_t1_size"] == 5
    assert overlap["churn_t2_size"] == 2
    assert overlap["intersection_size"] == 2

    p_t1_given_t2 = overlap["p_churn_t1_given_churn_t2"]
    jaccard = overlap["jaccard_secondary"]
    assert p_t1_given_t2 == 1.0
    assert jaccard == 2 / 5

    # This is the documented B6 fix: the abolished Jaccard-only rule would
    # have failed (< 0.5) while the conditional-probability rule passes.
    assert jaccard < a1_flips.VERDICT_THRESHOLD
    assert p_t1_given_t2 >= a1_flips.VERDICT_THRESHOLD

    assert report["verdict"]["verdict"] == "un phenomene (instabilite partagee)"
    assert report["verdict"]["seeds_meeting_condition"] == 3


def test_two_phenomenes_when_conditional_below_threshold(tmp_path):
    runtime_dir = _build_synthetic_runtime(tmp_path)
    # churn_T1 and churn_T2 barely overlap in either direction.
    specs = [
        dict(row_id="r1", label=1, b0_topk=WRONG, c1_topk=CORRECT, candidate_topk=CORRECT),
        dict(row_id="r2", label=1, b0_topk=WRONG, c1_topk=CORRECT, candidate_topk=CORRECT),
        dict(row_id="r3", label=1, b0_topk=WRONG, c1_topk=CORRECT, candidate_topk=CORRECT),
        dict(row_id="r4", label=1, b0_topk=CORRECT, c1_topk=CORRECT, candidate_topk=WRONG),
        dict(row_id="r5", label=1, b0_topk=CORRECT, c1_topk=CORRECT, candidate_topk=WRONG),
        dict(row_id="r6", label=1, b0_topk=CORRECT, c1_topk=CORRECT, candidate_topk=WRONG),
        # one shared churn row keeps both conditionals well under 0.5
        dict(row_id="r7", label=1, b0_topk=WRONG, c1_topk=CORRECT, candidate_topk=WRONG),
    ]
    rows = _rows_for_all_seeds(specs)
    report = _run(rows, runtime_dir)

    seed_report = report["per_seed"]["17"]
    overlap = seed_report["churn_overlap"]
    assert overlap["p_churn_t1_given_churn_t2"] < 0.5
    assert report["verdict"]["verdict"] == "deux phenomenes distincts"


# ---------------------------------------------------------------------------
# Arithmetic identity net(D) == net(T1) + net(T2)
# ---------------------------------------------------------------------------


def test_arithmetic_identity_net_d_equals_net_t1_plus_net_t2(tmp_path):
    runtime_dir = _build_synthetic_runtime(tmp_path)
    specs = [
        dict(row_id="r1", label=1, b0_topk=WRONG, c1_topk=CORRECT, candidate_topk=WRONG),
        dict(row_id="r2", label=1, b0_topk=CORRECT, c1_topk=WRONG, candidate_topk=CORRECT),
        dict(row_id="r3", label=1, b0_topk=WRONG, c1_topk=WRONG, candidate_topk=CORRECT),
        dict(row_id="r4", label=1, b0_topk=CORRECT, c1_topk=CORRECT, candidate_topk=WRONG),
        dict(row_id="r5", label=1, b0_topk=WRONG, c1_topk=CORRECT, candidate_topk=CORRECT),
    ]
    rows = _rows_for_all_seeds(specs)
    report = _run(rows, runtime_dir)  # would raise RuntimeError if the identity broke

    for seed_key in ("17", "42", "73"):
        transitions = report["per_seed"][seed_key]["transitions"]
        net_t1 = transitions["T1_B0_to_C1"]["net"]
        net_t2 = transitions["T2_C1_to_R2"]["net"]
        net_d = transitions["D_B0_to_R2_direct"]["net"]
        assert net_d == net_t1 + net_t2
        assert report["per_seed"][seed_key]["arithmetic_identity_net_d_eq_t1_plus_t2"] is True


# ---------------------------------------------------------------------------
# Malformed topk exclusion
# ---------------------------------------------------------------------------


def test_malformed_topk_row_excluded(tmp_path):
    runtime_dir = _build_synthetic_runtime(tmp_path)
    good_specs = [
        dict(row_id="r1", label=1, b0_topk=WRONG, c1_topk=CORRECT, candidate_topk=CORRECT),
    ]
    rows = _rows_for_all_seeds(good_specs)
    # Add one malformed row (duplicate index inside c1_topk) for seed 17 only.
    rows.append(
        _row(row_id="rbad", seed=17, label=1, b0_topk=CORRECT, c1_topk=[4, 4, 5], candidate_topk=CORRECT)
    )

    report = _run(rows, runtime_dir)

    assert report["data_hygiene"]["malformed_row_count"] == 1
    assert report["data_hygiene"]["malformed_rows"][0]["row_id"] == "rbad"
    assert report["data_hygiene"]["malformed_rows"][0]["malformed_fields"] == ["c1_topk"]
    # excluded row must not appear in any per-seed transition set
    seed_17 = report["per_seed"]["17"]
    assert seed_17["n_rows"] == 1


# ---------------------------------------------------------------------------
# Conditional probability values against a hand-computed fixture
# ---------------------------------------------------------------------------


def test_conditional_probabilities_match_hand_computation(tmp_path):
    runtime_dir = _build_synthetic_runtime(tmp_path)
    # churn_T1 = {a, b, c} ; churn_T2 = {b, c, d} ; intersection = {b, c}
    # P(T1|T2) = 2/3 ; P(T2|T1) = 2/3 ; Jaccard = 2/4 = 0.5
    specs = [
        dict(row_id="a", label=1, b0_topk=WRONG, c1_topk=CORRECT, candidate_topk=CORRECT),  # T1 only
        dict(row_id="b", label=1, b0_topk=WRONG, c1_topk=CORRECT, candidate_topk=WRONG),  # both
        dict(row_id="c", label=1, b0_topk=WRONG, c1_topk=CORRECT, candidate_topk=WRONG),  # both
        dict(row_id="d", label=1, b0_topk=CORRECT, c1_topk=CORRECT, candidate_topk=WRONG),  # T2 only
        dict(row_id="e", label=1, b0_topk=CORRECT, c1_topk=CORRECT, candidate_topk=CORRECT),  # neither
    ]
    rows = _rows_for_all_seeds(specs)
    report = _run(rows, runtime_dir)

    overlap = report["per_seed"]["17"]["churn_overlap"]
    assert overlap["p_churn_t1_given_churn_t2"] == 2 / 3
    assert overlap["p_churn_t2_given_churn_t1"] == 2 / 3
    assert overlap["jaccard_secondary"] == 0.5


# ---------------------------------------------------------------------------
# Determinism: identical fixture run twice -> identical canonical JSON bytes
# ---------------------------------------------------------------------------


def test_determinism_same_fixture_twice_identical_hash(tmp_path):
    runtime_dir = _build_synthetic_runtime(tmp_path)
    specs = [
        dict(row_id="r1", label=1, b0_topk=WRONG, c1_topk=CORRECT, candidate_topk=WRONG),
        dict(row_id="r2", label=1, b0_topk=CORRECT, c1_topk=WRONG, candidate_topk=CORRECT),
        dict(row_id="r3", label=1, b0_topk=WRONG, c1_topk=CORRECT, candidate_topk=CORRECT),
    ]
    rows = _rows_for_all_seeds(specs)

    report_1 = _run(rows, runtime_dir, bootstrap_draws=100, bootstrap_seed=777)
    report_2 = _run(rows, runtime_dir, bootstrap_draws=100, bootstrap_seed=777)

    assert common.canonical_json_dumps(report_1) == common.canonical_json_dumps(report_2)
    assert common.sha256_json(report_1) == common.sha256_json(report_2)


# ---------------------------------------------------------------------------
# Runtime guard
# ---------------------------------------------------------------------------


def test_runtime_guard_records_sha_and_detects_no_drift(tmp_path):
    runtime_dir = _build_synthetic_runtime(tmp_path)
    specs = [dict(row_id="r1", label=1, b0_topk=CORRECT, c1_topk=CORRECT, candidate_topk=CORRECT)]
    rows = _rows_for_all_seeds(specs)

    report = _run(rows, runtime_dir)

    assert report["runtime_guard"]["verified_unchanged_before_analysis"] is True
    assert report["runtime_guard"]["verified_unchanged_after_analysis"] is True
    assert set(report["runtime_guard"]["sha256_at_load"]) == {"config", "projection", "prototypes"}


# ---------------------------------------------------------------------------
# Markdown rendering smoke test
# ---------------------------------------------------------------------------


def test_render_markdown_contains_verdict_and_definitions(tmp_path):
    runtime_dir = _build_synthetic_runtime(tmp_path)
    specs = [
        dict(row_id="r1", label=1, b0_topk=WRONG, c1_topk=CORRECT, candidate_topk=WRONG),
        dict(row_id="r2", label=1, b0_topk=CORRECT, c1_topk=CORRECT, candidate_topk=CORRECT),
    ]
    rows = _rows_for_all_seeds(specs)
    report = _run(rows, runtime_dir)
    report["reproducibility"] = {"hash_run_1": "aa" * 32, "hash_run_2": "aa" * 32, "matched": True}

    markdown = a1_flips.render_markdown(report)

    assert "Regle de verdict pre-enregistree" in markdown
    assert report["verdict"]["verdict"] in markdown
    assert "Reproductibilite" in markdown
