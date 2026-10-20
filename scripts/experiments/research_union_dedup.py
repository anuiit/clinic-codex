"""Fixed inference-only cross-source deduplication on saved SAM/YOLO proposals."""
import argparse, collections, math
from pathlib import Path
from scripts.experiments import research_union_confusions as u
from scripts.experiments import research_full_proposals as g

ROOT = u.ROOT
RUN = ROOT / "backend/training_runs/20261005-union-dedup-v1"
INPUT = u.RUN / "union-results.json"
POLICIES = {
    "consensus": dict(iou=.70, same_name=True),
    "spatial": dict(iou=.85, same_name=False),
}
CUT = .35


def validate(rows):
    ids = [r["proposal_index"] for r in rows]
    assert len(ids) == len(set(ids))
    for r in rows:
        assert r["source"] in {"sam", "yolo"}
        assert len(r["bbox"]) == 4 and all(math.isfinite(x) for x in r["bbox"])
        assert r["bbox"][2] > 0 and r["bbox"][3] > 0
        assert 0 <= r["prediction"] < 286 and math.isfinite(r["similarity"])


def fuse(rows, policy):
    """Retain SAM, remove only YOLO proposals covered by a retained SAM box."""
    validate(rows)
    if policy == "union":
        return list(rows), []
    rule = POLICIES[policy]
    sam = [r for r in rows if r["source"] == "sam" and r["similarity"] >= CUT]
    kept, suppressed = [], []
    # ponytail: quadratic scan of about 100 proposals/page; index if page sizes grow.
    for r in rows:
        candidates = []
        if r["source"] == "yolo" and r["similarity"] >= CUT:
            for s in sam:
                if rule["same_name"] and r["prediction"] != s["prediction"]:
                    continue
                overlap = g.box_iou(g.box(r["bbox"])["box"], g.box(s["bbox"])["box"])
                if overlap >= rule["iou"]:
                    candidates.append((overlap, -s["proposal_index"], s))
        if candidates:
            overlap, _, s = max(candidates, key=lambda x: x[:2])
            suppressed.append(dict(proposal_index=r["proposal_index"],
                keeper_index=s["proposal_index"], iou=overlap,
                names_agree=r["prediction"] == s["prediction"]))
        else:
            kept.append(r)
    assert {r["proposal_index"] for r in kept if r["source"] == "sam"} == {
        r["proposal_index"] for r in rows if r["source"] == "sam"}
    assert len(kept) + len(suppressed) == len(rows)
    return kept, suppressed


def delta(before, after):
    b = {m["gt_index"] for m in before["matches"] if m["correct"]}
    a = {m["gt_index"] for m in after["matches"] if m["correct"]}
    bm = {m["gt_index"] for m in before["matches"]}
    am = {m["gt_index"] for m in after["matches"]}
    return dict(gained_correct=sorted(a-b), lost_correct=sorted(b-a),
        gained_matches=sorted(am-bm), lost_matches=sorted(bm-am))


def check():
    def row(i, source, bbox, label=1):
        return dict(proposal_index=i, source=source, bbox=bbox,
                    prediction=label, similarity=.8)
    rows = [row(7, "sam", [0,0,100,100]), row(8, "yolo", [0,0,100,100]),
            row(9, "yolo", [10,10,10,10]), row(10, "yolo", [200,0,100,100]),
            row(11, "yolo", [0,0,100,100], 2)]
    kept, removed = fuse(rows, "consensus")
    assert [r["proposal_index"] for r in kept] == [7,9,10,11]
    assert removed == [dict(proposal_index=8, keeper_index=7, iou=1., names_agree=True)]
    assert [r["proposal_index"] for r in fuse(rows, "spatial")[0]] == [7,9,10]
    low = [rows[0], dict(rows[1], similarity=.34)]
    assert fuse(low, "consensus")[1] == []
    low = [dict(rows[0], similarity=.34), rows[1]]
    assert fuse(low, "consensus")[1] == []
    assert fuse(rows, "union") == (rows, [])
    ties = [rows[0], row(3, "sam", [0,0,100,100]), rows[1]]
    assert fuse(ties, "consensus")[1][0]["keeper_index"] == 3
    gt = [dict(bbox=[0,0,100,100], label=1, natural_support=5)]
    before = u.safe_score(gt, [], rows[:2], CUT)
    after = u.safe_score(gt, [], fuse(rows[:2], "consensus")[0], CUT)
    assert before["counts"]["correct"] == after["counts"]["correct"] == 1
    assert before["counts"]["duplicate_predictions"] == 1
    assert after["counts"]["duplicate_predictions"] == 0
    try:
        fuse([dict(rows[0], bbox=[0,0,0,1])], "consensus")
    except AssertionError:
        pass
    else:
        raise AssertionError("Degenerate box accepted")
    return dict(sam_retained=True, nested_small_box_retained=True,
        different_names_retained_by_consensus=True, unmatched_yolo_retained=True,
        low_similarity_not_used_for_suppression=True, deterministic_ties=True,
        known_duplicate_removed_without_name_loss=True, invalid_box_rejected=True)


def main():
    checks = check()
    saved = u.read(INPUT)
    u.verify(saved["protocol"])
    paths = [INPUT, Path(__file__), Path(u.__file__), Path(g.__file__),
             ROOT / "scripts/evaluate_glyph_benchmark.py"] + u.active()
    protocol = dict(schema_version="union-dedup.v1", input_sha256=u.pin(paths),
        policies=POLICIES, primary="consensus", semantic_cut=CUT,
        retention="Every SAM box retained. YOLO removed only if an accepted SAM meets fixed rule. No same-source suppression. No coordinate/name averaging.",
        thresholds="Prespecified rules, no label-based calibration, no sweep and no held-out winner selection. Detector confidence cuts unchanged from saved run.",
        pairing="Same nested recognition heads as union comparator, seed42; raw=SAM mask+YOLO rectangle, mask=both masked",
        matching="Existing label-independent one-to-one greedy IoU>=.5, rematch each method, keep original proposal IDs/order",
        precision_F1_interpretable=False,
        scope="Same17 previously examined train pages343 known annotations; partial export; retrospective exploratory, no blind validation",
        optimizer_updates=0, activated=False, locked_test_scored=False, checks=checks)
    RUN.mkdir(parents=True, exist_ok=True)
    prereg = RUN / "preregistration.json"
    if prereg.exists():
        assert u.read(prereg) == protocol
    else:
        u.write(prereg, protocol)
    pages = []
    for page in saved["pages"]:
        views = {}
        for view in ["raw", "mask"]:
            methods = {}
            for method in ["union", *POLICIES]:
                rows, removed = fuse(page["proposals"][view], method)
                score = u.safe_score(page["gt"], page["ignore"], rows, CUT)
                methods[method] = dict(score=score,
                    kept_proposal_indices=[r["proposal_index"] for r in rows], suppressed=removed,
                    delta_union=delta(page["naming"][view]["union"], score),
                    delta_sam=delta(page["naming"][view]["sam"], score))
            assert methods["union"]["score"]["counts"] == page["naming"][view]["union"]["counts"]
            assert methods["union"]["score"]["matches"] == page["naming"][view]["union"]["matches"]
            views[view] = methods
        pages.append(dict(group=page["group"], fold=page["fold"], views=views))
    assert len(pages) == 17 and len({p["group"] for p in pages}) == 17
    summary = {}
    for view in ["raw", "mask"]:
        summary[view] = {}
        for method in ["union", *POLICIES]:
            records = [p["views"][view][method] for p in pages]
            result = u.aggregate([r["score"] for r in records])
            result["suppressed"] = sum(len(r["suppressed"]) for r in records)
            for reference in ["union", "sam"]:
                result["delta_"+reference] = {
                    k: sum(len(r["delta_"+reference][k]) for r in records)
                    for k in records[0]["delta_"+reference]}
                base = saved["naming"][view][reference]["counts"]
                d = result["delta_"+reference]
                assert d["gained_correct"]-d["lost_correct"] == result["counts"]["correct"]-base["correct"]
                assert d["gained_matches"]-d["lost_matches"] == result["counts"]["matched"]-base["matched"]
            assert result["counts"]["generated"]+result["suppressed"] == 1604
            summary[view][method] = result
    u.verify(protocol)
    u.verify(saved["protocol"])
    result = dict(protocol=protocol, summary=summary, pages=pages,
        optimizer_updates=0, activated=False, locked_test_scored=False,
        verification=dict(saved_union_reproduced_both_views=True,
            all_sam_proposals_preserved=True, inputs_and_active_weights_unchanged=True,
            thresholds_not_selected_on_scores=True))
    u.write(RUN/"results.json", result)
    print({v: {m: dict(proposals=r["counts"]["generated"], matched=r["counts"]["matched"],
        correct=r["counts"]["correct"], duplicates=r["counts"]["duplicate_predictions"],
        delta_union=r["delta_union"]) for m,r in ms.items()} for v,ms in summary.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(check()) if args.check else main()
