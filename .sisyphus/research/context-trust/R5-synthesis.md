# R5: Step 2 Implementation Scope

## Executive Summary

R1–R4 converge on a single, consistent picture: **the Codex classifier already computes everything needed for a compelling trust-and-similarity UI; the work is almost entirely in surfacing and visualizing data that currently disappears inside the backend.**

The model is a prototypical network with 286 class prototypes in a 128-d L2-normalized space. Every `/classify` call computes cosine similarities to all 286 prototypes, but only the top-3 are returned. The `top_k` array, `confidence` score, and `rejected` boolean together contain the raw material for five high-value trust signals (top-k breakdown, OOD/rejection, margin, distance-to-prototype, and entropy). None of these require model retraining or new dependencies.

The primary constraint is **missing visual assets**: only 16 of 286 classes have sample images in the repository. The full training dataset (`Elements/`, `metadata.csv`) is not present, so any feature that needs "show me the actual similar training images" is blocked. This rules out instance-level similarity search, per-class prototype grids, hard-negative galleries, and a visual class catalog for the vast majority of classes.

**Recommended scope**: ship a **frontend-first trust panel** using zero-backend-change signals, plus two lightweight backend endpoints (`/similar` and `/trust`) that formalize and enrich the data already flowing through the system. Defer all instance-level image retrieval and full visual galleries until the training dataset and canonical exemplars are available.

---

## What Ships (IN)

| Feature | IN/OUT | Reason | Effort |
|---------|--------|--------|--------|
| **Top-k Breakdown Panel** | **IN** | Zero backend changes. Uses existing `top_k` from `/classify` and `/segment`. Ranked list of competing classes with confidence bars. | S |
| **OOD / Rejection Indicator** | **IN** | Zero backend changes. Uses `rejected` boolean + `confidence`. Traffic-light badge + "Unknown glyph" banner. | S |
| **Margin / Ambiguity Warning** | **IN** | Zero backend changes. Computed from `top_k[0]` and `top_k[1]`. Most actionable ambiguity signal for scholars. | S |
| **Distance-to-Prototype Badge** | **IN** | Zero backend changes. Derived from `confidence` via `arccos` or `sqrt(2-2*confidence)`. Translates abstract score into geometric intuition. | S |
| **Confusion-Class Text List** | **IN** | Zero backend changes. Text-only ranked list of runner-up classes. High scholarly value even without images. | S |
| **POST /similar (prototype mode)** | **IN** | Minor backend change. Reuses existing `classify()` logic but returns a richer, contract-consistent response with similarity bands. Unblocks frontend from parsing `top_k` informally. | S |
| **POST /trust** | **IN** | Minor backend change. Derives all P0+P1 trust signals server-side (margin, rank, threshold check, ambiguity flag). Centralizes trust logic so frontend doesn't recompute. | S |
| **Sample-Image Enrichment (16 classes)** | **IN** | Minor backend change. At startup, embed the ~60 sample images and build an in-memory index. Add `/similar-samples` endpoint so the 16 classes with data can show actual thumbnails. Proof-of-concept for instance similarity without requiring the full dataset. | M |

### Total Effort Estimate

- **Wave 5 (Backend)**: 2 small endpoints + sample index ≈ **1–2 days**
- **Wave 6 (Frontend)**: 4–5 trust panels + confusion list + sample thumbnails ≈ **2–3 days**
- **Combined**: **3–5 days** for a complete Step 2 trust-and-similarity experience

---

## What is Deferred (OUT)

| Feature | IN/OUT | Reason | Unblocking Criteria |
|---------|--------|--------|---------------------|
| **Instance-level image retrieval** (`/similar?mode=instance`) | **OUT** | Requires full training dataset + precomputed 128-d embeddings. No `Elements/` or `metadata.csv` in repo. | Full dataset available; run `precompute_embeddings.py` variant for projected embeddings |
| **Per-class prototype grid** (Pattern 1) | **OUT** | Requires full dataset embeddings to find nearest training images per prototype. | Same as above |
| **Hard-negative examples** (Pattern 2) | **OUT** | Requires full dataset embeddings + cross-class filtering. | Same as above |
| **Visual class catalog** (286 images) | **OUT** | Only 16/286 classes have images. A visual catalog would be 94% empty. | Curate or auto-select one canonical image per class (needs dataset or manual curation) |
| **Calibration curve / reliability diagram** | **OUT** | Needs held-out validation set + offline ECE computation. Not a code problem; a data-pipeline problem. | Validation set assembled; ECE script run offline; endpoint added to serve static calibration data |
| **Gradient-based saliency (Grad-CAM, attention rollout)** | **OUT** | DINOv2 ViT backbone requires ViT-specific attribution hooks. Backend treats backbone as black box. High effort, high risk. | Expose attention/gradient hooks in inference engine; research ViT attribution methods for DINOv2 |
| **FAISS approximate nearest neighbor** | **OUT** | Adds `faiss-cpu`/`faiss-gpu` dependency. User constraint: no pip, no heavy dependencies. For current scale (<50k images), brute-force k-NN is sufficient anyway. | Dataset grows beyond ~100k images; user approves new dependency |
| **Full prototype activation chart (all 286 similarities)** | **OUT** | Computationally trivial (return 286 floats ≈ 1.1 KB), but UI value is unclear until scholars ask "what else did it consider?" Can be added later without breaking changes. | Scholar feedback requests full similarity view; backend adds `all_similarities` field to `/trust` |
| **"This Looks Like That" patch retrieval** | **OUT** | Requires precomputing nearest training-image patch per prototype + offline indexing. Medium effort, blocked by missing dataset. | Full dataset available; offline indexing script run; patch manifest generated |
| **GET /prototypes (with images)** | **OUT** | Requires exemplar manifest. Model prototype tensors are embeddings only; they don't point to renderable images. | Canonical exemplars curated or auto-generated; manifest JSON created and served |
| **Multi-prototype per class (k-means)** | **OUT** | Requires training data access to cluster embeddings per class. Changes model artifact format. | Full dataset available; offline clustering script run; `CodexClassifier` updated to load multi-prototype matrix |

---

## Implementation Plan

### Wave 5: Backend Endpoints

> Goal: formalize the trust/similarity data already computed by the model into stable API contracts.

- [ ] **S2.B1: Add `POST /similar` (prototype mode)** — effort: **S**
  - Decode `image_base64` to RGB array, crop bbox, call `CodexClassifier.classify(crop, top_k=limit)`.
  - Map `top_k` to R4 contract shape: `results[]` with `rank`, `match_type="class_prototype"`, `class_name`, `similarity`, `band`.
  - Bands: `high >= 0.60`, `moderate >= 0.35`, else `low`.
  - Dependencies: none (reuses existing classifier).
  - Blockers: none.

- [ ] **S2.B2: Add `POST /trust`** — effort: **S**
  - Decode image, crop bbox, classify with `top_k=num_classes` (or at least enough to rank `predicted_class`).
  - Return R4 contract: `predicted_class_rank`, `predicted_class_similarity`, `top1_similarity`, `margin_to_second`, `above_rejection_threshold`, `ambiguous` flag.
  - Standardize error envelope (`{"error": {"code": ..., "message": ...}}`) for 400/404/422/500.
  - Dependencies: none.
  - Blockers: verify whether `class_label` bug (R4 Gap #7) affects ranking; if so, fix mapping first.

- [ ] **S2.B3: Build sample-image in-memory index + `POST /similar-samples`** — effort: **M**
  - At backend startup, run `InferenceEngine.embed()` on all images in `data/elements_sample/` and `data/glyphs_sample/`.
  - Build `{class_name: [(path, embedding), ...]}` index.
  - Endpoint: given `image_base64` + `bbox`, embed query crop and return top-k nearest sample images (limit to classes present in sample data).
  - Return `{image_url, class_name, similarity}` tuples.
  - Dependencies: none.
  - Blockers: none.

- [ ] **S2.B4: Standardize preprocessing + bbox validation helper** — effort: **S**
  - Resolve gray `(128)` vs white `(255)` padding discrepancy between `CodexClassifier` and `InferenceEngine` (R4 Gap #8).
  - Add shared `validate_bbox(bbox, image_size)` utility for `/similar` and `/trust`.
  - Dependencies: S2.B1, S2.B2.
  - Blockers: none.

- [ ] **S2.B5: Add entropy to `/trust` (optional, if time permits)** — effort: **S**
  - Compute Shannon entropy over softmax-normalized top-k similarities (or all 286 if backend is adjusted to return them).
  - Add `"uncertainty": {"entropy": ..., "level": "low|moderate|high"}` to `/trust` response.
  - Dependencies: S2.B2.
  - Blockers: none.

### Wave 6: Frontend Panels

> Goal: turn the backend signals into scholar-facing trust UI.

- [ ] **S2.F1: Trust Summary Panel** — effort: **S**
  - Shows: predicted class, confidence score, traffic-light badge (green/amber/red), OOD banner if `rejected`.
  - Uses data from existing `/classify` or new `/trust`.
  - Dependencies: S2.B2 (can use existing `/classify` as fallback).

- [ ] **S2.F2: Top-k Breakdown + Margin Visualization** — effort: **S**
  - Horizontal bar chart of top-k classes with similarity scores.
  - Margin indicator: gap between #1 and #2. Amber warning chip if margin < 0.10.
  - Dependencies: S2.B1 or existing `/classify`.

- [ ] **S2.F3: Confusion-Class List** — effort: **S**
  - Compact panel below each classification result.
  - Ranked text list of runner-up classes with confidence bars (color-coded by band).
  - Optional: click class name to show tooltip/overlay with English gloss if available.
  - Dependencies: none (zero backend).

- [ ] **S2.F4: Distance-to-Prototype Badge** — effort: **S**
  - Small numeric badge: angular distance or Euclidean distance on unit sphere.
  - Tooltip explaining "smaller distance = stronger visual alignment with prototype."
  - Dependencies: none (zero backend).

- [ ] **S2.F5: Sample-Image Thumbnails (for 16 classes)** — effort: **M**
  - When a class in `top_k` is one of the 16 with sample data, show a small thumbnail grid of nearest sample images from `/similar-samples`.
  - Graceful fallback: text-only for the other 270 classes.
  - Dependencies: S2.B3.

- [ ] **S2.F6: Integration / wiring** — effort: **S**
  - Connect new endpoints to frontend state.
  - Add loading states, error handling for malformed images / invalid bboxes.
  - Ensure CORS and error envelopes work end-to-end.
  - Dependencies: all above.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Class-label mapping bug** (R4 Gap #7) corrupts `class_label` in API responses | Medium | Medium | Audit `CodexClassifier` and `InferenceEngine` label-index mapping before shipping `/trust` with `predicted_class_rank`. If bug confirmed, fix mapping first (effort: S). |
| **Preprocessing mismatch** (gray vs white padding) causes inconsistent embeddings between `/classify` and `/similar` | Medium | Medium | Resolve in S2.B4. Pick one padding strategy (recommend white `(255)` to match `InferenceEngine`, since it is closer to the training path). |
| **User expects instance-level image retrieval** and is disappointed by class-only similarity | Medium | High | Set clear expectation in UI copy: "Visual matches to class prototypes" vs "Similar historical examples." Ship sample-image thumbnails (S2.F5) as a tangible preview of instance-level UX. |
| **Sample data (16 classes) is too sparse** to make `/similar-samples` useful | Medium | Low | `/similar-samples` is explicitly a proof-of-concept. Document that it only covers classes with sample data. If user wants broader coverage, unblock by expanding sample data or obtaining full dataset. |
| **Entropy computation confusion** — scholars may misinterpret entropy as "model confidence" | Low | Medium | Label entropy clearly as "uncertainty" and pair it with concrete signals (margin, top-k bars) so it is never the sole trust metric. |
| **Backend latency** — `POST /trust` with `top_k=286` may be slower than default `top_k=3` | Low | Low | The classifier already computes all 286 similarities before `topk()`; returning more does not change model latency. Monitor actual latency in smoke tests. |

---

## Open Questions

1. **Does the user accept a text-first MVP for similarity, or is instance-level image retrieval a hard requirement for Step 2?**
   - If hard requirement: Step 2 is blocked until the full training dataset (`Elements/`, `metadata.csv`) is obtained.
   - If soft requirement: ship class-level similarity now, instance-level later.

2. **Where is the full training dataset?**
   - `precompute_embeddings.py` references `./Elements/` and `./metadata.csv`, but neither is in the repository. Are they stored externally? Can they be added?

3. **Should numeric `class_label` be exposed in new endpoints until the mapping bug is verified?**
   - R4 notes a potential bug where `class_label` is the prototype-row index, not the original sparse label. Recommend using `class_name` as the stable identifier in v1 API contracts.

4. **Who curates canonical exemplars for the 270 classes without sample images?**
   - Auto-selection (nearest training image to each prototype) is fast but may pick atypical examples. Domain-expert curation is better but takes time.

5. **Should the frontend use JSON/base64 for new endpoints, or do we also need multipart form-data for backward compatibility?**
   - R4 contracts use JSON/base64 because `bbox` is needed. Existing endpoints use multipart. Decide whether to support both or migrate all to JSON.

6. **What threshold defines "ambiguous" for the `ambiguous` flag in `/trust`?**
   - R4 suggests `margin_to_second < 0.05`. Is this scholar-friendly, or should it be configurable?

7. **Should similarity bands (`high`/`moderate`/`low`) be configurable thresholds, or hardcoded?**
   - Current model rejection threshold is 0.35. Band boundaries at 0.35 and 0.60 seem reasonable, but scholars may want to adjust.

---

## Synthesis Notes

- **R1, R2, R3, and R4 are in strong agreement**: the immediate win is frontend visualization of existing data, not backend invention.
- **The only material blocker is missing data/assets** (full dataset, canonical images). All code-path blockers are minor (preprocessing standardization, error envelope, label mapping verification).
- **No model retraining, no pip, no heavy dependencies** — all IN features respect these constraints.
- **The sample-image enrichment (S2.B3 / S2.F5) is a strategic bridge**: it proves the instance-similarity UX with real thumbnails for a subset of classes, keeping momentum high while the full dataset is located.
- **After R5 approval**: implementation can proceed in two waves (backend first, then frontend), with each wave independently verifiable via smoke tests.
