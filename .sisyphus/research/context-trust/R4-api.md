# R4: API Extension Contracts

## Existing Endpoints Audit

Current backend entrypoint: `frontend_integration_fix/frontend_integration/examples/flask_api.py`

| Endpoint | Method | Input | Response shape | Notes |
|---|---|---|---|---|
| `/classify` | `POST` | multipart form-data with `image` file | `{class_name, class_label, confidence, rejected, top_k[]}` | Uses `CodexClassifier.classify(PIL.Image)` directly. Returns 400 only when file is missing. |
| `/classify-batch` | `POST` | multipart form-data with repeated `images` files | `[{class_name, class_label, confidence, rejected, top_k[]}, ...]` | Converts each file to RGB numpy array, then calls `CodexClassifier.classify_batch()`. |
| `/segment` | `POST` | multipart form-data with `image` file | `{num_elements, image_size, elements[]}` where each element is `{bbox, class_name, class_label, confidence, rejected, top_k[]}` | Lazily instantiates `MobileSAMSegmenter(points_per_side=16)`, runs full-page segmentation, extracts crops, then classifies each crop. |
| `/classes` | `GET` | none | `{num_classes, class_names}` | Reads `codex_model/config.json` on each request. |

### Request handling / preprocessing patterns

- Images are loaded with `PIL.Image.open(...).convert("RGB")`.
- `/classify` passes a PIL image into `CodexClassifier`.
- `/classify-batch` and `/segment` convert to `numpy.uint8` RGB arrays first.
- `/segment` uses `MobileSAMSegmenter.extract_crops()` to whiten masked background pixels before classification.
- Bounding boxes in existing JSON are always `[x, y, w, h]` integers.

### CORS / errors

- CORS is added via `@app.after_request` with fixed origin `http://localhost:5173`.
- Allowed methods are `GET, POST, OPTIONS`, but there are no explicit `OPTIONS` routes.
- Error handling is minimal and inconsistent: only missing file fields return JSON `{"error": ...}` with HTTP 400.
- There is no standardized validation for malformed images, invalid bbox coordinates, unexpected inference errors, or class lookup failures.

## Model Methods Available

### `CodexClassifier` (`codex_model/classifier.py`)

- `classify(image: PIL.Image | np.ndarray, top_k: int = 3) -> dict`
- `classify_batch(images: list[PIL.Image | np.ndarray], top_k: int = 3) -> list[dict]`
- Internal preprocess: resize longest side to 224, pad to square with gray `(128,128,128)`, ImageNet normalize.
- Internal tensors available but not exposed as stable public API: `self._backbone`, `self._projection`, `self._prototypes`.

### `InferenceEngine` (`codex_pipeline/inference/engine.py`)

- `preprocess(image: np.ndarray) -> torch.Tensor`
- `embed(images: torch.Tensor) -> torch.Tensor`
- `classify_crop(image: np.ndarray) -> ClassificationResult`
- `classify_batch(images: list[np.ndarray]) -> list[ClassificationResult]`
- Loads full prototype artifact (`prototypes/prototypes.pt`) including `class_labels`, `class_names`, and potentially `class_meta` if present.
- Preprocess differs slightly from `CodexClassifier`: longest-side resize, white padding `(255,255,255)`, ImageNet normalize.

### `MobileSAMSegmenter` (`codex_pipeline/segmentation/mobilesam.py`)

- `segment_page(image: np.ndarray) -> list[RegionProposal]`
- `extract_crops(image: np.ndarray, proposals: list[RegionProposal], padding: int = 5) -> list[RegionProposal]`
- `RegionProposal` contains `bbox`, full-image `mask`, `area`, `confidence`, and optional `crop`.

## Proposed Endpoints

### POST /similar

- **Purpose**: given an image and region bbox, return visually similar matches for that region.
- **Recommended method/path**: `POST /similar`

- **Request**:

```json
{
  "image_base64": "<base64-encoded RGB/JPEG/PNG image>",
  "bbox": [120, 84, 56, 61],
  "limit": 10,
  "mode": "prototype",
  "include_crop": false
}
```

Request notes:

- `bbox` uses existing backend convention `[x, y, w, h]`.
- `mode` should initially support `"prototype"`; reserve `"instance"` for future dataset-level nearest-neighbor retrieval.
- JSON transport is required because this report is contract-first; implementation will need new image decoding logic because existing API only accepts multipart uploads.

- **Response 200**:

```json
{
  "query": {
    "bbox": [120, 84, 56, 61],
    "embedding_dim": 128,
    "mode": "prototype"
  },
  "best_match": {
    "class_name": "atl",
    "class_label": 282,
    "similarity": 0.699,
    "rejected": false
  },
  "results": [
    {
      "rank": 1,
      "match_type": "class_prototype",
      "class_name": "atl",
      "class_label": 282,
      "similarity": 0.699,
      "band": "high",
      "asset": null
    },
    {
      "rank": 2,
      "match_type": "class_prototype",
      "class_name": "tlalli",
      "class_label": 275,
      "similarity": 0.538,
      "band": "moderate",
      "asset": null
    }
  ]
}
```

- **Response 400 / 422 / 500**:

```json
{"error": {"code": "INVALID_REQUEST", "message": "bbox must be [x,y,w,h] within image bounds"}}
```

```json
{"error": {"code": "INVALID_IMAGE", "message": "image_base64 could not be decoded"}}
```

```json
{"error": {"code": "INFERENCE_FAILED", "message": "embedding or similarity search failed"}}
```

- **Implementation**:
  - Decode `image_base64` to RGB numpy array.
  - Crop bbox from the source image.
  - Immediate MVP: run `CodexClassifier.classify(crop, top_k=limit)` and map `top_k` to `results` as prototype-level similarity.
  - Future instance mode: use `InferenceEngine.preprocess()` + `InferenceEngine.embed()` and compare against a precomputed projected embedding index (`dataset_embeddings.pt` or FAISS) to return real similar images/regions.
  - Similarity bands can be derived from existing scores, e.g. `high >= 0.60`, `moderate >= 0.35`, else `low`.

- **Latency**:
  - Prototype mode: ~100-150 ms CPU, ~20-40 ms GPU, plus small image decode/crop overhead.
  - Instance mode with exact in-memory k-NN over current-scale embeddings: likely +1-10 ms over prototype mode once an embedding index exists.

- **Open questions**:
  - Does product need true image/region retrieval in v1, or are prototype/class matches sufficient?
  - Should results include canonical exemplar URLs for prototype matches?
  - Should the endpoint also accept multipart uploads for parity with current API?

### POST /trust

- **Purpose**: given an image, region bbox, and predicted class, return trust / confidence signals for that prediction.
- **Recommended method/path**: `POST /trust`

- **Request**:

```json
{
  "image_base64": "<base64-encoded RGB/JPEG/PNG image>",
  "bbox": [120, 84, 56, 61],
  "predicted_class": "atl",
  "top_k": 5
}
```

- **Response 200**:

```json
{
  "query": {
    "bbox": [120, 84, 56, 61],
    "predicted_class": "atl"
  },
  "trust": {
    "predicted_class_rank": 1,
    "predicted_class_similarity": 0.699,
    "top1_class": "atl",
    "top1_similarity": 0.699,
    "margin_to_second": 0.161,
    "above_rejection_threshold": true,
    "rejection_threshold": 0.35,
    "ambiguous": false,
    "class_consistency": {
      "num_examples": null,
      "intra_class_variance": null
    }
  },
  "top_candidates": [
    {"rank": 1, "class_name": "atl", "similarity": 0.699},
    {"rank": 2, "class_name": "tlalli", "similarity": 0.538},
    {"rank": 3, "class_name": "tlacuilolli", "similarity": 0.512}
  ]
}
```

- **Response 400 / 404 / 500**:

```json
{"error": {"code": "UNKNOWN_CLASS", "message": "predicted_class is not in class_names"}}
```

```json
{"error": {"code": "INVALID_REQUEST", "message": "bbox must be [x,y,w,h] within image bounds"}}
```

```json
{"error": {"code": "INFERENCE_FAILED", "message": "trust signals could not be computed"}}
```

- **Implementation**:
  - Decode image and crop bbox.
  - Call `CodexClassifier.classify(crop, top_k=num_classes)` so the server can rank `predicted_class` against the full 286-class similarity list without changing model code.
  - Derive trust signals from existing model outputs:
    - `predicted_class_rank`
    - `predicted_class_similarity`
    - `top1_similarity`
    - `margin_to_second`
    - `above_rejection_threshold`
    - `ambiguous` (e.g. `margin_to_second < 0.05` or top1 below threshold)
  - Optional enrichment: if backend loads full `prototypes/prototypes.pt` instead of the stripped `codex_model/weights/prototypes.pt`, it can also expose `class_meta[label].count` and `class_meta[label].variance` from `evaluate.py` exports.

- **Latency**:
  - ~100-170 ms CPU, ~20-50 ms GPU.
  - Asking for all 286 similarities is still cheap because the classifier already computes the full similarity vector before `topk()`.

- **Open questions**:
  - Is `predicted_class` always a class name string, or should class IDs also be accepted?
  - What threshold should define `ambiguous` for scholar-facing UI?
  - Should trust include dataset priors such as exemplar count or historical frequency?

### GET /prototypes

- **Purpose**: return exemplar images for a requested class.
- **Recommended method/path**: `GET /prototypes?class_name=atl&limit=12`

- **Request**:

Query params:

```text
class_name: string (required)
limit: integer (optional, default 12, max 50)
```

- **Response 200**:

```json
{
  "class_name": "atl",
  "class_label": 282,
  "count": 12,
  "items": [
    {
      "id": "atl-001",
      "image_url": "/assets/prototypes/atl/atl-001.jpg",
      "thumbnail_url": "/assets/prototypes/atl/atl-001-thumb.jpg",
      "bbox": null,
      "source": "canonical_exemplar"
    }
  ]
}
```

- **Response 404 / 500**:

```json
{"error": {"code": "CLASS_NOT_FOUND", "message": "No class named 'atlx'"}}
```

```json
{"error": {"code": "PROTOTYPE_STORE_MISSING", "message": "No exemplar manifest is available for this class"}}
```

- **Implementation**:
  - Validate `class_name` against `config.json` / loaded class names.
  - Read from an exemplar manifest rather than model weights. The model prototype tensor contains embeddings, not image assets.
  - Preferred backing store: a generated JSON/PT manifest mapping each class to canonical or training-set example images.
  - If later upgraded to region-level exemplars, each item can additionally include `bbox` and source page metadata.

- **Latency**:
  - In-memory manifest lookup: <10 ms.
  - If manifest is read from disk per request: ~10-30 ms.
  - Serving actual images is a separate static-file concern.

- **Open questions**:
  - Are exemplars meant to be canonical curated images, or arbitrary training examples?
  - Should the backend return signed URLs / filesystem-relative paths / embedded thumbnails?
  - Does the UI require region crops plus source manuscript metadata?

## Gaps & Blockers

1. **R2 and R3 context are not present in `.sisyphus/research/context-trust/`** at audit time, so trust and prototype contracts can only use code evidence plus R1 recommendations.
2. **No JSON image ingestion exists today.** Current Flask API only accepts multipart file uploads.
3. **No bbox validation / crop helper exists in the API layer.** `/similar` and `/trust` need shared validation logic.
4. **No dataset embedding index exists for true image/region similarity.** R1's recommended instance-retrieval path requires a precomputed projected embedding store (or FAISS index) plus image metadata.
5. **No exemplar manifest exists for `/prototypes`.** Model prototype tensors are embeddings only; they do not point to renderable exemplar images.
6. **Trust metadata is only partially exported.** `evaluate.py` can export `class_meta` (count + variance) into `prototypes/prototypes.pt`, but `export_model.py` strips that metadata from `codex_model/weights/prototypes.pt`.
7. **Potential class-label bug in current inference outputs.** Both `CodexClassifier` and `InferenceEngine` appear to return prototype-row indices as `class_label`, not the original sparse labels stored in `class_labels` / `_label_index`. Any new API should avoid promising stable numeric IDs until this is clarified.
8. **Preprocessing differs between wrappers.** `CodexClassifier` pads with gray `(128)` while `InferenceEngine` pads with white `(255)`. This should be standardized before mixing both wrappers in one API surface.
9. **Error envelope is not standardized.** New endpoints should use one JSON error shape across 400/404/422/500 responses.
10. **CORS is fixed to localhost:5173.** If frontend origin changes, new endpoints will inherit the same integration limitation.

## Open Questions

1. Should `/similar` ship as **prototype similarity first** and reserve true image retrieval for a later phase?
2. Should `/trust` be based only on model-native signals (score, rank, margin, threshold) or also include dataset-derived signals (class variance, exemplar count)?
3. What asset store will back `/prototypes` for all 286 classes?
4. Do clients need backward-compatible multipart support, or is a JSON/base64 contract acceptable for all new endpoints?
5. Should numeric `class_label` be exposed at all until the current label-index mapping is verified/fixed?
