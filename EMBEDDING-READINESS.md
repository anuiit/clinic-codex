# Embedding Similarity Design Doc

This document outlines the technical design for the embedding-similarity feature, which enables historical archetype lookup and trust signal analysis for Nahuatl glyph elements.

## Architecture

The embedding pipeline transforms raw glyph images into compact, semantically rich vectors using a foundation model backbone and a specialized projection head.

1. **Backbone**: DINOv2 (ViT-S/14) pretrained on ImageNet.
2. **Feature Extraction**: Extracts the CLS token (384-dimensional) from the backbone.
3. **Projection**: A two-layer MLP projects the 384-dim features into a 128-dimensional embedding space.
4. **Normalization**: Embeddings are L2-normalized to enable similarity measurement via cosine distance (dot product).

## Core API Endpoints

The features are exposed via two primary REST endpoints in `backend/examples/flask_api.py`.

### `POST /similar`
Finds historical archetypes similar to a cropped glyph element.

- **Request Body**:
  ```json
  {
    "image_base64": "...",
    "bbox": [x, y, w, h],
    "limit": 5
  }
  ```
- **Process**: Crops the image, generates a 128-dim embedding, and calculates cosine similarity against 286 class prototypes.
- **Returns**: Ranked list of matches with similarity scores and confidence bands (high, moderate, low).

### `POST /trust`
Calculates confidence and ambiguity signals for a given classification.

- **Request Body**:
  ```json
  {
    "image_base64": "...",
    "bbox": [x, y, w, h],
    "predicted_class": "atl"
  }
  ```
- **Signals Provided**:
  - `margin_to_second`: Difference between top-1 and top-2 similarity.
  - `entropy`: Distribution of similarity across top-K classes.
  - `ambiguous`: Flag set if the margin is below 0.05.
  - `above_rejection_threshold`: Comparison against the 0.35 baseline.

## Implementation Details

- **Classes**: 286 labeled Nahuatl element classes.
- **Prototypes**: One embedding per class, computed as the centroid of known exemplars.
- **Current Status**: Endpoints are implemented in the Flask API. Activation is awaiting the finalized dataset of labeled glyph images.
- **Deferred**: Instance-level embeddings (searching individual historical examples rather than class averages) is noted for future work.

## Requirements for Activation

To fully activate these features, the following are required:
1. **Labeled Dataset**: A curated set of high-quality, segmented glyph images for all 286 classes.
2. **Prototype Script**: A utility to re-compute and export `prototypes.pt` whenever the underlying dataset is updated.
