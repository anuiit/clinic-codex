# R1: Visual Similarity Strategy

## Current Model Capabilities

The Codex classifier is a **prototypical network** built on top of a frozen DINOv2-ViT-S/14 backbone. Here is what the model exposes today:

### Architecture
| Component | Details |
|-----------|---------|
| **Backbone** | DINOv2-ViT-S/14 (frozen, loaded via `torch.hub`) |
| **Feature dim** | 384-d CLS token |
| **Projection head** | 2-layer MLP: 384 → 384 → 128, GELU, dropout 0.1, L2-normalized |
| **Embedding space** | 128-d, L2-normalized (unit hypersphere) |
| **Prototypes** | 286 class prototypes, each 128-d, L2-normalized |
| **Classification** | Cosine similarity = dot product (since both are L2-normalized) |
| **Rejection threshold** | 0.35 (values below this mark the sample as unknown / OOD) |

### What the API Currently Returns
The `/classify` endpoint returns:
```json
{
  "class_name": "atl",
  "class_label": 282,
  "confidence": 0.699,
  "rejected": false,
  "top_k": [
    {"class_name": "atl", "confidence": 0.699},
    {"class_name": "tlalli", "confidence": 0.538},
    {"class_name": "tlacuilolli", "confidence": 0.512}
  ]
}
```

### Key Exposed Tensors / Methods
- `CodexClassifier.classify(image)` — top-k class predictions via prototype similarity
- `InferenceEngine.embed(images)` — raw 128-d L2-normalized embeddings `(B, 128)`
- `InferenceEngine._embedding_to_result(embedding)` — maps a single embedding to top-k prototype matches
- `self._prototypes` — `(286, 128)` tensor of class prototypes (directly accessible)
- `self._projection` — projection head module (can be used independently)
- `self._backbone` — DINOv2 backbone (can extract 384-d features before projection)

### Prototype Storage
- `prototypes/prototypes.pt` — full training artifact (prototypes + projection weights + metadata)
- `codex_model/weights/prototypes.pt` — exported prototypes tensor + class info
- `codex_model/weights/projection.pt` — projection head `state_dict` only

### Precomputed Features (Training Only)
- `precompute_embeddings.py` can cache 384-d DINOv2 features for the entire dataset
- Training operates on cached features; the projection head is trained with episodic prototypical loss

---

## Candidate Approaches

### 1. Exact Cosine k-NN on Dataset Embeddings
**Description**  
Pre-compute 128-d embeddings for every image in the training dataset and store them. At query time, compute the embedding of the input image and perform an exact brute-force cosine similarity search against all stored embeddings. Return the top-N most similar *individual images* along with their class labels.

**Pros**
- **Instance-level granularity**: Returns actual similar images, not just similar classes.
- **No additional dependencies**: Pure PyTorch / NumPy; no FAISS or other libraries needed.
- **Deterministic and exact**: No approximation error.
- **Leverages existing embed() method**: The `InferenceEngine.embed()` already outputs the exact vectors needed.

**Cons**
- **Scalability**: O(N) per query. With ~10k images this is still fast on GPU (~1-2ms), but slows as the dataset grows.
- **Storage**: ~10k × 128 × 4 bytes ≈ 5 MB; trivial for current scale, but grows linearly.
- **Maintenance**: New training data requires re-computing and storing embeddings.

**Integration Cost**: **Low**  
- Add a precomputation step in `precompute_embeddings.py` to also cache *projected* embeddings (or run the projection head on cached 384-d features).
- Store `{embedding: (N, 128), image_path: List[str], class_label: List[int]}` in a `.pt` file.
- Add a new API endpoint (e.g., `/similar`) that loads this tensor and runs `torch.mm(query_embedding, dataset_embeddings.t()).topk(k)`.

**Data Requirements**: None beyond existing training images.

---

### 2. FAISS Approximate Nearest Neighbor (ANN) Index
**Description**  
Build a FAISS index (e.g., `IndexFlatIP` for exact inner product, or `IndexIVFFlat` / `IndexHNSWFlat` for approximate search) on the dataset embeddings. FAISS is optimized for billion-scale search and can run on GPU (`faiss-gpu`).

**Pros**
- **Extremely fast**: Sub-millisecond queries even for millions of embeddings (GPU) or tens of milliseconds (CPU).
- **Scalable**: Handles dataset growth without linear query-time degradation.
- **GPU support**: `faiss-gpu` can keep the index on the same device as the model.
- **Mature ecosystem**: Well-documented, supports batch queries, filtering, and IVF/HNSQ quantization.

**Cons**
- **New dependency**: Adds `faiss-cpu` or `faiss-gpu` to `requirements.txt`.
- **Index build time**: IVF and HNSW indexes require an offline training step on a sample of embeddings.
- **Approximation trade-off**: IVF/HNSW can miss exact nearest neighbors (tunable recall vs. speed).

**Integration Cost**: **Low-Medium**  
- Install FAISS, build index at model load time or ship a pre-built index file.
- Add ~20 lines to the inference engine to query the index and map FAISS indices back to image paths / class labels.
- For 286 classes and likely <50k images, `IndexFlatIP` (exact, no training) is sufficient and trivial.

**Data Requirements**: Same as Approach 1.

---

### 3. Prototype-Space Similarity with Confidence Bands
**Description**  
Instead of searching individual images, use the **existing 286 class prototypes** to define visual similarity. The model already computes cosine similarity to every prototype; we simply expose this more richly. For a query embedding, return the top-K *classes* whose prototypes are closest, along with similarity scores and a confidence band (e.g., "high similarity" > 0.6, "moderate" 0.35-0.6, "low" < 0.35).

**Pros**
- **Zero infrastructure**: No precomputed dataset, no new libraries, no storage.
- **Instant integration**: The `top_k` field already contains this information; just needs richer API formatting.
- **Human-interpretable**: "This glyph looks like 'atl' (0.70), 'tlalli' (0.54), or 'tlacuilolli' (0.51)" is immediately useful to scholars.
- **Fastest possible**: A single matrix multiplication `(1, 128) × (128, 286)` is already done in every classify call.

**Cons**
- **Class-level only**: Cannot show *which specific historical example* is most similar.
- **Limited by prototype quality**: If a class has high intra-class variance, a single prototype may not capture all visual modes.

**Integration Cost**: **Very Low**  
- Add a new endpoint (e.g., `/similar-classes`) that reuses the existing `classify()` logic but formats `top_k` as the primary output.
- Optionally add similarity band labels (high/moderate/low) based on configurable thresholds.

**Data Requirements**: None.

---

### 4. Multi-Prototype per Class (k-Means Prototypes)
**Description**  
Compute multiple prototypes per class by clustering the training embeddings of each class (e.g., k-means with k=3-5). Replace the single `(286, 128)` prototype matrix with a `(286 × k, 128)` matrix. Classification and similarity search then operate against a finer-grained set of prototypes, capturing intra-class visual variation.

**Pros**
- **Captures visual diversity within classes**: A class like "xochitl" (flower) may have multiple stylistic variants; multiple prototypes represent each mode.
- **Better similarity signals**: Query embeddings map to a specific visual mode, not just a class centroid.
- **Still fast**: `(1, 128) × (128, 286k)` is still a single GPU matrix multiply.

**Cons**
- **Requires training data access**: Need the full dataset embeddings to run k-means per class.
- **Increases storage**: k× more prototypes (e.g., 5× = 1,430 prototypes; still tiny at ~730 KB).
- **Slightly complicates the mapping**: Need to track which prototype belongs to which class and mode.

**Integration Cost**: **Medium**  
- Add an offline script that loads cached embeddings, runs k-means per class (e.g., `sklearn.cluster.KMeans`), and exports a multi-prototype `.pt` file.
- Update `CodexClassifier` and `InferenceEngine` to load the multi-prototype matrix and handle the mapping.

**Data Requirements**: Full training dataset embeddings.

---

### 5. Hierarchical: Prototype → Embedding → Nearest Image
**Description**  
A two-stage approach:
1. **Stage 1 (fast)**: Use class prototypes to get the top-M candidate classes (~1ms).
2. **Stage 2 (optional)**: If instance-level similarity is needed, search a precomputed embedding index *only within the top-M classes*. This dramatically narrows the search space.

**Pros**
- **Best of both worlds**: Fast class-level similarity always available; instance-level similarity on demand.
- **Efficient**: Instance search runs on a tiny subset of the dataset (e.g., top 3 classes × ~50 images = 150 embeddings).
- **Progressive disclosure**: Frontend can show class matches instantly, then load actual similar images asynchronously.

**Cons**
- **Most complex**: Requires both prototype storage *and* per-image embedding storage.
- **Two storage files**: Prototypes + dataset embedding index.

**Integration Cost**: **Medium**  
- Combine the storage needs of Approaches 1 and 3.
- Add orchestration logic in the API to run Stage 1, optionally Stage 2.

**Data Requirements**: Training dataset embeddings.

---

## Recommendation

**Primary recommendation: Approach 3 (Prototype-Space Similarity) as the immediate MVP, with a migration path to Approach 1 (Exact k-NN) or Approach 5 (Hierarchical) if instance-level similarity becomes a requirement.**

### Justification
1. **The model is already a prototypical network**. The 128-d embedding space and 286 prototypes are the native representation. Fighting against this (e.g., by ignoring prototypes and doing raw pixel similarity) would be counter-productive.
2. **Zero new dependencies or data**. Approach 3 works with the exact artifacts already shipped (`prototypes.pt`, `projection.pt`). This is critical for a research/demo backend where simplicity matters.
3. **The `top_k` field is already the similarity signal**. The backend computes it on every request; the frontend just needs an endpoint that surfaces it as a first-class feature.
4. **Class-level similarity is high-value for scholars**. In a codex-studies context, knowing that an unknown glyph element is visually similar to "atl", "tlalli", or "tlacuilolli" is immediately actionable. Instance-level retrieval is a "nice to have" that can be added later.
5. **Performance is trivial**: The similarity computation is already done in every classify call and takes <1ms on GPU.

### Suggested API Addition
```python
@app.route("/similar", methods=["POST"])
def similar():
    img = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
    result = clf.classify(img, top_k=10)  # Increase k for similarity view
    return jsonify({
        "query_embedding_dim": 128,
        "similar_classes": result["top_k"],
        "best_match": result["class_name"],
        "best_confidence": result["confidence"],
    })
```

### Future Migration Path
If the product team later requires "show me the 5 most similar *images* from the training set":
- **Phase 2**: Implement Approach 1 (Exact k-NN). Precompute and ship a `dataset_embeddings.pt` file. Add an optional `mode=instance` parameter to `/similar`.
- **Phase 3**: If the dataset grows beyond ~100k images, swap the brute-force search for Approach 2 (FAISS `IndexFlatIP`). The API contract remains identical.

---

## Open Questions

1. **Does the frontend need instance-level similarity (actual similar images) or is class-level similarity sufficient for the first release?**
   - If instance-level is required, we should precompute embeddings for the training dataset now.

2. **Is there a curated "canonical example" image per class that could be shown when a prototype match is returned?**
   - If yes, we can enrich the `/similar` response with a `canonical_image_url` per class without needing full k-NN.

3. **What is the total size of the training dataset?**
   - The `precompute_embeddings.py` script suggests there may be thousands of images. Knowing the exact count helps estimate storage for Approach 1.

4. **Should similarity be exposed as a standalone endpoint, or should the frontend use the existing `top_k` field in `/classify` responses?**
   - A standalone endpoint is cleaner but requires a small backend change. Reusing `top_k` requires no backend work.

5. **Are there plans to expand beyond 286 classes?**
   - If the class count will grow to thousands, FAISS (Approach 2) becomes more attractive even for class-level search.
