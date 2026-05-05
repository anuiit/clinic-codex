# R3: Prototype Gallery Patterns

## Available Backend Assets

### What Exists in the Repository Today

| Asset | Location | Details |
|-------|----------|---------|
| **286 class prototypes** | `codex_model/weights/prototypes.pt`, `prototypes/prototypes.pt` | 128-d L2-normalized vectors, one per class |
| **Class name catalog** | `codex_model/config.json` | 286 Nahuatl class names (e.g., "atl", "calli", "xochitl") |
| **Projection head + DINOv2** | `codex_model/weights/projection.pt` + torch.hub | Can embed any image to 128-d on-the-fly |
| **Sample element images** | `data/elements_sample/` | 10 classes, ~3 BMP images each |
| **Sample glyph images** | `data/glyphs_sample/` | 6 classes, ~5 JPG images each |
| **API `top_k` response** | `flask_api.py` `/classify`, `/segment` | Already returns nearest competing classes with confidence scores |

### What Is Missing

| Missing Asset | Why It Matters | Blocked By |
|---------------|----------------|------------|
| **Full training dataset** | `precompute_embeddings.py` expects `./Elements/` dir and `./metadata.csv`; neither exists in repo | Dataset is external / not committed |
| **Precomputed dataset embeddings** | Required for any instance-level similarity search (finding actual similar *images*) | Missing full dataset + never run precomputation |
| **Canonical exemplar per class** | Required to show a representative image when a class is mentioned | No curation script or manual selection exists |
| **Per-class training image lists** | Required to know which images belong to which class | Missing metadata.csv |

### Key Constraint

> **Only 16 of 286 classes have sample images in the repository.** The other 270 classes have **prototype vectors and names only** — no visual examples are locally reachable.

The model can embed any image on-the-fly, but without the full training dataset or precomputed embeddings, we cannot retrieve "nearest training images" for arbitrary queries.

---

## Gallery Patterns

### 1. Per-Class Prototype Grid

**Description**  
For the predicted class, display a grid of the training images that are closest to that class's prototype vector in the 128-d embedding space. This answers: *"What does the model think this class looks like?"*

**Data Required**
- Embeddings for every image in the training dataset
- Mapping from embeddings back to image file paths
- Class labels for every image

**Available Now?** **No**
- The full training dataset (`./Elements/`, `./metadata.csv`) is not present in the repository.
- `precompute_embeddings.py` exists but has never been run; no `features.pt` or `dataset_embeddings.pt` exists.
- For the 16 classes with sample data, we *could* compute embeddings on-the-fly and show nearest samples, but this would be a toy demonstration covering <6% of classes.

**Pros**
- Highly interpretable: scholars see the visual archetype the model learned
- Builds trust by showing the "evidence" behind a prediction
- Standard pattern in prototype-based ML systems (e.g., Prototypical Networks papers)

**Cons**
- Requires full dataset access and offline precomputation
- Storage grows linearly with dataset size (~5 MB per 10k images, trivial at current scale)
- Maintenance burden: must recompute when training data changes

**Path to Availability**
1. Obtain/regenerate `metadata.csv` and `./Elements/` directory
2. Run `python -m codex_pipeline.scripts.precompute_embeddings` (or a variant that also projects to 128-d)
3. Add API endpoint that loads the embedding matrix and runs `torch.mm(query_proto, dataset_embeddings.t()).topk(k)`

---

### 2. Hard-Negative Examples

**Description**  
Show training images that are embedding-space neighbors of the query but belong to a *different* class. This surfaces borderline cases: *"This looks like X, but here are examples the model knows are actually Y."*

**Data Required**
- Full dataset embeddings (same as Pattern 1)
- Per-image class labels
- Ability to filter: high similarity to query, different class label

**Available Now?** **No**
- Same blocker as Pattern 1: no dataset embeddings.
- Additionally requires cross-class neighbor search, which is a secondary computation on top of the embeddings.

**Pros**
- Excellent for scholar education: explains *why* a class might be confused with another
- Surfaces genuine visual ambiguity in the codex (e.g., stylistic variants)
- Builds calibration: users learn the model's failure modes

**Cons**
- More complex than prototype grids: requires filtering by class mismatch
- Can be misleading if hard negatives are actually labeling errors
- Needs careful UI framing ("similar but different" rather than "the model is confused")

**Path to Availability**
1. Same precomputation as Pattern 1
2. Add backend logic: `dataset_embeddings @ query_embedding.t()`, then filter `label != predicted_class`, take top-k

---

### 3. Confusion-Class Examples

**Description**  
For a query, show representative images from the top-k competing classes (the classes the model considered but ranked lower). The API already returns these classes in `top_k`; this pattern visualizes them.

**Data Required**
- The `top_k` list (already returned by `/classify`)
- A representative/canonical image per class for the competing classes

**Available Now?** **Partially — text yes, images no**
- The API returns `top_k` with `class_name` and `confidence` for every classification. **Zero backend changes needed** to show the list.
- However, **images are only available for 16/286 classes** in the sample data. For the other 270 classes, no representative image exists in the repo.
- A text-only confusion view (class names + confidence bars) is fully buildable today.

**Pros**
- **Immediate MVP**: uses data already flowing over the wire
- High scholarly value: seeing that an unknown glyph is similar to "atl" (0.70), "tlalli" (0.54), or "tlacuilolli" (0.51) is actionable
- Low risk: no new dependencies, no precomputation, no storage

**Cons**
- Without images, it is less persuasive than a visual gallery
- Class names alone may not help non-expert users
- Doesn't show *why* the classes are confused — just *that* they are

**Variants by Effort**

| Variant | Effort | Description |
|---------|--------|-------------|
| **Text-only confusion list** | 1 hour frontend | Ranked list of competing class names with confidence bars |
| **Sample-image enrichment** | 2-3 hours | For the 16 classes with sample data, show their images; others get placeholder |
| **Canonical image backend** | 1-2 days backend | Add `canonical_image_url` to `/classes` or `/classify` response (requires curating 286 images) |

**Path to Availability**
- **Today**: build the text-only variant
- **Short-term**: curate one canonical image per class (could be the sample data image nearest to each prototype, or manually selected by domain experts)
- **Medium-term**: combine with Pattern 1 to show dynamic nearest images per competing class

---

### 4. Distance-Ranked Exemplars

**Description**  
Rank *all* training images by cosine similarity to the query embedding and display the top-N most similar images across all classes. This is a true "visual search" or "reverse image search" within the training set.

**Data Required**
- Full dataset embeddings (projected to 128-d, same space as query)
- Image paths for every training example
- Optional: class labels to color-code or group results

**Available Now?** **No**
- Same blocker as Patterns 1 and 2: no precomputed embeddings, no full dataset.
- The `InferenceEngine.embed()` method can compute query embeddings, but there is no index to search against.

**Pros**
- Most granular similarity signal: shows actual historical examples
- Useful for provenance: "This query looks most like folio 26r, 32v, and 72r"
- Familiar UX pattern (Google Images reverse search, Pinterest visual search)

**Cons**
- Highest data requirement of all patterns
- Scalability: O(N) brute-force search; fine for ~10k images, needs FAISS for 100k+
- Can surface low-quality or ambiguous training images if not filtered

**Path to Availability**
1. Same precomputation as Pattern 1
2. Add `/similar-images` endpoint: `torch.mm(query_embedding, dataset_embeddings.t()).topk(k)`
3. Return `{image_url, class_name, similarity}` tuples
4. Optional: build FAISS `IndexFlatIP` if dataset grows beyond ~50k images

---

### 5. Class Catalog Browser

**Description**  
A standalone browsable gallery showing all 286 classes, each with a representative image and class name. Users can scroll or search to explore the full taxonomy before uploading an image.

**Data Required**
- One representative image per class (canonical exemplar)
- Class names and optional descriptions/etymology

**Available Now?** **No for images; yes for names**
- `config.json` contains all 286 class names. A text-only catalog or autocomplete is trivial.
- Only 16 classes have any images in the repository. A visual catalog would be 94% empty.

**Pros**
- Educational: scholars learn the taxonomy by browsing
- Reference: users can check "what does `tlacuilolli` look like?" before analyzing
- Foundation for other features (click a class to see its prototype, confusion history, etc.)

**Cons**
- Requires the most manual curation: 286 canonical images must be selected
- Large UI surface: 286 items need search, filtering, or pagination
- Without good exemplars, can mislead users about class appearance

**Variants by Effort**

| Variant | Effort | Description |
|---------|--------|-------------|
| **Text catalog + search** | 2 hours | Alphabetic / searchable list of 286 class names |
| **Sample-data preview** | 3-4 hours | Show images for the 16 classes that have them; placeholder for rest |
| **Full visual catalog** | 2-3 days | Curate 286 canonical images, build grid with lazy loading |

**Path to Availability**
- **Today**: text catalog
- **Short-term**: auto-select canonical images by embedding all sample/training data and picking the image nearest each prototype
- **Long-term**: domain-expert curated gallery with multi-image carousels per class

---

## Cross-Pattern Asset Matrix

| Asset \ Pattern | P1 Prototype Grid | P2 Hard Negatives | P3 Confusion Classes | P4 Ranked Exemplars | P5 Catalog Browser |
|-----------------|-------------------|-------------------|----------------------|---------------------|--------------------|
| 286 prototypes | ✅ Used | ✅ Used | ✅ Used | ✅ Used | ✅ Used |
| 286 class names | ✅ Used | ✅ Used | ✅ Used | ✅ Used | ✅ Used |
| `top_k` API | ❌ No | ❌ No | ✅ Yes | ❌ No | ❌ No |
| Full dataset embeddings | ✅ Required | ✅ Required | ❌ No | ✅ Required | ⚠️ Helpful |
| Canonical image/class | ⚠️ Helpful | ❌ No | ✅ Required for images | ❌ No | ✅ Required for images |
| Sample data (16 classes) | ⚠️ Partial | ⚠️ Partial | ⚠️ Partial | ⚠️ Partial | ⚠️ Partial |

**Legend:** ✅ = fully available / required, ⚠️ = partially available / nice-to-have, ❌ = not available / not needed

---

## Recommendation

### MVP: Confusion-Class Text Visualization (Pattern 3, Text-Only Variant)

**Why this is the right first step:**

1. **Zero backend work**: The `/classify` and `/segment` endpoints already return `top_k` with class names and confidence scores. The frontend can render this immediately.
2. **High scholar value**: Knowing that an element is visually similar to "atl" (0.70), "tlalli" (0.54), or "tlacuilolli" (0.51) is immediately actionable for codex studies.
3. **Trust-building**: Transparency about the model's runner-up predictions builds user confidence, even without images.
4. **No missing assets**: Unlike every other pattern, this does not depend on the full training dataset or canonical images.

**Suggested MVP UI:**
- A compact panel below each classification result
- Ranked list of top-3 competing classes
- Horizontal confidence bar per class (color-coded: green >0.6, yellow 0.35-0.6, red <0.35)
- Optional: click a class name to open a tooltip with its English gloss / description (if available)

### Phase 2: Enrich with Sample Images + On-the-Fly Embeddings

Once the text MVP is live, add visual context for the 16 classes that have sample data:

1. **At backend startup**, compute embeddings for all ~60 sample images using `InferenceEngine.embed()`
2. **Build a lightweight in-memory index**: `{class_name: [(image_path, embedding), ...]}`
3. **Add `/similar-samples` endpoint**: Given a query image, return the nearest sample images (limited to the 16 sampled classes)
4. **Frontend**: Show a small thumbnail grid of the top-3 nearest sample images alongside the text `top_k` list

This is not a full solution (270 classes still have no images), but it:
- Proves the instance-similarity UX without requiring the full dataset
- Provides a fallback visual for common classes
- Can be extended seamlessly when the full dataset becomes available

### Phase 3: Full Dataset Precomputation (Patterns 1, 2, 4)

When the full training dataset (`Elements/` + `metadata.csv`) is available:

1. Run a modified `precompute_embeddings.py` that projects to 128-d (not just 384-d DINOv2 features)
2. Ship `dataset_embeddings.pt` alongside the model weights
3. Add `/similar` endpoint supporting multiple modes:
   - `mode=prototype`: nearest training images to the predicted class prototype (Pattern 1)
   - `mode=query`: nearest training images to the query embedding (Pattern 4)
   - `mode=hard_negative`: high-similarity, different-class images (Pattern 2)
4. Auto-generate canonical images per class by picking the training image closest to each prototype; this also unblocks Pattern 5 (Class Catalog Browser)

---

## Open Questions

1. **Where is the full training dataset?**
   - `default.yaml` references `./Elements/` and `./metadata.csv`, but neither exists in the repository. Are they stored externally (cloud storage, shared drive)? Can they be added to the repo, or should the backend load them from an external path?

2. **Who curates canonical exemplars?**
   - Auto-selecting the image nearest each prototype is fast but may pick a low-quality or atypical example. Domain-expert curation (one archaeographer picking the "best" example per class) would yield a more trustworthy catalog.

3. **Should sample data be expanded?**
   - The current 16-class sample is useful for smoke tests but insufficient for a visual gallery. Should we expand the sample to cover, say, 50-100 common classes? This would make Patterns 3 and 5 viable for a larger subset without needing the full dataset.

4. **Is the full dataset size known?**
   - `precompute_embeddings.py` handles arbitrary N, but knowing the image count helps decide between brute-force search (Pattern 4) and FAISS indexing. A rough estimate from the 286-class taxonomy suggests thousands to tens of thousands of images.

5. **Does the frontend need instance-level similarity now, or is class-level sufficient?**
   - The user's original request was to "visualize close embedded images from the analyzed image" — this implies instance-level (Pattern 4). However, class-level confusion (Pattern 3) is the only pattern reachable without the full dataset. Is the user willing to accept a text-first MVP?

---

*Research based on inspection of: `flask_api.py`, `classifier.py`, `engine.py`, `dataset.py`, `precompute_embeddings.py`, `config.json`, `metadata.py`, and the `data/` directory contents.*
