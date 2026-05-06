# R2: Trust Visualization Patterns

## Model Data Available Today

The Codex classifier is a prototypical network (DINOv2-ViT-S/14 → 384-d CLS → 2-layer MLP → 128-d L2-normalized embeddings) with 286 class prototypes on the unit hypersphere. Classification is cosine similarity (dot product) between the query embedding and each prototype.

### What the API Currently Returns

The `/classify` and `/segment` endpoints return:

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

### What the Model Computes But Does NOT Expose

- **Raw 128-d L2-normalized embedding** (`embedding`) — computed in every call
- **All 286 cosine similarities** (`similarities`) — computed as `torch.mm(embedding, prototypes.t())` but only `topk(k)` values are returned
- **Distance to predicted prototype** — `1 - confidence` (since cosine similarity ≈ 1 means identical direction)
- **Distance to nearest competing prototype** — difference between top-1 and top-2 similarity
- **Entropy of the full 286-class similarity distribution** — computable from `similarities`
- **384-d DINOv2 CLS token** (`features`) — available before projection

---

## Trust Signals

### 1. Top-k Probability Breakdown

- **Description**: A ranked list of the model’s most likely class predictions with their similarity scores. This is the simplest and most direct trust signal: if the top candidate has a score of 0.70 and the second has 0.54, the model is reasonably confident in its first choice.
- **Data Required**: `top_k` array with `class_name` and `confidence` per candidate.
- **Available Now**: **Yes** — already returned by `/classify` and `/segment`.
- **UX Pattern**:
  - **Horizontal bar chart** showing each candidate class as a labeled bar with similarity score.
  - **Donut / segmented gauge** showing top-1 score as a filled arc with color band (green ≥ 0.6, amber 0.35–0.6, red < 0.35).
  - **Expandable “alternatives” panel** beneath the primary prediction, similar to search-engine suggestion UI.
- **Precedents**:
  1. **PathAI AISight Dx** — Algorithm Impressions display ranked quantitative outputs (e.g., biomarker percentages) in categorized tables, letting pathologists scan multiple candidate results at once.
  2. **Paige OmniScreen** — Predicted genomic alteration panels show ranked probability bars for each gene alteration, with confidence thresholds color-coded into actionable tiers.

---

### 2. Calibration Curve / Reliability Diagram

- **Description**: A plot that checks whether the model’s confidence scores are *calibrated* — i.e., when the model says “0.80 confidence,” does it actually get 80% of those predictions correct? In a prototypical network with cosine similarities, the raw similarity score is not a true probability; a reliability diagram reveals how much the score can be trusted as a frequency estimate.
- **Data Required**: A held-out validation set of `(image, true_label)` pairs plus the model’s predicted confidence for each. Requires batch offline computation of ECE (Expected Calibration Error) and per-bin accuracy.
- **Available Now**: **Partial** — the model outputs confidence scores, but no calibration dataset or temperature scaling is currently computed.
- **UX Pattern**:
  - **Static reliability diagram** (x-axis: binned predicted confidence, y-axis: observed accuracy) with a 45° diagonal reference line and gap bars showing over/under-confidence per bin.
  - **Interactive “Calibrate”-style view** (Appleby et al., 2022): learned reliability diagram with histogram of prediction density beneath, letting users brush a confidence region and inspect instances inside it.
  - **Inline calibration badge** next to the confidence score: “Calibrated” (green) / “Overconfident” (amber) / “Underconfident” (blue) based on pre-computed bin statistics.
- **Precedents**:
  1. **Nature Medicine (2025)** — AI breast-cancer risk model publishes calibration curves with 95% confidence bands and loess-smoothed curves so clinicians can see how predicted malignancy risk maps to real outcomes.
  2. **scikit-learn `CalibrationDisplay`** — Standard reliability-diagram visualization used in clinical ML tutorials and medical-AI reproducibility checklists (e.g., BMJ Health & Care Informatics).

---

### 3. Distance to Predicted Prototype

- **Description**: In a cosine-similarity classifier, the confidence score *is* the cosine similarity to the winning prototype. Because both vectors are L2-normalized, the angular distance is `arccos(confidence)` and the Euclidean distance in the unit sphere is `sqrt(2 - 2·confidence)`. A smaller distance means the query embedding points in nearly the same direction as the prototype — the model is “close” to a known archetype.
- **Data Required**: The top-1 similarity score (already returned). Optionally, the raw 128-d embedding and the winning prototype vector to compute exact distance metrics.
- **Available Now**: **Yes** — `confidence` is the cosine similarity; distance is a one-line transform.
- **UX Pattern**:
  - **Radial gauge / polar plot** showing angular distance from the prototype (0° = identical, 90° = orthogonal, > 90° = opposite).
  - **“Proximity ring” overlay** around the glyph thumbnail: a thin colored ring whose thickness or saturation encodes closeness to the prototype.
  - **Numeric badge**: “0.12 rad from *atl* prototype” with a tooltip explaining that smaller angles mean stronger visual alignment.
- **Precedents**:
  1. **ProtoPNet (Chen et al., NeurIPS 2019)** — Prototype networks visualize the similarity score between an image patch and the nearest prototype as a heatmap overlay; the raw distance score is the core interpretability signal.
  2. **ScatterUQ (MIT Lincoln Labs, 2023)** — Distance-aware neural networks plot the semantic distance between a test sample and its nearest prototype (or training example) in a 2-D scatter view with confidence contours.

---

### 4. Distance to Nearest Competing Prototype (Margin / Separation)

- **Description**: The *margin* is the gap between the top-1 similarity and the top-2 similarity. A large margin means the model sees a clear winner; a tiny margin means the top two classes are nearly tied and the prediction is ambiguous. This is one of the most actionable trust signals for expert users because it directly quantifies “how close was the runner-up?”
- **Data Required**: Top-1 and top-2 cosine similarity values.
- **Available Now**: **Yes** — `top_k[0].confidence` and `top_k[1].confidence` are already returned.
- **UX Pattern**:
  - **“Margin bar”** — a thin horizontal bar showing top-1 and top-2 scores as stacked or adjacent blocks; the gap between them is the visual focus.
  - **Diverging arrow graphic** — two arrows pulling away from a center point; arrow length = similarity, gap width = margin.
  - **Inline warning chip** — when margin < 0.10, show an amber “Ambiguous — close runner-up” chip with the runner-up class name.
- **Precedents**:
  1. **Paige TissueMap** — When AI detects two adjacent cancer subtypes with similar probability, the UI highlights the boundary region and shows a side-by-side comparison of confidence scores for each subtype, letting the pathologist decide.
  2. **ScatterUQ (MIT Lincoln Labs)** — The “low-confidence in-distribution” use case explicitly visualizes test samples that sit near the decision boundary by comparing them to training examples from the top-2 competing classes.

---

### 5. Entropy of Top-k Distribution

- **Description**: Shannon entropy measures how “spread out” the model’s top-k similarity scores are. High entropy means the model is uncertain (scores are flat across candidates); low entropy means the model is confident (one candidate dominates). For a prototypical network, entropy is computed over the softmax-normalized similarities or directly over the top-k similarities.
- **Data Required**: The full vector of 286 similarities (or at least the top-k values). Entropy = `-Σ p_i · log(p_i)` where `p_i` can be `softmax(similarity_i / τ)`.
- **Available Now**: **Partial** — only top-3 similarities are returned by default. The backend computes all 286 similarities internally (`torch.mm(embedding, prototypes.t())`) but discards them.
- **UX Pattern**:
  - **Entropy badge** — a small pill showing “Low / Moderate / High uncertainty” derived from entropy thresholds.
  - **Thermometer / vertical bar** — color gradient from cool (low entropy, confident) to hot (high entropy, uncertain).
  - **Uncertainty distribution plot** (Nkululeko style) — histogram of entropy values for the dataset, with the current sample’s position marked; correct predictions typically cluster on the left, incorrect ones on the right.
- **Precedents**:
  1. **Nkululeko (speech-emotion ML toolkit)** — Automatically generates `uncertainty_distribution.png` showing entropy histograms for correct vs. incorrect predictions, with a user-configurable `uncertainty_threshold` that filters predictions.
  2. **Hercules (IEEE TII 2022)** — MC-dropout uncertainty quantification module produces entropy-based uncertainty scores that are visualized alongside Grad-CAM heatmaps to flag ambiguous cases for clinician review.

---

### 6. OOD / Rejection Signal

- **Description**: A binary or graded indicator that the input image is “out of distribution” — visually unlike anything the model was trained on. The current system uses a hard threshold (0.35): if the top-1 similarity is below this, the sample is rejected as unknown. A richer signal would show *how far* below the threshold the sample is (graded OOD severity) and the nearest in-distribution class distance.
- **Data Required**: Top-1 similarity score and the rejection threshold. For graded severity, the full 286 similarities help compute the maximum possible similarity (i.e., even the “best” match is poor).
- **Available Now**: **Yes** — `rejected` boolean is returned; threshold is fixed at 0.35.
- **UX Pattern**:
  - **Traffic-light badge** — Green (accepted, > 0.60), Amber (marginal, 0.35–0.60), Red (rejected, < 0.35).
  - **“Unknown glyph” panel** — When rejected, show a clear “Needs human review” banner with the top-k candidates grayed out and a note: “Model does not recognize this element. Closest match: *atl* (0.32).”
  - **Grayscale fade on thumbnail** — Rejected elements are rendered at 50% opacity or with a dashed border to visually distinguish them from confident detections.
- **Precedents**:
  1. **Paige.ai FullFocus** — FDA-cleared viewer uses confidence bands to flag regions below diagnostic thresholds, automatically graying out low-confidence regions so pathologists focus on high-certainty areas.
  2. **ScatterUQ (MIT Lincoln Labs)** — Explicit “OOD Use Case” filters samples into an outlier view, showing the test sample alongside its closest in-distribution training examples with an outlier score slider.

---

### 7. Gradient-based Saliency

- **Description**: A heatmap overlaid on the input image showing which pixels most influenced the model’s prediction. Methods like Grad-CAM, Grad-CAM++, or ScoreCAM backpropagate the class score through the convolutional layers to produce a spatial attribution map. For the DINOv2-ViT backbone, ViT-specific methods (e.g., attention rollout, LRP for transformers, or rollout with attribution) are needed because ViTs lack convolutional feature maps.
- **Data Required**: Access to the backbone’s attention maps or gradient flow. For DINOv2 ViT, this requires hooking into the transformer blocks and computing attention rollout or gradient-weighted attention aggregation. The 384-d CLS token alone is not enough.
- **Available Now**: **No** — The backbone is treated as a frozen black box; no attention or gradient hooks are exposed in the current API.
- **UX Pattern**:
  - **Semi-transparent heatmap overlay** (JET or viridis colormap) on the glyph crop, with high-intensity regions indicating influential image areas.
  - **Side-by-side comparison** — Original crop on the left, saliency overlay on the right, with a slider to fade between them.
  - **Paige-style “opaque cancer” adaptation** — Instead of a rainbow heatmap, keep the influential regions fully opaque and lighten/de-saturate non-influential regions. This avoids obscuring fine glyph details that scholars need to see.
- **Precedents**:
  1. **XBoundNet++ (OpenReview 2024)** — Custom Grad-CAM produces layer-wise attention maps for kidney ablation segmentation, showing clinicians exactly which image regions drove the boundary prediction.
  2. **Nature Machine Intelligence (2022)** — “Benchmarking saliency methods for chest X-ray interpretation” establishes Grad-CAM as the clinical standard for saliency visualization, despite noted limitations for small or complex pathologies.

---

### 8. Feature Contribution / Prototype Activation

- **Description**: Because the classifier is a prototypical network, the prediction is fundamentally a comparison between the query embedding and each class prototype. Instead of (or in addition to) pixel-level saliency, we can show *which prototypes are activated* and *how strongly*. For the Codex model, this means showing the similarity score to every prototype, perhaps highlighting the top-N prototypes and their associated class names. A more advanced version would project the 128-d embedding dimensions that contribute most to the winning similarity score.
- **Data Required**: The full `(286,)` similarity vector. Optionally, the raw 128-d embedding and the `(286, 128)` prototype matrix to compute per-dimension contributions (e.g., `embedding · prototype_j` element-wise products).
- **Available Now**: **Partial** — all 286 similarities are computed internally but only top-3 are returned. The prototype matrix is loaded server-side but not exposed through the API.
- **UX Pattern**:
  - **Prototype activation bar chart** — all 286 classes as a sortable, filterable bar chart; the top match is pinned at the top, with a zoomed-in view of the top 10.
  - **“This looks like that” panel** (ProtoPNet style) — Show the query crop next to the training-image patch that is nearest to the winning prototype, with a similarity score. Scholars can visually verify “this glyph looks like that known example.”
  - **Polar / radar chart** of top-k prototypes — each axis is a candidate class, radial distance = similarity.
- **Precedents**:
  1. **ProtoPNet (Chen et al., NeurIPS 2019)** — The canonical prototype-network visualization: for each activated prototype, the model shows the training patch that the prototype represents, placed next to the query image patch with the highest activation. This is case-based reasoning made visual.
  2. **ProtoConcepts (NeurIPS 2023)** — Extends ProtoPNet by visualizing each prototype with *multiple* training patches (a “prototypical ball”), reducing ambiguity about what the prototype means and letting users infer the shared visual concept.

---

## Recommended Signals to Ship (MVP)

| Priority | Signal | Why Ship It |
|----------|--------|-------------|
| **P0** | **Top-k Breakdown** | Already in the API; zero backend work. Front-end only. Gives scholars immediate alternatives when the top-1 feels wrong. |
| **P0** | **OOD / Rejection Signal** | Already in the API (`rejected` boolean). Front-end only. Critical for expert trust: knowing *when* the model admits ignorance is as important as knowing what it predicts. |
| **P0** | **Margin (Distance to Runner-up)** | Computable from existing `top_k` data (gap between #1 and #2). Front-end only. The single most actionable ambiguity signal for experts. |
| **P1** | **Distance to Predicted Prototype** | Computable from existing `confidence` score (`arccos` or `sqrt(2-2·confidence)`). Front-end only. Translates the abstract similarity score into an intuitive geometric distance. |
| **P1** | **Entropy of Top-k Distribution** | Requires returning more than top-3 similarities (or computing entropy server-side). Small backend change. Provides a principled, single-number uncertainty metric. |

**Rationale**: The P0 signals require **zero backend changes** — they are all derivable from data already returned by `/classify`. They can be implemented entirely in the front-end and provide immediate expert value. The P1 signals need minor backend additions (expose more similarities or compute a derived metric server-side) but still need no model retraining.

---

## Deferred Signals

| Signal | Blocker | Complexity |
|--------|---------|------------|
| **Calibration Curve** | Needs a held-out validation set and offline ECE computation. Also requires periodic re-computation as data drifts. | Medium — mostly data pipeline, not model. |
| **Gradient-based Saliency** | DINOv2 ViT backbone requires ViT-specific attribution methods (attention rollout, transformer LRP, or token-level gradients). Current backend treats backbone as a black box with no hooks. | High — requires architectural changes to expose attention/gradients and significant front-end work for overlay rendering. |
| **Full Prototype Activation (all 286)** | Requires returning the full similarity vector (286 floats ≈ 1.1 KB per element). Trivial computationally but increases payload size 100×. | Low-Medium — backend change + front-end list/chart view. Can be deferred until scholars ask for “what else did it consider?” |
| **“This Looks Like That” Patch Retrieval** | Requires pre-computing and storing the nearest training-image patch for each prototype, plus a lookup at inference time. | Medium — requires dataset access and an offline indexing step. |

---

## Data Requirements Summary

| Signal | Data Needed | Backend Change Needed? | Effort |
|--------|-------------|------------------------|--------|
| Top-k Breakdown | `top_k` array | **No** | Front-end only |
| OOD / Rejection | `confidence`, `rejected` | **No** | Front-end only |
| Margin (Runner-up gap) | `top_k[0]`, `top_k[1]` | **No** | Front-end only |
| Distance to Prototype | `confidence` (derive `arccos`) | **No** | Front-end only |
| Entropy | All 286 similarities (or top-N > 3) | **Yes** — return `all_similarities` or server-side entropy | Small |
| Calibration Curve | Validation set + ECE script | **Yes** — offline calibration dataset + API endpoint | Medium |
| Gradient Saliency | DINOv2 attention/gradient hooks | **Yes** — expose ViT attribution hooks | High |
| Prototype Activation (full) | All 286 similarities | **Yes** — return full vector | Small |
| “This Looks Like That” | Nearest training patch per prototype | **Yes** — offline index + retrieval endpoint | Medium |

---

## References

1. Chen, C., Li, O., Tao, D., Barnett, A., Rudin, C., & Su, J. K. (2019). *This Looks Like That: Deep Learning for Interpretable Image Recognition.* NeurIPS.
2. Nauta, M., van Bree, R., & Seifert, C. (2021). *Neural Prototype Trees for Interpretable Fine-Grained Image Recognition.* CVPR.
3. Appleby, A., et al. (2022). *Calibrate: Interactive Reliability Diagrams.* IEEE VIS Short Papers.
4. Adams, R. P., et al. (2021). *Uncertainty-aware Visualization in Medical Imaging — A Survey.* Computer Graphics Forum.
5. XBoundNet++ (OpenReview 2024). Kidney ablation segmentation with Grad-CAM, probability maps, and MC-dropout uncertainty.
6. ScatterUQ (MIT Lincoln Labs, 2023). *Interactive Uncertainty Visualizations for Multiclass Deep Learning.* arXiv:2308.04588.
7. Hercules (IEEE TII, 2022). *Deep Hierarchical Attentive Multi-Level Fusion with Uncertainty Quantification.*
8. Paige.ai Blog (2023). *Designing Products of Empowerment: The UX Approach at Paige.*
9. PathAI AISight Dx v2.19 Release Notes (2026). Guided Algorithm Review, Algorithm Impressions, multi-algorithm workflows.
10. Nkululeko Documentation. *Using Uncertainty in Predictions.*
11. Hollance (2020). *Reliability Diagrams.* GitHub: `hollance/reliability-diagrams`.
12. ProtoConcepts (NeurIPS 2023). *Interpretable Image Classification with Prototypical Concepts.*
13. ProtoPFaith (NeurIPS 2023). *Faithful Explanations in Convolutional Neural Networks for Case-Based Reasoning.*
14. BUCAN (2026). *Bayesian Uncertainty-aware Classification with Attention Networks for Medical Images.*
15. Deep Evidential Learning for Radiotherapy (2024). Uncertainty heatmaps and DVH confidence bands.
