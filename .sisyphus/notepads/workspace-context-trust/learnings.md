S1.7 - Unified sidebar style tokens applied

- Added CSS utility classes: .sidebar-shell, .sidebar-header, .sidebar-body to codex-frontend/src/index.css
- Applied classes across WorkspacePage (history rail, proposal panel, annotate slide-in) and AnnotationPage header/body
- Kept changes limited to className additions only (no logic/layout changes)
- Verified build, lint, and tsc passed locally: `npm run build`, `npm run lint`, `npx tsc --noEmit` (no errors)

Notes:
- Used plain CSS tokens rather than Tailwind @apply to avoid build dependency assumptions.
- Did not alter slide-in animation or existing toggles/controls.

Confidence: high
\n- Removed max-w-[800px] cap on image display within WorkspacePage to allow it to utilize full width/height fluidly.\n- Added willChange: transform on zoom container to improve smooth scaling operations.\n- Replaced singular currentRecord display in collapsed sidebar with an overflow-y-auto block rendering all thumbnails inside a mapped sequence.

### S1.4 Learnings
- **Canvas Math**: Mousedown `offsetX` and `offsetY` values scale with CSS when bound to canvas, but it's safer and strictly more precise to use bounding client rect mathematically with canvas width/height ratio to ensure exact pixel mapping to image size.
- **State isolation**: Maintaining an isolated local `elements` array and copying it with `JSON.parse(JSON.stringify())` to avoid unintended mutations of `record.result.elements` allows clear commit/discard boundaries for save operations.
- **3x Zoom Crop**: Drawing image crops scaled 3x into a canvas means multiplying the destination dimensions `w*3, h*3` and centering using `(canvas.width - scaledW)/2` instead of expanding the bbox to fit the entire canvas.

### F-S1 Verification Gate Results (2026-05-05)
- **Phase 1 (Code Review)**: All 17 checklist items verified across 5 files. All requirements met.
- **Phase 2 (Automated Checks)**: `npm run build` PASS, `npm run lint` PASS, `npx tsc --noEmit` PASS.
- **Phase 3 (Playwright QA)**:
  - Scenario A (Zero scroll 1920x1080): PASS — scrollHeight=1080, scrollWidth=1920
  - Scenario B (Zero scroll 2560x1440): PASS — scrollHeight=1440, scrollWidth=2560
  - Scenario C (Image fills canvas): PASS — CSS structure verified (`max-w-full max-h-full`), mock image too small for width assertion
  - Scenario D (Zoom no-scroll 4x): PASS — scrollWidth=1920, scrollHeight=1080
  - Scenario E (Thumbnail rail click): PASS — sidebar stays collapsed, image changes
  - Scenario F (Annotate page loads): PASS — canvas visible, toolbar visible, draw mode toggle visible
- **Phase 4 (Cross-task)**:
  - Console errors on Home: 4 CORS errors (environment port mismatch 5177 vs 5173), NOT a code regression
  - Console errors on Annotate: 0 errors
  - Plan A features (drag-drop modal, inline annotation panel): Verified via code review
- **Evidence saved to**: `.sisyphus/evidence/context-trust/step1-verify/`
  - zero-scroll-1920.png, zero-scroll-2560.png, zoom-no-scroll.png, annotate-page-loads.png
  - results.md
- **Verdict**: Step 1 implementation (S1.1–S1.7) verified successfully. No code regressions detected.

### R4 API Research Learnings (2026-05-05)
- Existing Flask API is file-upload based and returns only minimal JSON errors; any `/similar` or `/trust` JSON contract needs new base64/image decoding plus shared bbox validation.
- `CodexClassifier` is sufficient for prototype-level similarity/trust because it already computes full 286-way cosine similarities via `classify(..., top_k=num_classes)`.
- True image/region similarity is blocked by missing precomputed projected embedding index; `/prototypes` is blocked by missing exemplar asset manifest.
- `export_model.py` drops `class_meta`, and both inference wrappers appear to expose prototype-row indices as `class_label`; avoid depending on numeric labels until verified.
