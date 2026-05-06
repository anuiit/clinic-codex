# T16 Browser QA Learnings

## Date: 2026-05-05

### Dev Server
- Vite dev server started on port 5200 (5173-5175 were occupied by other projects)
- Used `setsid` to daemonize the process so it survives bash session end

### QA Results
- Empty workspace loads correctly with compact chrome (no KPI cards)
- History toggle (Collapse/Expand) works correctly
- File input (`<input type="file">`) is present for upload flow
- `/annotate/fake-id-12345` shows graceful "Analysis not found" message
- Tab keyboard navigation works without crash
- No real console errors on page load (CORS errors are expected when backend not running)

### Zoom Controls
- Zoom controls (ZoomIn/ZoomOut/Fit) are implemented in WorkspacePage.tsx (lines 778-786)
- They only render when `currentRecord` is set (image loaded) — this is correct behavior
- Cannot test zoom without backend since image upload requires backend processing

### CORS Note
- Backend hardcodes `http://localhost:5173` in CORS config
- Running on port 5200 triggers CORS errors — these are NOT real errors, just port mismatch
- On the canonical port 5173, CORS would work fine

### Build
- `npm run build` passes: exit 0, 305.78 kB JS bundle, 26.86 kB CSS
