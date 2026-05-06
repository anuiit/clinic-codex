## 2026-05-05
- WorkspacePage overlay alignment stays correct under CSS zoom when the overlay canvas internal resolution matches `currentRecord.result.image_size` and boxes are drawn in original bbox coordinates.
- Canvas hit-testing remains zoom-safe by using `event.nativeEvent.offsetX/offsetY` directly against original bbox coordinates instead of remapping through rendered image bounds.
- Global file import works cleanly when drag-and-drop and file-picker flows share one modal state machine (`modalOpen`, `modalFile`, preview URL, dimensions) instead of separate inline upload UI.
- Revoking object URLs from a ref-backed preview URL on modal close/unmount avoids preview leaks while preserving a separate base64 image payload for saved analysis records.
