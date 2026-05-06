
## [2026-05-06] Task: T7 — Venv corruption (pre-existing)
- Multiple packages had empty `__init__.py` files: jinja2, cv2 (opencv-python-headless), pydantic, pydantic-core, typing-inspection
- scipy had a truncated `.so` file (`libscipy_openblas-*.so: file too short`)
- Root cause: these were corrupted BEFORE our work (old venv state from T1 triage)
- Fix: `pip install --no-cache-dir --prefer-binary --force-reinstall <pkg>` for each
- install.sh sanity check correctly caught the failures — the script logic is sound
- After force-reinstall of all 5 corrupted packages: `all imports OK`
- torch.version.cuda = None (CPU-only confirmed)
- Backend /classes endpoint: HTTP 200, returned 286 class names
