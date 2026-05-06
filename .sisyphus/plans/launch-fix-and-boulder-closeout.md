# Clinic Codex — Launch-Fix & Boulder Close-Out

## TL;DR

> **Quick Summary**: The user has been crashed by `scripts/install.sh` multiple times on WSL (during `pip install`). Root cause: `torch>=2.1.0` with no CPU-only index pulls 3+GB of CUDA wheels, plus `git+https://...` packages trigger source builds during pip resolution, plus a monolithic install with no staging. Fix: pin CPU-only torch, replace git+https with PyPI versions, stage the install in 4 phases with `--no-cache-dir --prefer-binary`, harden `run-dev.sh` with pre-flight checks, and write a manual launch runbook as the *primary* documented path. Then close out the stale `codex-cleanup-and-polish` boulder.
>
> **Deliverables**:
> - Hardened `scripts/install.sh` (staged, Python-3.11-locked, CPU-only torch, idempotent npm)
> - Hardened `scripts/run-dev.sh` (pre-flight venv + port checks, no `source activate`, fixed `\n` echo bug)
> - Updated `backend/requirements.txt` (CPU torch index, PyPI mobile-sam/segment-anything, no git+https)
> - Updated `scripts/install.ps1` (Windows parity)
> - New `INSTALL.md` "Manual launch (primary path)" section — the runbook the user can copy-paste even if scripts fail
> - Reconciled `.sisyphus/plans/codex-cleanup-and-polish.md` (checkboxes 15-21 → checked)
> - Updated `.sisyphus/boulder.json` (boulder marked complete; switch active to this plan)
> - Verified end-to-end: backend `/classes` returns 200, frontend loads at :7118
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES — 3 waves
> **Critical Path**: Wave 1 (env triage + requirements.txt fix) → Wave 2 (install.sh + run-dev.sh + install.ps1 in parallel) → Wave 3 (manual runbook + end-to-end verification + boulder reconciliation)

---

## Context

### Original Request (verbatim, French + intent)
> "1 and 3: fix the script not working or give me concrete steps to launch the app. Do not crash the WSL environment (happened multiple times earlier because of the installation script for some reason)."

Combines: (1) close out the active `codex-cleanup-and-polish` boulder pragmatically (skip F1-F4 per user direction) and (3) deliver concrete, safe launch instructions that don't crash WSL.

### Environment Probe (verified 2026-05-06)

| Check | Status |
|---|---|
| OS | WSL2 Linux 6.6.87.2 on `DESKTOP-1SP5H8H` |
| RAM | 15Gi total, 12Gi available, 4Gi swap |
| System python3 | **3.14.3** (Linuxbrew) — ⚠️ no torch wheels exist for 3.14 |
| Venv python | 3.11 at `backend/.venv/bin/python` ✅ |
| Node | v22.22.0, npm 11.12.1 ✅ |
| Frontend `node_modules` | 272MB, complete ✅ |
| Model weights | All 4 files present ✅ |

### Venv State (the crash evidence)

**Installed in venv** (working): numpy 2.4.4, PIL, scipy, yaml, tqdm
**Installed but broken**: mobile_sam 1.0 (import fails)
**MISSING**: flask, torch, torchvision, albumentations, segment_anything
**Bloat (unused)**: cuda-toolkit 13.0.2, nvidia-cublas, nvidia-cudnn, nvidia-nccl, nvidia-cufft (~GBs of CUDA wheels pulled by previous `pip install torch` with default index)

### Crash Root Cause Analysis (high confidence)

1. **`requirements.txt` line 1: `torch>=2.1.0`** without `--extra-index-url https://download.pytorch.org/whl/cpu` → pip pulls CUDA-12 wheels (~3GB)
2. **WIP changes added `git+https://github.com/.../MobileSAM.git`** and `git+https://.../segment-anything.git` → pip clones during resolve, runs setup.py, multiplies memory pressure
3. **`install.sh` does monolithic `pip install -r requirements.txt`** with no staging, no `--no-cache-dir`, no `--prefer-binary`, no progress checkpoints → if pip dies, no recovery
4. **System python3 is 3.14.3** — no torch wheels exist → if user re-creates venv, source compile attempted → guaranteed WSL meltdown
5. **`run-dev.sh` uses `set -e` + `source .venv/bin/activate`** → fails silently if venv broken; `echo "\n..."` doesn't expand `\n` (cosmetic but signals carelessness)

### Boulder Status (`codex-cleanup-and-polish`)

- **14/21 tasks committed to git** (1-14 done with full commits)
- **Tasks 15-21**: code on disk and committed (`d3daa4e`, `5e68110`, `335ffe2`, `ab0fd8f`, `2664e40`, `8b78935`, `1fa55b4`) — but plan checkboxes still show `[ ]` (stale state)
- **F1-F4 final verification**: never run — user explicitly says SKIP
- **WIP files uncommitted on master**: `.sisyphus/boulder.json`, `backend/requirements.txt`, `scripts/install.sh`, `scripts/run-dev.sh` — these are exactly the files we need to fix

### Self-Review Findings (gap analysis)

**Identified Gaps (addressed in plan)**:
- `backend/pyproject.toml` requires `>=3.11` AND pulls `mobile-sam` from git source via `[tool.uv.sources]` — pip won't honor this, but document the dual-tooling reality
- `backend/uv.lock` (253KB) shows previous `uv` usage — explicitly note in install.sh that we use pip not uv (avoid mixing)
- `pip install --upgrade pip` can itself hang — make it best-effort, time-bounded
- `set -e` in install.sh kills on any non-zero exit — replace with explicit error-checking on critical commands only
- Disk space not just memory — verify ≥2GB free before install
- Frontend `.env` exists alongside `.env.example` — must NOT be overwritten
- The user lost trust in scripts; **the manual runbook in INSTALL.md must become the primary documented path**, scripts secondary
- `mobile_sam` PyPI vs git source may have feature divergence — document fallback
- CI workflow uses old paths but is non-blocking — out of scope here

---

## Work Objectives

### Core Objective
Restore the user's ability to launch the Clinic Codex app on WSL2 with **zero risk of system crash**, by fixing the install/run scripts AND providing a manual fallback runbook. Then formally close the stale `codex-cleanup-and-polish` boulder.

### Concrete Deliverables
- `scripts/install.sh` (rewritten — staged, Python-locked, CPU-only torch, idempotent)
- `scripts/run-dev.sh` (rewritten — defensive, no `source activate`, pre-flight checks)
- `scripts/install.ps1` (updated — Windows parity)
- `backend/requirements.txt` (CPU torch index, PyPI packages, no git+https)
- `INSTALL.md` (new "Manual Launch — Primary Path" section)
- `.sisyphus/plans/codex-cleanup-and-polish.md` (checkboxes 15-21 → `[x]`)
- `.sisyphus/boulder.json` (active_plan switched, old plan archived in completed list)
- Evidence files in `.sisyphus/evidence/launch-fix/` proving each script works

### Definition of Done
- [ ] `bash scripts/install.sh` on this machine completes without crashing WSL
- [ ] `bash scripts/run-dev.sh` starts both servers; `curl http://localhost:7117/classes` returns 200; `curl http://localhost:7118/` returns frontend HTML
- [ ] Manual runbook in `INSTALL.md` produces the same result, copy-pasted line by line
- [ ] Old boulder marked complete in `boulder.json`; this plan registered as new active

### Must Have
- CPU-only torch install (no CUDA wheel downloads)
- Staged install with checkpoint messages between stages
- Explicit Python 3.10/3.11 detection (refuse 3.14 with clear message)
- Pre-flight checks in `run-dev.sh` (venv exists, ports free)
- Manual runbook that works without ANY scripts
- Each script tested end-to-end on this WSL2 box before marking task done

### Must NOT Have (Guardrails)
- ❌ Any `pip install torch` that could pull CUDA wheels
- ❌ Any `git+https://...` in `requirements.txt`
- ❌ Source compilation (always `--prefer-binary`)
- ❌ `source .venv/bin/activate` in shell scripts (call `.venv/bin/python` directly)
- ❌ Touching tasks 1-14 of the cleanup boulder (they're done and committed)
- ❌ Running F1-F4 final verification wave (user explicitly skipped)
- ❌ Touching `pyproject.toml`, `uv.lock`, or trying `uv` (pip-only path)
- ❌ Overwriting user's existing `frontend/.env` or `backend/.env`
- ❌ New features, refactors, embedding work (out of scope)
- ❌ Any single command that could exceed 2GB RAM during install
- ❌ AI-slop: excessive comments, defensive checks for things that can't fail, abstraction wrappers

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — all verification is agent-executed.

### Test Decision
- **Infrastructure exists**: Partial (vitest+pytest scaffolds committed in tasks 17-18, but not the focus here)
- **Automated tests**: NONE for this plan (out of scope — fixing scripts, not adding tests)
- **Verification mode**: 100% agent-executed QA scenarios via Bash + actual script execution
- **Why no tests**: shell scripts are best verified by running them; adding bats/shellcheck tests is scope creep

### QA Policy
Every task includes ONE primary QA scenario that runs the actual deliverable end-to-end on this WSL2 environment. Evidence saved to `.sisyphus/evidence/launch-fix/task-{N}-*.{ext}`.

- **Shell scripts**: Run with `bash -x` capturing output to log file
- **Backend launch**: `curl http://localhost:7117/classes` returns JSON with `num_classes: 286`
- **Frontend launch**: `curl http://localhost:7118/` returns HTML containing `<div id="root">`
- **WSL safety**: Monitor `free -m` before/after install; abort if available memory drops below 1GB

### Final Verification (lightweight, NOT F1-F4)
A single end-to-end smoke test in Wave 3 that:
1. Wipes `backend/.venv` to simulate fresh install
2. Runs `bash scripts/install.sh` to completion
3. Runs `bash scripts/run-dev.sh` in background
4. Hits both endpoints
5. Cleanup: kills servers, restores venv if needed

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation - run sequentially, blocks all others):
└── Task 1: Triage existing venv + clean CUDA bloat decision

Wave 2 (Script fixes - MAX PARALLEL after Wave 1):
├── Task 2: Fix backend/requirements.txt (CPU torch + PyPI packages) [quick]
├── Task 3: Rewrite scripts/install.sh (staged, defensive) [quick]
├── Task 4: Rewrite scripts/run-dev.sh (pre-flight, no activate) [quick]
└── Task 5: Update scripts/install.ps1 (Windows parity) [quick]

Wave 3 (Documentation + verification + close-out):
├── Task 6: Add Manual Launch runbook to INSTALL.md [quick]
├── Task 7: End-to-end smoke test on this WSL box [unspecified-high]
└── Task 8: Reconcile boulder + plan checkboxes + boulder.json [quick]

Critical Path: Task 1 → Task 2 → Task 3 → Task 7 → Task 8
Parallel Speedup: ~50% vs sequential
Max Concurrent: 4 (Wave 2)
```

### Dependency Matrix

| Task | Depends On | Blocks |
|---|---|---|
| 1 | — | 2, 3, 4, 5 |
| 2 | 1 | 3, 7 |
| 3 | 2 | 7 |
| 4 | 1 | 7 |
| 5 | 2 | (Windows-only, doesn't block 7) |
| 6 | 1, 2, 3, 4 | 7 |
| 7 | 2, 3, 4, 6 | 8 |
| 8 | 7 | (final) |

### Agent Dispatch Summary

| Wave | Tasks | Categories |
|---|---|---|
| 1 | T1 | `quick` (read/inspect, no writes) |
| 2 | T2-T5 | T2-T5 → `quick` (small file edits) |
| 3 | T6 → `writing`, T7 → `unspecified-high` (multi-step verification), T8 → `quick` |

---

## TODOs

- [x] 1. Triage venv + decide CUDA bloat strategy (READ-ONLY inspection task)

  **What to do**:
  - Run `backend/.venv/bin/pip list --format=freeze > /tmp/venv-before.txt` — snapshot current state
  - Run `du -sh backend/.venv` — baseline size
  - Run `backend/.venv/bin/python -c "import flask, torch, mobile_sam, segment_anything, albumentations" 2>&1 | head -20` — confirm what's broken
  - Run `df -h /home/sina | tail -1` — confirm ≥2GB free
  - Decision: Keep venv (incremental repair) vs. wipe-and-rebuild
    - **Default**: KEEP venv. Incremental install only adds missing packages. Faster, lower crash risk.
    - **Wipe only if**: venv python version != 3.10 or 3.11, OR venv size > 5GB
  - Document decision in `.sisyphus/evidence/launch-fix/task-1-triage.md`
  - **DO NOT** run any `pip install` in this task. Inspection only.

  **Must NOT do**:
  - No `pip uninstall` (could cascade into broken state)
  - No `rm -rf .venv` (user would lose 5+ minutes of progress)
  - No installing anything

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Pure read/snapshot work, no decision logic beyond a 2-branch check
  - **Skills**: none
  - **Skills Evaluated but Omitted**:
    - `playwright`: no UI involved
    - `git-master`: no git ops in this task

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 1 (sequential, blocks Wave 2)
  - **Blocks**: Tasks 2, 3, 4, 5
  - **Blocked By**: None — can start immediately

  **References**:

  **Pattern References**:
  - `backend/.venv/bin/python` — call interpreter directly, never `source activate`
  - `scripts/run-dev.sh:36` — current bad pattern (`source .venv/bin/activate`) — DO NOT replicate

  **External References**:
  - `pip` docs: https://pip.pypa.io/en/stable/cli/pip_list/ — `--format=freeze` produces requirements.txt-compatible output

  **WHY Each Reference Matters**:
  - Snapshot via `pip list --format=freeze` lets us diff before/after if anything goes wrong — recovery aid
  - Calling `.venv/bin/python` directly avoids subshell activation issues that have caused silent failures

  **Acceptance Criteria**:
  - [ ] `/tmp/venv-before.txt` exists and has ≥10 lines
  - [ ] `.sisyphus/evidence/launch-fix/task-1-triage.md` exists with sections: "Venv Size", "Python Version", "Missing Packages", "Decision (KEEP|WIPE)", "Rationale"
  - [ ] Free disk verified ≥2GB

  **QA Scenarios**:

  ```
  Scenario: Venv triage produces actionable report
    Tool: Bash
    Preconditions: backend/.venv exists with python3.11
    Steps:
      1. mkdir -p .sisyphus/evidence/launch-fix
      2. backend/.venv/bin/pip list --format=freeze > /tmp/venv-before.txt
      3. backend/.venv/bin/python --version > /tmp/venv-pyver.txt
      4. du -sh backend/.venv > /tmp/venv-size.txt
      5. backend/.venv/bin/python -c "import flask" 2>&1 | tee /tmp/import-flask.txt
      6. Write evidence file with all 4 facts + decision
    Expected Result: Evidence file lists at least flask, torch, segment_anything as missing; venv python is 3.11.x; decision = KEEP
    Failure Indicators: Venv python is 3.14, OR venv >5GB → decision must be WIPE
    Evidence: .sisyphus/evidence/launch-fix/task-1-triage.md
  ```

  **Evidence to Capture**:
  - [ ] `.sisyphus/evidence/launch-fix/task-1-triage.md` (decision report)
  - [ ] `/tmp/venv-before.txt` (package snapshot for rollback reference)

  **Commit**: NO (inspection only, no file changes)

- [x] 2. Fix `backend/requirements.txt` — CPU torch index + PyPI SAM packages

  **What to do**:
  - Replace current contents (which includes WIP `git+https://github.com/ChaoningZhang/MobileSAM.git` and `git+https://github.com/facebookresearch/segment-anything.git`) with this exact structure:
    ```
    # CPU-only torch wheels — DO NOT REMOVE THIS LINE (prevents 3GB CUDA download → WSL OOM)
    --extra-index-url https://download.pytorch.org/whl/cpu

    # Core
    numpy>=1.24,<2.5
    pillow>=10.0
    pyyaml>=6.0
    scipy>=1.11
    tqdm>=4.66

    # Web
    flask>=3.0,<4.0
    flask-cors>=4.0

    # ML — CPU torch (resolved from extra-index-url above)
    torch>=2.1,<2.6
    torchvision>=0.16,<0.21

    # SAM family — PyPI only, NO git+https (causes pip resolver to clone+build → memory spike)
    segment-anything==1.0
    mobile-sam==1.0

    # Augmentation — pinned <2.0 (2.x has breaking API changes for our pipeline)
    albumentations>=1.4,<2.0

    # Vision backbones (DINOv2 via timm)
    timm>=0.9
    ```
  - Preserve any other lines that exist beyond these categories (read current file first; only replace ML/web/SAM-related lines)
  - Add a comment block at top documenting the CPU-only intent

  **Must NOT do**:
  - No `git+https://...` lines anywhere in the file
  - No `torch==2.x.x+cu121` (CUDA suffix)
  - No `--index-url` (replaces PyPI entirely — only `--extra-index-url` is safe)
  - No removing existing pinned versions for unrelated packages

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single-file targeted edit, well-defined contents
  - **Skills**: none

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T3, T4, T5 in Wave 2)
  - **Parallel Group**: Wave 2
  - **Blocks**: T3 (install.sh references this file), T7 (smoke test consumes this)
  - **Blocked By**: T1

  **References**:

  **Pattern References**:
  - `backend/requirements.txt` (current state) — read first to preserve any non-ML lines
  - `backend/pyproject.toml:9-22` — uv-managed dep list (mirrors what we want, but we're using pip path)

  **External References**:
  - PyTorch CPU wheels index: `https://download.pytorch.org/whl/cpu` — official, version-specific
  - mobile-sam PyPI: `https://pypi.org/project/mobile-sam/` — confirm version 1.0 exists
  - segment-anything PyPI: `https://pypi.org/project/segment-anything/` — confirm 1.0

  **WHY Each Reference Matters**:
  - The CPU torch index is the SINGLE most important line — without it pip pulls CUDA wheels (3GB) and WSL OOMs
  - PyPI versions of mobile-sam/segment-anything avoid `pip install git+https://...` which clones repos AND runs `setup.py` during dependency resolution, multiplying memory pressure 5-10x
  - The albumentations `<2.0` cap matches what `backend/codex_model/` expects (the WIP downgrade was intentional)

  **Acceptance Criteria**:
  - [ ] File contains exactly one `--extra-index-url https://download.pytorch.org/whl/cpu` line
  - [ ] `grep -E 'git\+https|cu1[12][0-9]|--index-url[^a-z]' backend/requirements.txt` returns ZERO matches
  - [ ] `grep -c '^[a-z]' backend/requirements.txt` ≥ 10 (at least 10 package lines)
  - [ ] File ends with newline

  **QA Scenarios**:

  ```
  Scenario: requirements.txt is well-formed and CUDA-safe
    Tool: Bash
    Preconditions: New requirements.txt written
    Steps:
      1. grep -c 'git+https' backend/requirements.txt → expect 0
      2. grep -c 'extra-index-url.*pytorch.org/whl/cpu' backend/requirements.txt → expect 1
      3. grep -E 'torch[<>=]' backend/requirements.txt → expect output containing torch and torchvision
      4. backend/.venv/bin/pip install --dry-run -r backend/requirements.txt 2>&1 | tee /tmp/dry-run.log
      5. grep -i 'cuda\|nvidia' /tmp/dry-run.log → expect 0 matches (no CUDA wheels resolved)
    Expected Result: Dry-run resolves cleanly with CPU-only wheels; zero CUDA references
    Failure Indicators: ANY `nvidia-*` or `cu12*` package appears in dry-run output → STOP, do not proceed to T3
    Evidence: .sisyphus/evidence/launch-fix/task-2-dry-run.log

  Scenario: File preserves non-ML content
    Tool: Bash
    Preconditions: Original requirements.txt backed up
    Steps:
      1. diff /tmp/venv-before.txt backend/requirements.txt | head -50
      2. Verify no unrelated packages were silently dropped
    Expected Result: Only ML/web/SAM-related lines changed; comments added at top
    Evidence: .sisyphus/evidence/launch-fix/task-2-diff.txt
  ```

  **Evidence to Capture**:
  - [ ] `.sisyphus/evidence/launch-fix/task-2-dry-run.log` (proves no CUDA resolution)
  - [ ] `.sisyphus/evidence/launch-fix/task-2-diff.txt` (proves no unintended changes)

  **Commit**: YES
  - Message: `fix(backend): pin CPU-only torch + PyPI mobile-sam to prevent WSL OOM`
  - Files: `backend/requirements.txt`
  - Pre-commit: `grep -c 'git+https' backend/requirements.txt` → must be 0

- [x] 3. Rewrite `scripts/install.sh` — staged, defensive, Python-3.11-locked

  **What to do**:
  Rewrite `scripts/install.sh` with this structure (NOT verbatim — adapt to project conventions found in current file):

  ```bash
  #!/usr/bin/env bash
  # Clinic Codex installer — staged, CPU-only, WSL-safe.
  # Each stage is checkpointed. If interrupted, re-run is safe (idempotent).

  set -u  # NOT set -e — we want explicit error handling per stage
  set -o pipefail

  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  cd "$REPO_ROOT"

  log()  { printf '\n[install] %s\n' "$*"; }
  fail() { printf '\n[install] ERROR: %s\n' "$*" >&2; exit 1; }

  # --- Stage 0: Pre-flight checks ---
  log "Stage 0/5: pre-flight checks"

  # Find Python 3.10 or 3.11 (refuse 3.12+, 3.14, system fallbacks)
  PYTHON=""
  for candidate in python3.11 python3.10; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON="$candidate"; break
    fi
  done
  [ -n "$PYTHON" ] || fail "Need python3.10 or python3.11. Install: sudo apt install python3.11 python3.11-venv"

  # Verify version (defense against shims)
  PYVER=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  case "$PYVER" in
    3.10|3.11) ;;
    *) fail "Found $PYTHON but reports version $PYVER. Need 3.10 or 3.11." ;;
  esac
  log "Using $PYTHON ($PYVER)"

  # Disk space check (need ~2GB)
  AVAIL_KB=$(df -k . | awk 'NR==2 {print $4}')
  [ "$AVAIL_KB" -gt 2000000 ] || fail "Need ≥2GB free disk. Have $((AVAIL_KB/1024))MB."

  # --- Stage 1: venv ---
  log "Stage 1/5: venv at backend/.venv"
  if [ ! -x backend/.venv/bin/python ]; then
    "$PYTHON" -m venv backend/.venv || fail "venv creation failed"
  else
    log "  reusing existing venv"
  fi
  PIP="backend/.venv/bin/pip"
  PY="backend/.venv/bin/python"

  # Best-effort pip upgrade (don't fail install if this fails)
  "$PIP" install --quiet --upgrade pip 2>/dev/null || log "  pip upgrade skipped (network or permission)"

  # --- Stage 2: core utils (small, fast, low risk) ---
  log "Stage 2/5: core utils (numpy, pillow, scipy, pyyaml, tqdm)"
  "$PIP" install --no-cache-dir --prefer-binary \
    "numpy>=1.24,<2.5" "pillow>=10.0" "pyyaml>=6.0" "scipy>=1.11" "tqdm>=4.66" \
    || fail "Stage 2 failed — core utils"

  # --- Stage 3: web (flask) ---
  log "Stage 3/5: flask + flask-cors"
  "$PIP" install --no-cache-dir --prefer-binary \
    "flask>=3.0,<4.0" "flask-cors>=4.0" \
    || fail "Stage 3 failed — flask"

  # --- Stage 4: torch CPU (the big one — ~200MB, this is where WSL crashed before) ---
  log "Stage 4/5: torch CPU wheels (~200MB download — be patient)"
  "$PIP" install --no-cache-dir --prefer-binary \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.1,<2.6" "torchvision>=0.16,<0.21" \
    || fail "Stage 4 failed — torch. Re-run script to resume."

  # Sanity: confirm CPU not CUDA
  HAS_CUDA=$("$PY" -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo "import-failed")
  [ "$HAS_CUDA" = "None" ] || fail "torch installed with CUDA ($HAS_CUDA) — should be CPU. Wipe venv and retry."

  # --- Stage 5: SAM family + augmentation + timm ---
  log "Stage 5/5: segment-anything, mobile-sam, albumentations, timm"
  "$PIP" install --no-cache-dir --prefer-binary \
    "segment-anything==1.0" "mobile-sam==1.0" \
    "albumentations>=1.4,<2.0" "timm>=0.9" \
    || fail "Stage 5 failed — SAM/augmentation"

  # --- Frontend (idempotent npm) ---
  log "Frontend: npm install (idempotent)"
  if [ ! -d frontend/node_modules ] || [ -z "$(ls -A frontend/node_modules 2>/dev/null)" ]; then
    (cd frontend && npm install) || fail "npm install failed"
  else
    log "  frontend/node_modules present — skipping (delete it to force reinstall)"
  fi

  # --- Final import sanity ---
  log "Sanity: importing all critical modules"
  "$PY" -c "import flask, torch, torchvision, mobile_sam, segment_anything, albumentations, timm; print('all imports OK')" \
    || fail "Sanity import failed — see error above"

  log "DONE. Launch with: bash scripts/run-dev.sh"
  ```

  **Must NOT do**:
  - No `set -e` (kills on any non-zero, masks recoverable errors)
  - No `source .venv/bin/activate` anywhere
  - No `pip install -r requirements.txt` (monolithic — defeats the staging)
  - No `python3` (unversioned — could resolve to 3.14)
  - No `--upgrade-strategy eager` (forces unrelated upgrades, balloons memory)
  - No `npx` invocations
  - No interactive prompts (`read -p` etc.)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single-file rewrite with concrete spec above; no complex logic
  - **Skills**: none

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T2, T4, T5)
  - **Parallel Group**: Wave 2
  - **Blocks**: T7 (smoke test runs this script)
  - **Blocked By**: T1 (decision on venv keep/wipe), T2 (requirements.txt — though install.sh hardcodes packages, it must agree with requirements.txt)

  **References**:

  **Pattern References**:
  - `scripts/install.sh` (current state, WIP) — reference for any project-specific paths/conventions; do NOT preserve its monolithic install pattern
  - `scripts/download-weights.sh` — example of a working defensive script (curl with retry, dir checks)

  **External References**:
  - PyTorch install matrix: `https://pytorch.org/get-started/locally/` — confirms CPU-only command
  - pip `--prefer-binary` docs: `https://pip.pypa.io/en/stable/cli/pip_install/#cmdoption-prefer-binary`

  **WHY Each Reference Matters**:
  - `download-weights.sh` shows the project's preferred bash style (no fancy frameworks, defensive defaults)
  - `--prefer-binary` is critical: without it, pip may build wheels from source even when binary exists, causing memory spikes
  - `--no-cache-dir` prevents pip from filling `~/.cache/pip` with wheels we'll never reuse (saves 1-2GB on disk over time)

  **Acceptance Criteria**:
  - [ ] Script is executable (`chmod +x` confirmed)
  - [ ] `bash -n scripts/install.sh` → no syntax errors
  - [ ] `grep -c 'source.*activate' scripts/install.sh` → 0
  - [ ] `grep -c 'set -e' scripts/install.sh` → 0
  - [ ] `grep -c -- '--prefer-binary' scripts/install.sh` → ≥4 (one per pip stage)

  **QA Scenarios**:

  ```
  Scenario: install.sh runs to completion on this WSL box
    Tool: Bash
    Preconditions: T2 complete (requirements.txt fixed); venv state from T1 known
    Steps:
      1. mkdir -p .sisyphus/evidence/launch-fix
      2. bash -x scripts/install.sh 2>&1 | tee .sisyphus/evidence/launch-fix/task-3-run.log
      3. grep -c 'Stage [0-5]/5' .sisyphus/evidence/launch-fix/task-3-run.log → expect 6 (Stage 0-5 all hit)
      4. grep 'all imports OK' .sisyphus/evidence/launch-fix/task-3-run.log → expect 1 match
      5. backend/.venv/bin/python -c "import torch; assert torch.version.cuda is None, 'CUDA leaked in!'"
    Expected Result: Script exits 0, all stages complete, final sanity import passes, torch is CPU-only
    Failure Indicators: WSL becomes unresponsive (free -m shows <500MB available) → IMMEDIATELY abort with Ctrl+C; do NOT proceed
    Evidence: .sisyphus/evidence/launch-fix/task-3-run.log

  Scenario: install.sh is idempotent (second run is fast no-op)
    Tool: Bash
    Preconditions: First run of install.sh succeeded
    Steps:
      1. time bash scripts/install.sh 2>&1 | tee .sisyphus/evidence/launch-fix/task-3-rerun.log
      2. Verify "reusing existing venv" appears
      3. Verify "skipping" appears for npm
      4. Total runtime < 60 seconds
    Expected Result: Re-run completes in <60s with no installations triggered
    Evidence: .sisyphus/evidence/launch-fix/task-3-rerun.log

  Scenario: install.sh refuses Python 3.14
    Tool: Bash
    Preconditions: System has python3.14 in PATH; backup python3.11 path
    Steps:
      1. PATH="/home/linuxbrew/.linuxbrew/bin" bash scripts/install.sh 2>&1 | head -20
    Expected Result: Exits with error message naming Python version requirement; does NOT proceed to venv creation
    Evidence: .sisyphus/evidence/launch-fix/task-3-py314-rejected.log
  ```

  **Evidence to Capture**:
  - [ ] `.sisyphus/evidence/launch-fix/task-3-run.log` (full first-run trace)
  - [ ] `.sisyphus/evidence/launch-fix/task-3-rerun.log` (idempotency proof)
  - [ ] `.sisyphus/evidence/launch-fix/task-3-py314-rejected.log` (refusal proof)

  **Commit**: YES
  - Message: `fix(scripts): rewrite install.sh — staged, Python-3.11-locked, --prefer-binary`
  - Files: `scripts/install.sh`
  - Pre-commit: `bash -n scripts/install.sh && [ -x scripts/install.sh ]`

- [x] 4. Rewrite `scripts/run-dev.sh` — pre-flight checks, no `source activate`, fix `\n` bug

  **What to do**:
  Rewrite `scripts/run-dev.sh` with this structure:

  ```bash
  #!/usr/bin/env bash
  # Clinic Codex dev launcher — backend on :7117, frontend on :7118.
  # Defensive: fails fast with clear errors if env not set up.

  set -u
  set -o pipefail

  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  cd "$REPO_ROOT"

  log()  { printf '[run-dev] %s\n' "$*"; }
  fail() { printf '[run-dev] ERROR: %s\n' "$*" >&2; exit 1; }

  # --- Pre-flight: venv ---
  PY="backend/.venv/bin/python"
  [ -x "$PY" ] || fail "backend/.venv missing. Run: bash scripts/install.sh"
  "$PY" -c "import flask, torch, mobile_sam" 2>/dev/null \
    || fail "venv incomplete. Run: bash scripts/install.sh"

  # --- Pre-flight: frontend deps ---
  [ -d frontend/node_modules ] || fail "frontend/node_modules missing. Run: bash scripts/install.sh"

  # --- Pre-flight: ports ---
  BACKEND_PORT="${BACKEND_PORT:-7117}"
  FRONTEND_PORT="${FRONTEND_PORT:-7118}"
  for port in "$BACKEND_PORT" "$FRONTEND_PORT"; do
    if (echo > "/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
      fail "Port $port already in use. Stop other process or override BACKEND_PORT/FRONTEND_PORT env."
    fi
  done

  # --- Cleanup on exit ---
  PIDS=()
  cleanup() {
    log "shutting down (pids: ${PIDS[*]:-none})"
    for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
  }
  trap cleanup EXIT INT TERM

  # --- Launch backend (call python directly, no source activate) ---
  log "starting backend on :$BACKEND_PORT"
  PORT="$BACKEND_PORT" "$PY" backend/examples/flask_api.py &
  PIDS+=($!)

  # --- Launch frontend ---
  log "starting frontend on :$FRONTEND_PORT"
  (cd frontend && PORT="$FRONTEND_PORT" npm run dev) &
  PIDS+=($!)

  # --- Wait for backend ready ---
  for i in $(seq 1 30); do
    sleep 1
    if curl -sf "http://127.0.0.1:$BACKEND_PORT/classes" >/dev/null 2>&1; then
      log "backend ready: http://localhost:$BACKEND_PORT"
      break
    fi
    [ "$i" = 30 ] && fail "backend did not start within 30s — see logs above"
  done

  printf '\n[run-dev] both services up.\n'
  printf '[run-dev]   backend:  http://localhost:%s\n' "$BACKEND_PORT"
  printf '[run-dev]   frontend: http://localhost:%s\n' "$FRONTEND_PORT"
  printf '[run-dev] Ctrl-C to stop.\n\n'

  wait
  ```

  **Must NOT do**:
  - No `source .venv/bin/activate` (call `.venv/bin/python` directly)
  - No `echo "\n..."` (use `printf '\n...\n'`)
  - No `set -e` (we want explicit error handling on pre-flight)
  - No `wait -n` (not POSIX-portable)
  - No backgrounding without storing PID for cleanup
  - No hardcoded ports (must respect `BACKEND_PORT` / `FRONTEND_PORT` env)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single-file rewrite with full spec
  - **Skills**: none

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T2, T3, T5)
  - **Parallel Group**: Wave 2
  - **Blocks**: T7 (smoke test runs this)
  - **Blocked By**: T1

  **References**:

  **Pattern References**:
  - `scripts/run-dev.sh` (current WIP) — note buggy `echo "\n..."` ~line 15 and `source activate` ~line 36; replace, don't preserve
  - `backend/examples/flask_api.py` — confirms it reads `PORT` env var (env probe verified)
  - `frontend/package.json:scripts.dev` — Vite 5+ respects `PORT` env

  **External References**:
  - Bash `/dev/tcp` port check: built-in, works on bash 3.2+
  - `trap` cleanup pattern: standard POSIX shell idiom

  **WHY Each Reference Matters**:
  - Calling `.venv/bin/python` directly avoids the activation-script subshell that swallows errors silently
  - Pre-flight import check (`python -c "import flask, torch, mobile_sam"`) catches the EXACT failure mode that bit us before (venv exists but is broken — the current state)
  - Port check prevents confusing "address already in use" mid-launch failures
  - `trap cleanup EXIT` ensures Ctrl-C kills backend + frontend together, never orphaning one

  **Acceptance Criteria**:
  - [ ] `bash -n scripts/run-dev.sh` → no syntax errors
  - [ ] `grep -c 'source.*activate' scripts/run-dev.sh` → 0
  - [ ] `grep -cE 'echo[^a-z]*"\\\\n' scripts/run-dev.sh` → 0
  - [ ] Script is executable
  - [ ] Pre-flight check runs before any service launch

  **QA Scenarios**:

  ```
  Scenario: run-dev.sh launches both services
    Tool: Bash
    Preconditions: T2, T3 complete; venv working; ports 7117/7118 free
    Steps:
      1. bash scripts/run-dev.sh > /tmp/run-dev.log 2>&1 &
      2. RUN_PID=$!
      3. for i in $(seq 1 30); do sleep 1; curl -sf http://localhost:7117/classes >/dev/null && break; done
      4. curl -sf http://localhost:7117/classes | python3 -c 'import sys,json; d=json.load(sys.stdin); assert d["num_classes"]==286, d'
      5. curl -sf http://localhost:7118/ | grep -q 'id="root"'
      6. kill -INT $RUN_PID; wait $RUN_PID 2>/dev/null
      7. cp /tmp/run-dev.log .sisyphus/evidence/launch-fix/task-4-run.log
    Expected Result: Backend returns valid /classes JSON with 286 classes; frontend serves HTML containing root div; clean shutdown
    Failure Indicators: curl times out at step 3 → backend didn't start; investigate /tmp/run-dev.log
    Evidence: .sisyphus/evidence/launch-fix/task-4-run.log

  Scenario: run-dev.sh refuses to start when venv is broken
    Tool: Bash
    Preconditions: Temporarily move venv: mv backend/.venv backend/.venv.bak
    Steps:
      1. bash scripts/run-dev.sh 2>&1 | head -5
      2. echo "Exit: $?"
      3. mv backend/.venv.bak backend/.venv  # CLEANUP — must always run
    Expected Result: Exits non-zero with "backend/.venv missing" message; never attempts service launch
    Evidence: .sisyphus/evidence/launch-fix/task-4-no-venv.log

  Scenario: run-dev.sh refuses when port occupied
    Tool: Bash
    Preconditions: Bind port 7117 with a sleeper process
    Steps:
      1. python3 -c 'import socket,time; s=socket.socket(); s.bind(("127.0.0.1",7117)); s.listen(); time.sleep(10)' &
      2. BLOCKER_PID=$!; sleep 1
      3. bash scripts/run-dev.sh 2>&1 | head -5
      4. echo "Exit: $?"
      5. kill $BLOCKER_PID 2>/dev/null  # CLEANUP
    Expected Result: Exits non-zero with "Port 7117 already in use" message
    Evidence: .sisyphus/evidence/launch-fix/task-4-port-busy.log
  ```

  **Evidence to Capture**:
  - [ ] `.sisyphus/evidence/launch-fix/task-4-run.log` (success run)
  - [ ] `.sisyphus/evidence/launch-fix/task-4-no-venv.log` (defensive check)
  - [ ] `.sisyphus/evidence/launch-fix/task-4-port-busy.log` (defensive check)

  **Commit**: YES
  - Message: `fix(scripts): harden run-dev.sh — pre-flight checks, drop source activate`
  - Files: `scripts/run-dev.sh`
  - Pre-commit: `bash -n scripts/run-dev.sh && [ -x scripts/run-dev.sh ]`

- [x] 5. Update `scripts/install.ps1` — Windows parity with hardened install.sh

  **What to do**:
  Mirror the staged install pattern from `install.sh` to PowerShell:
  - Detect Python 3.10/3.11 via `py -3.11`, `py -3.10`, or fallback `python` (verify version)
  - Refuse 3.12+/3.14 with clear error message
  - Create venv at `backend\.venv` if missing
  - Install in same 5 stages with `--no-cache-dir --prefer-binary`
  - Use `--extra-index-url https://download.pytorch.org/whl/cpu` for the torch stage
  - Idempotent npm via `frontend\node_modules` directory check
  - Final sanity import test
  - Use try/catch for per-stage error handling with informative messages
  - DO NOT call `Activate.ps1`; use `backend\.venv\Scripts\python.exe` directly
  - Use `$PSScriptRoot` to anchor paths (works regardless of cwd)

  **Must NOT do**:
  - No `pip install -r requirements.txt` (use staged inline like install.sh)
  - No `& .\.venv\Scripts\Activate.ps1`
  - No interactive prompts (`Read-Host` etc.)
  - No assumption about cwd

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Mechanical translation of install.sh pattern to PowerShell
  - **Skills**: none

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T2, T3, T4)
  - **Parallel Group**: Wave 2
  - **Blocks**: nothing in this plan (Windows-only path; no smoke test on WSL)
  - **Blocked By**: T1

  **References**:

  **Pattern References**:
  - `scripts/install.sh` (after T3 complete) — canonical pattern to mirror
  - `scripts/install.ps1` (current state) — preserve any Windows-specific quirks

  **External References**:
  - PowerShell `$PSScriptRoot` automatic var — repo-relative paths
  - Python launcher (`py.exe`) docs — preferred way to select Python version on Windows

  **WHY Each Reference Matters**:
  - install.sh defines the canonical staged install; install.ps1 must produce equivalent end state
  - Windows users typically have the `py` launcher, which handles Python version selection cleanly without needing PATH manipulation

  **Acceptance Criteria**:
  - [ ] File contains `--extra-index-url https://download.pytorch.org/whl/cpu` for torch stage
  - [ ] No `Activate.ps1` invocations
  - [ ] No `pip install -r` (must be staged)
  - [ ] If `pwsh` available on this WSL, syntax check passes

  **QA Scenarios**:

  ```
  Scenario: install.ps1 syntax-validates and uses correct patterns
    Tool: Bash
    Preconditions: pwsh may or may not be installed on this WSL
    Steps:
      1. if command -v pwsh >/dev/null 2>&1; then \
           pwsh -NoProfile -Command "\$null = [ScriptBlock]::Create((Get-Content -Raw scripts/install.ps1)); Write-Host 'PARSE OK'"; \
         else \
           echo "pwsh not installed — syntax check skipped, manual review only"; \
         fi
      2. grep -c 'extra-index-url.*pytorch.org/whl/cpu' scripts/install.ps1
      3. grep -c 'Activate.ps1' scripts/install.ps1
      4. grep -c 'pip install -r' scripts/install.ps1
    Expected Result: Step 2 returns 1; steps 3-4 return 0; step 1 prints "PARSE OK" or skip notice
    Evidence: .sisyphus/evidence/launch-fix/task-5-validate.log
  ```

  **Evidence to Capture**:
  - [ ] `.sisyphus/evidence/launch-fix/task-5-validate.log` (parse + grep results)

  **Commit**: YES
  - Message: `fix(scripts): mirror install.sh hardening to install.ps1`
  - Files: `scripts/install.ps1`
  - Pre-commit: pattern grep checks all pass

- [x] 6. Add Manual Launch runbook to `INSTALL.md` (the PRIMARY documented path)

  **What to do**:
  Insert a new top-level section in `INSTALL.md` titled "Manual Launch (Primary Path)" that lets a user copy-paste their way to a running app WITHOUT relying on any script. Structure:

  ```markdown
  ## Manual Launch (Primary Path)

  > Use these steps if `scripts/install.sh` doesn't work for you, or if you want
  > to understand exactly what's happening. Each command is independent and safe to re-run.

  ### Prerequisites
  - Python 3.10 or 3.11 (NOT 3.12+ — torch wheels not yet available)
  - Node.js 20+ and npm
  - ~2GB free disk space
  - On Linux/WSL: `python3.11 --version` should print `Python 3.11.x`. If not: `sudo apt install python3.11 python3.11-venv`.

  ### Backend (one-time setup)

  ```bash
  cd /path/to/clinic-codex

  # 1. Create venv with Python 3.11 (skip if backend/.venv already exists)
  python3.11 -m venv backend/.venv

  # 2. Upgrade pip (best-effort)
  backend/.venv/bin/pip install --upgrade pip

  # 3. Install core utils (small, fast)
  backend/.venv/bin/pip install --no-cache-dir --prefer-binary \
    "numpy>=1.24,<2.5" "pillow>=10.0" "pyyaml>=6.0" "scipy>=1.11" "tqdm>=4.66"

  # 4. Install Flask
  backend/.venv/bin/pip install --no-cache-dir --prefer-binary \
    "flask>=3.0,<4.0" "flask-cors>=4.0"

  # 5. Install torch CPU wheels (~200MB — be patient, do NOT use plain `pip install torch`)
  backend/.venv/bin/pip install --no-cache-dir --prefer-binary \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.1,<2.6" "torchvision>=0.16,<0.21"

  # 6. Install SAM family + augmentation
  backend/.venv/bin/pip install --no-cache-dir --prefer-binary \
    "segment-anything==1.0" "mobile-sam==1.0" \
    "albumentations>=1.4,<2.0" "timm>=0.9"

  # 7. Sanity check
  backend/.venv/bin/python -c "import flask, torch, mobile_sam, segment_anything; print('OK')"
  ```

  ### Frontend (one-time setup)

  ```bash
  cd frontend
  npm install
  cd ..
  ```

  ### Launch (every time)

  Open two terminals:

  **Terminal 1 — backend** (port 7117):
  ```bash
  PORT=7117 backend/.venv/bin/python backend/examples/flask_api.py
  ```

  **Terminal 2 — frontend** (port 7118):
  ```bash
  cd frontend && PORT=7118 npm run dev
  ```

  Then open `http://localhost:7118` in your browser.

  ### Verify it's working

  ```bash
  # Backend should return JSON with num_classes: 286
  curl http://localhost:7117/classes | python3 -m json.tool | head -5

  # Frontend should return HTML containing <div id="root">
  curl -s http://localhost:7118/ | grep -o 'id="root"'
  ```

  ### Common issues

  | Symptom | Cause | Fix |
  |---|---|---|
  | `pip` downloads several GB and WSL freezes | Used plain `pip install torch` (CUDA wheels) | Use the `--extra-index-url https://download.pytorch.org/whl/cpu` flag exactly as shown in step 5 |
  | `ModuleNotFoundError: mobile_sam` | Skipped step 6, or installed from git source | Run step 6 exactly; do NOT use `git+https://...` versions |
  | `Address already in use :7117` | Backend already running | `lsof -ti:7117 \| xargs -r kill` then retry |
  | `python3.11: command not found` | Python 3.11 not installed | `sudo apt install python3.11 python3.11-venv` (Ubuntu/WSL) |
  ```

  **Must NOT do**:
  - No `pip install torch` without the CPU index URL anywhere in the doc
  - No `source .venv/bin/activate` (use `.venv/bin/python` directly throughout)
  - No `git+https://` references
  - No removing or rewriting the existing INSTALL.md content (only ADD this new section near the top)
  - No screenshots/badges/marketing fluff

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Documentation task; writing-focused agent better at clear technical prose
  - **Skills**: none

  **Parallelization**:
  - **Can Run In Parallel**: NO (Wave 3, but lightweight)
  - **Parallel Group**: Wave 3 (with T7, T8 — but T7 depends on T6 if T6 is the source of truth)
  - **Blocks**: T7 (smoke test references this runbook)
  - **Blocked By**: T1, T2, T3, T4 (need final command shapes locked in)

  **References**:

  **Pattern References**:
  - `INSTALL.md` (current state) — read entire file first; insert NEW section without removing existing content
  - `scripts/install.sh` (after T3) — commands in runbook MUST match what install.sh does exactly

  **External References**:
  - None needed beyond what install.sh references

  **WHY Each Reference Matters**:
  - The runbook is the user's escape hatch when scripts fail; it MUST be byte-equivalent to what scripts do, or users will hit different failures
  - Existing INSTALL.md likely has researcher-friendly content that should be preserved

  **Acceptance Criteria**:
  - [ ] New section "Manual Launch (Primary Path)" exists in `INSTALL.md`
  - [ ] Section includes ALL 7 backend steps + frontend setup + launch commands + verify + common issues table
  - [ ] `grep -c 'extra-index-url.*pytorch.org/whl/cpu' INSTALL.md` ≥ 1
  - [ ] `grep -c 'git+https' INSTALL.md` = 0
  - [ ] `grep -c 'source .*activate' INSTALL.md` = 0
  - [ ] Original INSTALL.md content preserved (line count after >= line count before + 50)

  **QA Scenarios**:

  ```
  Scenario: Manual runbook commands actually work end-to-end
    Tool: Bash
    Preconditions: T2, T3, T4 complete (install.sh and run-dev.sh in final form); user has working venv
    Steps:
      1. Extract every code block from "Manual Launch (Primary Path)" section into /tmp/runbook-commands.sh
      2. Run the "Verify it's working" commands while a manual launch is active (or after Task 7 verification)
      3. Confirm both verify commands output expected results
    Expected Result: Both verify commands succeed; runbook is internally consistent
    Evidence: .sisyphus/evidence/launch-fix/task-6-runbook-verify.log

  Scenario: Documentation is self-contained
    Tool: Bash
    Preconditions: Section written
    Steps:
      1. wc -l INSTALL.md → record line count
      2. grep -A 200 'Manual Launch (Primary Path)' INSTALL.md | head -200 → visual review
      3. Count code blocks: grep -c '^```' INSTALL.md → expect ≥ (existing + 6)
    Expected Result: Section is complete and well-formatted; no orphaned references to scripts
    Evidence: .sisyphus/evidence/launch-fix/task-6-doc-check.log
  ```

  **Evidence to Capture**:
  - [ ] `.sisyphus/evidence/launch-fix/task-6-runbook-verify.log` (commands work)
  - [ ] `.sisyphus/evidence/launch-fix/task-6-doc-check.log` (structure check)

  **Commit**: YES
  - Message: `docs(install): add Manual Launch runbook as primary path`
  - Files: `INSTALL.md`
  - Pre-commit: `grep -c 'git+https' INSTALL.md` = 0

- [x] 7. End-to-end smoke test on this WSL2 box (the SINGLE verification gate)

  **What to do**:
  Run a comprehensive end-to-end test that proves the user can launch the app from this exact environment:

  1. **Pre-state snapshot**:
     - `free -m > .sisyphus/evidence/launch-fix/task-7-mem-before.txt`
     - `du -sh backend/.venv > .sisyphus/evidence/launch-fix/task-7-venv-before.txt`
  2. **Run install.sh** (idempotent — should be fast since T1-T6 left things ready):
     - `bash scripts/install.sh 2>&1 | tee .sisyphus/evidence/launch-fix/task-7-install.log`
     - Verify exit code 0
  3. **Confirm no CUDA leaked in**:
     - `backend/.venv/bin/python -c "import torch; assert torch.version.cuda is None, f'CUDA leaked: {torch.version.cuda}'"`
     - `du -sh backend/.venv > .sisyphus/evidence/launch-fix/task-7-venv-after.txt` — confirm not bloated
  4. **Run run-dev.sh in background**:
     - `bash scripts/run-dev.sh > .sisyphus/evidence/launch-fix/task-7-rundev.log 2>&1 &`
     - `RUN_PID=$!`
  5. **Wait for backend ready** (max 30s):
     - Loop curl `http://localhost:7117/classes` until 200 or timeout
  6. **Verify backend response**:
     - `curl -sf http://localhost:7117/classes | python3 -c 'import sys,json; d=json.load(sys.stdin); assert d["num_classes"]==286'`
  7. **Verify frontend response**:
     - `curl -sf http://localhost:7118/ | grep -q 'id="root"'`
  8. **Memory check during run**:
     - `free -m > .sisyphus/evidence/launch-fix/task-7-mem-during.txt`
     - Confirm available RAM > 2GB (no OOM pressure)
  9. **Clean shutdown**:
     - `kill -INT $RUN_PID; wait $RUN_PID 2>/dev/null`
     - Verify ports 7117 and 7118 are free again
  10. **Write summary**: `.sisyphus/evidence/launch-fix/task-7-summary.md` with PASS/FAIL per step

  **Must NOT do**:
  - No `rm -rf backend/.venv` (don't simulate fresh install — that would re-trigger crash risk; user already has working venv post-install.sh)
  - No `npm install` from scratch (idempotent check should skip)
  - No leaving processes running
  - No proceeding to T8 if any step fails — surface failure immediately

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Multi-step verification with branching error handling; needs careful evidence capture and cleanup
  - **Skills**: none

  **Parallelization**:
  - **Can Run In Parallel**: NO (must run alone — uses ports, monitors memory)
  - **Parallel Group**: Wave 3
  - **Blocks**: T8
  - **Blocked By**: T2, T3, T4, T6

  **References**:

  **Pattern References**:
  - `scripts/install.sh` (T3 final form) — the script under test
  - `scripts/run-dev.sh` (T4 final form) — the script under test
  - `INSTALL.md` "Manual Launch" section (T6 final form) — verify commands match

  **External References**:
  - None — this is purely local execution

  **WHY Each Reference Matters**:
  - This task is the GATE: if it fails, the user is back to square one
  - Memory monitoring catches the EXACT failure mode that crashed the user's WSL before
  - Idempotency check (re-running install.sh) confirms no destructive operations

  **Acceptance Criteria**:
  - [ ] All 10 steps complete with PASS verdict
  - [ ] Backend returns `num_classes: 286`
  - [ ] Frontend serves HTML with `<div id="root">`
  - [ ] Available RAM during run remains > 2GB
  - [ ] No leftover processes (`pgrep -f flask_api.py` returns nothing after cleanup)
  - [ ] All 5 evidence files written

  **QA Scenarios**:

  ```
  Scenario: Full launch cycle succeeds
    Tool: Bash
    Preconditions: T2, T3, T4, T6 all complete and committed
    Steps:
      1. mkdir -p .sisyphus/evidence/launch-fix
      2. free -m > .sisyphus/evidence/launch-fix/task-7-mem-before.txt
      3. bash scripts/install.sh 2>&1 | tee .sisyphus/evidence/launch-fix/task-7-install.log
      4. test ${PIPESTATUS[0]} -eq 0 || { echo FAIL-INSTALL; exit 1; }
      5. backend/.venv/bin/python -c "import torch; assert torch.version.cuda is None"
      6. bash scripts/run-dev.sh > .sisyphus/evidence/launch-fix/task-7-rundev.log 2>&1 &
      7. RUN_PID=$!
      8. for i in $(seq 1 30); do sleep 1; curl -sf http://localhost:7117/classes >/dev/null && BACKEND_UP=1 && break; done
      9. test "$BACKEND_UP" = 1 || { kill $RUN_PID; echo FAIL-BACKEND-START; exit 1; }
      10. curl -sf http://localhost:7117/classes | python3 -c 'import sys,json; d=json.load(sys.stdin); assert d["num_classes"]==286, d'
      11. curl -sf http://localhost:7118/ | grep -q 'id="root"'
      12. free -m > .sisyphus/evidence/launch-fix/task-7-mem-during.txt
      13. AVAIL=$(awk '/^Mem:/ {print $7}' .sisyphus/evidence/launch-fix/task-7-mem-during.txt)
      14. test "$AVAIL" -gt 2000 || echo "WARN: low memory $AVAIL MB"
      15. kill -INT $RUN_PID; wait $RUN_PID 2>/dev/null
      16. sleep 2
      17. ! (echo > /dev/tcp/127.0.0.1/7117) 2>/dev/null && ! (echo > /dev/tcp/127.0.0.1/7118) 2>/dev/null
      18. Write .sisyphus/evidence/launch-fix/task-7-summary.md with PASS verdicts
    Expected Result: All steps pass; both endpoints respond correctly; memory stays healthy; clean shutdown
    Failure Indicators:
      - free -m shows available RAM dropping below 1GB → ABORT, kill all processes, do not commit anything
      - install.sh hangs >5min → ABORT, investigate stage that stalled
      - Backend never becomes ready → check task-7-rundev.log for stack trace
    Evidence: .sisyphus/evidence/launch-fix/task-7-summary.md (consolidated PASS/FAIL report)

  Scenario: Cleanup leaves no orphan processes
    Tool: Bash
    Preconditions: Main scenario above completed
    Steps:
      1. pgrep -f 'flask_api.py' → expect empty
      2. pgrep -f 'vite.*7118' → expect empty
      3. ss -tlnp 2>/dev/null | grep -E ':(7117|7118)' → expect empty
    Expected Result: Zero matches in all 3 checks
    Evidence: .sisyphus/evidence/launch-fix/task-7-cleanup-check.log
  ```

  **Evidence to Capture**:
  - [ ] `.sisyphus/evidence/launch-fix/task-7-mem-before.txt`
  - [ ] `.sisyphus/evidence/launch-fix/task-7-mem-during.txt`
  - [ ] `.sisyphus/evidence/launch-fix/task-7-venv-before.txt`
  - [ ] `.sisyphus/evidence/launch-fix/task-7-venv-after.txt`
  - [ ] `.sisyphus/evidence/launch-fix/task-7-install.log`
  - [ ] `.sisyphus/evidence/launch-fix/task-7-rundev.log`
  - [ ] `.sisyphus/evidence/launch-fix/task-7-cleanup-check.log`
  - [ ] `.sisyphus/evidence/launch-fix/task-7-summary.md` (consolidated report)

  **Commit**: NO (this is a verification task, no source changes)

- [x] 8. Reconcile boulder + plan checkboxes + boulder.json (close-out)

  **What to do**:
  1. **Update old plan checkboxes**: In `.sisyphus/plans/codex-cleanup-and-polish.md`, change `[ ]` to `[x]` for tasks 15-21 (they are committed in git: `d3daa4e`, `5e68110`, `335ffe2`, `ab0fd8f`, `2664e40`, `8b78935`, `1fa55b4`). Verify each task's `[ ]` checkbox by searching for the task title; do NOT mark anything not actually committed.
  2. **Skip F1-F4**: In the same plan file, locate the "Final Verification Wave" section. Add a brief note at the top of that section: `> **SKIPPED** per user direction (2026-05-06): close-out without final verification.` Leave the F1-F4 checkboxes unchecked.
  3. **Update `.sisyphus/boulder.json`**:
     - Read current state (it tracks active boulder + task_sessions)
     - Mark `codex-cleanup-and-polish` as complete (set `status: "completed"`, add `completed_at` timestamp)
     - If schema supports it, move to `completed_boulders` array
     - Set new active boulder = `launch-fix-and-boulder-closeout` (this plan)
     - Preserve all `task_sessions` data (audit trail)
  4. **Verify**: Read both files back, confirm changes are correct.

  **Must NOT do**:
  - No checking `[x]` for tasks not actually committed (verify each via `git log --oneline | grep`)
  - No checking F1-F4 boxes (they are skipped, not done)
  - No deleting `task_sessions` data from boulder.json (audit history preserved)
  - No editing tasks 1-14 of the cleanup plan (already correctly checked)
  - No fabricating completion timestamps (use `date -u +%Y-%m-%dT%H:%M:%SZ`)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Mechanical state-file updates with verification
  - **Skills**: none

  **Parallelization**:
  - **Can Run In Parallel**: NO (must be last — depends on T7 verification passing)
  - **Parallel Group**: Wave 3 final
  - **Blocks**: nothing (final task)
  - **Blocked By**: T7

  **References**:

  **Pattern References**:
  - `.sisyphus/plans/codex-cleanup-and-polish.md` — read entire file; locate task 15-21 checkboxes (around line 1214+, per env probe)
  - `.sisyphus/boulder.json` — read entire file first; understand schema before editing
  - Git log: `git log --oneline -30` — confirms which tasks are actually committed

  **External References**:
  - None

  **WHY Each Reference Matters**:
  - The boulder.json schema must be respected exactly (other Sisyphus tooling reads it)
  - Cross-checking checkboxes against git log prevents marking unfinished work as done
  - Preserving `task_sessions` keeps the audit trail for any future analysis

  **Acceptance Criteria**:
  - [ ] In `codex-cleanup-and-polish.md`: tasks 15, 16, 17, 18, 19, 20, 21 are `[x]`
  - [ ] In `codex-cleanup-and-polish.md`: F1, F2, F3, F4 remain `[ ]` with SKIPPED note added
  - [ ] In `codex-cleanup-and-polish.md`: tasks 1-14 still `[x]` (unchanged)
  - [ ] In `boulder.json`: `codex-cleanup-and-polish` has `status: "completed"` and `completed_at` timestamp
  - [ ] In `boulder.json`: new active boulder reference = `launch-fix-and-boulder-closeout`
  - [ ] In `boulder.json`: all 21 entries in `task_sessions` preserved
  - [ ] `git diff --stat` shows only the 2 expected files modified

  **QA Scenarios**:

  ```
  Scenario: Plan and boulder.json correctly reflect close-out state
    Tool: Bash
    Preconditions: T7 passed
    Steps:
      1. grep -E '^- \[x\] (15|16|17|18|19|20|21)\.' .sisyphus/plans/codex-cleanup-and-polish.md | wc -l → expect 7
      2. grep -E '^- \[ \] F[1-4]\.' .sisyphus/plans/codex-cleanup-and-polish.md | wc -l → expect 4 (still unchecked)
      3. grep 'SKIPPED per user direction' .sisyphus/plans/codex-cleanup-and-polish.md | wc -l → expect ≥1
      4. python3 -c "import json; d=json.load(open('.sisyphus/boulder.json')); print(d)" — must parse cleanly
      5. python3 -c "import json; d=json.load(open('.sisyphus/boulder.json')); assert any(b.get('name')=='codex-cleanup-and-polish' and b.get('status')=='completed' for b in d.get('completed_boulders', [d.get('active_boulder',{})]))"
      6. test $(git diff --name-only | wc -l) -le 2 → only expected files modified
    Expected Result: All assertions pass
    Evidence: .sisyphus/evidence/launch-fix/task-8-reconcile.log
  ```

  **Evidence to Capture**:
  - [ ] `.sisyphus/evidence/launch-fix/task-8-reconcile.log` (verification output)

  **Commit**: YES
  - Message: `chore(sisyphus): close codex-cleanup-and-polish boulder; reconcile checkboxes`
  - Files: `.sisyphus/plans/codex-cleanup-and-polish.md`, `.sisyphus/boulder.json`
  - Pre-commit: `python3 -c "import json; json.load(open('.sisyphus/boulder.json'))"` (valid JSON)

---

## Final Verification (lightweight)

**SKIPPED** per user direction. The single end-to-end smoke test in Task 7 serves as the verification gate.

If issues are found after delivery, the user can request a F1-F4 audit pass separately.

---

## Commit Strategy

| Task | Commit | Message |
|---|---|---|
| 2 | YES | `fix(backend): pin CPU-only torch + PyPI mobile-sam to prevent WSL OOM` |
| 3 | YES | `fix(scripts): rewrite install.sh — staged, Python-3.11-locked, --prefer-binary` |
| 4 | YES | `fix(scripts): harden run-dev.sh — pre-flight checks, drop source activate` |
| 5 | YES | `fix(scripts): mirror install.sh hardening to install.ps1` |
| 6 | YES | `docs(install): add Manual Launch runbook as primary path` |
| 8 | YES | `chore(sisyphus): close codex-cleanup-and-polish boulder; reconcile checkboxes` |

All commits go to `master` (the user's working branch). No new feature branch — this is hot-fix territory.

---

## Success Criteria

### Verification Commands
```bash
# Backend launches and serves
PORT=7117 backend/.venv/bin/python backend/examples/flask_api.py &
sleep 3
curl -sf http://localhost:7117/classes | python3 -c 'import sys,json; d=json.load(sys.stdin); assert d["num_classes"]==286'  # Expected: silent success

# Frontend launches
(cd frontend && npm run dev) &
sleep 5
curl -sf http://localhost:7118/ | grep -q 'id="root"'  # Expected: silent success

# Install script idempotent (re-run doesn't break)
bash scripts/install.sh && bash scripts/install.sh  # Expected: both runs exit 0
```

### Final Checklist
- [ ] All "Must Have" present in deliverables
- [ ] All "Must NOT Have" absent (verified by `grep -E 'git\+https|cuda|source.*activate' scripts/*.sh backend/requirements.txt` returns 0 matches in install context)
- [ ] User can copy-paste manual runbook from INSTALL.md and reach a running app
- [ ] WSL never freezes during any task execution
- [ ] Old boulder `codex-cleanup-and-polish` marked complete in boulder.json
- [ ] All 6 commits land cleanly on master
