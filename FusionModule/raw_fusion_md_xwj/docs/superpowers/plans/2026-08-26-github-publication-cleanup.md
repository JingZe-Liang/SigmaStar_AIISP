# GitHub Publication Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a clean data-first V2 repository with curated baseline evidence and no active Formal V2 path.

**Architecture:** Keep the data-first runtime and its tests. Treat `/data1/wangzepu/Jaime/Projects/DERIVED/v2_data_first` as the external experiment store, copying only a small safe-q baseline bundle into `results/`.

**Tech Stack:** Python 3.10 (`aaa_312`), PyTorch, NumPy, OpenCV, pytest, Git.

**Spec:** `docs/superpowers/specs/2026-08-26-github-publication-cleanup-design.md`

## Global Constraints

- MOG2 is training supervision only, never a model input or inference dependency.
- Do not add RAW datasets, MOG2 masks, full inference arrays, or intermediate checkpoints to Git.
- The checked-in safe-q result is a baseline and must state its all-zero 645x q-map limitation.
- Preserve current uncommitted data-first implementation changes.

---

### Task 1: Remove Formal-Only Dead Code

**Files:**
- Delete: `src/raw_fusion/v2/composer.py`
- Delete: `src/raw_fusion/v2/tiling.py`
- Modify: `src/raw_fusion/v2/schemas/common.py`

**Interfaces:**
- Consumes: active imports reported by `rg` across runtime and tests.
- Produces: data-first-only source with no unused Formal schema registry entries.

- [ ] **Step 1: Verify no code imports obsolete modules**

Run `rg -n "from \\.composer|from \\.tiling|import .*composer|import .*tiling" src/raw_fusion/v2 src/tests/v2`.
Expected: no active import.

- [ ] **Step 2: Delete Formal-only source and registry entries**

Remove `composer.py` and `tiling.py`. Remove these Formal-only schema names from `SCHEMA_FIELDS` in `common.py`: `FinalHardStopConditionV2`, `FinalHardStopReportV2`, `EvaluationRawFrameV2`, `EvaluationRawConditionV2`, `EvaluationRawBundleV2`, `EvaluationDiagnosticsV2`, `EvaluationRunResultV2`, `AuditShardV2`, `AuditClaimV2`, `EvaluationResultV2`, `EvaluationPlanV2`, `SamplerManifestV2`, `AuditBundleV2`, and `LabelBundleV2`.

- [ ] **Step 3: Validate the removal**

Run `PYTHONPATH=src /data1/wangzepu/.conda/envs/aaa_312/bin/python -m compileall -q src` and `PYTHONPATH=src /data1/wangzepu/.conda/envs/aaa_312/bin/python -m pytest src/tests/v2 -q`.
Expected: both exit zero.

- [ ] **Step 4: Commit the cleanup**

Run `git add src/raw_fusion/v2 src/tests/v2` followed by `git commit -m "refactor: remove obsolete Formal V2 runtime"`.

### Task 2: Add Curated Baseline Evidence

**Files:**
- Create: `results/2026-08-26-safe-q/data_first_v2.pt`
- Create: `results/2026-08-26-safe-q/metrics.jsonl`
- Create: `results/2026-08-26-safe-q/training_manifest.json`
- Create: `results/2026-08-26-safe-q/comparison.json`
- Create: `results/2026-08-26-safe-q/inference_128x.jsonl`
- Create: `results/2026-08-26-safe-q/inference_645x.jsonl`
- Create: `results/2026-08-26-safe-q/inference_manifest.json`
- Create: `results/2026-08-26-safe-q/fusion_2dnr_3dnr_fixed_isp_128x.mp4`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: safe-q run files from external `DERIVED`.
- Produces: a tracked, compact baseline bundle.

- [ ] **Step 1: Validate selected sources**

Read the safe-q manifest and comparison summary. Confirm the checkpoint and MP4 are smaller than the GitHub per-file limit. Exclude `.npy` arrays, MOG2 cache data, intermediate checkpoints, balanced-run outputs, and smoke outputs.

- [ ] **Step 2: Copy the exact curated bundle**

Copy the final checkpoint and metric log from `train_gpu1_safe_q_20260826`, naming its manifest `training_manifest.json`; copy the two traces and name the inference manifest `inference_manifest.json`; then copy `comparison_safe_q_20260826.json` and the video from `visual_128x_fixed_isp`.

- [ ] **Step 3: Permit curated evidence only**

Add ordered `.gitignore` negations that allow `results/2026-08-26-safe-q/` while keeping generic checkpoints and videos ignored elsewhere.

- [ ] **Step 4: Verify result contents**

Run `find results/2026-08-26-safe-q -type f -printf '%s %p\\n' | sort -nr`.
Expected: the curated checkpoint, logs, summaries, and video only; no `.npy` files or MOG2 masks.

- [ ] **Step 5: Commit the evidence**

Run `git add .gitignore results/2026-08-26-safe-q` followed by `git commit -m "docs: add safe-q baseline evidence"`.

### Task 3: Replace Stale Documentation

**Files:**
- Modify: `README.md`
- Create: `docs/protocol.md`
- Create: `docs/results.md`
- Delete: `docs/superpowers/specs/2026-08-26-data-first-v2-design.md`
- Delete: `docs/superpowers/plans/2026-08-26-data-first-v2-implementation.md`

**Interfaces:**
- Consumes: active CLI commands in `pyproject.toml` and `cli.py`, and the curated result bundle.
- Produces: public documentation that names only existing data-first behavior.

- [ ] **Step 1: Rewrite README around current commands**

Document validation, MOG2 cache generation, train, resume, inference, and comparison. Use `../DERIVED/v2_data_first/...` for runtime output. Use `pytest src/tests/v2 -q` for validation and remove old Formal test references.

- [ ] **Step 2: Add protocol and results documents**

Document the four deployment inputs, MOG2-only supervision, 32x32 selector, and range limiter in `docs/protocol.md`. Record result provenance, metrics, 128x q activity, and the 645x all-zero-q limitation in `docs/results.md`.

- [ ] **Step 3: Remove superseded historical plans**

Delete the earlier data-first design and implementation-plan records; retain the publication-cleanup design and plan as repository provenance.

- [ ] **Step 4: Verify stale references are absent**

Run `rg -n "test_evaluation_plan|test_evaluate|test_audit|test_smoke|test_delivery|test_cli|test_pipeline|label_bundle|sampler_manifest|Formal V2" README.md docs`.
Expected: no obsolete workflow instructions outside the cleanup records.

- [ ] **Step 5: Commit documentation**

Run `git add README.md docs` followed by `git commit -m "docs: document data-first V2 workflow"`.

### Task 4: Final GitHub Readiness Verification

**Files:**
- Verify: complete repository publication surface

**Interfaces:**
- Consumes: cleaned source, curated evidence, and public documentation.
- Produces: a validated repository ready for a remote push.

- [ ] **Step 1: Run source and test verification**

Run `PYTHONPATH=src /data1/wangzepu/.conda/envs/aaa_312/bin/python -m compileall -q src`, `PYTHONPATH=src /data1/wangzepu/.conda/envs/aaa_312/bin/python -m pytest src/tests/v2 -q`, and `git diff --check`.
Expected: all exit zero.

- [ ] **Step 2: Inspect the Git surface**

Run `git status --short` and `git ls-files | rg "(^results/|\\.(npy|raw|pt|mp4)$)"`.
Expected: no untracked required source; generated files are limited to the curated result bundle.

- [ ] **Step 3: Commit all remaining intended publication changes**

Run `git add README.md pyproject.toml src docs .gitignore results` followed by `git commit -m "chore: prepare data-first V2 for publication"`.
