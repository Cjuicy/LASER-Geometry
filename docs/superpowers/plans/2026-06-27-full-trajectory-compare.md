# Full Trajectory Comparison Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Viser comparison for every frame in two standard LASER result directories.

**Architecture:** A strict manifest loader validates numeric RGB/depth/confidence files, intrinsics, and TUM poses. Pure geometry helpers align geometry to baseline frame 0 and construct capped overview/current-frame clouds. A thin Viser layer renders complete trajectory splines and replaces two detail clouds on slider updates.

**Tech Stack:** Python, NumPy, SciPy Rotation, imageio, Viser, pytest.

---

### Task 1: Standard Output Loading and Geometry

**Files:**
- Create: `tests/test_full_trajectory_compare.py`
- Create: `eval/vis_full_trajectory_compare.py`

- [x] Write failing tests for numeric sorting, TUM `wxyz` parsing, strict frame-count validation, first-frame alignment, and depth unprojection.
- [x] Run `conda run -n vggt-dem python -m pytest tests/test_full_trajectory_compare.py -q`; expect missing-module failure.
- [x] Implement `numeric_paths()`, `load_tum_poses()`, `load_run()`, `align_geometry_to_baseline()`, and `unproject_depth()`.
- [x] Rerun the tests; expect all Task 1 tests to pass.

### Task 2: Confidence Filtering and Bounded Overview

**Files:**
- Modify: `tests/test_full_trajectory_compare.py`
- Modify: `eval/vis_full_trajectory_compare.py`

- [x] Write failing tests proving confidence masks preserve RGB/point correspondence and deterministic sampling applies one global cap.
- [x] Run focused tests; expect undefined helper failures.
- [x] Implement `frame_cloud()`, `sample_cloud()`, and `overview_cloud()` with `pixel_stride`, quantile filtering, frame stride, and deterministic seeds.
- [x] Rerun the complete test file; expect all tests to pass.

### Task 3: Independent Viser UI

**Files:**
- Modify: `eval/vis_full_trajectory_compare.py`
- Modify: `tests/test_full_trajectory_compare.py`

- [x] Write a failing unit test for robust overview camera bounds.
- [x] Implement CLI arguments, complete baseline/geometry splines, start markers, capped overview clouds, a frame slider that removes/recreates exactly two detail nodes, visibility controls, and an automatic overview camera.
- [x] Run `conda run -n vggt-dem python -m pytest tests/test_full_trajectory_compare.py tests/test_alignment_debug_viewer.py -q`.
- [x] Run `conda run -n vggt-dem python -m py_compile eval/vis_full_trajectory_compare.py` and `git diff --check`.

### Task 4: Real Output Verification and Publication

**Files:**
- Modify: `docs/superpowers/plans/2026-06-27-full-trajectory-compare.md`

- [x] Launch against `outputs/viser/desk_depth` and `outputs/viser/desk_geometry` and verify all 123 frames are reported rather than the old 100-frame limit.
- [x] Inspect first, middle, and last slider frames in the browser; verify trajectory splines and no new console errors.
- [x] Commit the new script, tests, specification, and plan; push `main`.
- [x] Provide cloud packaging and local launch commands for KITTI 09 standard outputs.
