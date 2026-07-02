# Geometry Baseline Parameter Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in Geometry segmentation profile that changes only Felzenszwalb parameters to the LASER depth baseline values.

**Architecture:** Keep the legacy Geometry functions untouched and add two thin baseline-parameter wrappers. Route an explicit profile through the Geometry graph builder and streaming engine; the depth builder remains unaware of the profile. Expose the profile in `demo.py` with `legacy` as the compatibility default.

**Tech Stack:** Python, NumPy, scikit-image, PyTorch, argparse, pytest

---

### Task 1: Add Isolated Baseline-Parameter Wrappers

**Files:**
- Modify: `inference_engine/utils/geometry_segmentation.py`
- Test: `tests/test_geometry_segmentation.py`

- [ ] **Step 1: Write the failing wrapper tests**

Add tests that monkeypatch `segment_geometry_felzenszwalb_rag_stages`, invoke both new wrappers, and assert that the delegated call contains:

```python
assert calls["seg_scale"] == 300
assert calls["seg_sigma"] == 1.1
assert calls["seg_min_size"] == 500
assert calls["normal_method"] == "sobel"
```

The labels wrapper must return `stages.merged_labels` without changing it.

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
conda run -n vggt-dem pytest tests/test_geometry_segmentation.py -q
```

Expected: failure because `segment_geometry_felzenszwalb_rag_baseline_params_stages` and `segment_geometry_felzenszwalb_rag_baseline_params` do not exist.

- [ ] **Step 3: Implement the two wrappers**

Add wrappers with the same public inputs as the legacy functions. The stages wrapper delegates to the existing implementation:

```python
return segment_geometry_felzenszwalb_rag_stages(
    depth_map,
    conf_map=conf_map,
    intrinsic=intrinsic,
    point_map=point_map,
    top_conf_percentile=top_conf_percentile,
    depth_merge_thresh=depth_merge_thresh,
    normal_thresh_deg=normal_thresh_deg,
    seg_scale=300,
    seg_sigma=1.1,
    seg_min_size=500,
    normal_method=normal_method,
    batch_idx=batch_idx,
)
```

The labels wrapper returns `.merged_labels` from the new stages wrapper. Do not change the defaults or body of either legacy function.

- [ ] **Step 4: Run the focused tests and verify pass**

Run:

```bash
conda run -n vggt-dem pytest tests/test_geometry_segmentation.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the wrapper unit**

```bash
git add inference_engine/utils/geometry_segmentation.py tests/test_geometry_segmentation.py
git commit -m "feat: add geometry baseline parameter wrappers"
```

### Task 2: Route the Geometry Segmentation Profile

**Files:**
- Modify: `inference_engine/utils/lsa.py`
- Test: `tests/test_lsa_graph_builders.py`

- [ ] **Step 1: Write failing profile-selection tests**

Extend the graph-builder tests to monkeypatch all four Geometry functions and assert:

```python
lsa.build_geometry_sp_graph(depth, geometry_seg_profile="legacy")
assert calls["op"] is lsa.segment_geometry_felzenszwalb_rag

lsa.build_geometry_sp_graph(depth, geometry_seg_profile="baseline_params")
assert calls["op"] is lsa.segment_geometry_felzenszwalb_rag_baseline_params
```

Repeat the baseline assertion for the trace path, where `ordered_batch_apply` must receive `segment_geometry_felzenszwalb_rag_baseline_params_stages`. Assert that an unknown profile raises `ValueError`.

- [ ] **Step 2: Run the focused tests and verify failure**

```bash
conda run -n vggt-dem pytest tests/test_lsa_graph_builders.py -q
```

Expected: failure because the profile argument and imported wrappers are absent.

- [ ] **Step 3: Implement explicit profile dispatch**

Import the wrappers and add:

```python
GEOMETRY_SEGMENTATION_PROFILES = {
    "legacy": (
        segment_geometry_felzenszwalb_rag,
        segment_geometry_felzenszwalb_rag_stages,
    ),
    "baseline_params": (
        segment_geometry_felzenszwalb_rag_baseline_params,
        segment_geometry_felzenszwalb_rag_baseline_params_stages,
    ),
}
```

Add `geometry_seg_profile="legacy"` to `build_geometry_sp_graph`, validate it, and select the ordinary and stages operations before the trace branch. Do not add this argument to `build_depth_sp_graph`.

- [ ] **Step 4: Run graph-builder tests and verify pass**

```bash
conda run -n vggt-dem pytest tests/test_lsa_graph_builders.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit profile routing**

```bash
git add inference_engine/utils/lsa.py tests/test_lsa_graph_builders.py
git commit -m "feat: route geometry segmentation profiles"
```

### Task 3: Expose the Profile Through Engine and CLI

**Files:**
- Modify: `inference_engine/streaming_window_engine.py`
- Modify: `demo.py`
- Test: `tests/test_streaming_window_engine.py`
- Test: `tests/test_demo_cli.py`

- [ ] **Step 1: Write failing engine and CLI tests**

Add engine tests that construct with `geometry_seg_profile="baseline_params"`, assert the value is forwarded only by `_build_geometry_segment_graph`, and assert an invalid value raises `ValueError`. Extend the CLI test to assert:

```python
assert parser.parse_args([]).geometry_seg_profile == "legacy"
assert parser.parse_args([
    "--geometry_seg_profile", "baseline_params"
]).geometry_seg_profile == "baseline_params"
```

Update the fake engine assertion to confirm `load_model()` forwards the selected profile.

- [ ] **Step 2: Run the focused tests and verify failure**

```bash
conda run -n vggt-dem pytest \
  tests/test_streaming_window_engine.py \
  tests/test_demo_cli.py -q
```

Expected: failure because the engine and CLI do not yet accept the profile.

- [ ] **Step 3: Add engine validation and forwarding**

Add `geometry_seg_profile: str = "legacy"` to `StreamingWindowEngine.__init__`, validate membership in `("legacy", "baseline_params")`, store it, and pass it from `_build_geometry_segment_graph`:

```python
kwargs = {
    "conf_map": conf.cpu().numpy(),
    "top_conf_percentile": self.top_conf_percentile,
    "point_map": local_points_np,
    "intrinsic": intrinsic_np,
    "normal_method": self.normal_method,
    "geometry_seg_profile": self.geometry_seg_profile,
}
```

Leave `_build_depth_segment_graph` unchanged.

- [ ] **Step 4: Add the CLI option and connect it**

Add:

```python
parser.add_argument(
    "--geometry_seg_profile",
    default="legacy",
    choices=["legacy", "baseline_params"],
    help="geometry segmentation implementation profile",
)
```

Pass `geometry_seg_profile=args.geometry_seg_profile` to `StreamingWindowEngine`.

- [ ] **Step 5: Run focused and full regression tests**

```bash
conda run -n vggt-dem pytest \
  tests/test_geometry_segmentation.py \
  tests/test_lsa_graph_builders.py \
  tests/test_streaming_window_engine.py \
  tests/test_demo_cli.py -q

conda run -n vggt-dem pytest -q
```

Expected: all tests pass. Existing tests must continue to exercise `legacy` by default.

- [ ] **Step 6: Verify CLI help and repository diff**

```bash
conda run -n vggt-dem python demo.py --help
git diff --check
git status --short
```

Expected: help lists both profile choices; diff check is empty; only intended tracked files plus the pre-existing untracked `.superpowers/` and `viser/` appear.

- [ ] **Step 7: Commit the engine and CLI unit**

```bash
git add demo.py inference_engine/streaming_window_engine.py \
  tests/test_streaming_window_engine.py tests/test_demo_cli.py
git commit -m "feat: expose geometry baseline parameter profile"
```

### Task 4: Publish and Run the A1 Experiment

**Files:**
- Modify: `docs/CLOUD_RUN.md`

- [ ] **Step 1: Add the non-debug KITTI A1 commands**

Document the `s1-w75-o30` Geometry command using `--geometry_seg_profile baseline_params`, a unique Viser scene/cache name, and the matching `quick_eval_local.py` command with KITTI GT stride `1`. Explicitly omit `--debug_alignment`.

- [ ] **Step 2: Verify the documented commands against CLI help**

```bash
conda run -n vggt-dem python demo.py --help
conda run -n vggt-dem python eval/quick_eval_local.py --help
```

Expected: every documented option is accepted.

- [ ] **Step 3: Commit documentation and push**

```bash
git add docs/CLOUD_RUN.md
git commit -m "docs: add geometry parameter parity experiment"
git push origin main
```

Expected: `origin/main` advances to the final local commit.

