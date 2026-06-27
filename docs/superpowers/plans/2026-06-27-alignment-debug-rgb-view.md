# Alignment Debug RGB View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional RGB, mutual-confidence-filtered Viser mode that shows one alignment pair or all recorded alignment regions without changing inference or trace files.

**Architecture:** Keep solid-color `key` and `process` rendering unchanged. Add pure helpers that map `pair_NNNN` to local sampled images, construct aligned point/color arrays, aggregate all pairs with one deterministic per-method cap, and derive an initial source-camera view. `main()` selects the existing or RGB path from CLI arguments.

**Tech Stack:** Python, NumPy, PyTorch image preprocessing, Viser, pytest.

---

### Task 1: Pair Metadata and RGB Image Mapping

**Files:**
- Modify: `tests/test_alignment_debug_viewer.py`
- Modify: `eval/vis_alignment_debug.py`

- [ ] **Step 1: Write failing range and RGB-shape tests**

```python
def test_pair_sampled_image_range_uses_window_stride():
    viewer = _load_viewer_module()
    start, stop = viewer._pair_sampled_image_range(
        "pair_0003", {"window_size": 30, "overlap": 10}
    )
    assert (start, stop) == (60, 70)


def test_load_pair_rgb_uses_matching_sampled_images():
    viewer = _load_viewer_module()
    seen = {}

    def fake_loader(paths):
        seen["paths"] = paths
        return np.ones((2, 3, 4, 5), dtype=np.float32)

    rgb = viewer._load_pair_rgb(
        "pair_0001",
        [f"frame_{i}" for i in range(8)],
        {"window_size": 4, "overlap": 2},
        image_loader=fake_loader,
        expected_shape=(2, 4, 5),
    )
    assert seen["paths"] == ["frame_2", "frame_3"]
    assert rgb.shape == (2, 4, 5, 3)
    assert rgb.dtype == np.uint8
```

- [ ] **Step 2: Run tests and verify RED**

```bash
conda run -n vggt-dem python -m pytest \
  tests/test_alignment_debug_viewer.py::test_pair_sampled_image_range_uses_window_stride \
  tests/test_alignment_debug_viewer.py::test_load_pair_rgb_uses_matching_sampled_images -q
```

Expected: FAIL because both helpers are undefined.

- [ ] **Step 3: Implement metadata and image helpers**

```python
def _load_debug_metadata(debug_dir):
    path = Path(debug_dir) / "meta.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing alignment debug metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _pair_sampled_image_range(pair_name, metadata):
    pair_number = int(pair_name.rsplit("_", 1)[1])
    window_size = int(metadata["window_size"])
    overlap = int(metadata["overlap"])
    start = pair_number * (window_size - overlap)
    return start, start + overlap


def _load_pair_rgb(pair_name, sampled_paths, metadata, *, image_loader, expected_shape):
    start, stop = _pair_sampled_image_range(pair_name, metadata)
    paths = sampled_paths[start:stop]
    if len(paths) != stop - start:
        raise ValueError(f"Not enough sampled images for {pair_name}: need [{start}:{stop}].")
    images = image_loader(paths)
    if hasattr(images, "detach"):
        images = images.detach().cpu().numpy()
    images = np.asarray(images, dtype=np.float32)
    rgb = np.clip(images.transpose(0, 2, 3, 1) * 255.0, 0, 255).astype(np.uint8)
    if rgb.shape[:3] != tuple(expected_shape):
        raise ValueError(f"RGB shape {rgb.shape[:3]} does not match {expected_shape}.")
    return rgb
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Expected: `2 passed`.

### Task 2: High-Confidence RGB Cloud Construction

**Files:**
- Modify: `tests/test_alignment_debug_viewer.py`
- Modify: `eval/vis_alignment_debug.py`

- [ ] **Step 1: Write a failing point/color alignment test**

```python
def test_build_rgb_cloud_filters_points_and_colors_with_same_mask():
    viewer = _load_viewer_module()
    points = np.arange(24, dtype=np.float32).reshape(2, 2, 2, 3)
    rgb = np.arange(24, dtype=np.uint8).reshape(2, 2, 2, 3)
    mask = np.array([[[True, False], [False, True]], [[False, True], [False, False]]])
    cloud_points, cloud_colors = viewer._build_rgb_cloud(points, rgb, mask, frame_index=None)
    np.testing.assert_array_equal(cloud_points, points[mask])
    np.testing.assert_array_equal(cloud_colors, rgb[mask])
```

- [ ] **Step 2: Run test and verify RED**

```bash
conda run -n vggt-dem python -m pytest \
  tests/test_alignment_debug_viewer.py::test_build_rgb_cloud_filters_points_and_colors_with_same_mask -q
```

Expected: FAIL because `_build_rgb_cloud` is undefined.

- [ ] **Step 3: Implement aligned filtering**

```python
def _build_rgb_cloud(points, rgb, mask, *, frame_index):
    points = _select_frame(points, frame_index)
    rgb = _select_frame(rgb, frame_index)
    mask = _select_frame(mask, frame_index)
    if points.shape[:-1] != rgb.shape[:-1] or points.shape[:-1] != mask.shape:
        raise ValueError("Point, RGB, and confidence-mask shapes must match.")
    return flatten_points(points, mask=mask), np.asarray(rgb, dtype=np.uint8)[mask].reshape(-1, 3)
```

- [ ] **Step 4: Run focused test and verify GREEN**

Expected: `1 passed`.

### Task 3: RGB and All-Pairs CLI Contract

**Files:**
- Modify: `tests/test_alignment_debug_viewer.py`
- Modify: `eval/vis_alignment_debug.py`

- [ ] **Step 1: Write failing CLI tests**

```python
import pytest


def test_parse_args_accepts_rgb_all_pairs_options():
    viewer = _load_viewer_module()
    args = viewer.parse_args([
        "--debug_dir", "scene", "--layer_mode", "rgb",
        "--image_dir", "data/09/image_2", "--sample_interval", "10",
        "--camera_view", "source", "--all_pairs",
    ])
    viewer._validate_args(args)
    assert args.layer_mode == "rgb"
    assert args.all_pairs is True


def test_rgb_mode_requires_image_dir():
    viewer = _load_viewer_module()
    args = viewer.parse_args(["--debug_dir", "scene", "--layer_mode", "rgb"])
    with pytest.raises(ValueError, match="--image_dir"):
        viewer._validate_args(args)
```

- [ ] **Step 2: Run tests and verify RED**

Expected: parsing fails because the RGB options are absent.

- [ ] **Step 3: Add arguments and validation**

Add `rgb` to `--layer_mode`; add `--image_dir`, positive integer `--sample_interval`, `--camera_view {default,source}`, and `--all_pairs`. Validate with:

```python
def _validate_args(args):
    if args.sample_interval <= 0:
        raise ValueError("--sample_interval must be positive.")
    if args.layer_mode == "rgb" and not args.image_dir:
        raise ValueError("--image_dir is required for RGB mode.")
    if args.all_pairs and args.layer_mode != "rgb":
        raise ValueError("--all_pairs is only supported in RGB mode.")
    if args.all_pairs and args.coordinate_space == "local":
        raise ValueError("--all_pairs requires world coordinates.")
    return args
```

- [ ] **Step 4: Run all viewer tests**

```bash
conda run -n vggt-dem python -m pytest tests/test_alignment_debug_viewer.py -q
```

Expected: all viewer tests pass.

### Task 4: All-Pairs Global Sampling

**Files:**
- Modify: `tests/test_alignment_debug_viewer.py`
- Modify: `eval/vis_alignment_debug.py`

- [ ] **Step 1: Write a failing global-cap test**

```python
def test_aggregate_rgb_clouds_applies_one_global_point_cap():
    viewer = _load_viewer_module()
    clouds = [
        (np.arange(18, dtype=np.float32).reshape(6, 3), np.full((6, 3), 10, dtype=np.uint8)),
        (np.arange(18, 36, dtype=np.float32).reshape(6, 3), np.full((6, 3), 20, dtype=np.uint8)),
    ]
    points, colors = viewer._aggregate_rgb_clouds(clouds, max_points=5, seed=7)
    assert points.shape == (5, 3)
    assert colors.shape == (5, 3)
    assert set(np.unique(colors[:, 0])).issubset({10, 20})
```

- [ ] **Step 2: Run test and verify RED**

Expected: FAIL because `_aggregate_rgb_clouds` is undefined.

- [ ] **Step 3: Implement one deterministic global sample**

```python
def _aggregate_rgb_clouds(clouds, *, max_points, seed):
    points = np.concatenate([cloud[0] for cloud in clouds], axis=0)
    colors = np.concatenate([cloud[1] for cloud in clouds], axis=0)
    return sample_points(points, colors, max_points=max_points, seed=seed)
```

Add `_prepare_pair_rgb_cloud()` to transform refined target points to world coordinates and apply the matching RGB and mutual mask. In all-pairs mode, prepare each pair, concatenate once, sample once, and add one cloud per method.

- [ ] **Step 4: Run all viewer tests and verify GREEN**

```bash
conda run -n vggt-dem python -m pytest tests/test_alignment_debug_viewer.py -q
```

### Task 5: Detail and Overview Camera Initialization

**Files:**
- Modify: `tests/test_alignment_debug_viewer.py`
- Modify: `eval/vis_alignment_debug.py`

- [ ] **Step 1: Write a failing camera-spec test**

```python
def test_source_camera_spec_uses_camera_forward_and_up_axes():
    viewer = _load_viewer_module()
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    spec = viewer._source_camera_spec(pose, look_distance=4.0)
    np.testing.assert_allclose(spec["position"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(spec["look_at"], [1.0, 2.0, 7.0])
    np.testing.assert_allclose(spec["up_direction"], [0.0, -1.0, 0.0])
```

- [ ] **Step 2: Run test and verify RED**

Expected: FAIL because `_source_camera_spec` is undefined.

- [ ] **Step 3: Implement camera spec and connection callback**

```python
def _source_camera_spec(pose, *, look_distance=4.0):
    pose = np.asarray(pose, dtype=np.float32)
    position = pose[:3, 3]
    return {
        "position": position,
        "look_at": position + pose[:3, 2] * float(look_distance),
        "up_direction": -pose[:3, 1],
        "fov": np.deg2rad(60.0),
    }
```

Register `on_client_connect` only for `--camera_view source`. Use the selected
target pose in detail mode. For all-pairs mode, use `_overview_camera_spec()`.
Compute 1st/99th percentile bounds from both displayed, offset clouds, look at
their center, and place the camera on a stable oblique direction far enough to
fit the full extent. Keep `up_direction=(0, 0, -1)`.

- [ ] **Step 4: Run all viewer tests and verify GREEN**

Expected: all tests pass.

### Task 6: Main Integration and Browser Verification

**Files:**
- Modify: `eval/vis_alignment_debug.py`
- Modify: `tests/test_alignment_debug_viewer.py`

- [ ] **Step 1: Integrate the RGB route**

Load sampled paths once with `list_image_paths(args.image_dir, args.sample_interval)`. Cache RGB arrays by pair name so baseline and geometry share identical colors. Keep `_add_pair_layers()` unchanged for `key` and `process`; route `rgb` to the single-pair or all-pairs helpers. GUI text states RGB filtering, pair scope, cap, and coordinate space.

- [ ] **Step 2: Run the relevant regression suite**

```bash
conda run -n vggt-dem python -m pytest \
  tests/test_alignment_debug_viewer.py \
  tests/test_streaming_window_engine.py \
  tests/test_alignment_debug.py \
  tests/test_demo_cli.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Run syntax and diff checks**

```bash
conda run -n vggt-dem python -m py_compile eval/vis_alignment_debug.py
git diff --check
```

Expected: both commands exit 0.

- [ ] **Step 4: Launch detail mode**

```bash
conda run -n vggt-dem python eval/vis_alignment_debug.py \
  --baseline_debug_dir outputs/debug_alignment/kitti09_pi3_depth_s10_w30_o10_world_debug \
  --geometry_debug_dir outputs/debug_alignment/kitti09_pi3_geometry_s10_w30_o10_world_debug \
  --layer_mode rgb --image_dir data/09/image_2 --sample_interval 10 \
  --coordinate_space world --camera_view source --all_frames \
  --compare_axis x --compare_offset 2.5 --port 8097
```

Expected: road and vegetation are recognizable for one pair.

- [ ] **Step 5: Launch all-pairs mode**

```bash
conda run -n vggt-dem python eval/vis_alignment_debug.py \
  --baseline_debug_dir outputs/debug_alignment/kitti09_pi3_depth_s10_w30_o10_world_debug \
  --geometry_debug_dir outputs/debug_alignment/kitti09_pi3_geometry_s10_w30_o10_world_debug \
  --layer_mode rgb --image_dir data/09/image_2 --sample_interval 10 \
  --coordinate_space world --camera_view source --all_pairs \
  --compare_axis x --compare_offset 2.5 --max_points 200000 --port 8098
```

Expected: seven alignment regions appear as one capped cloud per method and the browser remains responsive.

- [ ] **Step 6: Inspect both pages in the in-app browser**

Verify recognizable scene content, separate baseline/geometry folders, all-pairs status text, a nonblank canvas, and no actionable console errors. Preserve 8098 for user review and stop temporary probe services.

- [ ] **Step 7: Commit the implementation**

```bash
git add eval/vis_alignment_debug.py tests/test_alignment_debug_viewer.py docs/superpowers/plans/2026-06-27-alignment-debug-rgb-view.md
git commit -m "feat: add RGB all-pairs alignment viewer"
```
