# Ground-Truth Trajectory Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional KITTI ground truth, independent baseline/geometry Sim(3) alignment, two ATE values, and single/pair/triple visibility controls to the standalone full-trajectory Viser viewer.

**Architecture:** Extend only `eval/vis_full_trajectory_compare.py` with pure ground-truth, similarity, and visibility helpers, then connect those helpers to the existing Viser layer. Without `--gt_traj`, preserve the current baseline-frame behavior byte-for-byte at the decision boundary; with ground truth, transform each method's poses and point clouds using its own Umeyama similarity to the same sampled reference.

**Tech Stack:** Python, NumPy, SciPy Rotation, evo Umeyama geometry, Viser, pytest.

---

### Task 1: KITTI Ground-Truth Loading and Exact Sampling

**Files:**
- Modify: `tests/test_full_trajectory_compare.py`
- Modify: `eval/vis_full_trajectory_compare.py`

- [x] **Step 1: Write failing KITTI parsing and sampling tests**

Add focused tests that require 12-column KITTI rows to become homogeneous 4x4 poses and require stride sampling to match the prediction count exactly:

```python
def test_load_kitti_poses_builds_homogeneous_matrices(tmp_path):
    viewer = _load_module()
    path = tmp_path / "09.txt"
    np.savetxt(path, [[1, 0, 0, 2, 0, 1, 0, 3, 0, 0, 1, 4]])
    poses = viewer.load_kitti_poses(path)
    np.testing.assert_allclose(poses[0], [[1, 0, 0, 2], [0, 1, 0, 3], [0, 0, 1, 4], [0, 0, 0, 1]])


def test_sample_ground_truth_requires_exact_count():
    viewer = _load_module()
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 21, axis=0)
    assert len(viewer.sample_ground_truth(poses, stride=2, expected_count=11)) == 11
    with pytest.raises(ValueError, match="sampled ground-truth count"):
        viewer.sample_ground_truth(poses, stride=3, expected_count=11)
```

- [x] **Step 2: Run focused tests and verify RED**

Run:

```bash
conda run -n vggt-dem python -m pytest \
  tests/test_full_trajectory_compare.py::test_load_kitti_poses_builds_homogeneous_matrices \
  tests/test_full_trajectory_compare.py::test_sample_ground_truth_requires_exact_count -q
```

Expected: both tests fail because the helpers do not exist.

- [x] **Step 3: Implement strict loading and sampling**

Add pure helpers:

```python
def load_kitti_poses(path):
    rows = np.atleast_2d(np.loadtxt(path, dtype=np.float32))
    if rows.shape[1] != 12:
        raise ValueError(f"Expected 12 columns in KITTI trajectory {path}, got {rows.shape[1]}")
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], len(rows), axis=0)
    poses[:, :3, :] = rows.reshape(-1, 3, 4)
    return poses


def sample_ground_truth(poses, *, stride, expected_count):
    if stride <= 0:
        raise ValueError("ground-truth stride must be positive")
    sampled = np.asarray(poses, dtype=np.float32)[::stride]
    if len(sampled) != expected_count:
        raise ValueError(
            f"sampled ground-truth count {len(sampled)} from {len(poses)} poses "
            f"with stride {stride}; expected {expected_count}"
        )
    return sampled
```

- [x] **Step 4: Run the complete viewer test file and verify GREEN**

Run: `conda run -n vggt-dem python -m pytest tests/test_full_trajectory_compare.py -q`

Expected: all tests pass.

### Task 2: Sim(3), Pose/Point Transformation, and ATE

**Files:**
- Modify: `tests/test_full_trajectory_compare.py`
- Modify: `eval/vis_full_trajectory_compare.py`

- [x] **Step 1: Write failing similarity and ATE tests**

Use a synthetic reference and a prediction related by known rotation, scale,
and translation. Tests must prove that pose translations, orientations, and
ordinary points receive the same similarity and that aligned ATE is zero:

```python
def test_align_sim3_transforms_poses_points_and_computes_ate():
    viewer = _load_module()
    reference = np.repeat(np.eye(4, dtype=np.float32)[None], 4, axis=0)
    reference[:, :3, 3] = [[0, 0, 0], [1, 0, 0], [1, 2, 0], [1, 2, 3]]
    prediction = reference.copy()
    prediction[:, :3, 3] = (reference[:, :3, 3] - [4, -2, 1]) / 2
    aligned, similarity = viewer.align_poses_sim3(prediction, reference)
    np.testing.assert_allclose(aligned[:, :3, 3], reference[:, :3, 3], atol=1e-5)
    assert viewer.translation_ate(aligned, reference) == pytest.approx(0.0, abs=1e-5)
    points = viewer.apply_similarity_to_points(np.array([[0, 0, 0]], np.float32), similarity)
    np.testing.assert_allclose(points, [similarity[1]], atol=1e-5)
```

- [x] **Step 2: Run the new test and verify RED**

Run:

```bash
conda run -n vggt-dem python -m pytest \
  tests/test_full_trajectory_compare.py::test_align_sim3_transforms_poses_points_and_computes_ate -q
```

Expected: failure because `align_poses_sim3` is undefined.

- [x] **Step 3: Implement similarity helpers with evo's Umeyama solver**

Add:

```python
from evo.core.geometry import umeyama_alignment


def apply_similarity_to_points(points, similarity):
    rotation, translation, scale = similarity
    return (scale * (np.asarray(points) @ rotation.T) + translation).astype(np.float32)


def apply_similarity_to_poses(poses, similarity):
    rotation, translation, scale = similarity
    aligned = np.asarray(poses, dtype=np.float32).copy()
    aligned[:, :3, 3] = apply_similarity_to_points(aligned[:, :3, 3], similarity)
    aligned[:, :3, :3] = rotation[None] @ aligned[:, :3, :3]
    return aligned


def align_poses_sim3(prediction, reference):
    rotation, translation, scale = umeyama_alignment(
        np.asarray(prediction)[:, :3, 3].T,
        np.asarray(reference)[:, :3, 3].T,
        with_scale=True,
    )
    similarity = (rotation.astype(np.float32), translation.astype(np.float32), float(scale))
    return apply_similarity_to_poses(prediction, similarity), similarity


def translation_ate(prediction, reference):
    errors = np.asarray(prediction)[:, :3, 3] - np.asarray(reference)[:, :3, 3]
    return float(np.sqrt(np.mean(np.sum(errors * errors, axis=1))))
```

Reject non-finite solver outputs with a clear `ValueError` before applying the
similarity.

- [x] **Step 4: Route all predicted point clouds through the solved similarity**

Extend `frame_cloud()` and `overview_cloud()` with an optional `similarity=None`.
After camera-to-world projection, apply the similarity when present:

```python
if similarity is not None:
    points_world = apply_similarity_to_points(points_world.reshape(-1, 3), similarity).reshape(points_world.shape)
```

This keeps the no-ground-truth path unchanged and ensures trajectory and cloud
coordinates agree in ground-truth mode.

- [x] **Step 5: Run focused and regression tests**

Run:

```bash
conda run -n vggt-dem python -m pytest \
  tests/test_full_trajectory_compare.py \
  tests/test_alignment_debug_viewer.py -q
```

Expected: all tests pass.

### Task 3: Three Quick Comparisons and Independent Visibility

**Files:**
- Modify: `tests/test_full_trajectory_compare.py`
- Modify: `eval/vis_full_trajectory_compare.py`

- [x] **Step 1: Write a failing visibility-preset test**

```python
@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        ("GT vs Baseline", (True, True, False)),
        ("GT vs Geometry", (True, False, True)),
        ("Baseline vs Geometry", (False, True, True)),
    ],
)
def test_comparison_visibility(preset, expected):
    viewer = _load_module()
    assert viewer.comparison_visibility(preset) == expected
```

- [x] **Step 2: Run the test and verify RED**

Run: `conda run -n vggt-dem python -m pytest tests/test_full_trajectory_compare.py::test_comparison_visibility -q`

Expected: three failures because the helper does not exist.

- [x] **Step 3: Implement preset mapping and CLI**

Add `comparison_visibility()` with an explicit mapping. Extend `parse_args()`:

```python
parser.add_argument("--gt_traj", type=Path)
parser.add_argument("--gt_format", choices=("kitti",), default="kitti")
parser.add_argument("--gt_stride", type=int, default=1)
```

Validate positive stride. In `build_viewer()`, load and sample KITTI ground
truth only when `--gt_traj` is present; solve baseline and geometry similarities
independently, compute both ATE values, and retain the existing first-frame path
otherwise.

- [x] **Step 4: Add ground truth and controls to Viser**

Add the gray ground-truth spline and start marker. In the Display folder add an
exclusive three-option button group for quick comparisons plus three independent
checkboxes. A preset callback sets the checkbox values, and the existing shared
visibility updater controls each complete method group. Do not add a GT point
cloud. Add a `Fit full trajectory` button that restores a camera computed from
the complete trajectory min/max bounds, not percentile or point-cloud bounds.

The information markdown in GT mode includes:

```text
Alignment: independent Sim(3) to Ground Truth
LASER baseline ATE: <value>
LASER-Geometry ATE: <value>
```

- [x] **Step 5: Verify tests and compilation**

Run:

```bash
conda run -n vggt-dem python -m pytest tests -q
conda run -n vggt-dem python -m py_compile eval/vis_full_trajectory_compare.py
git diff --check
```

Expected: all tests pass, compilation exits zero, and diff check is empty.

### Task 4: KITTI 09 Browser Verification and Publication

**Files:**
- Modify: `docs/superpowers/plans/2026-06-28-ground-truth-trajectory-view.md`

- [x] **Step 1: Launch the three-trajectory KITTI 09 viewer**

Run:

```bash
conda run -n vggt-dem python eval/vis_full_trajectory_compare.py \
  --baseline_dir outputs/viser/kitti09_pi3_depth_s10_w30_o10_world_debug \
  --geometry_dir outputs/viser/kitti09_pi3_geometry_s10_w30_o10_world_debug \
  --gt_traj data/dataset/poses/09.txt \
  --gt_format kitti \
  --gt_stride 10 \
  --port 8101 \
  --frame_stride 1 \
  --pixel_stride 6 \
  --detail_pixel_stride 2 \
  --max_points 200000 \
  --conf_quantile 0.7
```

Expected: 160 frames, gray/blue/green trajectories, and two finite ATE values.

- [x] **Step 2: Exercise all visibility modes**

In the browser, select each quick comparison and verify its exact pair. Then
manually show Ground Truth only, baseline only, geometry only, and all three.
Move the frame control to 0, 80, and 159, use wheel zoom, and verify
`Fit full trajectory` restores the complete scene beside the GUI panel. Also
start the viewer once without `--gt_traj` and confirm the original two-method
mode reaches `Viewer ready`.

- [x] **Step 3: Commit and push**

Stage only the viewer, focused tests, design, and implementation plan; never
stage the untracked `viser/` directory:

```bash
git add eval/vis_full_trajectory_compare.py tests/test_full_trajectory_compare.py docs/superpowers/plans/2026-06-28-ground-truth-trajectory-view.md
git commit -m "feat: compare trajectories with ground truth"
git push origin main
```

- [x] **Step 4: Provide the updated local/cloud launch command**

Report the pushed commit, test count, ATE values observed locally, and the exact
KITTI 09 command from Step 1.
