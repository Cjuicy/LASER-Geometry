# Ground-Truth Trajectory Comparison Design

## Goal

Extend the standalone full-trajectory Viser viewer with an optional ground-truth
trajectory. The resulting scene supports direct comparison among Ground Truth,
the LASER baseline, and LASER-Geometry without changing inference, saved model
outputs, or the existing two-trajectory behavior.

## Scope and Isolation

Only `eval/vis_full_trajectory_compare.py`, its focused tests, and documentation
change. `demo.py`, `StreamingWindowEngine`, alignment/refinement code, and the
standard `outputs/viser/<scene>` format remain untouched.

Ground truth is optional. When `--gt_traj` is absent, the viewer keeps its
current behavior: baseline remains in its saved frame and geometry receives the
existing first-frame rigid alignment to baseline.

## Inputs and Frame Matching

The new CLI accepts:

- `--gt_traj`: optional ground-truth trajectory path;
- `--gt_format`: initially `kitti`, with strict 12-value pose rows;
- `--gt_stride`: positive source-frame stride, default `1`.

KITTI rows are parsed as camera-to-world 3x4 matrices and completed to 4x4
poses. Ground truth is sampled with `poses[::gt_stride]`. The sampled count must
equal both prediction counts; otherwise startup fails with a message containing
the source count, stride, sampled count, and prediction count. No silent
truncation or nearest-neighbor resampling is allowed.

For KITTI 09 stride-10 experiments, the expected mapping is 1591 source poses
to 160 prediction poses using `--gt_stride 10`.

## Sim(3) Alignment and ATE

When ground truth is present, the baseline and geometry predictions are aligned
independently to the same sampled ground truth using Umeyama Sim(3), matching
the project's ATE setting of `align=True` and `correct_scale=True`:

```text
baseline  --Sim(3)--> ground truth
geometry  --Sim(3)--> ground truth
```

Each solved similarity `(R, t, s)` transforms camera centers and point clouds
as `s * R * x + t`. Camera orientations transform as `R * R_camera`; scale is
not applied to rotation. This keeps each method's trajectory and scene cloud in
one consistent ground-truth coordinate system.

ATE is translation RMSE after alignment:

```text
sqrt(mean(sum((prediction_aligned_xyz - ground_truth_xyz)^2)))
```

The script computes and reports baseline ATE and geometry ATE at startup and in
the Viser information panel. Ground truth itself is not assigned an ATE.

## Scene and Controls

Trajectory colors are fixed:

- Ground Truth: gray;
- LASER baseline: blue;
- LASER-Geometry: green.

The Display folder adds three quick comparison buttons:

1. `GT vs Baseline`;
2. `GT vs Geometry`;
3. `Baseline vs Geometry`.

Each button sets the three independent visibility checkboxes to the matching
pair. The same checkboxes remain manually editable:

- `Ground Truth`;
- `LASER baseline`;
- `LASER-Geometry`.

This supports every single-trajectory view, all three pairwise comparisons, and
all three trajectories at once. Method visibility controls its trajectory,
start marker, overview cloud, and current-frame detail cloud together. Ground
truth controls only its trajectory and start marker because no ground-truth
depth cloud is loaded.

The information panel always shows frame count, alignment mode, baseline ATE,
geometry ATE, confidence quantile, and point cap. Without ground truth it keeps
the current first-frame-alignment text and omits ATE values and ground-truth
controls.

The Display folder also provides `Fit full trajectory`. It restores the camera
after arbitrary wheel zooming and is independent of the selected frame or
visibility preset.

## Camera and Resource Behavior

Automatic camera bounds use the complete min/max extent of every visible-method
trajectory, including both endpoints. The camera stays inside Viser's useful
clipping distance and uses a wide field of view so the complete trajectory fits
beside the GUI panel. Point-cloud outliers do not determine this camera.
Overview point clouds retain the existing per-method streaming point cap. Only
the two predicted current-frame detail clouds are replaced on frame changes, so
adding ground truth has negligible browser memory cost.

## Errors and Validation

Startup rejects:

- missing or malformed ground-truth files;
- unsupported ground-truth formats;
- non-positive ground-truth stride;
- sampled ground-truth and prediction count mismatches;
- degenerate Sim(3) inputs that cannot produce a finite transform.

Errors are raised before the Viser server starts, avoiding a connected but empty
page.

## Verification

Automated tests cover KITTI pose parsing, stride/count validation, Sim(3)
recovery, pose and point transformation, ATE RMSE, optional no-ground-truth
compatibility, and visibility preset selection.

Manual verification uses KITTI 09:

```text
baseline: outputs/viser/kitti09_pi3_depth_s10_w30_o10_world_debug
geometry: outputs/viser/kitti09_pi3_geometry_s10_w30_o10_world_debug
ground truth: data/dataset/poses/09.txt
ground-truth stride: 10
```

The browser check exercises all three quick comparisons, each single-trajectory
view, the all-three view, frames 0, 80, and 159, wheel zoom recovery through
`Fit full trajectory`, and the original no-ground-truth mode.
