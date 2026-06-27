# Full Trajectory Comparison Viewer Design

## Goal

Add a standalone, read-only Viser script that compares complete LASER depth and
geometry results from their existing `outputs/viser/<scene>` directories. The
script must not modify inference, alignment, segmentation, existing viewers,
or the saved result format.

## Inputs

The script accepts `--baseline_dir` and `--geometry_dir`. Each directory must
contain the standard files produced by `save_for_viser()`:

- `pred_traj.txt` in TUM order: timestamp, translation, quaternion `wxyz`;
- `pred_intrinsics.txt`, one 3x3 matrix per frame;
- `frame_NNNN.png`, `frame_NNNN.npy`, and `conf_N.npy`.

Files are sorted numerically. The loader requires equal RGB, depth, confidence,
intrinsic, and pose counts within each run, and equal frame counts between the
two runs. It does not drop the last frame and has no 100-frame limit.

## Coordinate Alignment

Baseline poses remain in their saved coordinates. Geometry is transformed once
with:

```text
T_align = T_baseline[0] @ inverse(T_geometry[0])
T_geometry_aligned[i] = T_align @ T_geometry[i]
```

This makes frame 0 coincide while preserving every later relative difference
and accumulated drift. Geometry point clouds use the aligned geometry poses.
No independent middle-frame normalization is allowed.

## Point Clouds

Depth maps are unprojected with their saved intrinsics, then transformed by the
corresponding camera-to-world poses. RGB supplies point colors. Per-frame
confidence filtering keeps values at or above a configurable quantile.

The viewer has two complementary representations:

1. **Overview:** all frames contribute downsampled, high-confidence points.
   Deterministic global sampling applies one `--max_points` cap per method.
2. **Frame detail:** a GUI slider shows one full-detail frame per method. On a
   slider update, the previous two detail point-cloud nodes are removed and
   replaced, so browser memory does not grow with sequence length.

`--pixel_stride` controls overview density without changing the trajectories.

## Trajectories and UI

Both complete camera-center trajectories are shown as separate colored splines,
with small start markers. GUI controls provide frame selection, overview
visibility, baseline visibility, geometry visibility, and confidence quantile.
The information panel reports frame count, alignment rule, and active point
caps.

An automatic overview camera uses robust bounds from both trajectory lines and
overview point clouds. Baseline and geometry stay in the same coordinate system;
they are not separated by a decorative offset.

## Files and Isolation

Create only:

- `eval/vis_full_trajectory_compare.py`;
- `tests/test_full_trajectory_compare.py`;
- design and implementation documentation.

Existing Python modules are imported only for generic helpers when doing so does
not alter their state. The viewer treats all experiment outputs as read-only.

## Resource Limits

Only the two capped overview clouds and two current-frame detail clouds reside
in the browser. Default limits target ordinary local machines; users may lower
`--max_points` or increase `--pixel_stride` for constrained systems. Complete
trajectory splines are always retained because their memory cost is negligible.

## Verification

Automated tests cover numeric file ordering, manifest validation, TUM pose
parsing, first-frame geometry alignment, depth unprojection, confidence
filtering, deterministic global sampling, and frame-node replacement logic.

Manual verification uses the downloaded KITTI 09 depth and geometry standard
outputs, confirms the complete frame count, checks both trajectory splines,
uses the frame slider at the beginning/middle/end, and verifies a responsive
browser with no new console errors.
