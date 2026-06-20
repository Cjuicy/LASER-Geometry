# Geometry-Aware LASER M0

## Goal

M0 follows the shared ChatGPT plan: replace only the source of LASER segment labels.

Original LASER:

```text
model prediction -> depth map -> depth segmentation -> make_sp_graph
-> LASER scale estimation -> LASER scale propagation -> refined points
```

M0 geometry mode:

```text
model prediction -> depth + confidence + camera/point map
-> geometry features -> geometry-aware segmentation -> make_sp_graph
-> LASER scale estimation -> LASER scale propagation -> refined points
```

Scale estimation, scale propagation, and loop closure are intentionally unchanged.

## Implemented

- `inference_engine/utils/geometry.py`
  - `depth_to_local_points_np`
  - `compute_normals_cross_np`
  - `compute_normals_sobel_np`
  - `compute_depth_edge_np`
  - `compute_normal_edge_np`
  - `build_geometry_info_np`

- `inference_engine/utils/depth.py`
  - region geometry descriptors
  - depth/normal/confidence merge decision
  - union-find based geometry region merging
  - `segment_geometry_felzenszwalb_rag`

- `inference_engine/utils/lsa.py`
  - `make_sp_graph(..., segment_mode="depth" | "geometry")`
  - `depth` mode keeps the original LASER path
  - `geometry` mode only changes labels before graph construction

- `demo.py` and `demo_lc.py`
  - `--segment_mode depth|geometry`
  - `--normal_method cross|sobel`

- `eval/quick_vis_geometry.py`
  - saves depth, confidence, normal, edge, segment, and RGB overlay debug images

- `eval/kitti_pose.py`
  - reads KITTI odometry `poses.txt` into the trajectory tuple expected by `eval.vo_eval`

## Current Scope

Use `cross` first. `sobel` is available for quick ablation. `pca` is deliberately reserved for a later, slower ablation and is not exposed in the demo CLIs.

Recommended first comparison:

```text
A0: --segment_mode depth    --depth_refine
A2: --segment_mode geometry --normal_method cross --depth_refine
```

Loop closure should be tested after the non-loop baseline/geometry runs are stable.
