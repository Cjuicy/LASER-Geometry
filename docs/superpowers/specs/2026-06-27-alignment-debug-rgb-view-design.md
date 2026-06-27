# Alignment Debug RGB View Design

## Goal

Make the standalone alignment-debug Viser view show recognizable KITTI scene
content without changing `demo.py`, `StreamingWindowEngine`, recorded `.npz`
traces, or normal LASER outputs.

The existing solid-color `key` and `process` views remain available. The new
mode reads the original local image sequence only when explicitly requested.

## Command-Line Interface

`eval/vis_alignment_debug.py` gains:

- `--image_dir`: optional source image directory, such as `data/09/image_2`.
- `--sample_interval`: image sampling interval used by the cloud run.
- `--layer_mode rgb`: show the refined target point cloud with RGB colors and
  retain only points selected by `mutual_conf_mask`.
- `--camera_view source`: initialize the browser from the selected overlap
  frame camera pose so the point cloud is recognizable as a scene.
- `--all_pairs`: aggregate the RGB high-confidence regions from every recorded
  alignment pair instead of selecting one pair with `--pair_index`.

When `--layer_mode rgb` is selected, `--image_dir` is required. Existing modes
do not require image files and preserve their current behavior.

## Image Mapping

The viewer reads `window_size` and `overlap` from the trace `meta.json`. For a
trace named `pair_NNNN`, the first sampled image used by its target overlap is:

```text
pair_number * (window_size - overlap)
```

The viewer takes the next `overlap` sampled images. For KITTI 09 pair 1 with
`window_size=30`, `overlap=10`, and `sample_interval=10`, this resolves sampled
frames 20 through 29, corresponding to original frames 200 through 290.

Images are loaded through the project's existing
`load_and_preprocess_images()` helper so their height and width match the point
maps exactly. A clear error is raised if metadata, images, or shapes do not
match.

In `--all_pairs` mode, the mapping is evaluated independently for each trace.
For the current KITTI 09 data this covers seven disjoint alignment regions,
each containing ten sampled overlap frames. It intentionally does not claim to
represent the 90 sampled non-overlap frames that are absent from the debug
traces.

## Rendering

RGB colors are attached to `tgt_points_after_refine_overlap`. The
`mutual_conf_mask` removes the low-confidence points that currently fill the
camera frustum and obscure the road, vegetation, and object boundaries.

In comparison mode, baseline and geometry retain separate scene-tree folders
and the configured comparison offset. Source-camera view uses the selected
target pose and a wider field of view so the two offset scenes remain visible.
The original solid-color alignment layers remain unchanged for inspecting
before/Sim3/refine states.

Detailed mode keeps the current single-pair behavior. Sequence-alignment mode
concatenates the high-confidence RGB points from all pairs into one cloud per
method, then applies `--max_points` once per method. Sampling is deterministic,
so baseline and geometry remain stable between viewer runs. With the existing
default cap, the browser receives at most 200,000 points per method rather than
the approximately 1.4 million mutual-confidence points stored per method.

## Isolation

Only `eval/vis_alignment_debug.py` and its viewer tests change. No RGB arrays
are added to debug traces, so existing downloaded data remains usable and no
cloud inference rerun is required.

## Verification

Automated tests cover:

- pair-number to sampled-image-range mapping;
- RGB point-map shape and frame selection;
- mutual-confidence filtering while preserving point/color alignment;
- argument validation for RGB mode;
- source-camera pose selection with comparison offsets;
- all-pairs aggregation and a global per-method point cap.

Manual verification opens the existing KITTI 09 depth/geometry traces in Viser
and confirms that road, roadside structures, and vegetation are recognizable,
while baseline and geometry stay separately toggleable. It verifies both a
single detailed pair and all seven alignment regions. Viewing the complete
160-frame sampled reconstruction remains the responsibility of the standard
`outputs/viser` data and is outside this debug-view change.
