# Geometry Baseline Parameter Profile Design

## Goal

Add an isolated A1 ablation that changes only the Geometry Felzenszwalb initial-segmentation parameters to match the LASER depth baseline. The experiment must preserve the existing Geometry implementation and default behavior.

## Scope

The A1 profile uses:

- `seg_scale=300`
- `seg_sigma=1.1`
- `seg_min_size=500`

It does not change geometry features, normal estimation, confidence handling, region merging, temporal segment matching, scale anchors, scale propagation, loop closure, or visualization tracing.

## Interface

`demo.py` adds:

```text
--geometry_seg_profile {legacy,baseline_params}
```

The default is `legacy`, which must reproduce the current code path. `baseline_params` selects new Geometry wrapper functions whose only responsibility is to call the existing Geometry segmentation implementation with the three baseline parameters above.

The profile is passed explicitly through `StreamingWindowEngine` and `build_geometry_sp_graph`. It is ignored by the depth branch. Invalid profiles fail during engine construction.

## Isolation

The existing `segment_geometry_felzenszwalb_rag*` functions and their defaults remain unchanged. New `segment_geometry_felzenszwalb_rag_baseline_params*` wrappers provide the A1 behavior. Future reliable-geometry segmentation work will use another independent function/profile rather than replacing either A1 or legacy behavior.

## Outputs

The cloud experiment runs without `--debug_alignment`. Normal execution therefore produces the standard Viser result under `outputs/viser/<scene>` and trajectory metrics under `outputs/eval/<scene>`. Cache files remain implementation intermediates and use an experiment-specific directory.

## Verification

Automated tests must verify:

1. The legacy profile selects the existing Geometry functions.
2. The baseline profile selects the new wrappers.
3. The wrappers pass exactly `300`, `1.1`, and `500` while retaining all other arguments.
4. The depth path is unchanged.
5. The CLI default remains `legacy` and accepts `baseline_params`.

