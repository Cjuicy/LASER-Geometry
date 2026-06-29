# Alignment Pipeline Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record the real depth/geometry segmentation and scale-propagation pipeline, then generate a scrollable static HTML report with ten stage images per target-window frame.

**Architecture:** Keep algorithm outputs unchanged and add optional trace sinks that are instantiated only when `--debug_alignment` is enabled. Segmentation functions expose initial and merged labels, LSA records direct anchors and temporal propagation provenance, and `AlignmentDebugRecorder` writes one compressed numeric trace per window. A standalone renderer validates the baseline and geometry runs, creates lazy-loaded WebP assets, and writes a single-entry `index.html` plus `data.json`.

**Tech Stack:** Python 3, NumPy, PyTorch, OpenCV, scikit-image, pytest, static HTML/CSS/JavaScript.

---

## File Structure

- Create `inference_engine/utils/segmentation_trace.py`: immutable per-frame segmentation stage result and confidence-mask helper.
- Modify `inference_engine/utils/depth.py`: explicit depth stages function; compatibility function still returns merged labels.
- Modify `inference_engine/utils/geometry_segmentation.py`: explicit geometry stages function; compatibility function still returns merged labels.
- Modify `inference_engine/utils/batch_threading.py`: ordered generic batch helper reused by stage collection.
- Modify `inference_engine/utils/lsa.py`: optional segmentation and scale trace sinks.
- Modify `inference_engine/utils/scale_anchor.py`: record every accepted overlap match without changing anchor estimation.
- Create `inference_engine/alignment_pipeline_trace.py`: typed match, propagation, segment-state collector and numeric-array conversion.
- Modify `inference_engine/alignment_debug.py`: save `pipeline/window_XXXX.npz` and pipeline metadata.
- Modify `inference_engine/streaming_window_engine.py`: create trace collectors only in debug mode and record every window.
- Modify `demo.py`: pass sample interval into debug metadata and validate confidence retention.
- Create `eval/build_alignment_pipeline_report.py`: validation, rendering, asset generation, JSON manifest, HTML generation, CLI.
- Create `tests/test_segmentation_trace.py`: depth/geometry stages and adjustable confidence tests.
- Modify `tests/test_lsa_graph_builders.py`: anchor/propagation provenance and output-invariance tests.
- Modify `tests/test_alignment_debug.py`: pipeline window serialization tests.
- Modify `tests/test_streaming_window_engine.py`: debug collector wiring and metadata tests.
- Modify `tests/test_demo_cli.py`: sample interval/debug metadata argument coverage.
- Create `tests/test_alignment_pipeline_report.py`: two-run validation, ten-cell rows, assets, and empty-state report tests.

### Task 1: Expose Real Segmentation Stages

**Files:**
- Create: `inference_engine/utils/segmentation_trace.py`
- Modify: `inference_engine/utils/batch_threading.py`
- Modify: `inference_engine/utils/depth.py`
- Modify: `inference_engine/utils/geometry_segmentation.py`
- Create: `tests/test_segmentation_trace.py`

- [ ] **Step 1: Write failing tests for adjustable confidence and compatibility**

```python
from dataclasses import replace

import numpy as np

from inference_engine.utils.depth import (
    segment_depth_felzenszwalb_rag,
    segment_depth_felzenszwalb_rag_stages,
)
from inference_engine.utils.geometry_segmentation import (
    segment_geometry_felzenszwalb_rag,
    segment_geometry_felzenszwalb_rag_stages,
)


def test_depth_stages_keep_original_merged_labels(monkeypatch):
    initial = np.array([[0, 0], [1, 1]], dtype=np.intp)
    merged = np.array([[0, 0], [0, 0]], dtype=np.intp)
    monkeypatch.setattr("inference_engine.utils.depth.felzenszwalb", lambda *a, **k: initial)
    monkeypatch.setattr("inference_engine.utils.depth.merge_regions", lambda *a, **k: merged)
    depth = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    conf = np.array([[[0.1, 0.2], [0.8, 0.9]]], dtype=np.float32)

    stages = segment_depth_felzenszwalb_rag_stages(
        depth, 0.1, conf_map=conf, top_conf_percentile=0.5, batch_idx=0
    )
    labels = segment_depth_felzenszwalb_rag(
        depth, 0.1, conf_map=conf, top_conf_percentile=0.5, batch_idx=0
    )

    np.testing.assert_array_equal(stages.initial_labels, initial)
    np.testing.assert_array_equal(stages.merged_labels, merged)
    np.testing.assert_array_equal(labels, merged)
    assert stages.confidence_threshold == 0.2
    assert stages.high_confidence_mask.sum() == 3


def test_confidence_retention_is_not_hard_coded(monkeypatch):
    monkeypatch.setattr(
        "inference_engine.utils.depth.felzenszwalb",
        lambda depth, **kwargs: np.zeros(depth.shape, dtype=np.intp),
    )
    monkeypatch.setattr(
        "inference_engine.utils.depth.merge_regions",
        lambda labels, depth, threshold: labels,
    )
    depth = np.ones((2, 5), dtype=np.float32)
    conf = np.arange(10, dtype=np.float32).reshape(1, 2, 5)

    top_20 = segment_depth_felzenszwalb_rag_stages(
        depth, 0.1, conf_map=conf, top_conf_percentile=0.8, batch_idx=0
    )
    top_40 = segment_depth_felzenszwalb_rag_stages(
        depth, 0.1, conf_map=conf, top_conf_percentile=0.6, batch_idx=0
    )

    assert top_20.high_confidence_mask.sum() == 2
    assert top_40.high_confidence_mask.sum() == 4
```

- [ ] **Step 2: Run the tests and verify the new APIs are missing**

Run:

```bash
conda run -n vggt-dem python -m pytest tests/test_segmentation_trace.py -q
```

Expected: collection fails because the stages functions do not exist.

- [ ] **Step 3: Add the stage result and ordered batch helper**

```python
# inference_engine/utils/segmentation_trace.py
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SegmentationStages:
    initial_labels: np.ndarray
    merged_labels: np.ndarray
    confidence_threshold: float
    high_confidence_mask: np.ndarray


def confidence_selection(conf, quantile):
    conf = np.asarray(conf)
    if quantile is None:
        return float("nan"), np.ones(conf.shape, dtype=bool)
    threshold = float(np.quantile(conf.reshape(-1), quantile, method="nearest"))
    return threshold, np.isfinite(conf) & (conf >= threshold)
```

Refactor `batched_image_op_wrapper` so it calls a new `ordered_batch_apply(images, op_func, n_jobs=None, *args, **kwargs)` that returns an ordered Python list. Keep `batched_image_op_wrapper` as `np.stack(ordered_batch_apply(images, op_func, n_jobs=n_jobs, *args, **kwargs), axis=0).astype(np.intp, copy=False)` so every existing caller keeps the same type and ordering.

- [ ] **Step 4: Add depth and geometry stages wrappers**

Implement `segment_depth_felzenszwalb_rag_stages` with the same signature as `segment_depth_felzenszwalb_rag` by moving the current initial Felzenszwalb and merge code into the stages function. The existing `segment_depth_felzenszwalb_rag` must return `.merged_labels` from that function.

Implement `segment_geometry_felzenszwalb_rag_stages` with the same signature as `segment_geometry_felzenszwalb_rag`: retain the current depth+normal initial segmentation and `merge_regions_geometry`, return both label maps, and keep `segment_geometry_felzenszwalb_rag` returning only `.merged_labels`.

Both functions must use `confidence_selection` with the actual internal quantile supplied by the engine; no literal `0.7` is allowed.

- [ ] **Step 5: Run focused and existing segmentation tests**

```bash
conda run -n vggt-dem python -m pytest \
  tests/test_segmentation_trace.py \
  tests/test_geometry_segmentation.py \
  tests/test_lsa_graph_builders.py -q
```

Expected: all tests pass and the old public segmentation functions still return arrays.

- [ ] **Step 6: Commit segmentation trace support**

```bash
git add inference_engine/utils/segmentation_trace.py \
  inference_engine/utils/batch_threading.py \
  inference_engine/utils/depth.py \
  inference_engine/utils/geometry_segmentation.py \
  tests/test_segmentation_trace.py
git commit -m "feat: expose segmentation pipeline stages"
```

### Task 2: Record Overlap Anchors and Temporal Propagation

**Files:**
- Create: `inference_engine/alignment_pipeline_trace.py`
- Modify: `inference_engine/utils/scale_anchor.py`
- Modify: `inference_engine/utils/lsa.py`
- Modify: `tests/test_lsa_graph_builders.py`

- [ ] **Step 1: Write failing provenance and invariance tests**

```python
import numpy as np

from inference_engine.alignment_pipeline_trace import ScaleTraceCollector
from inference_engine.utils.lsa import align_adjacent_windows_depth_segments
from inference_engine.utils.segment_graph import match_segmentation_seq


def test_scale_trace_distinguishes_anchor_propagated_and_identity():
    src_labels = np.array([[[0, 0], [1, 1]]], dtype=np.intp)
    tgt_labels = np.array(
        [
            [[0, 0], [1, 1]],
            [[0, 0], [1, 1]],
            [[0, 0], [2, 2]],
        ],
        dtype=np.intp,
    )
    src_graph = match_segmentation_seq(src_labels, iou_thresh=0.3)
    tgt_graph = match_segmentation_seq(tgt_labels, iou_thresh=0.3)
    collector = ScaleTraceCollector()

    scale_mask = align_adjacent_windows_depth_segments(
        np.ones((1, 2, 2), dtype=np.float32),
        np.ones((3, 2, 2), dtype=np.float32) * 2.0,
        src_graph,
        tgt_graph,
        overlap=1,
        corr_iou_thresh=0.4,
        trace=collector,
    )

    states = {(x.frame, x.segment): x.role for x in collector.segment_states}
    assert states[(0, 0)] == "A"
    assert states[(1, 0)] == "P"
    assert states[(2, 2)] == "I"
    assert collector.matches
    assert collector.propagation_edges
    assert scale_mask.shape == (3, 2, 2)


def test_trace_sink_does_not_change_scale_mask():
    labels = np.array([[[0, 0], [1, 1]], [[0, 0], [1, 1]]], dtype=np.intp)
    src_depth = np.ones((2, 2, 2), dtype=np.float32)
    tgt_depth = np.ones((2, 2, 2), dtype=np.float32) * 2.0
    src_graph_a = match_segmentation_seq(labels, iou_thresh=0.3)
    tgt_graph_a = match_segmentation_seq(labels, iou_thresh=0.3)
    src_graph_b = match_segmentation_seq(labels, iou_thresh=0.3)
    tgt_graph_b = match_segmentation_seq(labels, iou_thresh=0.3)
    without_trace = align_adjacent_windows_depth_segments(
        src_depth, tgt_depth, src_graph_a, tgt_graph_a, overlap=1
    )
    collector = ScaleTraceCollector()
    with_trace = align_adjacent_windows_depth_segments(
        src_depth, tgt_depth, src_graph_b, tgt_graph_b, overlap=1, trace=collector
    )
    np.testing.assert_allclose(with_trace, without_trace)
```

- [ ] **Step 2: Run the tests and verify the collector is missing**

```bash
conda run -n vggt-dem python -m pytest \
  tests/test_lsa_graph_builders.py -q
```

Expected: failure importing `ScaleTraceCollector` or passing `trace`.

- [ ] **Step 3: Implement typed trace records**

```python
# inference_engine/alignment_pipeline_trace.py
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class MatchRecord:
    frame: int
    src_segment: int
    tgt_segment: int
    iou: float
    scale: float


@dataclass(frozen=True)
class PropagationRecord:
    parent_frame: int
    parent_segment: int
    child_frame: int
    child_segment: int
    iou: float
    scale: float


@dataclass(frozen=True)
class SegmentState:
    frame: int
    segment: int
    role: str
    scale: float


@dataclass
class ScaleTraceCollector:
    matches: list[MatchRecord] = field(default_factory=list)
    propagation_edges: list[PropagationRecord] = field(default_factory=list)
    segment_states: list[SegmentState] = field(default_factory=list)
    direct_anchor_keys: set[tuple[int, int]] = field(default_factory=set)

    def to_arrays(self):
        role_codes = {"I": 0, "P": 1, "A": 2}
        return {
            "match_frame": np.asarray([x.frame for x in self.matches], dtype=np.int32),
            "match_src_segment": np.asarray([x.src_segment for x in self.matches], dtype=np.int32),
            "match_tgt_segment": np.asarray([x.tgt_segment for x in self.matches], dtype=np.int32),
            "match_iou": np.asarray([x.iou for x in self.matches], dtype=np.float32),
            "match_scale": np.asarray([x.scale for x in self.matches], dtype=np.float32),
            "prop_parent_frame": np.asarray([x.parent_frame for x in self.propagation_edges], dtype=np.int32),
            "prop_parent_segment": np.asarray([x.parent_segment for x in self.propagation_edges], dtype=np.int32),
            "prop_child_frame": np.asarray([x.child_frame for x in self.propagation_edges], dtype=np.int32),
            "prop_child_segment": np.asarray([x.child_segment for x in self.propagation_edges], dtype=np.int32),
            "prop_iou": np.asarray([x.iou for x in self.propagation_edges], dtype=np.float32),
            "prop_scale": np.asarray([x.scale for x in self.propagation_edges], dtype=np.float32),
            "segment_frame": np.asarray([x.frame for x in self.segment_states], dtype=np.int32),
            "segment_id": np.asarray([x.segment for x in self.segment_states], dtype=np.int32),
            "segment_role": np.asarray([role_codes[x.role] for x in self.segment_states], dtype=np.uint8),
            "segment_scale": np.asarray([x.scale for x in self.segment_states], dtype=np.float32),
        }
```

`to_arrays()` must emit empty arrays with stable dtypes when no matches or propagation edges exist.

- [ ] **Step 4: Instrument the existing anchor worker**

Add optional `trace=None`, `frame_idx`, `src_segment_idx`, and a target vertex-index map to `_edge_scale_worker`. Immediately after appending the existing target cache values, append a `MatchRecord` with the same IoU and estimated scale. Pass those indices from `assign_overlap_window_depth_scale`; do not change estimator arguments, graph edges, cache writes, or `n_jobs` behavior.

- [ ] **Step 5: Instrument existing LSA propagation**

Add `trace=None` to `refine_segment_scales` and `align_adjacent_windows_depth_segments`.

After overlap assignment, snapshot all target `(frame, segment)` keys whose cache already contains a scale. During `_propagate_scale_cache`, record an edge only when the existing code actually appends a propagated scale. After the scale mask is built, classify every target segment as `A`, `P`, or `I` and compute the same IoU-weighted mean scale used by `_get_scale_mask`.

- [ ] **Step 6: Run LSA and scale-anchor tests**

```bash
conda run -n vggt-dem python -m pytest \
  tests/test_lsa_graph_builders.py \
  tests/test_scale_anchor.py -q
```

Expected: all tests pass; traced and untraced scale masks are equal.

- [ ] **Step 7: Commit scale provenance**

```bash
git add inference_engine/alignment_pipeline_trace.py \
  inference_engine/utils/scale_anchor.py \
  inference_engine/utils/lsa.py \
  tests/test_lsa_graph_builders.py
git commit -m "feat: trace scale anchors and propagation"
```

### Task 3: Save One Real Trace Per Window

**Files:**
- Modify: `inference_engine/alignment_debug.py`
- Modify: `inference_engine/utils/lsa.py`
- Modify: `inference_engine/streaming_window_engine.py`
- Modify: `demo.py`
- Modify: `tests/test_alignment_debug.py`
- Modify: `tests/test_streaming_window_engine.py`
- Modify: `tests/test_demo_cli.py`

- [ ] **Step 1: Write failing recorder tests**

```python
def test_recorder_writes_numeric_pipeline_window(tmp_path):
    recorder = AlignmentDebugRecorder(enabled=True, root_dir=tmp_path, scene_name="scene")
    path = recorder.record_pipeline_window(
        window_index=1,
        payload={
            "global_frame_indices": np.array([2, 3], dtype=np.int32),
            "initial_labels": np.zeros((2, 3, 4), dtype=np.uint16),
            "merged_labels": np.ones((2, 3, 4), dtype=np.uint16),
            "high_confidence_masks": np.ones((2, 3, 4), dtype=bool),
        },
        metadata={
            "schema_version": 1,
            "confidence_retained_fraction": 0.3,
            "confidence_quantile": 0.7,
            "sample_interval": 10,
        },
    )

    assert path == tmp_path / "scene" / "pipeline" / "window_0001.npz"
    with np.load(path, allow_pickle=False) as arrays:
        assert arrays["initial_labels"].dtype == np.uint16
    meta = json.loads((path.parent / "meta.json").read_text())
    assert meta["confidence_retained_fraction"] == 0.3
    assert meta["confidence_quantile"] == 0.7
```

Add engine tests asserting that debug mode passes a segmentation trace sink to exactly one selected graph builder, passes a `ScaleTraceCollector` to refinement, records the first window, and records a later window with mutual confidence masks. Add a parser/load-model test asserting `sample_interval` is passed as debug metadata.

- [ ] **Step 2: Run focused tests and verify missing window recording**

```bash
conda run -n vggt-dem python -m pytest \
  tests/test_alignment_debug.py \
  tests/test_streaming_window_engine.py \
  tests/test_demo_cli.py -q
```

Expected: failures for `record_pipeline_window` and missing trace wiring.

- [ ] **Step 3: Add segmentation trace collection to graph builders**

Add an optional mutable `segmentation_trace` mapping to `build_depth_sp_graph` and `build_geometry_sp_graph`. When it is `None`, keep the existing batched labels path. When provided, call the ordered stage functions once per frame and fill:

```python
segmentation_trace.update(
    initial_labels=np.stack([stage.initial_labels for stage in stages]),
    merged_labels=np.stack([stage.merged_labels for stage in stages]),
    confidence_thresholds=np.asarray([stage.confidence_threshold for stage in stages]),
    high_confidence_masks=np.stack([stage.high_confidence_mask for stage in stages]),
)
```

Build the graph from `merged_labels` in both paths.

- [ ] **Step 4: Add pipeline window serialization**

Extend `AlignmentDebugRecorder` with `record_pipeline_window`. It writes metadata to `scene_dir/pipeline/meta.json`, converts labels to the smallest safe unsigned integer dtype, rejects object arrays, and saves `window_XXXX.npz` with `np.savez_compressed`.

Keep `record_pair` unchanged so the existing Viser debug viewer remains compatible.

- [ ] **Step 5: Wire the engine without changing normal execution**

Store both forms of confidence configuration:

```python
self.confidence_retained_fraction = top_conf_percentile
self.top_conf_percentile = 1 - top_conf_percentile
self.debug_sample_interval = debug_sample_interval
```

For each registration-worker item, create segmentation/scale collectors only when both `debug_alignment` and `depth_refine` are true. Record every window after refinement and before cache update. Use:

```python
window_start = self.cache_id * (self.window_size - self.overlap)
global_frame_indices = np.arange(window_start, window_start + len(working_window["conf"]))
```

For the first window, write empty match/propagation arrays and identity segment states. For later windows, include the existing mutual confidence mask and `ScaleTraceCollector.to_arrays()`.

- [ ] **Step 6: Pass and validate sample interval**

Add `debug_sample_interval: int = 1` to `StreamingWindowEngine.__init__`, require it to be positive, and pass `args.sample_interval` from `demo.load_model`. Validate `0 < top_conf_percentile <= 1` so retained fraction and quantile metadata are unambiguous.

- [ ] **Step 7: Run engine and recorder tests**

```bash
conda run -n vggt-dem python -m pytest \
  tests/test_alignment_debug.py \
  tests/test_streaming_window_engine.py \
  tests/test_demo_cli.py \
  tests/test_lsa_graph_builders.py -q
```

Expected: all focused tests pass; legacy pair tracing remains green.

- [ ] **Step 8: Commit window trace recording**

```bash
git add inference_engine/alignment_debug.py \
  inference_engine/utils/lsa.py \
  inference_engine/streaming_window_engine.py \
  demo.py \
  tests/test_alignment_debug.py \
  tests/test_streaming_window_engine.py \
  tests/test_demo_cli.py
git commit -m "feat: record alignment pipeline windows"
```

### Task 4: Render the Five Real Stage Images

**Files:**
- Create: `eval/build_alignment_pipeline_report.py`
- Create: `tests/test_alignment_pipeline_report.py`

- [ ] **Step 1: Write failing renderer tests**

```python
from eval.build_alignment_pipeline_report import (
    render_confidence_stage,
    render_segmentation_stage,
    render_overlap_stage,
    render_propagation_stage,
)


def test_confidence_stage_dims_rejected_pixels():
    rgb = np.full((4, 5, 3), 200, dtype=np.uint8)
    mask = np.zeros((4, 5), dtype=bool)
    mask[:, 3:] = True
    rendered = render_confidence_stage(rgb, mask, mutual_mask=None)
    assert rendered[:, :3].mean() < rendered[:, 3:].mean()


def test_segmentation_stage_draws_full_boundaries_but_emphasizes_anchors():
    rgb = np.full((4, 6, 3), 160, dtype=np.uint8)
    labels = np.array(
        [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1],
         [2, 2, 2, 3, 3, 3], [2, 2, 2, 3, 3, 3]],
        dtype=np.int32,
    )
    conf_mask = np.zeros((4, 6), dtype=bool)
    conf_mask[:, 3:] = True
    anchor_ids = np.array([1, 3], dtype=np.int32)
    rendered = render_segmentation_stage(
        rgb, labels, conf_mask, emphasized_merged_labels=labels, anchor_ids=anchor_ids
    )
    assert rendered.shape == rgb.shape
    assert not np.array_equal(rendered, rgb)


def test_overlap_stage_renders_source_target_and_all_matches():
    rgb = np.full((4, 6, 3), 160, dtype=np.uint8)
    src_labels = np.tile(np.array([0, 0, 0, 1, 1, 1], dtype=np.int32), (4, 1))
    tgt_labels = np.tile(np.array([0, 0, 2, 2, 1, 1], dtype=np.int32), (4, 1))
    matches = [
        {"src_segment": 0, "tgt_segment": 0, "iou": 0.67, "scale": 1.04},
        {"src_segment": 1, "tgt_segment": 1, "iou": 0.51, "scale": 0.97},
    ]
    image, details = render_overlap_stage(
        rgb=rgb,
        src_labels=src_labels,
        tgt_labels=tgt_labels,
        matches=matches,
    )
    assert image.shape[1] == rgb.shape[1] * 2
    assert len(details["matches"]) == len(matches)


def test_propagation_stage_reports_a_p_i_roles():
    rgb = np.full((4, 6, 3), 160, dtype=np.uint8)
    previous_labels = np.tile(np.array([0, 0, 0, 1, 1, 1], dtype=np.int32), (4, 1))
    current_labels = np.tile(np.array([0, 0, 2, 2, 1, 1], dtype=np.int32), (4, 1))
    states = [
        {"segment": 0, "role": "A", "scale": 0.95},
        {"segment": 1, "role": "P", "scale": 1.05},
        {"segment": 2, "role": "I", "scale": 1.0},
    ]
    edges = [
        {"parent_segment": 0, "child_segment": 0, "iou": 0.7, "scale": 0.95},
        {"parent_segment": 1, "child_segment": 1, "iou": 0.6, "scale": 1.05},
    ]
    image, details = render_propagation_stage(
        previous_rgb=rgb,
        current_rgb=rgb,
        previous_labels=previous_labels,
        current_labels=current_labels,
        segment_states=states,
        edges=edges,
    )
    assert image.shape[1] == rgb.shape[1] * 2
    assert {item["role"] for item in details["segments"]} == {"A", "P", "I"}
```

- [ ] **Step 2: Run renderer tests and verify the module is missing**

```bash
conda run -n vggt-dem python -m pytest \
  tests/test_alignment_pipeline_report.py -q
```

Expected: import failure for `eval.build_alignment_pipeline_report`.

- [ ] **Step 3: Implement shared image helpers**

Add deterministic label colors, boundary extraction, centroids, confidence dimming, text badges, and WebP writing. All renderers accept BGR `uint8` arrays and numeric trace arrays; they must not import model code or mutate inputs.

Use the fixed visual contract:

```python
ROLE_CODES = {0: "I", 1: "P", 2: "A"}
SOURCE_BOUNDARY_BGR = (255, 220, 40)
TARGET_BOUNDARY_BGR = (210, 80, 255)
LOW_CONF_BRIGHTNESS = 0.22
```

- [ ] **Step 4: Implement confidence and segmentation renderers**

`render_confidence_stage` keeps high-confidence RGB unchanged, dims the rest, outlines the high-confidence boundary in white, and outlines the mutual mask in cyan when present.

`render_segmentation_stage` draws every label boundary. It uses the merged labels plus direct-anchor IDs to determine which initial or merged regions receive strong colored boundaries; other boundaries remain thin gray. It returns count metadata for the modal.

- [ ] **Step 5: Implement overlap and propagation renderers**

`render_overlap_stage` creates a two-panel image, draws source and target boundaries, computes segment centroids, and draws every recorded match using IoU-dependent line thickness. Empty matches render a readable `No accepted IoU match` panel.

`render_propagation_stage` creates a two-panel previous/current view when a previous target frame exists. It colors current segments with a blue-white-red scale map centered on `1.0`, marks `A/P/I`, and draws every recorded temporal edge from parent to child centroid. The first frame uses a single current panel and a `No previous target frame` badge.

- [ ] **Step 6: Run renderer tests**

```bash
conda run -n vggt-dem python -m pytest \
  tests/test_alignment_pipeline_report.py -q
```

Expected: focused renderer tests pass.

- [ ] **Step 7: Commit stage rendering**

```bash
git add eval/build_alignment_pipeline_report.py \
  tests/test_alignment_pipeline_report.py
git commit -m "feat: render alignment pipeline stages"
```

### Task 5: Build the Ten-Image Static HTML Report

**Files:**
- Modify: `eval/build_alignment_pipeline_report.py`
- Modify: `tests/test_alignment_pipeline_report.py`
- Modify: `docs/CLOUD_RUN.md`

- [ ] **Step 1: Write failing report validation and DOM tests**

```python
def _write_synthetic_run(root, segment_mode):
    pipeline_dir = root / "pipeline"
    pipeline_dir.mkdir(parents=True)
    meta = {
        "schema_version": 1,
        "segment_mode": segment_mode,
        "normal_method": "cross",
        "scale_anchor_mode": "depth_irls",
        "window_size": 2,
        "overlap": 1,
        "sample_interval": 1,
        "confidence_retained_fraction": 0.3,
        "confidence_quantile": 0.7,
        "graph_iou_threshold": 0.3,
        "anchor_iou_threshold": 0.4,
    }
    (pipeline_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    labels = np.array(
        [
            [[0, 0, 0, 1, 1, 1]] * 4,
            [[0, 0, 2, 2, 1, 1]] * 4,
        ],
        dtype=np.uint16,
    )
    np.savez_compressed(
        pipeline_dir / "window_0000.npz",
        global_frame_indices=np.array([0, 1], dtype=np.int32),
        confidence_thresholds=np.array([0.7, 0.7], dtype=np.float32),
        high_confidence_masks=np.ones((2, 4, 6), dtype=bool),
        initial_labels=labels,
        merged_labels=labels,
        mutual_confidence_masks=np.empty((0, 4, 6), dtype=bool),
        match_frame=np.empty(0, dtype=np.int32),
        match_src_segment=np.empty(0, dtype=np.int32),
        match_tgt_segment=np.empty(0, dtype=np.int32),
        match_iou=np.empty(0, dtype=np.float32),
        match_scale=np.empty(0, dtype=np.float32),
        prop_parent_frame=np.empty(0, dtype=np.int32),
        prop_parent_segment=np.empty(0, dtype=np.int32),
        prop_child_frame=np.empty(0, dtype=np.int32),
        prop_child_segment=np.empty(0, dtype=np.int32),
        prop_iou=np.empty(0, dtype=np.float32),
        prop_scale=np.empty(0, dtype=np.float32),
        segment_frame=np.array([0, 0, 1, 1, 1], dtype=np.int32),
        segment_id=np.array([0, 1, 0, 1, 2], dtype=np.int32),
        segment_role=np.zeros(5, dtype=np.uint8),
        segment_scale=np.ones(5, dtype=np.float32),
    )
    return root


def write_synthetic_pipeline_runs(tmp_path):
    return (
        _write_synthetic_run(tmp_path / "depth", "depth"),
        _write_synthetic_run(tmp_path / "geometry", "geometry"),
    )


def synthetic_image_dir(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir(exist_ok=True)
    for index in range(2):
        image = np.full((4, 6, 3), 80 + index * 40, dtype=np.uint8)
        cv2.imwrite(str(image_dir / f"{index:06d}.png"), image)
    return image_dir


def test_report_rejects_mismatched_confidence_fraction(tmp_path):
    baseline, geometry = write_synthetic_pipeline_runs(tmp_path)
    geometry_meta = json.loads((geometry / "pipeline" / "meta.json").read_text())
    geometry_meta["confidence_retained_fraction"] = 0.4
    (geometry / "pipeline" / "meta.json").write_text(json.dumps(geometry_meta))
    with pytest.raises(ValueError, match="confidence_retained_fraction"):
        build_report(baseline, geometry, synthetic_image_dir(tmp_path), tmp_path / "report")


def test_report_writes_ten_assets_per_row_and_single_entry_html(tmp_path):
    baseline, geometry = write_synthetic_pipeline_runs(tmp_path)
    out_dir = tmp_path / "report"
    result = build_report(
        baseline_debug_dir=baseline,
        geometry_debug_dir=geometry,
        image_dir=synthetic_image_dir(tmp_path),
        out_dir=out_dir,
    )
    manifest = json.loads((out_dir / "data.json").read_text())
    assert all(len(row["stages"]) == 10 for row in manifest["rows"])
    assert (out_dir / "index.html").is_file()
    html = (out_dir / "index.html").read_text()
    assert 'loading="lazy"' in html
    assert 'class="pipeline-row"' in html
    for row in manifest["rows"]:
        for stage in row["stages"]:
            assert (out_dir / stage["asset"]).is_file()
```

- [ ] **Step 2: Run report tests and verify report APIs are missing**

```bash
conda run -n vggt-dem python -m pytest \
  tests/test_alignment_pipeline_report.py -q
```

Expected: failures for `build_report` and metadata validation.

- [ ] **Step 3: Implement trace loading and strict comparison**

Implement:

```python
COMPARABLE_FIELDS = (
    "window_size",
    "overlap",
    "sample_interval",
    "confidence_retained_fraction",
    "confidence_quantile",
    "graph_iou_threshold",
    "anchor_iou_threshold",
    "scale_anchor_mode",
)


def validate_comparable_runs(baseline_meta, geometry_meta):
    if baseline_meta["segment_mode"] != "depth":
        raise ValueError("baseline segment_mode must be depth")
    if geometry_meta["segment_mode"] != "geometry":
        raise ValueError("geometry segment_mode must be geometry")
    if baseline_meta["scale_anchor_mode"] != "depth_irls":
        raise ValueError("this report requires scale_anchor_mode=depth_irls")
    for field in COMPARABLE_FIELDS:
        if baseline_meta[field] != geometry_meta[field]:
            raise ValueError(f"Mismatched {field}")
```

Load all `window_*.npz` with `allow_pickle=False` and require matching window numbers and frame shapes.

- [ ] **Step 4: Build ordered rows and ten assets**

For each target window and local frame, map the sampled RGB using `global_frame_indices`. Render stages in exactly this order:

```python
STAGE_ORDER = (
    ("depth", "confidence"),
    ("depth", "initial"),
    ("depth", "merged"),
    ("depth", "overlap"),
    ("depth", "propagation"),
    ("geometry", "confidence"),
    ("geometry", "initial"),
    ("geometry", "merged"),
    ("geometry", "overlap"),
    ("geometry", "propagation"),
)
```

For source overlap labels, read the previous window's last `overlap` merged frames. First-window and empty-match assets use explicit placeholder renderers rather than missing paths.

- [ ] **Step 5: Generate JSON and HTML**

Write UTF-8 `data.json` and an `index.html` that embeds the same lightweight manifest as JSON. The HTML must contain:

- two sticky method headers spanning five columns each;
- five sticky stage headers per method;
- one `.pipeline-row` with ten lazy `<img>` elements per target-window frame;
- overlap/non-overlap row badges;
- a click-to-open modal with the full asset and numeric details;
- horizontal row scrolling below the minimum report width;
- no network dependencies.

- [ ] **Step 6: Add CLI and cloud commands**

Expose:

```bash
python eval/build_alignment_pipeline_report.py \
  --baseline_debug_dir PATH \
  --geometry_debug_dir PATH \
  --image_dir PATH \
  --out_dir PATH
```

Add optional `--sample_interval` only as a validator: if provided, it must equal metadata. Add `--window_start`, `--window_stop`, and `--frame_step` for preview reports; defaults include all windows and every frame.

Document the two debug runs and report command in `docs/CLOUD_RUN.md` using `--scale_anchor_mode depth_irls` and the same confidence/window parameters.

- [ ] **Step 7: Run report and CLI tests**

```bash
conda run -n vggt-dem python -m pytest \
  tests/test_alignment_pipeline_report.py \
  tests/test_demo_cli.py -q
conda run -n vggt-dem python -m py_compile \
  eval/build_alignment_pipeline_report.py
```

Expected: report tests pass, every asset referenced by HTML exists, and the CLI compiles.

- [ ] **Step 8: Commit the report generator**

```bash
git add eval/build_alignment_pipeline_report.py \
  tests/test_alignment_pipeline_report.py \
  docs/CLOUD_RUN.md
git commit -m "feat: generate alignment pipeline report"
```

### Task 6: Full Regression and Delivery Commands

**Files:**
- Modify only files required by failures discovered during verification.

- [ ] **Step 1: Run the focused feature suite**

```bash
conda run -n vggt-dem python -m pytest \
  tests/test_segmentation_trace.py \
  tests/test_geometry_segmentation.py \
  tests/test_lsa_graph_builders.py \
  tests/test_scale_anchor.py \
  tests/test_alignment_debug.py \
  tests/test_streaming_window_engine.py \
  tests/test_demo_cli.py \
  tests/test_alignment_pipeline_report.py -q
```

Expected: all feature and directly affected regression tests pass.

- [ ] **Step 2: Run the complete test suite**

```bash
conda run -n vggt-dem python -m pytest tests -q
```

Expected: all tests pass. If an environment-only optional dependency prevents collection, record the exact failing test and still run every unaffected test module.

- [ ] **Step 3: Run syntax and diff checks**

```bash
conda run -n vggt-dem python -m py_compile \
  inference_engine/utils/segmentation_trace.py \
  inference_engine/alignment_pipeline_trace.py \
  inference_engine/alignment_debug.py \
  inference_engine/streaming_window_engine.py \
  eval/build_alignment_pipeline_report.py \
  demo.py
git diff --check
git status --short
```

Expected: compilation succeeds, no whitespace errors, and only intentional changes plus the pre-existing untracked `viser/` remain.

- [ ] **Step 4: Verify the documented cloud workflow**

Run both methods with the same `--sample_interval`, `--window_size`, `--overlap`, `--top_conf_percentile`, `--scale_anchor_mode depth_irls`, and `--debug_alignment`, then build the report. Verify `index.html` opens with ten images per row, overlap rows show source-target matches, and non-overlap rows show propagation/identity roles.

- [ ] **Step 5: Commit any verification-only fixes**

```bash
git add inference_engine/utils/segmentation_trace.py \
  inference_engine/utils/batch_threading.py \
  inference_engine/utils/depth.py \
  inference_engine/utils/geometry_segmentation.py \
  inference_engine/alignment_pipeline_trace.py \
  inference_engine/utils/scale_anchor.py \
  inference_engine/utils/lsa.py \
  inference_engine/alignment_debug.py \
  inference_engine/streaming_window_engine.py \
  eval/build_alignment_pipeline_report.py \
  demo.py \
  tests/test_segmentation_trace.py \
  tests/test_lsa_graph_builders.py \
  tests/test_alignment_debug.py \
  tests/test_streaming_window_engine.py \
  tests/test_demo_cli.py \
  tests/test_alignment_pipeline_report.py \
  docs/CLOUD_RUN.md
git commit -m "fix: harden alignment pipeline report"
```

Skip this commit when verification requires no fixes.
