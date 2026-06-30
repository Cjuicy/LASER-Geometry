# Depth-Paired Pipeline Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the real pre-refine segmentation depth to pipeline traces, show it beside initial/merged segmentation, and compare Baseline and Geometry together in the detail modal.

**Architecture:** The graph builders copy their exact input depth only when a segmentation trace sink is active, and pipeline schema v2 stores it as float32. The report computes one per-frame color range shared by both methods, composes depth and segmentation into the existing initial/merged cards, and pairs matching stages in a responsive two-column modal. Algorithm outputs and the ten-card row contract remain unchanged.

**Tech Stack:** Python 3, NumPy, OpenCV, static HTML/CSS/JavaScript, pytest.

---

## File Structure

- Modify `inference_engine/utils/lsa.py`: add an isolated float32 copy of the graph-builder depth to the optional trace.
- Modify `inference_engine/streaming_window_engine.py`: write pipeline schema version 2.
- Modify `tests/test_lsa_graph_builders.py`: verify exact depth capture and copy semantics for both methods.
- Modify `tests/test_streaming_window_engine.py`: verify schema v2 metadata and serialized depth.
- Modify `eval/build_alignment_pipeline_report.py`: validate v2, compute shared depth ranges, render heatmaps/composites, and generate paired modal markup.
- Modify `tests/test_alignment_pipeline_report.py`: cover shared colors, composite dimensions, v1 rejection, ten cards, and paired modal.
- Modify `docs/CLOUD_RUN.md`: update rerun names and explain that old v1 traces cannot supply segmentation depth.

### Task 1: Record the Exact Segmentation Input Depth

**Files:**
- Modify: `inference_engine/utils/lsa.py`
- Modify: `inference_engine/streaming_window_engine.py`
- Modify: `tests/test_lsa_graph_builders.py`
- Modify: `tests/test_streaming_window_engine.py`

- [ ] **Step 1: Write failing graph-builder depth tests**

Add tests using the existing `ordered_batch_apply` stubs:

```python
def test_depth_graph_trace_copies_float32_segmentation_depth(monkeypatch):
    depth = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    stages = [
        SegmentationStages(
            np.zeros((2, 2), dtype=np.intp),
            np.zeros((2, 2), dtype=np.intp),
            0.7,
            np.ones((2, 2), dtype=bool),
        )
        for _ in range(2)
    ]
    monkeypatch.setattr(lsa, "ordered_batch_apply", lambda *args, **kwargs: stages)
    monkeypatch.setattr(lsa, "match_segmentation_seq", lambda labels, iou_thresh: labels)
    trace = {}

    lsa.build_depth_sp_graph(depth, segmentation_trace=trace)
    depth[:] = -1

    assert trace["segmentation_depths"].dtype == np.float32
    np.testing.assert_array_equal(
        trace["segmentation_depths"],
        np.arange(8, dtype=np.float32).reshape(2, 2, 2),
    )


def test_geometry_graph_trace_copies_float32_segmentation_depth(monkeypatch):
    depth = np.arange(4, dtype=np.float64).reshape(1, 2, 2)
    stage = SegmentationStages(
        np.zeros((2, 2), dtype=np.intp),
        np.zeros((2, 2), dtype=np.intp),
        0.7,
        np.ones((2, 2), dtype=bool),
    )
    monkeypatch.setattr(lsa, "ordered_batch_apply", lambda *args, **kwargs: [stage])
    monkeypatch.setattr(lsa, "match_segmentation_seq", lambda labels, iou_thresh: labels)
    trace = {}

    lsa.build_geometry_sp_graph(depth, segmentation_trace=trace)

    assert trace["segmentation_depths"].dtype == np.float32
    np.testing.assert_array_equal(trace["segmentation_depths"], depth.astype(np.float32))
```

- [ ] **Step 2: Write a failing pipeline serialization assertion**

Extend `test_streaming_engine_records_pipeline_window_metadata` so its synthetic trace contains:

```python
"segmentation_depths": np.arange(8, dtype=np.float32).reshape(2, 2, 2),
```

Then assert:

```python
assert metadata["schema_version"] == 2
assert arrays["segmentation_depths"].dtype == np.float32
np.testing.assert_array_equal(
    arrays["segmentation_depths"],
    np.arange(8, dtype=np.float32).reshape(2, 2, 2),
)
```

- [ ] **Step 3: Run the tests and verify the new trace field is absent**

```bash
conda run -n vggt-dem python -m pytest \
  tests/test_lsa_graph_builders.py \
  tests/test_streaming_window_engine.py -q
```

Expected: failures for missing `segmentation_depths` and schema version still equal to 1.

- [ ] **Step 4: Add depth to both optional graph traces**

In both `build_depth_sp_graph` and `build_geometry_sp_graph`, add this field only inside the `segmentation_trace is not None` branch:

```python
segmentation_trace.update(
    segmentation_depths=np.asarray(depth, dtype=np.float32).copy(),
    initial_labels=np.stack([stage.initial_labels for stage in stages], axis=0),
    merged_labels=labels,
    confidence_thresholds=np.asarray(
        [stage.confidence_threshold for stage in stages], dtype=np.float32
    ),
    high_confidence_masks=np.stack(
        [stage.high_confidence_mask for stage in stages], axis=0
    ),
)
```

The explicit `.copy()` is required so later scale refinement cannot mutate the recorded input.

- [ ] **Step 5: Upgrade pipeline metadata to schema v2**

Change only the pipeline metadata emitted by `_record_alignment_pipeline_window`:

```python
"schema_version": 2,
```

Keep existing pair debug metadata and filenames unchanged.

- [ ] **Step 6: Run focused trace tests**

```bash
conda run -n vggt-dem python -m pytest \
  tests/test_lsa_graph_builders.py \
  tests/test_streaming_window_engine.py \
  tests/test_alignment_debug.py -q
```

Expected: all tests pass and serialized depth loads with `allow_pickle=False`.

- [ ] **Step 7: Commit runtime depth tracing**

```bash
git add inference_engine/utils/lsa.py \
  inference_engine/streaming_window_engine.py \
  tests/test_lsa_graph_builders.py \
  tests/test_streaming_window_engine.py
git commit -m "feat: record segmentation input depth"
```

### Task 2: Render Shared-Scale Depth and Segmentation Composites

**Files:**
- Modify: `eval/build_alignment_pipeline_report.py`
- Modify: `tests/test_alignment_pipeline_report.py`

- [ ] **Step 1: Upgrade synthetic report traces to schema v2**

Change `_metadata` to return `schema_version: 2`. In `_write_window`, save a deterministic float32 depth sequence:

```python
segmentation_depths=np.stack(
    [
        np.linspace(1.0, 5.0, 96, dtype=np.float32).reshape(8, 12),
        np.linspace(2.0, 6.0, 96, dtype=np.float32).reshape(8, 12),
    ],
    axis=0,
),
```

- [ ] **Step 2: Write failing shared-range and composite tests**

```python
from eval.build_alignment_pipeline_report import (
    compute_shared_depth_range,
    colorize_depth,
    compose_depth_segmentation,
)


def test_depth_color_is_shared_between_methods():
    baseline = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    geometry = np.array([[1.0, 2.0], [30.0, 40.0]], dtype=np.float32)
    display_range = compute_shared_depth_range(baseline, geometry)

    baseline_color, _ = colorize_depth(baseline, display_range)
    geometry_color, _ = colorize_depth(geometry, display_range)

    np.testing.assert_array_equal(baseline_color[0, 1], geometry_color[0, 1])


def test_depth_segmentation_composite_doubles_width():
    depth_color = np.zeros((8, 12, 3), dtype=np.uint8)
    segmentation = np.ones((8, 12, 3), dtype=np.uint8)

    composite = compose_depth_segmentation(depth_color, segmentation)

    assert composite.shape == (8, 24, 3)
```

Add a finite-value test:

```python
def test_shared_depth_range_ignores_non_finite_values():
    baseline = np.array([[1.0, np.nan]], dtype=np.float32)
    geometry = np.array([[3.0, np.inf]], dtype=np.float32)
    lo, hi = compute_shared_depth_range(baseline, geometry)
    assert np.isfinite(lo)
    assert np.isfinite(hi)
    assert lo <= 1.0 < 3.0 <= hi
```

- [ ] **Step 3: Write a failing v1 rejection test**

```python
def test_report_rejects_v1_trace_without_segmentation_depth(tmp_path):
    baseline = _write_run(tmp_path, "depth", "depth")
    geometry = _write_run(tmp_path, "geometry", "geometry")
    for root in (baseline, geometry):
        path = root / "pipeline" / "meta.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["schema_version"] = 1
        path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="Pipeline trace v2.*rerun"):
        build_report(baseline, geometry, _write_images(tmp_path), tmp_path / "report")
```

- [ ] **Step 4: Run tests and verify rendering APIs are missing**

```bash
conda run -n vggt-dem python -m pytest tests/test_alignment_pipeline_report.py -q
```

Expected: import failures for the three new rendering helpers or schema validation failure.

- [ ] **Step 5: Implement shared range, colorization, and composition**

Implement these pure functions:

```python
def compute_shared_depth_range(*depth_maps):
    values = np.concatenate(
        [np.asarray(depth, dtype=np.float32)[np.isfinite(depth)] for depth in depth_maps]
    )
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(values, (2.0, 98.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(values))
        hi = float(np.max(values))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def colorize_depth(depth, display_range):
    depth = np.asarray(depth, dtype=np.float32)
    lo, hi = display_range
    finite = np.isfinite(depth)
    normalized = np.zeros(depth.shape, dtype=np.float32)
    normalized[finite] = np.clip((depth[finite] - lo) / (hi - lo), 0.0, 1.0)
    image = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    image[~finite] = 0
    valid = depth[finite]
    details = {
        "valid_pixels": int(valid.size),
        "depth_min": float(np.min(valid)) if valid.size else None,
        "depth_p02": float(np.percentile(valid, 2)) if valid.size else None,
        "depth_p50": float(np.percentile(valid, 50)) if valid.size else None,
        "depth_p98": float(np.percentile(valid, 98)) if valid.size else None,
        "depth_max": float(np.max(valid)) if valid.size else None,
        "shared_color_min": float(lo),
        "shared_color_max": float(hi),
    }
    return image, details


def compose_depth_segmentation(depth_color, segmentation):
    if depth_color.shape != segmentation.shape:
        raise ValueError("depth and segmentation images must have the same shape")
    return np.concatenate([depth_color, segmentation], axis=1)
```

- [ ] **Step 6: Use one range for both methods and compose initial/merged assets**

Inside each report row, compute:

```python
shared_depth_range = compute_shared_depth_range(
    base_arrays["segmentation_depths"][local_frame],
    geom_arrays["segmentation_depths"][local_frame],
)
```

Pass this range into both `_render_method_stages` calls. In that function, create one depth heatmap, compose it with `initial_image` and `merged_image`, and merge depth statistics into both stage detail mappings. Confidence, overlap, and propagation assets remain unchanged.

- [ ] **Step 7: Validate schema and shapes before rendering**

Require both metadata versions to equal 2 and each window to include `segmentation_depths`. Check its shape against `merged_labels`; raise the exact rerun message for v1 or missing arrays.

- [ ] **Step 8: Run report rendering tests**

```bash
conda run -n vggt-dem python -m pytest tests/test_alignment_pipeline_report.py -q
```

Expected: shared colors match, composite images double width, v1 is rejected, and existing report tests pass.

- [ ] **Step 9: Commit depth composites**

```bash
git add eval/build_alignment_pipeline_report.py tests/test_alignment_pipeline_report.py
git commit -m "feat: pair segmentation with input depth"
```

### Task 3: Pair Baseline and Geometry in the Detail Modal

**Files:**
- Modify: `eval/build_alignment_pipeline_report.py`
- Modify: `tests/test_alignment_pipeline_report.py`
- Modify: `docs/CLOUD_RUN.md`

- [ ] **Step 1: Write failing paired-modal HTML assertions**

Extend the complete report test:

```python
assert 'id="modal-depth-image"' in html
assert 'id="modal-geometry-image"' in html
assert 'id="modal-depth-details"' in html
assert 'id="modal-geometry-details"' in html
assert 'data-stage-name=' in html
assert 'find((candidate) => candidate.stage === stageName)' in html
assert '@media (max-width: 760px)' in html
```

Keep the existing assertion that every row has exactly ten stages.

- [ ] **Step 2: Run the report test and verify the old single-image modal fails**

```bash
conda run -n vggt-dem python -m pytest \
  tests/test_alignment_pipeline_report.py::test_report_writes_ten_assets_per_row_and_single_entry_html -q
```

Expected: failures because only `modal-image` and `modal-details` exist.

- [ ] **Step 3: Generate stage-name card metadata**

Change each card button to include:

```html
data-row="ROW_INDEX" data-stage-name="STAGE_NAME"
```

Do not derive pairing from the card's numeric position.

- [ ] **Step 4: Replace the modal body with two method panels**

Generate this semantic structure:

```html
<div class="modal-compare">
  <section class="method-panel">
    <h3>Baseline / Depth</h3>
    <img id="modal-depth-image" alt="Baseline stage detail">
    <pre id="modal-depth-details"></pre>
  </section>
  <section class="method-panel">
    <h3>Geometry</h3>
    <img id="modal-geometry-image" alt="Geometry stage detail">
    <pre id="modal-geometry-details"></pre>
  </section>
</div>
```

Use a two-column grid on desktop and add:

```css
@media (max-width: 760px) {
  .modal-compare { grid-template-columns: 1fr; }
}
```

- [ ] **Step 5: Pair entries by stage name in JavaScript**

The click handler must use:

```javascript
const row = report.rows[Number(card.dataset.row)];
const stageName = card.dataset.stageName;
const depthStage = row.stages.find(
  (candidate) => candidate.method === 'depth' && candidate.stage === stageName
);
const geometryStage = row.stages.find(
  (candidate) => candidate.method === 'geometry' && candidate.stage === stageName
);
```

Populate both images and both JSON detail blocks. Use the same handler regardless of which method card was clicked.

- [ ] **Step 6: Update cloud documentation**

State that traces generated before schema v2 must be rerun. Replace example scene/report suffixes with the new implementation commit suffix after the final commit; keep baseline and geometry parameters identical.

- [ ] **Step 7: Run focused and complete verification**

```bash
conda run -n vggt-dem python -m pytest tests/test_alignment_pipeline_report.py -q
conda run -n vggt-dem python -m pytest tests -q
conda run -n vggt-dem python -m py_compile \
  inference_engine/utils/lsa.py \
  inference_engine/streaming_window_engine.py \
  eval/build_alignment_pipeline_report.py
git diff --check
```

Expected: all tests pass, syntax compilation succeeds, and the diff has no whitespace errors.

- [ ] **Step 8: Commit the paired modal and documentation**

```bash
git add eval/build_alignment_pipeline_report.py \
  tests/test_alignment_pipeline_report.py \
  docs/CLOUD_RUN.md
git commit -m "feat: compare pipeline stages in paired modal"
```
