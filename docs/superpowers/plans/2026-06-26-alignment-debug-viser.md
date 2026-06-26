# Alignment Debug Viser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional, non-LC alignment debug pipeline that records LASER window-alignment internals and visualizes them in an independent Viser viewer without changing the default demo output path.

**Architecture:** The main inference path remains unchanged unless `--debug_alignment` is explicitly enabled. `demo.py` and `StreamingWindowEngine` only receive a thin optional config and a guarded debug-recorder call; all trace serialization, segment/IoU extraction, point-cloud sampling, and Viser rendering live in new standalone modules. The default `outputs/viser/<scene>` format stays compatible with the existing viewer.

**Tech Stack:** Python, NumPy, PyTorch tensors converted at the boundary, existing `viser` package, existing LASER segment graph utilities, pytest.

---

## File Structure

- Create `inference_engine/alignment_debug.py`
  - Owns trace serialization.
  - Converts tensors/graphs into compact NumPy arrays.
  - Saves one `pair_XXXX.npz` plus `meta.json`.
  - Has no dependency on Viser.

- Create `eval/vis_alignment_debug.py`
  - Owns interactive 3D visualization.
  - Reads debug trace directories.
  - Can load one run or baseline-vs-geometry paired runs.
  - Does not import `demo.py` or `StreamingWindowEngine`.

- Modify `inference_engine/streaming_window_engine.py`
  - Add optional constructor args:
    - `debug_alignment=False`
    - `debug_alignment_path=None`
    - `debug_alignment_scene=None`
  - Instantiate a debug recorder only when enabled.
  - Insert one guarded recorder call after Sim3 registration and segment refinement.
  - Default behavior and cache summary remain unchanged.

- Modify `demo.py`
  - Add optional CLI flags:
    - `--debug_alignment`
    - `--debug_alignment_path`
  - Pass them through to `StreamingWindowEngine`.
  - Default values preserve existing commands exactly.

- Create `tests/test_alignment_debug.py`
  - Tests recorder save format, graph extraction, and default-off behavior.

- Modify existing CLI tests only if needed to assert new flags are optional.

---

### Task 1: Alignment Debug Trace Module

**Files:**
- Create: `inference_engine/alignment_debug.py`
- Test: `tests/test_alignment_debug.py`

- [ ] **Step 1: Write failing test for disabled recorder**

```python
from inference_engine.alignment_debug import AlignmentDebugRecorder


def test_alignment_debug_recorder_disabled_is_noop(tmp_path):
    recorder = AlignmentDebugRecorder(enabled=False, root_dir=tmp_path, scene_name="demo")
    recorder.record_pair(pair_index=1, payload={"scale": 1.2})
    assert not (tmp_path / "demo").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
conda run -n vggt-dem python -m pytest tests/test_alignment_debug.py::test_alignment_debug_recorder_disabled_is_noop -q
```

Expected: FAIL because `inference_engine.alignment_debug` does not exist.

- [ ] **Step 3: Implement minimal disabled recorder**

Create `inference_engine/alignment_debug.py`:

```python
from pathlib import Path


class AlignmentDebugRecorder:
    def __init__(self, *, enabled=False, root_dir=None, scene_name=None):
        self.enabled = bool(enabled)
        self.root_dir = None if root_dir is None else Path(root_dir)
        self.scene_name = scene_name or "alignment_debug"

    @property
    def scene_dir(self):
        if self.root_dir is None:
            return None
        return self.root_dir / self.scene_name

    def record_pair(self, *, pair_index, payload):
        if not self.enabled:
            return None
        raise NotImplementedError("Enabled debug recording is implemented in the next step.")
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
conda run -n vggt-dem python -m pytest tests/test_alignment_debug.py::test_alignment_debug_recorder_disabled_is_noop -q
```

Expected: PASS.

- [ ] **Step 5: Write failing test for enabled recorder save format**

```python
import json
import numpy as np

from inference_engine.alignment_debug import AlignmentDebugRecorder


def test_alignment_debug_recorder_writes_npz_and_meta(tmp_path):
    recorder = AlignmentDebugRecorder(enabled=True, root_dir=tmp_path, scene_name="scene")

    saved_path = recorder.record_pair(
        pair_index=2,
        payload={
            "sim3_scale": 1.25,
            "src_points": np.zeros((2, 3), dtype=np.float32),
            "tgt_points_before": np.ones((2, 3), dtype=np.float32),
            "mutual_conf_mask": np.array([True, False]),
        },
        metadata={"segment_mode": "geometry", "window_size": 30},
    )

    assert saved_path == tmp_path / "scene" / "pair_0002.npz"
    assert saved_path.is_file()
    assert (tmp_path / "scene" / "meta.json").is_file()

    arrays = np.load(saved_path)
    assert arrays["src_points"].shape == (2, 3)
    assert arrays["mutual_conf_mask"].tolist() == [True, False]

    meta = json.loads((tmp_path / "scene" / "meta.json").read_text(encoding="utf-8"))
    assert meta["segment_mode"] == "geometry"
    assert meta["window_size"] == 30
```

- [ ] **Step 6: Run test to verify it fails**

Run:

```bash
conda run -n vggt-dem python -m pytest tests/test_alignment_debug.py::test_alignment_debug_recorder_writes_npz_and_meta -q
```

Expected: FAIL because enabled recording raises `NotImplementedError`.

- [ ] **Step 7: Implement enabled recorder**

Add to `inference_engine/alignment_debug.py`:

```python
import json
import numpy as np


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _jsonable(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
```

Update `record_pair`:

```python
    def record_pair(self, *, pair_index, payload, metadata=None):
        if not self.enabled:
            return None
        if self.scene_dir is None:
            raise ValueError("root_dir is required when alignment debug recording is enabled.")

        self.scene_dir.mkdir(parents=True, exist_ok=True)
        if metadata:
            meta_path = self.scene_dir / "meta.json"
            meta_path.write_text(
                json.dumps({k: _jsonable(v) for k, v in metadata.items()}, indent=2, sort_keys=True),
                encoding="utf-8",
            )

        np_payload = {key: _to_numpy(value) for key, value in payload.items()}
        out_path = self.scene_dir / f"pair_{pair_index:04d}.npz"
        np.savez_compressed(out_path, **np_payload)
        return out_path
```

- [ ] **Step 8: Run recorder tests**

Run:

```bash
conda run -n vggt-dem python -m pytest tests/test_alignment_debug.py -q
```

Expected: PASS.

---

### Task 2: Segment Graph Extraction Helpers

**Files:**
- Modify: `inference_engine/alignment_debug.py`
- Test: `tests/test_alignment_debug.py`

- [ ] **Step 1: Write failing test for graph extraction**

```python
import numpy as np

from inference_engine.alignment_debug import summarize_graph_layer
from pi3.utils.graph import Vertex


def test_summarize_graph_layer_extracts_masks_scales_and_iou():
    v0 = Vertex(data=np.array([[True, False], [False, False]]), default_cache={"iou": [0.7], "scale": [1.2]})
    v1 = Vertex(data=np.array([[False, True], [True, True]]), default_cache={"iou": [], "scale": []})

    summary = summarize_graph_layer([v0, v1])

    assert summary["masks"].shape == (2, 2, 2)
    assert summary["has_scale"].tolist() == [True, False]
    assert summary["mean_iou"].tolist() == [0.7, 0.0]
    assert summary["mean_scale"].tolist() == [1.2, 1.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
conda run -n vggt-dem python -m pytest tests/test_alignment_debug.py::test_summarize_graph_layer_extracts_masks_scales_and_iou -q
```

Expected: FAIL because `summarize_graph_layer` does not exist.

- [ ] **Step 3: Implement graph extraction**

Add:

```python
def summarize_graph_layer(graph_layer):
    masks = []
    has_scale = []
    mean_iou = []
    mean_scale = []

    for vertex in graph_layer:
        masks.append(np.asarray(vertex.data, dtype=bool))
        scales = np.asarray(vertex.cache.get("scale", []), dtype=np.float32)
        ious = np.asarray(vertex.cache.get("iou", []), dtype=np.float32)
        has_scale.append(scales.size > 0)
        mean_scale.append(float(scales.mean()) if scales.size else 1.0)
        mean_iou.append(float(ious.mean()) if ious.size else 0.0)

    return {
        "masks": np.stack(masks, axis=0) if masks else np.zeros((0, 0, 0), dtype=bool),
        "has_scale": np.asarray(has_scale, dtype=bool),
        "mean_iou": np.asarray(mean_iou, dtype=np.float32),
        "mean_scale": np.asarray(mean_scale, dtype=np.float32),
    }
```

- [ ] **Step 4: Run graph extraction tests**

Run:

```bash
conda run -n vggt-dem python -m pytest tests/test_alignment_debug.py -q
```

Expected: PASS.

---

### Task 3: Optional Streaming Engine Hook

**Files:**
- Modify: `inference_engine/streaming_window_engine.py`
- Test: `tests/test_streaming_window_engine.py` or `tests/test_alignment_debug.py`

- [ ] **Step 1: Write failing test that default engine does not create debug recorder output**

Use constructor-only test with a dummy delegate. Assert new args default to disabled and no debug dir is created.

```python
import torch

from inference_engine.streaming_window_engine import StreamingWindowEngine


class DummyModel(torch.nn.Module):
    def forward(self, x):
        raise RuntimeError("not used")


def test_streaming_engine_alignment_debug_is_default_off(tmp_path):
    engine = StreamingWindowEngine(
        DummyModel(),
        inference_device="cpu",
        dtype=torch.float32,
        intermediate_device="cpu",
        process_device="cpu",
        cache_root=str(tmp_path / "cache"),
        benchmark_latency=False,
    )

    assert engine.debug_alignment is False
    assert not (tmp_path / "debug").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
conda run -n vggt-dem python -m pytest tests/test_streaming_window_engine.py::test_streaming_engine_alignment_debug_is_default_off -q
```

Expected: FAIL because `debug_alignment` does not exist.

- [ ] **Step 3: Add constructor-only optional debug fields**

In `StreamingWindowEngine.__init__`, add keyword args after existing debug-safe options:

```python
            debug_alignment: bool = False,
            debug_alignment_path: str | None = None,
            debug_alignment_scene: str | None = None,
```

Import:

```python
from .alignment_debug import AlignmentDebugRecorder
```

Set:

```python
        self.debug_alignment = bool(debug_alignment)
        self.alignment_debug_recorder = AlignmentDebugRecorder(
            enabled=self.debug_alignment,
            root_dir=debug_alignment_path,
            scene_name=debug_alignment_scene,
        )
```

- [ ] **Step 4: Run default-off test**

Run:

```bash
conda run -n vggt-dem python -m pytest tests/test_streaming_window_engine.py::test_streaming_engine_alignment_debug_is_default_off -q
```

Expected: PASS.

- [ ] **Step 5: Add guarded recorder call after refinement**

Inside `_registration_worker`, after `working_window['local_points']` has received Sim3 and optional segment refinement, call a new helper:

```python
                self._record_alignment_debug_pair(
                    pair_index=self.cache_id,
                    sim3_scale=s_d,
                    sim3_R=R,
                    sim3_t=t,
                    prev_local_points=self.prev_window_cache['local_points'][-self.overlap:],
                    cur_local_points_before=cur_local_points,
                    cur_local_points_after=working_window['local_points'][:self.overlap],
                    prev_conf=self.prev_window_cache['conf'][-self.overlap:],
                    cur_conf=working_window['conf'][:self.overlap],
                    mutual_conf_mask=conf_mask,
                    tgt_sp_graph=tgt_sp_graph,
                )
```

Add helper method:

```python
    def _record_alignment_debug_pair(self, **kwargs):
        if not self.debug_alignment:
            return
        payload = {
            "sim3_scale": kwargs["sim3_scale"],
            "sim3_R": kwargs["sim3_R"],
            "sim3_t": kwargs["sim3_t"],
            "src_points_overlap": kwargs["prev_local_points"],
            "tgt_points_before_overlap": kwargs["cur_local_points_before"],
            "tgt_points_after_overlap": kwargs["cur_local_points_after"],
            "src_conf_overlap": kwargs["prev_conf"],
            "tgt_conf_overlap": kwargs["cur_conf"],
            "mutual_conf_mask": kwargs["mutual_conf_mask"],
        }
        if kwargs["tgt_sp_graph"] is not None:
            from .alignment_debug import summarize_graph_layer
            graph_summary = summarize_graph_layer(kwargs["tgt_sp_graph"][0])
            payload.update({
                "tgt_segment_masks_frame0": graph_summary["masks"],
                "tgt_segment_has_scale_frame0": graph_summary["has_scale"],
                "tgt_segment_mean_iou_frame0": graph_summary["mean_iou"],
                "tgt_segment_mean_scale_frame0": graph_summary["mean_scale"],
            })
        self.alignment_debug_recorder.record_pair(
            pair_index=kwargs["pair_index"],
            payload=payload,
            metadata={
                "segment_mode": self.segment_mode,
                "normal_method": self.normal_method,
                "scale_anchor_mode": self.scale_anchor_mode,
                "window_size": self.window_size,
                "overlap": self.overlap,
                "top_conf_percentile": self.top_conf_percentile,
            },
        )
```

- [ ] **Step 6: Run streaming tests**

Run:

```bash
conda run -n vggt-dem python -m pytest tests/test_streaming_window_engine.py tests/test_alignment_debug.py -q
```

Expected: PASS.

---

### Task 4: Demo CLI Optional Flags

**Files:**
- Modify: `demo.py`
- Test: `tests/test_demo_vggt_cli.py` or create `tests/test_demo_cli.py`

- [ ] **Step 1: Write failing parser test**

```python
from demo import get_args_parser


def test_demo_parser_accepts_alignment_debug_flags():
    parser = get_args_parser()
    args = parser.parse_args([
        "--data_path", "data/09/image_2",
        "--debug_alignment",
        "--debug_alignment_path", "outputs/debug_alignment",
    ])

    assert args.debug_alignment is True
    assert args.debug_alignment_path == "outputs/debug_alignment"
```

- [ ] **Step 2: Run parser test to verify it fails**

Run:

```bash
conda run -n vggt-dem python -m pytest tests/test_demo_cli.py::test_demo_parser_accepts_alignment_debug_flags -q
```

Expected: FAIL because flags do not exist.

- [ ] **Step 3: Add flags to `demo.py` parser**

Add:

```python
    parser.add_argument('--debug_alignment', action='store_true', help='save optional alignment debug traces')
    parser.add_argument('--debug_alignment_path', default='outputs/debug_alignment', type=str, help='alignment debug trace root')
```

- [ ] **Step 4: Pass flags to engine**

In `load_model(args)` pass:

```python
        debug_alignment=args.debug_alignment,
        debug_alignment_path=args.debug_alignment_path,
        debug_alignment_scene=args.scene_name,
```

- [ ] **Step 5: Run parser and engine tests**

Run:

```bash
conda run -n vggt-dem python -m pytest tests/test_demo_cli.py tests/test_streaming_window_engine.py tests/test_alignment_debug.py -q
```

Expected: PASS.

---

### Task 5: Independent Viser Debug Viewer

**Files:**
- Create: `eval/vis_alignment_debug.py`
- Test: `tests/test_alignment_debug_viewer.py`

- [ ] **Step 1: Write failing loader test**

```python
import numpy as np

from eval.vis_alignment_debug import load_debug_pairs


def test_load_debug_pairs_reads_npz_files_in_order(tmp_path):
    debug_dir = tmp_path / "scene"
    debug_dir.mkdir()
    np.savez_compressed(debug_dir / "pair_0002.npz", sim3_scale=np.array(2.0))
    np.savez_compressed(debug_dir / "pair_0001.npz", sim3_scale=np.array(1.0))

    pairs = load_debug_pairs(debug_dir)

    assert [pair["pair_name"] for pair in pairs] == ["pair_0001", "pair_0002"]
    assert pairs[0]["arrays"]["sim3_scale"].item() == 1.0
```

- [ ] **Step 2: Run loader test to verify it fails**

Run:

```bash
conda run -n vggt-dem python -m pytest tests/test_alignment_debug_viewer.py::test_load_debug_pairs_reads_npz_files_in_order -q
```

Expected: FAIL because viewer module does not exist.

- [ ] **Step 3: Implement loader-only viewer foundation**

Create `eval/vis_alignment_debug.py`:

```python
import argparse
import time
from pathlib import Path

import numpy as np


def load_debug_pairs(debug_dir):
    debug_dir = Path(debug_dir)
    pairs = []
    for path in sorted(debug_dir.glob("pair_*.npz")):
        pairs.append({"pair_name": path.stem, "path": path, "arrays": np.load(path)})
    return pairs


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_dir", required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max_points", type=int, default=200000)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    pairs = load_debug_pairs(args.debug_dir)
    if not pairs:
        raise FileNotFoundError(f"No pair_*.npz files found in {args.debug_dir}")

    import viser
    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-z")
    server.gui.add_markdown(f"Loaded {len(pairs)} alignment debug pairs from `{args.debug_dir}`")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run loader test**

Run:

```bash
conda run -n vggt-dem python -m pytest tests/test_alignment_debug_viewer.py -q
```

Expected: PASS.

- [ ] **Step 5: Add point-cloud preparation helpers**

Add tests for `make_confidence_colors(conf)` and `sample_points(points, colors, max_points)`.

- [ ] **Step 6: Add Viser layers**

Use debug arrays:

```text
/baseline/src_overlap
/baseline/tgt_before
/baseline/tgt_after
/baseline/mutual_conf
/baseline/scale_anchor_segments
/geometry/src_overlap
/geometry/tgt_before
/geometry/tgt_after
/geometry/mutual_conf
/geometry/scale_anchor_segments
```

Do not import inference engine here.

- [ ] **Step 7: Add compare-mode CLI**

Add:

```bash
--baseline_debug_dir outputs/debug_alignment/kitti08_depth_s10
--geometry_debug_dir outputs/debug_alignment/kitti08_geometry_s10
```

When both are present, load both and display side-by-side with an x-axis offset.

---

### Task 6: Verification and Cloud Commands

**Files:**
- No new files unless docs update is desired.

- [ ] **Step 1: Run focused tests**

Run:

```bash
conda run -n vggt-dem python -m pytest tests/test_alignment_debug.py tests/test_alignment_debug_viewer.py tests/test_streaming_window_engine.py tests/test_demo_cli.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full tests**

Run:

```bash
conda run -n vggt-dem python -m pytest -q
```

Expected: PASS.

- [ ] **Step 3: Compile changed Python files**

Run:

```bash
conda run -n vggt-dem python -m py_compile inference_engine/alignment_debug.py eval/vis_alignment_debug.py demo.py inference_engine/streaming_window_engine.py
```

Expected: exit code 0.

- [ ] **Step 4: Check git diff**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intended tracked files changed plus existing untracked `viser/`.

- [ ] **Step 5: Commit**

Run:

```bash
git add inference_engine/alignment_debug.py eval/vis_alignment_debug.py demo.py inference_engine/streaming_window_engine.py tests/test_alignment_debug.py tests/test_alignment_debug_viewer.py tests/test_demo_cli.py
git commit -m "feat: add optional alignment debug viewer"
git push origin main
```

- [ ] **Step 6: Cloud smoke commands**

Run depth baseline trace:

```bash
python demo.py \
  --model_ckpt weights/model.safetensors \
  --data_path data/08/image_2 \
  --scene_name kitti08_pi3_depth_s10_w30_o10_debug \
  --output_path outputs/viser \
  --cache_path cache/kitti08_pi3_depth_s10_w30_o10_debug \
  --sample_interval 10 \
  --window_size 30 \
  --overlap 10 \
  --depth_refine \
  --segment_mode depth \
  --debug_alignment \
  --debug_alignment_path outputs/debug_alignment
```

Run geometry trace:

```bash
python demo.py \
  --model_ckpt weights/model.safetensors \
  --data_path data/08/image_2 \
  --scene_name kitti08_pi3_geometry_s10_w30_o10_debug \
  --output_path outputs/viser \
  --cache_path cache/kitti08_pi3_geometry_s10_w30_o10_debug \
  --sample_interval 10 \
  --window_size 30 \
  --overlap 10 \
  --depth_refine \
  --segment_mode geometry \
  --normal_method cross \
  --debug_alignment \
  --debug_alignment_path outputs/debug_alignment
```

Open single-run viewer:

```bash
python eval/vis_alignment_debug.py \
  --debug_dir outputs/debug_alignment/kitti08_pi3_geometry_s10_w30_o10_debug \
  --port 8080
```

Open comparison viewer:

```bash
python eval/vis_alignment_debug.py \
  --baseline_debug_dir outputs/debug_alignment/kitti08_pi3_depth_s10_w30_o10_debug \
  --geometry_debug_dir outputs/debug_alignment/kitti08_pi3_geometry_s10_w30_o10_debug \
  --port 8080
```

---

## Self-Review

- Spec coverage: The plan preserves default `demo.py` and `StreamingWindowEngine` behavior, makes debug optional, excludes LC, records alignment internals, and adds an independent Viser viewer.
- Unfinished-marker scan: No unfinished markers remain; each task has concrete files, test commands, and expected outcomes.
- Type consistency: `AlignmentDebugRecorder`, `summarize_graph_layer`, `debug_alignment`, `debug_alignment_path`, and `debug_alignment_scene` names are consistent across tasks.
