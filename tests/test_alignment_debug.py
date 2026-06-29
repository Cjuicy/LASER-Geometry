import json

import numpy as np

from inference_engine.alignment_debug import (
    AlignmentDebugRecorder,
    summarize_graph_layer,
)
from pi3.utils.graph import Vertex


def test_alignment_debug_recorder_disabled_is_noop(tmp_path):
    recorder = AlignmentDebugRecorder(enabled=False, root_dir=tmp_path, scene_name="demo")

    recorder.record_pair(pair_index=1, payload={"scale": 1.2})

    assert not (tmp_path / "demo").exists()


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


def test_summarize_graph_layer_extracts_masks_scales_and_iou():
    v0 = Vertex(
        data=np.array([[True, False], [False, False]]),
        default_cache={"iou": [0.7], "scale": [1.2]},
    )
    v1 = Vertex(
        data=np.array([[False, True], [True, True]]),
        default_cache={"iou": [], "scale": []},
    )

    summary = summarize_graph_layer([v0, v1])

    assert summary["masks"].shape == (2, 2, 2)
    assert summary["has_scale"].tolist() == [True, False]
    np.testing.assert_allclose(summary["mean_iou"], [0.7, 0.0], rtol=1e-6)
    np.testing.assert_allclose(summary["mean_scale"], [1.2, 1.0], rtol=1e-6)


def test_recorder_writes_numeric_pipeline_window(tmp_path):
    recorder = AlignmentDebugRecorder(enabled=True, root_dir=tmp_path, scene_name="scene")

    path = recorder.record_pipeline_window(
        window_index=1,
        payload={
            "global_frame_indices": np.array([2, 3], dtype=np.int32),
            "initial_labels": np.zeros((2, 3, 4), dtype=np.int64),
            "merged_labels": np.ones((2, 3, 4), dtype=np.int64),
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
        assert arrays["merged_labels"].dtype == np.uint16
    meta = json.loads((path.parent / "meta.json").read_text(encoding="utf-8"))
    assert meta["confidence_retained_fraction"] == 0.3
    assert meta["confidence_quantile"] == 0.7
    assert meta["sample_interval"] == 10
