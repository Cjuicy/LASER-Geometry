import inspect

import numpy as np
import torch

from pi3.utils.graph import Vertex
from inference_engine.alignment_pipeline_trace import ScaleTraceCollector
from inference_engine.utils import depth as depth_module
from inference_engine.utils import lsa
from inference_engine.utils import scale_anchor as scale_anchor_module


def test_depth_graph_builder_keeps_original_laser_signature():
    signature = inspect.signature(lsa.build_depth_sp_graph)

    assert "point_map" not in signature.parameters
    assert "intrinsic" not in signature.parameters
    assert "normal_method" not in signature.parameters


def test_depth_module_only_owns_depth_segmentation_entrypoints():
    assert hasattr(depth_module, "segment_depth_felzenszwalb_rag")
    assert not hasattr(depth_module, "align_depth_irls")
    assert not hasattr(depth_module, "align_depth_irls_conf_weighted")
    assert not hasattr(depth_module, "assign_overlap_window_depth_scale")
    assert not hasattr(depth_module, "match_segmentation_seq")
    assert not hasattr(depth_module, "connect_bipartite_sp_graphs")


def test_depth_graph_builder_uses_depth_segmentation_only(monkeypatch):
    calls = {}
    labels = np.array([[[0, 0], [1, 1]]], dtype=np.int32)

    def fake_batched_image_op_wrapper(depth, op, **kwargs):
        calls["depth"] = depth
        calls["op"] = op
        calls["kwargs"] = kwargs
        return labels

    def fake_match_segmentation_seq(labels_arg, iou_thresh):
        calls["labels"] = labels_arg
        calls["iou_thresh"] = iou_thresh
        return "depth_graph"

    monkeypatch.setattr(lsa, "batched_image_op_wrapper", fake_batched_image_op_wrapper)
    monkeypatch.setattr(lsa, "match_segmentation_seq", fake_match_segmentation_seq)

    depth = np.ones((1, 2, 2), dtype=np.float32)
    conf = np.ones((1, 2, 2), dtype=np.float32)
    graph = lsa.build_depth_sp_graph(
        depth,
        depth_merge_thresh=0.25,
        conf_map=conf,
        top_conf_percentile=0.7,
        corr_iou_thresh=0.33,
    )

    assert graph == "depth_graph"
    np.testing.assert_array_equal(calls["depth"], depth)
    assert calls["op"] is lsa.segment_depth_felzenszwalb_rag
    assert calls["kwargs"] == {
        "depth_merge_thresh": 0.25,
        "conf_map": conf,
        "top_conf_percentile": 0.7,
    }
    np.testing.assert_array_equal(calls["labels"], labels)
    assert calls["iou_thresh"] == 0.33


def test_geometry_graph_builder_uses_geometry_segmentation_inputs(monkeypatch):
    calls = {}
    labels = np.array(
        [
            [[0, 0], [1, 1]],
            [[2, 2], [3, 3]],
        ],
        dtype=np.int32,
    )

    def fake_batched_image_op_wrapper(depth, op, **kwargs):
        calls["depth"] = depth
        calls["op"] = op
        calls["kwargs"] = kwargs
        return labels

    def fake_match_segmentation_seq(labels_arg, iou_thresh):
        calls["labels"] = labels_arg
        calls["iou_thresh"] = iou_thresh
        return "geometry_graph"

    monkeypatch.setattr(
        lsa,
        "batched_image_op_wrapper",
        fake_batched_image_op_wrapper,
    )
    monkeypatch.setattr(lsa, "match_segmentation_seq", fake_match_segmentation_seq)

    depth = np.ones((2, 2, 2), dtype=np.float32)
    conf = np.full((2, 2, 2), 0.8, dtype=np.float32)
    point_map = np.ones((2, 2, 2, 3), dtype=np.float32)
    intrinsic = np.stack([np.eye(3), np.eye(3) * 2], axis=0).astype(np.float32)
    graph = lsa.build_geometry_sp_graph(
        depth,
        depth_merge_thresh=0.2,
        conf_map=conf,
        top_conf_percentile=0.6,
        corr_iou_thresh=0.44,
        point_map=point_map,
        intrinsic=intrinsic,
        normal_method="sobel",
    )

    assert graph == "geometry_graph"
    np.testing.assert_array_equal(calls["depth"], depth)
    assert calls["op"] is lsa.segment_geometry_felzenszwalb_rag
    assert calls["kwargs"]["depth_merge_thresh"] == 0.2
    assert calls["kwargs"]["top_conf_percentile"] == 0.6
    assert calls["kwargs"]["normal_method"] == "sobel"
    np.testing.assert_array_equal(calls["kwargs"]["conf_map"], conf)
    np.testing.assert_array_equal(calls["kwargs"]["point_map"], point_map)
    np.testing.assert_array_equal(calls["kwargs"]["intrinsic"], intrinsic)
    np.testing.assert_array_equal(calls["labels"], labels)
    assert calls["iou_thresh"] == 0.44


def test_refine_segment_scales_is_mode_neutral_name(monkeypatch):
    calls = {}

    def fake_align(
        src_depth,
        tgt_depth,
        src_graphs,
        tgt_graphs,
        overlap,
        corr_iou_thresh,
        src_conf=None,
        tgt_conf=None,
        scale_anchor_mode="depth_irls",
    ):
        calls["src_depth"] = src_depth
        calls["tgt_depth"] = tgt_depth
        calls["src_graphs"] = src_graphs
        calls["tgt_graphs"] = tgt_graphs
        calls["overlap"] = overlap
        calls["corr_iou_thresh"] = corr_iou_thresh
        calls["src_conf"] = src_conf
        calls["tgt_conf"] = tgt_conf
        calls["scale_anchor_mode"] = scale_anchor_mode
        return np.full(tgt_depth.shape, 2.0, dtype=np.float32)

    monkeypatch.setattr(lsa, "align_adjacent_windows_depth_segments", fake_align)

    src_pcd = np.ones((2, 1, 1, 3), dtype=np.float32)
    tgt_pcd = np.ones((3, 1, 1, 3), dtype=np.float32) * 4.0
    src_conf = np.ones((2, 1, 1), dtype=np.float32) * 0.5
    tgt_conf = np.ones((3, 1, 1), dtype=np.float32) * 0.8
    scale_mask = lsa.refine_segment_scales(
        src_pcd,
        tgt_pcd,
        "src_graph",
        "tgt_graph",
        overlap=1,
        corr_iou_thresh=0.9,
        src_conf=src_conf,
        tgt_conf=tgt_conf,
        scale_anchor_mode="conf_weighted_irls",
    )

    assert isinstance(scale_mask, torch.Tensor)
    assert scale_mask.shape == (3, 1, 1, 1)
    torch.testing.assert_close(scale_mask, torch.full((3, 1, 1, 1), 2.0))
    np.testing.assert_array_equal(calls["src_depth"], src_pcd[..., -1])
    np.testing.assert_array_equal(calls["tgt_depth"], tgt_pcd[..., -1])
    assert calls["src_graphs"] == "src_graph"
    assert calls["tgt_graphs"] == "tgt_graph"
    assert calls["overlap"] == 1
    assert calls["corr_iou_thresh"] == 0.9
    np.testing.assert_array_equal(calls["src_conf"], src_conf)
    np.testing.assert_array_equal(calls["tgt_conf"], tgt_conf)
    assert calls["scale_anchor_mode"] == "conf_weighted_irls"


def test_refine_depth_segments_keeps_backward_compatible_alias(monkeypatch):
    calls = []

    def fake_refine_segment_scales(*args, **kwargs):
        calls.append((args, kwargs))
        return "scale_mask"

    monkeypatch.setattr(lsa, "refine_segment_scales", fake_refine_segment_scales)

    result = lsa.refine_depth_segments(
        "src_pcd",
        "tgt_pcd",
        "src_graph",
        "tgt_graph",
        3,
        corr_iou_thresh=0.8,
    )

    assert result == "scale_mask"
    assert calls == [
        (
            ("src_pcd", "tgt_pcd", "src_graph", "tgt_graph", 3),
            {"corr_iou_thresh": 0.8},
        )
    ]


def test_confidence_weighted_irls_prefers_high_confidence_depth_pair():
    src_depth = np.array([1.0, 1.0], dtype=np.float32)
    tgt_depth = np.array([2.0, 10.0], dtype=np.float32)
    src_conf = np.array([10.0, -10.0], dtype=np.float32)
    tgt_conf = np.array([8.0, -8.0], dtype=np.float32)

    unweighted = scale_anchor_module.align_depth_irls(src_depth, tgt_depth)
    weighted = scale_anchor_module.align_depth_irls_conf_weighted(
        src_depth,
        tgt_depth,
        src_conf,
        tgt_conf,
    )

    assert unweighted > 5.0
    assert weighted < 3.0


def test_overlap_scale_assignment_can_use_confidence_weighted_anchor_mode():
    src_depth = np.array([[[2.0, 10.0]]], dtype=np.float32)
    tgt_depth = np.array([[[1.0, 1.0]]], dtype=np.float32)
    src_conf = np.array([[[10.0, -10.0]]], dtype=np.float32)
    tgt_conf = np.array([[[8.0, -8.0]]], dtype=np.float32)

    src_vertex = Vertex(
        data=np.array([[True, True]]),
        default_cache={"iou": [], "scale": []},
    )
    tgt_vertex = Vertex(
        data=np.array([[True, True]]),
        default_cache={"iou": [], "scale": []},
    )

    scale_anchor_module.assign_overlap_window_depth_scale(
        src_depth,
        tgt_depth,
        [[src_vertex]],
        [[tgt_vertex]],
        src_conf_overlap=src_conf,
        tgt_conf_overlap=tgt_conf,
        scale_anchor_mode="conf_weighted_irls",
    )

    assert len(tgt_vertex.cache["scale"]) == 1
    assert tgt_vertex.cache["scale"][0] < 3.0


def test_scale_trace_distinguishes_anchor_propagated_and_identity():
    src_labels = np.zeros((1, 2, 2), dtype=np.intp)
    tgt_labels = np.array(
        [
            [[0, 0], [0, 0]],
            [[0, 0], [0, 0]],
            [[0, 0], [0, 1]],
        ],
        dtype=np.intp,
    )
    src_graph = lsa.build_sp_graph_from_labels(src_labels, corr_iou_thresh=0.3)
    tgt_graph = lsa.build_sp_graph_from_labels(tgt_labels, corr_iou_thresh=0.3)
    collector = ScaleTraceCollector()

    scale_mask = lsa.align_adjacent_windows_depth_segments(
        np.full((1, 2, 2), 2.0, dtype=np.float32),
        np.ones((3, 2, 2), dtype=np.float32),
        src_graph,
        tgt_graph,
        overlap=1,
        corr_iou_thresh=0.4,
        trace=collector,
    )

    states = {(state.frame, state.segment): state for state in collector.segment_states}
    assert states[(0, 0)].role == "A"
    assert states[(1, 0)].role == "P"
    assert states[(2, 1)].role == "I"
    assert states[(2, 1)].scale == 1.0
    assert collector.matches
    assert collector.propagation_edges
    assert scale_mask.shape == (3, 2, 2)


def test_scale_trace_does_not_change_scale_mask():
    labels = np.array(
        [
            [[0, 0], [1, 1]],
            [[0, 0], [1, 1]],
        ],
        dtype=np.intp,
    )
    src_depth = np.ones((2, 2, 2), dtype=np.float32)
    tgt_depth = np.ones((2, 2, 2), dtype=np.float32) * 2.0

    without_trace = lsa.align_adjacent_windows_depth_segments(
        src_depth,
        tgt_depth,
        lsa.build_sp_graph_from_labels(labels),
        lsa.build_sp_graph_from_labels(labels),
        overlap=1,
    )
    collector = ScaleTraceCollector()
    with_trace = lsa.align_adjacent_windows_depth_segments(
        src_depth,
        tgt_depth,
        lsa.build_sp_graph_from_labels(labels),
        lsa.build_sp_graph_from_labels(labels),
        overlap=1,
        trace=collector,
    )

    np.testing.assert_allclose(with_trace, without_trace)
