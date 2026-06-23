import numpy as np
import torch

from inference_engine import streaming_window_engine as swe_module
from inference_engine.streaming_window_engine import StreamingWindowEngine


def _engine(tmp_path, segment_mode="depth", normal_method="cross"):
    return StreamingWindowEngine(
        torch.nn.Identity(),
        inference_device="cpu",
        dtype=torch.float32,
        process_device="cpu",
        window_size=1,
        overlap=1,
        depth_refine=True,
        cache_root=str(tmp_path),
        benchmark_latency=False,
        segment_mode=segment_mode,
        normal_method=normal_method,
    )


def _worker_window():
    return {
        "local_points": torch.ones(1, 1, 1, 1, 3),
        "camera_poses": torch.eye(4).reshape(1, 1, 4, 4),
        "conf": torch.ones(1, 1, 1, 1),
    }


def test_depth_segment_graph_uses_original_laser_inputs(monkeypatch, tmp_path):
    calls = []

    def fake_build_depth_sp_graph(depth, **kwargs):
        calls.append((depth, kwargs))
        return "depth_graph"

    monkeypatch.setattr(swe_module, "build_depth_sp_graph", fake_build_depth_sp_graph)
    engine = _engine(tmp_path, segment_mode="depth", normal_method="sobel")

    local_points = torch.tensor(
        [
            [[[1.0, 2.0, 3.0]]],
            [[[4.0, 5.0, 6.0]]],
        ]
    )
    conf = torch.ones(2, 1, 1)
    graph = engine._build_depth_segment_graph(local_points, conf)

    assert graph == "depth_graph"
    assert len(calls) == 1
    depth, kwargs = calls[0]
    np.testing.assert_array_equal(depth, local_points.numpy()[..., -1])
    assert set(kwargs) == {"conf_map", "top_conf_percentile"}
    np.testing.assert_array_equal(kwargs["conf_map"], conf.numpy())
    assert kwargs["top_conf_percentile"] == engine.top_conf_percentile


def test_geometry_segment_graph_uses_geometry_inputs(monkeypatch, tmp_path):
    calls = []

    def fake_build_geometry_sp_graph(depth, **kwargs):
        calls.append((depth, kwargs))
        return "geometry_graph"

    monkeypatch.setattr(swe_module, "build_geometry_sp_graph", fake_build_geometry_sp_graph)
    engine = _engine(tmp_path, segment_mode="geometry", normal_method="sobel")

    local_points = torch.tensor(
        [
            [[[1.0, 2.0, 3.0]]],
            [[[4.0, 5.0, 6.0]]],
        ]
    )
    conf = torch.ones(2, 1, 1)
    intrinsic = torch.eye(3)
    graph = engine._build_geometry_segment_graph(local_points, conf, intrinsic)

    assert graph == "geometry_graph"
    assert len(calls) == 1
    depth, kwargs = calls[0]
    np.testing.assert_array_equal(depth, local_points.numpy()[..., -1])
    assert set(kwargs) == {
        "conf_map",
        "top_conf_percentile",
        "point_map",
        "intrinsic",
        "normal_method",
    }
    np.testing.assert_array_equal(kwargs["conf_map"], conf.numpy())
    np.testing.assert_array_equal(kwargs["point_map"], local_points.numpy())
    np.testing.assert_array_equal(kwargs["intrinsic"], intrinsic.numpy())
    assert kwargs["top_conf_percentile"] == engine.top_conf_percentile
    assert kwargs["normal_method"] == "sobel"


def test_registration_worker_uses_geometry_specific_refinement_branch(monkeypatch, tmp_path):
    monkeypatch.setattr(
        swe_module,
        "estimate_pseudo_depth_and_intrinsics",
        lambda local_points: (
            local_points[..., -1],
            torch.eye(3).repeat(local_points.shape[0], 1, 1),
        ),
    )
    monkeypatch.setattr(
        swe_module,
        "unproject_depth_to_local_points",
        lambda depth, intrinsic: torch.ones(*depth.shape, 3),
    )
    monkeypatch.setattr(
        swe_module,
        "register_adjacent_windows",
        lambda *args, **kwargs: (torch.tensor(2.0), torch.eye(3), torch.zeros(3)),
    )
    monkeypatch.setattr(
        swe_module,
        "apply_sim3_to_pose",
        lambda camera_poses, *args, **kwargs: camera_poses,
    )

    refine_calls = []

    def fake_refine(prev_points, tgt_points, anchor_graph, tgt_graph, overlap):
        refine_calls.append((anchor_graph, tgt_graph, overlap))
        return torch.full((1, 1, 1, 1), 1.5)

    monkeypatch.setattr(swe_module, "refine_segment_scales", fake_refine)

    engine = _engine(tmp_path, segment_mode="geometry")
    build_calls = []

    def fake_build_segment_graph(local_points, conf, ref_intrinsic):
        build_calls.append((local_points.clone(), conf.clone(), ref_intrinsic.clone()))
        return f"graph_{len(build_calls)}"

    engine._build_geometry_segment_graph = fake_build_segment_graph
    engine.temp_cache_dir = tmp_path
    engine.registration_queue.put((_worker_window(), 0.0))
    engine.registration_queue.put((_worker_window(), 0.0))
    engine.registration_queue.put(swe_module.STOP_SIGNAL)

    engine._registration_worker()

    assert len(build_calls) == 2
    assert refine_calls == [("graph_1", "graph_2", 1)]
    second_cache = torch.load(tmp_path / "window_cache_1.pt", map_location="cpu", weights_only=False)
    torch.testing.assert_close(second_cache["local_points"], torch.full((1, 1, 1, 3), 3.0))
