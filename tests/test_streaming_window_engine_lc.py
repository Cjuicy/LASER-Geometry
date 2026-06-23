import torch

from inference_engine import streaming_window_engine_lc as lc_module
from inference_engine.streaming_window_engine_lc import StreamingWindowEngineLC


def _pose(x):
    pose = torch.eye(4)
    pose[0, 3] = x
    return pose


def _worker_window():
    return {
        "local_points": torch.ones(1, 1, 1, 1, 3),
        "camera_poses": torch.eye(4).reshape(1, 1, 4, 4),
        "conf": torch.ones(1, 1, 1, 1),
    }


def test_lc_aggregate_keeps_scale_mask_out_of_camera_pose_sim3():
    cache0 = {
        "sim3": (torch.tensor(2.0), torch.eye(3), torch.zeros(3)),
        "local_points": torch.ones(1, 1, 1, 3),
        "camera_poses": torch.eye(4).repeat(1, 1, 1),
        "conf": torch.ones(1, 1, 1),
    }
    cache1 = {
        "sim3": (torch.tensor(2.0), torch.eye(3), torch.zeros(3)),
        "scale_mask": torch.full((1, 1, 1, 1), 3.0),
        "local_points": torch.ones(1, 1, 1, 3),
        "camera_poses": torch.eye(4).repeat(1, 1, 1),
        "conf": torch.ones(1, 1, 1),
    }

    aggregated = StreamingWindowEngineLC.aggregate_caches([cache0, cache1])

    torch.testing.assert_close(
        aggregated["local_points"][0, 1],
        torch.full((1, 1, 3), 6.0),
    )
    torch.testing.assert_close(
        aggregated["camera_poses"][0, 1, :3, 3],
        torch.zeros(3),
    )


def test_lc_registration_worker_keeps_segment_scale_out_of_saved_sim3(monkeypatch, tmp_path):
    monkeypatch.setattr(
        lc_module,
        "estimate_pseudo_depth_and_intrinsics",
        lambda local_points: (
            local_points[..., -1],
            torch.eye(3).repeat(local_points.shape[0], 1, 1),
        ),
    )
    monkeypatch.setattr(
        lc_module,
        "unproject_depth_to_local_points",
        lambda depth, intrinsic: torch.ones(*depth.shape, 3),
    )

    monkeypatch.setattr(
        lc_module,
        "register_adjacent_windows",
        lambda *args, **kwargs: (torch.tensor(2.0), torch.eye(3), torch.zeros(3)),
    )
    monkeypatch.setattr(
        StreamingWindowEngineLC,
        "_build_depth_segment_graph",
        lambda self, *args, **kwargs: ["graph"],
    )
    monkeypatch.setattr(
        lc_module,
        "refine_segment_scales",
        lambda *args, **kwargs: torch.full((1, 1, 1, 1), 1.5),
    )

    engine = StreamingWindowEngineLC(
        torch.nn.Identity(),
        inference_device="cpu",
        dtype=torch.float32,
        process_device="cpu",
        window_size=1,
        overlap=1,
        depth_refine=True,
        cache_root=str(tmp_path),
    )
    engine.temp_cache_dir = tmp_path
    engine.registration_queue.put((_worker_window(), 0.0))
    engine.registration_queue.put((_worker_window(), 0.0))
    engine.registration_queue.put(lc_module.STOP_SIGNAL)

    engine._registration_worker()

    second_cache = torch.load(tmp_path / "window_cache_1.pt", map_location="cpu", weights_only=False)

    torch.testing.assert_close(second_cache["sim3"][0], torch.tensor(2.0))
    torch.testing.assert_close(second_cache["scale_mask"], torch.full((1, 1, 1, 1), 1.5))
    assert "registration_local_points" not in second_cache
    assert "registration_camera_poses" not in second_cache
