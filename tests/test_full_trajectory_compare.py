import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "eval" / "vis_full_trajectory_compare.py"
    spec = importlib.util.spec_from_file_location("vis_full_trajectory_compare", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_numeric_paths_sorts_frame_numbers(tmp_path):
    viewer = _load_module()
    for name in ("frame_10.png", "frame_2.png", "frame_1.png"):
        (tmp_path / name).touch()
    assert [p.name for p in viewer.numeric_paths(tmp_path, "frame_*.png")] == [
        "frame_1.png", "frame_2.png", "frame_10.png"
    ]


def test_load_tum_poses_parses_wxyz(tmp_path):
    viewer = _load_module()
    path = tmp_path / "pred_traj.txt"
    np.savetxt(path, [[0.0, 1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]])
    poses = viewer.load_tum_poses(path)
    np.testing.assert_allclose(poses[0], np.array([
        [1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]
    ], dtype=np.float32))


def test_load_kitti_poses_builds_homogeneous_matrices(tmp_path):
    viewer = _load_module()
    path = tmp_path / "09.txt"
    np.savetxt(path, [[1, 0, 0, 2, 0, 1, 0, 3, 0, 0, 1, 4]])

    poses = viewer.load_kitti_poses(path)

    np.testing.assert_allclose(poses[0], np.array([
        [1, 0, 0, 2], [0, 1, 0, 3], [0, 0, 1, 4], [0, 0, 0, 1]
    ], dtype=np.float32))


def test_sample_ground_truth_requires_exact_count():
    viewer = _load_module()
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 21, axis=0)

    sampled = viewer.sample_ground_truth(poses, stride=2, expected_count=11)

    assert len(sampled) == 11
    with pytest.raises(ValueError, match="sampled ground-truth count"):
        viewer.sample_ground_truth(poses, stride=3, expected_count=11)


def test_align_geometry_matches_first_baseline_pose():
    viewer = _load_module()
    baseline = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    baseline[0, 0, 3] = 4.0
    geometry = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
    geometry[:, 1, 3] = [2.0, 5.0]
    aligned = viewer.align_geometry_to_baseline(baseline, geometry)
    np.testing.assert_allclose(aligned[0], baseline[0])
    np.testing.assert_allclose(aligned[1, :3, 3], [4.0, 3.0, 0.0])


def test_align_sim3_transforms_poses_points_and_computes_ate():
    viewer = _load_module()
    rotation = np.array(
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32
    )
    translation = np.array([4, -2, 1], dtype=np.float32)
    scale = 2.0
    prediction = np.repeat(np.eye(4, dtype=np.float32)[None], 4, axis=0)
    prediction[:, :3, 3] = [
        [0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]
    ]
    reference = prediction.copy()
    reference[:, :3, 3] = (
        scale * (prediction[:, :3, 3] @ rotation.T) + translation
    )
    reference[:, :3, :3] = rotation

    aligned, similarity = viewer.align_poses_sim3(prediction, reference)

    np.testing.assert_allclose(aligned, reference, atol=1e-5)
    assert viewer.translation_ate(aligned, reference) == pytest.approx(0.0, abs=1e-5)
    point = np.array([[1, 2, 3]], dtype=np.float32)
    transformed = viewer.apply_similarity_to_points(point, similarity)
    np.testing.assert_allclose(
        transformed, scale * (point @ rotation.T) + translation, atol=1e-5
    )


def test_unproject_depth_uses_intrinsics():
    viewer = _load_module()
    depth = np.full((2, 2), 2.0, dtype=np.float32)
    points = viewer.unproject_depth(depth, np.eye(3, dtype=np.float32))
    np.testing.assert_allclose(points[1, 1], [2.0, 2.0, 2.0])


def test_sample_cloud_is_deterministic_and_capped():
    viewer = _load_module()
    points = np.arange(60, dtype=np.float32).reshape(20, 3)
    colors = np.arange(60, dtype=np.uint8).reshape(20, 3)
    first = viewer.sample_cloud(points, colors, 5, seed=9)
    second = viewer.sample_cloud(points, colors, 5, seed=9)
    assert first[0].shape == (5, 3)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_frame_cloud_preserves_confident_point_color_pair(tmp_path):
    viewer = _load_module()
    rgb = np.array(
        [[[10, 0, 0], [20, 0, 0]], [[30, 0, 0], [40, 0, 0]]],
        dtype=np.uint8,
    )
    depth = np.ones((2, 2), dtype=np.float32)
    confidence = np.array([[1, 2], [3, 4]], dtype=np.float32)
    viewer.iio.imwrite(tmp_path / "frame_0000.png", rgb)
    np.save(tmp_path / "frame_0000.npy", depth)
    np.save(tmp_path / "conf_0.npy", confidence)
    run = {
        "rgb_paths": [tmp_path / "frame_0000.png"],
        "depth_paths": [tmp_path / "frame_0000.npy"],
        "conf_paths": [tmp_path / "conf_0.npy"],
        "intrinsics": np.eye(3, dtype=np.float32)[None],
        "num_frames": 1,
    }
    poses = np.eye(4, dtype=np.float32)[None]

    points, colors = viewer.frame_cloud(
        run, poses, 0, conf_quantile=0.75, pixel_stride=1
    )

    np.testing.assert_allclose(points, [[1, 1, 1]])
    np.testing.assert_array_equal(colors, [[40, 0, 0]])


def test_frame_cloud_applies_world_similarity(tmp_path):
    viewer = _load_module()
    rgb = np.full((2, 2, 3), 50, dtype=np.uint8)
    viewer.iio.imwrite(tmp_path / "frame_0000.png", rgb)
    np.save(tmp_path / "frame_0000.npy", np.ones((2, 2), dtype=np.float32))
    np.save(tmp_path / "conf_0.npy", np.ones((2, 2), dtype=np.float32))
    run = {
        "rgb_paths": [tmp_path / "frame_0000.png"],
        "depth_paths": [tmp_path / "frame_0000.npy"],
        "conf_paths": [tmp_path / "conf_0.npy"],
        "intrinsics": np.eye(3, dtype=np.float32)[None],
        "num_frames": 1,
    }
    poses = np.eye(4, dtype=np.float32)[None]
    similarity = (
        np.eye(3, dtype=np.float32),
        np.array([1, 2, 3], dtype=np.float32),
        2.0,
    )

    points, _ = viewer.frame_cloud(
        run,
        poses,
        0,
        conf_quantile=0.0,
        pixel_stride=2,
        similarity=similarity,
    )

    np.testing.assert_allclose(points, [[1, 2, 5]])


def test_overview_camera_centers_bounds():
    viewer = _load_module()
    spec = viewer.overview_camera(np.array([[-2, -1, -4], [2, 3, 0]], dtype=np.float32))
    np.testing.assert_allclose(spec["look_at"], [0, 1, -2])
    assert np.linalg.norm(spec["position"] - spec["look_at"]) > 4


def test_overview_camera_keeps_complete_trajectory_endpoints():
    viewer = _load_module()
    points = np.array([[0, 0, 0], [1, 0, 0], [100, 0, 0]], dtype=np.float32)

    spec = viewer.overview_camera(points)

    np.testing.assert_allclose(spec["look_at"], [50, 0, 0])
    assert np.linalg.norm(spec["position"] - spec["look_at"]) <= 100
    assert spec["fov"] >= 1.7


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        ("GT vs Baseline", (True, True, False)),
        ("GT vs Geometry", (True, False, True)),
        ("Baseline vs Geometry", (False, True, True)),
    ],
)
def test_comparison_visibility(preset, expected):
    viewer = _load_module()
    assert viewer.comparison_visibility(preset) == expected


def test_register_comparison_callback_uses_button_click_event():
    viewer = _load_module()

    class ButtonGroup:
        callback = None

        def on_click(self, callback):
            self.callback = callback
            return callback

    button_group = ButtonGroup()
    callback = lambda _: None

    registered = viewer.register_comparison_callback(button_group, callback)

    assert registered is callback
    assert button_group.callback is callback


def test_playback_controller_advances_wraps_and_uses_manual_frame():
    viewer = _load_module()
    state = {"frame": 1, "fps": 2.0}
    controller = viewer.PlaybackController(
        frame_count=3,
        get_frame=lambda: state["frame"],
        set_frame=lambda value: state.__setitem__("frame", value),
        get_fps=lambda: state["fps"],
    )
    try:
        assert controller.advance_once() == 2
        assert state["frame"] == 2
        assert controller.advance_once() == 0
        state["frame"] = 1
        assert controller.advance_once() == 2
    finally:
        controller.stop()


def test_playback_controller_play_pause_are_idempotent():
    viewer = _load_module()
    state = {"frame": 0}
    controller = viewer.PlaybackController(
        frame_count=2,
        get_frame=lambda: state["frame"],
        set_frame=lambda value: state.__setitem__("frame", value),
        get_fps=lambda: 0.5,
    )
    try:
        worker = controller._worker
        controller.play()
        controller.play()
        assert controller.is_playing
        assert controller._worker is worker
        controller.pause()
        controller.pause()
        assert not controller.is_playing
        assert state["frame"] == 0
    finally:
        controller.stop()


def test_playback_controller_worker_advances_while_playing():
    viewer = _load_module()
    advanced = viewer.threading.Event()
    state = {"frame": 0}

    def set_frame(value):
        state["frame"] = value
        advanced.set()

    controller = viewer.PlaybackController(
        frame_count=3,
        get_frame=lambda: state["frame"],
        set_frame=set_frame,
        get_fps=lambda: 10.0,
    )
    try:
        controller.play()
        assert advanced.wait(timeout=1.0)
        assert state["frame"] == 1
    finally:
        controller.stop()


@pytest.mark.parametrize(("frame_count", "fps"), [(0, 2.0), (2, 0.0)])
def test_playback_controller_rejects_non_positive_values(frame_count, fps):
    viewer = _load_module()
    with pytest.raises(ValueError, match="positive"):
        viewer.PlaybackController(
            frame_count=frame_count,
            get_frame=lambda: 0,
            set_frame=lambda _: None,
            get_fps=lambda: fps,
        )


def test_register_playback_callbacks_connects_play_and_pause():
    viewer = _load_module()

    class Button:
        callback = None

        def on_click(self, callback):
            self.callback = callback
            return callback

    class Controller:
        playing = False

        def play(self):
            self.playing = True

        def pause(self):
            self.playing = False

    play_button = Button()
    pause_button = Button()
    controller = Controller()
    viewer.register_playback_callbacks(play_button, pause_button, controller)

    play_button.callback(None)
    assert controller.playing
    pause_button.callback(None)
    assert not controller.playing


def test_load_run_rejects_mismatched_counts(tmp_path):
    viewer = _load_module()
    np.savetxt(tmp_path / "pred_traj.txt", [[0, 0, 0, 0, 1, 0, 0, 0]])
    np.savetxt(tmp_path / "pred_intrinsics.txt", np.eye(3).reshape(1, 9))
    (tmp_path / "frame_0000.png").touch()
    np.save(tmp_path / "frame_0000.npy", np.ones((2, 2)))
    with pytest.raises(ValueError, match="counts"):
        viewer.load_run(tmp_path)
