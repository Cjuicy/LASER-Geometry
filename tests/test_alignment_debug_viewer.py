import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_viewer_module():
    script_path = Path(__file__).resolve().parents[1] / "eval" / "vis_alignment_debug.py"
    spec = importlib.util.spec_from_file_location("vis_alignment_debug", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_viewer_adds_project_root_to_python_path(monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    filtered_path = [
        entry
        for entry in sys.path
        if Path(entry or ".").resolve() != project_root
    ]
    monkeypatch.setattr(sys, "path", filtered_path)

    _load_viewer_module()

    assert str(project_root) in sys.path


def test_load_debug_pairs_reads_npz_files_in_order(tmp_path):
    viewer = _load_viewer_module()
    debug_dir = tmp_path / "scene"
    debug_dir.mkdir()
    np.savez_compressed(debug_dir / "pair_0002.npz", sim3_scale=np.array(2.0))
    np.savez_compressed(debug_dir / "pair_0001.npz", sim3_scale=np.array(1.0))

    pairs = viewer.load_debug_pairs(debug_dir)

    assert [pair["pair_name"] for pair in pairs] == ["pair_0001", "pair_0002"]
    assert pairs[0]["arrays"]["sim3_scale"].item() == 1.0


def test_flatten_points_and_mask_keeps_matching_points():
    viewer = _load_viewer_module()
    points = np.arange(24, dtype=np.float32).reshape(2, 2, 2, 3)
    mask = np.array(
        [
            [[True, False], [False, True]],
            [[False, True], [False, False]],
        ]
    )

    flattened = viewer.flatten_points(points, mask=mask)

    assert flattened.shape == (3, 3)
    np.testing.assert_array_equal(flattened[0], points[0, 0, 0])
    np.testing.assert_array_equal(flattened[-1], points[1, 0, 1])


def test_sample_points_is_deterministic_and_preserves_color_alignment():
    viewer = _load_viewer_module()
    points = np.arange(30, dtype=np.float32).reshape(10, 3)
    colors = np.arange(30, dtype=np.uint8).reshape(10, 3)

    sampled_points, sampled_colors = viewer.sample_points(points, colors, max_points=4, seed=3)
    sampled_points_again, sampled_colors_again = viewer.sample_points(points, colors, max_points=4, seed=3)

    assert sampled_points.shape == (4, 3)
    np.testing.assert_array_equal(sampled_points, sampled_points_again)
    np.testing.assert_array_equal(sampled_colors, sampled_colors_again)
    for point, color in zip(sampled_points, sampled_colors):
        row = np.where((points == point).all(axis=1))[0][0]
        np.testing.assert_array_equal(color, colors[row])


def test_offset_points_can_use_requested_axis_without_mutating_input():
    viewer = _load_viewer_module()
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    shifted = viewer._offset_points(points, 2.5, axis="y")

    np.testing.assert_array_equal(points, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
    np.testing.assert_array_equal(shifted, np.array([[1.0, 4.5, 3.0], [4.0, 7.5, 6.0]], dtype=np.float32))


def test_parse_args_accepts_compare_axis():
    viewer = _load_viewer_module()

    args = viewer.parse_args(["--debug_dir", "scene", "--compare_axis", "z"])

    assert args.compare_axis == "z"


def test_select_frame_keeps_one_overlap_frame_by_default():
    viewer = _load_viewer_module()
    points = np.arange(2 * 3 * 4 * 3, dtype=np.float32).reshape(2, 3, 4, 3)

    selected = viewer._select_frame(points, frame_index=1)

    assert selected.shape == (3, 4, 3)
    np.testing.assert_array_equal(selected, points[1])


def test_parse_args_defaults_to_single_frame_and_all_frames_can_restore_old_view():
    viewer = _load_viewer_module()

    default_args = viewer.parse_args(["--debug_dir", "scene"])
    all_frame_args = viewer.parse_args(["--debug_dir", "scene", "--all_frames"])

    assert default_args.frame_index == 0
    assert default_args.all_frames is False
    assert all_frame_args.all_frames is True


def test_parse_args_defaults_to_key_layers_and_accepts_process_layers():
    viewer = _load_viewer_module()

    default_args = viewer.parse_args(["--debug_dir", "scene"])
    process_args = viewer.parse_args(["--debug_dir", "scene", "--layer_mode", "process"])

    assert default_args.layer_mode == "key"
    assert process_args.layer_mode == "process"


def test_points_to_world_applies_camera_pose():
    viewer = _load_viewer_module()
    points = np.array([[[[1.0, 2.0, 3.0]]]], dtype=np.float32)
    poses = np.eye(4, dtype=np.float32)[None]
    poses[0, :3, 3] = np.array([10.0, 20.0, 30.0], dtype=np.float32)

    transformed = viewer._points_to_world(points, poses)

    np.testing.assert_array_equal(transformed, np.array([[[[11.0, 22.0, 33.0]]]], dtype=np.float32))


def test_parse_args_defaults_to_auto_coordinate_space_and_accepts_world():
    viewer = _load_viewer_module()

    default_args = viewer.parse_args(["--debug_dir", "scene"])
    world_args = viewer.parse_args(["--debug_dir", "scene", "--coordinate_space", "world"])

    assert default_args.coordinate_space == "auto"
    assert world_args.coordinate_space == "world"


def test_pair_sampled_image_range_uses_window_stride():
    viewer = _load_viewer_module()

    start, stop = viewer._pair_sampled_image_range(
        "pair_0003",
        {"window_size": 30, "overlap": 10},
    )

    assert (start, stop) == (60, 70)


def test_load_pair_rgb_uses_matching_sampled_images():
    viewer = _load_viewer_module()
    seen = {}

    def fake_loader(paths):
        seen["paths"] = paths
        return np.ones((2, 3, 4, 5), dtype=np.float32)

    rgb = viewer._load_pair_rgb(
        "pair_0001",
        [f"frame_{index}" for index in range(8)],
        {"window_size": 4, "overlap": 2},
        image_loader=fake_loader,
        expected_shape=(2, 4, 5),
    )

    assert seen["paths"] == ["frame_2", "frame_3"]
    assert rgb.shape == (2, 4, 5, 3)
    assert rgb.dtype == np.uint8


def test_build_rgb_cloud_filters_points_and_colors_with_same_mask():
    viewer = _load_viewer_module()
    points = np.arange(24, dtype=np.float32).reshape(2, 2, 2, 3)
    rgb = np.arange(24, dtype=np.uint8).reshape(2, 2, 2, 3)
    mask = np.array(
        [
            [[True, False], [False, True]],
            [[False, True], [False, False]],
        ]
    )

    cloud_points, cloud_colors = viewer._build_rgb_cloud(
        points,
        rgb,
        mask,
        frame_index=None,
    )

    np.testing.assert_array_equal(cloud_points, points[mask])
    np.testing.assert_array_equal(cloud_colors, rgb[mask])


def test_parse_args_accepts_rgb_all_pairs_options():
    viewer = _load_viewer_module()

    args = viewer.parse_args(
        [
            "--debug_dir",
            "scene",
            "--layer_mode",
            "rgb",
            "--image_dir",
            "data/09/image_2",
            "--sample_interval",
            "10",
            "--camera_view",
            "source",
            "--all_pairs",
        ]
    )
    viewer._validate_args(args)

    assert args.layer_mode == "rgb"
    assert args.all_pairs is True
    assert args.sample_interval == 10


def test_rgb_mode_requires_image_dir():
    viewer = _load_viewer_module()
    args = viewer.parse_args(["--debug_dir", "scene", "--layer_mode", "rgb"])

    with pytest.raises(ValueError, match="--image_dir"):
        viewer._validate_args(args)


def test_aggregate_rgb_clouds_applies_one_global_point_cap():
    viewer = _load_viewer_module()
    clouds = [
        (
            np.arange(18, dtype=np.float32).reshape(6, 3),
            np.full((6, 3), 10, dtype=np.uint8),
        ),
        (
            np.arange(18, 36, dtype=np.float32).reshape(6, 3),
            np.full((6, 3), 20, dtype=np.uint8),
        ),
    ]

    points, colors = viewer._aggregate_rgb_clouds(clouds, max_points=5, seed=7)

    assert points.shape == (5, 3)
    assert colors.shape == (5, 3)
    assert set(np.unique(colors[:, 0])).issubset({10, 20})


def test_source_camera_spec_uses_camera_forward_and_up_axes():
    viewer = _load_viewer_module()
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = [1.0, 2.0, 3.0]

    spec = viewer._source_camera_spec(pose, look_distance=4.0)

    np.testing.assert_allclose(spec["position"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(spec["look_at"], [1.0, 2.0, 7.0])
    np.testing.assert_allclose(spec["up_direction"], [0.0, -1.0, 0.0])


def test_prepare_pair_rgb_cloud_uses_refined_target_and_mutual_mask():
    viewer = _load_viewer_module()
    points = np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3)
    rgb = np.arange(12, dtype=np.uint8).reshape(1, 2, 2, 3)
    mask = np.array([[[True, False], [False, True]]])
    pair = {
        "pair_name": "pair_0001",
        "arrays": {
            "tgt_points_after_refine_overlap": points,
            "mutual_conf_mask": mask,
        },
    }

    cloud_points, cloud_colors = viewer._prepare_pair_rgb_cloud(
        pair,
        rgb,
        frame_index=None,
        coordinate_space="local",
    )

    np.testing.assert_array_equal(cloud_points, points[mask])
    np.testing.assert_array_equal(cloud_colors, rgb[mask])


def test_overview_camera_spec_centers_scene_bounds():
    viewer = _load_viewer_module()
    points = np.array(
        [
            [-2.0, -1.0, -4.0],
            [2.0, 3.0, 0.0],
        ],
        dtype=np.float32,
    )

    spec = viewer._overview_camera_spec(points)

    np.testing.assert_allclose(spec["look_at"], [0.0, 1.0, -2.0])
    np.testing.assert_allclose(spec["up_direction"], [0.0, 0.0, -1.0])
    assert np.linalg.norm(spec["position"] - spec["look_at"]) > 4.0
