import importlib.util
from pathlib import Path

import numpy as np


def _load_viewer_module():
    script_path = Path(__file__).resolve().parents[1] / "eval" / "vis_alignment_debug.py"
    spec = importlib.util.spec_from_file_location("vis_alignment_debug", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
