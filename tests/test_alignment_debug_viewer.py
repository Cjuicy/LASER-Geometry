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
