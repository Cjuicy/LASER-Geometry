import numpy as np

from inference_engine.utils.depth import merge_regions_geometry
from inference_engine.utils.geometry import build_geometry_info_np


def test_merge_regions_geometry_merges_adjacent_regions_with_similar_depth_and_normals():
    labels = np.array(
        [
            [10, 10, 20, 20],
            [10, 10, 20, 20],
            [10, 10, 20, 20],
            [10, 10, 20, 20],
        ],
        dtype=np.int32,
    )
    depth = np.full(labels.shape, 2.0, dtype=np.float32)
    normals = np.zeros((*labels.shape, 3), dtype=np.float32)
    normals[..., 2] = 1.0
    geometry_info = {
        "normal": normals,
        "valid_mask": np.ones(labels.shape, dtype=bool),
    }

    merged = merge_regions_geometry(
        labels,
        depth,
        geometry_info,
        depth_thresh=0.2,
        normal_thresh_deg=5.0,
    )

    assert merged.shape == labels.shape
    assert np.unique(merged).tolist() == [0]


def test_merge_regions_geometry_keeps_depth_discontinuity_separate_and_relabels():
    labels = np.array(
        [
            [3, 3, 9, 9],
            [3, 3, 9, 9],
            [3, 3, 9, 9],
            [3, 3, 9, 9],
        ],
        dtype=np.int32,
    )
    depth = np.array(
        [
            [1.0, 1.0, 3.0, 3.0],
            [1.0, 1.0, 3.0, 3.0],
            [1.0, 1.0, 3.0, 3.0],
            [1.0, 1.0, 3.0, 3.0],
        ],
        dtype=np.float32,
    )
    normals = np.zeros((*labels.shape, 3), dtype=np.float32)
    normals[..., 2] = 1.0
    geometry_info = {
        "normal": normals,
        "valid_mask": np.ones(labels.shape, dtype=bool),
    }

    merged = merge_regions_geometry(
        labels,
        depth,
        geometry_info,
        depth_thresh=0.2,
        normal_thresh_deg=5.0,
    )

    assert np.unique(merged).tolist() == [0, 1]
    assert np.all(merged[:, :2] == 0)
    assert np.all(merged[:, 2:] == 1)


def test_build_geometry_info_supports_sobel_normals():
    depth = np.full((5, 5), 2.0, dtype=np.float32)
    intrinsic = np.array(
        [
            [100.0, 0.0, 2.0],
            [0.0, 100.0, 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    geometry_info = build_geometry_info_np(
        depth=depth,
        intrinsic=intrinsic,
        normal_method="sobel",
    )

    assert geometry_info["normal"].shape == (5, 5, 3)
    assert np.isfinite(geometry_info["normal"]).all()
    assert geometry_info["normal_edge"].shape == (5, 5)
