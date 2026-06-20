import numpy as np

from eval.quick_vis_geometry import (
    save_conf_vis,
    save_depth_vis,
    save_edge_vis,
    save_normal_vis,
    save_segment_overlay,
    save_segment_vis,
)


def test_quick_vis_geometry_writes_expected_images(tmp_path):
    depth = np.linspace(1.0, 4.0, 16, dtype=np.float32).reshape(4, 4)
    conf = np.linspace(0.1, 0.9, 16, dtype=np.float32).reshape(4, 4)
    normal = np.zeros((4, 4, 3), dtype=np.float32)
    normal[..., 2] = 1.0
    edge = np.zeros((4, 4), dtype=np.float32)
    edge[:, 2:] = 1.0
    labels = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3],
        ],
        dtype=np.int32,
    )
    rgb = np.full((4, 4, 3), 128, dtype=np.uint8)

    save_depth_vis(depth, tmp_path / "depth.png")
    save_conf_vis(conf, tmp_path / "conf.png")
    save_normal_vis(normal, tmp_path / "normal.png")
    save_edge_vis(edge, tmp_path / "edge.png")
    save_segment_vis(labels, tmp_path / "segments.png")
    save_segment_overlay(rgb, labels, tmp_path / "overlay.png")

    for name in ["depth.png", "conf.png", "normal.png", "edge.png", "segments.png", "overlay.png"]:
        path = tmp_path / name
        assert path.exists()
        assert path.stat().st_size > 0


def test_load_traj_supports_kitti_pose_files(tmp_path):
    from eval.kitti_pose import load_kitti_traj

    pose_file = tmp_path / "poses.txt"
    pose_file.write_text(
        "\n".join(
            [
                "1 0 0 0 0 1 0 0 0 0 1 0",
                "1 0 0 1 0 1 0 0 0 0 1 0",
            ]
        )
    )

    traj, timestamps = load_kitti_traj(str(pose_file))

    assert traj.shape == (2, 7)
    assert timestamps.tolist() == [0.0, 1.0]
    np.testing.assert_allclose(traj[0, :3], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(traj[1, :3], [1.0, 0.0, 0.0])
