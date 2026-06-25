import importlib.util
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def _load_vis_depth_conf_module():
    script_path = Path(__file__).resolve().parents[1] / "eval" / "vis_depth_conf.py"
    spec = importlib.util.spec_from_file_location("vis_depth_conf", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_visualize_depth_conf_scene_outputs_depth_conf_and_overlays(tmp_path):
    vis_depth_conf = _load_vis_depth_conf_module()
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()

    depth = np.linspace(1.0, 5.0, 20, dtype=np.float32).reshape(4, 5)
    conf = np.linspace(0.1, 0.9, 20, dtype=np.float32).reshape(4, 5)
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    rgb[..., 0] = 255

    np.save(scene_dir / "frame_0003.npy", depth)
    np.save(scene_dir / "conf_3.npy", conf)
    cv2.imwrite(str(scene_dir / "frame_0003.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    out_dir = tmp_path / "out"

    result = vis_depth_conf.visualize_depth_conf(
        scene_dir=scene_dir,
        frame_idx=3,
        out_dir=out_dir,
        conf_quantile=0.7,
        alpha=0.5,
    )

    assert result["out_dir"] == out_dir
    assert (out_dir / "depth.png").is_file()
    assert (out_dir / "confidence.png").is_file()
    assert (out_dir / "depth_conf_overlay.png").is_file()
    assert (out_dir / "rgb_conf_overlay.png").is_file()
    assert (out_dir / "high_conf_mask.png").is_file()
    assert (out_dir / "depth_high_conf_overlay.png").is_file()

    summary = (out_dir / "summary.txt").read_text(encoding="utf-8")
    assert "pearson_corr_depth_conf" in summary
    assert "high_conf_threshold" in summary


def test_visualize_depth_conf_explicit_inputs_do_not_require_rgb(tmp_path):
    vis_depth_conf = _load_vis_depth_conf_module()

    depth_path = tmp_path / "depth.npy"
    conf_path = tmp_path / "conf.npy"
    np.save(depth_path, np.ones((3, 3), dtype=np.float32))
    np.save(conf_path, np.eye(3, dtype=np.float32))

    out_dir = tmp_path / "explicit"
    vis_depth_conf.visualize_depth_conf(
        depth_path=depth_path,
        conf_path=conf_path,
        out_dir=out_dir,
    )

    assert (out_dir / "depth.png").is_file()
    assert (out_dir / "confidence.png").is_file()
    assert (out_dir / "depth_conf_overlay.png").is_file()
    assert not (out_dir / "rgb_conf_overlay.png").exists()


def test_vis_depth_conf_cli_runs_from_script_path(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    depth_path = tmp_path / "depth.npy"
    conf_path = tmp_path / "conf.npy"
    out_dir = tmp_path / "cli"
    np.save(depth_path, np.ones((3, 3), dtype=np.float32))
    np.save(conf_path, np.eye(3, dtype=np.float32))

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "eval" / "vis_depth_conf.py"),
            "--depth",
            str(depth_path),
            "--conf",
            str(conf_path),
            "--out_dir",
            str(out_dir),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (out_dir / "depth_conf_overlay.png").is_file()


def test_visualize_depth_conf_can_compare_depth_and_geometry_segments(tmp_path, monkeypatch):
    vis_depth_conf = _load_vis_depth_conf_module()
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()

    depth = np.array(
        [
            [1.0, 1.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
            [3.0, 3.0, 4.0, 4.0],
            [3.0, 3.0, 4.0, 4.0],
        ],
        dtype=np.float32,
    )
    conf = np.full(depth.shape, 0.9, dtype=np.float32)
    rgb = np.zeros((*depth.shape, 3), dtype=np.uint8)

    np.save(scene_dir / "frame_0001.npy", depth)
    np.save(scene_dir / "conf_1.npy", conf)
    np.savetxt(scene_dir / "pred_intrinsics.txt", np.tile(np.eye(3).reshape(1, 9), (2, 1)))
    cv2.imwrite(str(scene_dir / "frame_0001.png"), rgb)

    depth_labels = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [2, 2, 3, 3],
            [2, 2, 3, 3],
        ],
        dtype=np.int32,
    )
    geometry_labels = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.int32,
    )

    monkeypatch.setattr(
        vis_depth_conf,
        "segment_depth_felzenszwalb_rag",
        lambda *args, **kwargs: depth_labels,
    )
    monkeypatch.setattr(
        vis_depth_conf,
        "segment_geometry_felzenszwalb_rag",
        lambda *args, **kwargs: geometry_labels,
    )

    out_dir = tmp_path / "segments"
    vis_depth_conf.visualize_depth_conf(
        scene_dir=scene_dir,
        frame_idx=1,
        out_dir=out_dir,
        vis_segments=True,
    )

    assert (out_dir / "depth_segment.png").is_file()
    assert (out_dir / "geometry_segment.png").is_file()
    assert (out_dir / "depth_segment_overlay.png").is_file()
    assert (out_dir / "geometry_segment_overlay.png").is_file()
    assert (out_dir / "segment_boundary_compare.png").is_file()
    assert (out_dir / "segment_difference.png").is_file()

    summary = (out_dir / "summary.txt").read_text(encoding="utf-8")
    assert "depth_segment_count: 4" in summary
    assert "geometry_segment_count: 2" in summary
    assert "segment_boundary_iou" in summary
