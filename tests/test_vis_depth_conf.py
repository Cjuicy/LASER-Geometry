import importlib.util
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
