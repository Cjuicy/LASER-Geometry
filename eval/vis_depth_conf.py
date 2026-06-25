"""
Visualize the relationship between saved depth maps and confidence maps.

The script reads the files written by eval.save_func.save_for_viser:
  - frame_0000.npy  depth
  - conf_0.npy      depth confidence
  - frame_0000.png  optional RGB

It writes standalone images that make it easy to inspect where confidence is
high and how those regions align with the depth structure.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def _as_path(path):
    if path is None:
        return None
    return Path(path)


def _load_npy_2d(path, name):
    arr = np.load(path)
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.squeeze(arr, axis=-1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D map, got shape {arr.shape} from {path}")
    return arr.astype(np.float32, copy=False)


def _finite_values(arr):
    finite = np.isfinite(arr)
    return arr[finite], finite


def _normalize_to_uint8(arr, percentiles=(2.0, 98.0)):
    arr = np.asarray(arr, dtype=np.float32)
    values, finite = _finite_values(arr)
    if values.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)

    lo, hi = np.percentile(values, percentiles)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(values))
        hi = float(np.max(values))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)

    norm = (arr - lo) / (hi - lo + 1e-8)
    norm = np.where(finite, norm, 0.0)
    return np.clip(norm * 255.0, 0, 255).astype(np.uint8)


def _colorize(arr, colormap, percentiles=(2.0, 98.0)):
    return cv2.applyColorMap(_normalize_to_uint8(arr, percentiles), colormap)


def _load_rgb_bgr(path, target_shape):
    if path is None or not path.exists():
        return None

    rgb_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise ValueError(f"Failed to read RGB image: {path}")

    height, width = target_shape
    if rgb_bgr.shape[:2] != (height, width):
        rgb_bgr = cv2.resize(rgb_bgr, (width, height), interpolation=cv2.INTER_AREA)
    return rgb_bgr


def _blend(base_bgr, heat_bgr, alpha):
    return cv2.addWeighted(base_bgr, 1.0 - alpha, heat_bgr, alpha, 0.0)


def _high_conf_mask(conf, quantile):
    values, finite = _finite_values(conf)
    if values.size == 0:
        return np.zeros(conf.shape, dtype=bool), float("nan")

    threshold = float(np.quantile(values, quantile))
    return finite & (conf >= threshold), threshold


def _save_high_conf_mask(mask, path):
    mask_u8 = (mask.astype(np.uint8) * 255)
    cv2.imwrite(str(path), mask_u8)


def _save_depth_high_conf_overlay(depth_color, mask, path):
    overlay = (depth_color.astype(np.float32) * 0.35).astype(np.uint8)
    overlay[mask] = depth_color[mask]

    boundary = cv2.Canny((mask.astype(np.uint8) * 255), 50, 150) > 0
    overlay[boundary] = np.array([255, 255, 255], dtype=np.uint8)
    cv2.imwrite(str(path), overlay)


def _pearson_corr(depth, conf):
    valid = np.isfinite(depth) & np.isfinite(conf)
    if np.count_nonzero(valid) < 2:
        return float("nan")

    depth_values = depth[valid].astype(np.float64)
    conf_values = conf[valid].astype(np.float64)
    if np.std(depth_values) <= 1e-12 or np.std(conf_values) <= 1e-12:
        return float("nan")

    return float(np.corrcoef(depth_values, conf_values)[0, 1])


def _stats_lines(depth, conf, high_mask, high_threshold):
    depth_values, _ = _finite_values(depth)
    conf_values, _ = _finite_values(conf)
    high_depth = depth[np.isfinite(depth) & high_mask]

    lines = [
        f"depth_shape: {depth.shape}",
        f"confidence_shape: {conf.shape}",
        f"depth_valid_pixels: {depth_values.size}",
        f"confidence_valid_pixels: {conf_values.size}",
        f"high_conf_threshold: {high_threshold:.8g}",
        f"high_conf_pixels: {int(np.count_nonzero(high_mask))}",
        f"pearson_corr_depth_conf: {_pearson_corr(depth, conf):.8g}",
    ]

    if depth_values.size:
        lines.extend(
            [
                f"depth_min: {np.min(depth_values):.8g}",
                f"depth_p02: {np.percentile(depth_values, 2):.8g}",
                f"depth_p50: {np.percentile(depth_values, 50):.8g}",
                f"depth_p98: {np.percentile(depth_values, 98):.8g}",
                f"depth_max: {np.max(depth_values):.8g}",
            ]
        )

    if conf_values.size:
        lines.extend(
            [
                f"confidence_min: {np.min(conf_values):.8g}",
                f"confidence_p50: {np.percentile(conf_values, 50):.8g}",
                f"confidence_p70: {np.percentile(conf_values, 70):.8g}",
                f"confidence_p90: {np.percentile(conf_values, 90):.8g}",
                f"confidence_max: {np.max(conf_values):.8g}",
            ]
        )

    if high_depth.size:
        lines.extend(
            [
                f"high_conf_depth_min: {np.min(high_depth):.8g}",
                f"high_conf_depth_p50: {np.percentile(high_depth, 50):.8g}",
                f"high_conf_depth_max: {np.max(high_depth):.8g}",
            ]
        )

    return lines


def _resolve_inputs(scene_dir=None, frame_idx=0, depth_path=None, conf_path=None, rgb_path=None):
    scene_dir = _as_path(scene_dir)
    depth_path = _as_path(depth_path)
    conf_path = _as_path(conf_path)
    rgb_path = _as_path(rgb_path)

    if scene_dir is not None:
        if depth_path is None:
            depth_path = scene_dir / f"frame_{frame_idx:04d}.npy"
        if conf_path is None:
            conf_path = scene_dir / f"conf_{frame_idx}.npy"
        if rgb_path is None:
            candidate_rgb = scene_dir / f"frame_{frame_idx:04d}.png"
            rgb_path = candidate_rgb if candidate_rgb.exists() else None

    if depth_path is None or conf_path is None:
        raise ValueError("Provide --scene_dir, or provide both --depth and --conf.")

    return depth_path, conf_path, rgb_path


def _default_out_dir(scene_dir, depth_path, frame_idx):
    if scene_dir is not None:
        return Path("outputs") / "depth_conf_vis" / Path(scene_dir).name / f"frame_{frame_idx:04d}"
    return Path(depth_path).parent / "depth_conf_vis"


def visualize_depth_conf(
    *,
    scene_dir=None,
    frame_idx=0,
    depth_path=None,
    conf_path=None,
    rgb_path=None,
    out_dir=None,
    alpha=0.45,
    conf_quantile=0.7,
    depth_percentiles=(2.0, 98.0),
):
    depth_path, conf_path, rgb_path = _resolve_inputs(
        scene_dir=scene_dir,
        frame_idx=frame_idx,
        depth_path=depth_path,
        conf_path=conf_path,
        rgb_path=rgb_path,
    )
    out_dir = _as_path(out_dir) or _default_out_dir(scene_dir, depth_path, frame_idx)

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1].")
    if not 0.0 <= conf_quantile <= 1.0:
        raise ValueError("conf_quantile must be in [0, 1].")

    depth = _load_npy_2d(depth_path, "depth")
    conf = _load_npy_2d(conf_path, "confidence")
    if depth.shape != conf.shape:
        raise ValueError(f"depth and confidence shapes differ: {depth.shape} vs {conf.shape}")

    out_dir.mkdir(parents=True, exist_ok=True)

    depth_color = _colorize(depth, cv2.COLORMAP_TURBO, depth_percentiles)
    conf_color = _colorize(conf, cv2.COLORMAP_VIRIDIS, (0.0, 100.0))
    high_mask, high_threshold = _high_conf_mask(conf, conf_quantile)

    cv2.imwrite(str(out_dir / "depth.png"), depth_color)
    cv2.imwrite(str(out_dir / "confidence.png"), conf_color)
    cv2.imwrite(str(out_dir / "depth_conf_overlay.png"), _blend(depth_color, conf_color, alpha))
    _save_high_conf_mask(high_mask, out_dir / "high_conf_mask.png")
    _save_depth_high_conf_overlay(depth_color, high_mask, out_dir / "depth_high_conf_overlay.png")

    rgb_bgr = _load_rgb_bgr(rgb_path, depth.shape)
    if rgb_bgr is not None:
        cv2.imwrite(str(out_dir / "rgb_conf_overlay.png"), _blend(rgb_bgr, conf_color, alpha))

    summary_lines = [
        f"depth_path: {depth_path}",
        f"confidence_path: {conf_path}",
        f"rgb_path: {rgb_path if rgb_path is not None else 'None'}",
        f"alpha: {alpha}",
        f"confidence_quantile: {conf_quantile}",
        "",
        *_stats_lines(depth, conf, high_mask, high_threshold),
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "out_dir": out_dir,
        "depth_path": depth_path,
        "conf_path": conf_path,
        "rgb_path": rgb_path,
        "high_conf_threshold": high_threshold,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Visualize saved depth and confidence maps, plus simple overlays."
    )
    parser.add_argument("--scene_dir", default=None, help="Directory under outputs/viser/<scene>.")
    parser.add_argument("--frame_idx", type=int, default=0, help="Frame index to read from scene_dir.")
    parser.add_argument("--depth", dest="depth_path", default=None, help="Explicit depth .npy path.")
    parser.add_argument("--conf", dest="conf_path", default=None, help="Explicit confidence .npy path.")
    parser.add_argument("--rgb", dest="rgb_path", default=None, help="Optional explicit RGB image path.")
    parser.add_argument("--out_dir", default=None, help="Output directory for visualization images.")
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha for confidence heatmap.")
    parser.add_argument(
        "--conf_quantile",
        type=float,
        default=0.7,
        help="Quantile threshold for the high-confidence mask.",
    )
    parser.add_argument(
        "--depth_percentiles",
        type=float,
        nargs=2,
        default=(2.0, 98.0),
        metavar=("LOW", "HIGH"),
        help="Percentile range for depth color normalization.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = visualize_depth_conf(
        scene_dir=args.scene_dir,
        frame_idx=args.frame_idx,
        depth_path=args.depth_path,
        conf_path=args.conf_path,
        rgb_path=args.rgb_path,
        out_dir=args.out_dir,
        alpha=args.alpha,
        conf_quantile=args.conf_quantile,
        depth_percentiles=tuple(args.depth_percentiles),
    )
    print(f"Saved depth-confidence visualization to: {result['out_dir']}")


if __name__ == "__main__":
    main()
