"""Interactive Viser viewer for optional LASER alignment debug traces."""

import argparse
import time
from pathlib import Path

import numpy as np


COLOR_GREY = np.array([150, 150, 150], dtype=np.uint8)
COLOR_RED = np.array([230, 80, 80], dtype=np.uint8)
COLOR_BLUE = np.array([80, 150, 255], dtype=np.uint8)
COLOR_GREEN = np.array([80, 220, 120], dtype=np.uint8)
COLOR_CYAN = np.array([80, 230, 230], dtype=np.uint8)
COLOR_ORANGE = np.array([255, 170, 40], dtype=np.uint8)


def load_debug_pairs(debug_dir):
    debug_dir = Path(debug_dir)
    pairs = []
    for path in sorted(debug_dir.glob("pair_*.npz")):
        pairs.append({"pair_name": path.stem, "path": path, "arrays": np.load(path)})
    if not pairs:
        raise FileNotFoundError(f"No pair_*.npz files found in {debug_dir}")
    return pairs


def flatten_points(points, mask=None):
    points = np.asarray(points, dtype=np.float32)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        return points[mask].reshape(-1, 3)
    return points.reshape(-1, 3)


def solid_colors(num_points, color):
    return np.repeat(np.asarray(color, dtype=np.uint8)[None, :], num_points, axis=0)


def sample_points(points, colors, max_points=200000, seed=0):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if points.shape[0] <= max_points:
        return points, colors

    rng = np.random.default_rng(seed)
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[indices], colors[indices]


def confidence_colors(conf):
    conf = np.asarray(conf, dtype=np.float32).reshape(-1)
    finite = np.isfinite(conf)
    if not np.any(finite):
        norm = np.zeros_like(conf)
    else:
        lo = np.min(conf[finite])
        hi = np.max(conf[finite])
        norm = (conf - lo) / (hi - lo + 1e-8)
        norm = np.where(finite, norm, 0.0)
    colors = np.zeros((conf.shape[0], 3), dtype=np.uint8)
    colors[:, 0] = np.clip(255 * norm, 0, 255).astype(np.uint8)
    colors[:, 1] = np.clip(255 * (1.0 - np.abs(norm - 0.5) * 2.0), 0, 255).astype(np.uint8)
    colors[:, 2] = np.clip(255 * (1.0 - norm), 0, 255).astype(np.uint8)
    return colors


def _offset_points(points, x_offset):
    points = np.asarray(points, dtype=np.float32).copy()
    if points.size:
        points[:, 0] += float(x_offset)
    return points


def _add_cloud(server, name, points, colors, *, point_size, max_points, x_offset=0.0, seed=0):
    points, colors = sample_points(points, colors, max_points=max_points, seed=seed)
    points = _offset_points(points, x_offset)
    if points.shape[0] == 0:
        return None
    return server.scene.add_point_cloud(
        name=name,
        points=points,
        colors=colors,
        point_size=point_size,
        point_shape="rounded",
    )


def _add_pair_layers(server, pair, *, prefix, x_offset, max_points, point_size):
    arrays = pair["arrays"]
    src = flatten_points(arrays["src_points_overlap"])
    tgt_before = flatten_points(arrays["tgt_points_before_overlap"])
    tgt_after_sim3 = flatten_points(arrays["tgt_points_after_sim3_overlap"])
    tgt_after_refine = flatten_points(arrays["tgt_points_after_refine_overlap"])

    _add_cloud(
        server,
        f"/{prefix}/{pair['pair_name']}/src_overlap_grey",
        src,
        solid_colors(len(src), COLOR_GREY),
        point_size=point_size,
        max_points=max_points,
        x_offset=x_offset,
        seed=1,
    )
    _add_cloud(
        server,
        f"/{prefix}/{pair['pair_name']}/target_before_red",
        tgt_before,
        solid_colors(len(tgt_before), COLOR_RED),
        point_size=point_size,
        max_points=max_points,
        x_offset=x_offset,
        seed=2,
    )
    _add_cloud(
        server,
        f"/{prefix}/{pair['pair_name']}/target_after_sim3_blue",
        tgt_after_sim3,
        solid_colors(len(tgt_after_sim3), COLOR_BLUE),
        point_size=point_size,
        max_points=max_points,
        x_offset=x_offset,
        seed=3,
    )
    _add_cloud(
        server,
        f"/{prefix}/{pair['pair_name']}/target_after_refine_green",
        tgt_after_refine,
        solid_colors(len(tgt_after_refine), COLOR_GREEN),
        point_size=point_size,
        max_points=max_points,
        x_offset=x_offset,
        seed=4,
    )

    if "mutual_conf_mask" in arrays:
        mutual_points = flatten_points(arrays["tgt_points_after_refine_overlap"], mask=arrays["mutual_conf_mask"])
        _add_cloud(
            server,
            f"/{prefix}/{pair['pair_name']}/mutual_conf_cyan",
            mutual_points,
            solid_colors(len(mutual_points), COLOR_CYAN),
            point_size=point_size * 1.5,
            max_points=max_points,
            x_offset=x_offset,
            seed=5,
        )

    if "tgt_segment_masks_frame0" in arrays and "tgt_segment_has_scale_frame0" in arrays:
        frame0_points = np.asarray(arrays["tgt_points_after_refine_overlap"])[0]
        masks = np.asarray(arrays["tgt_segment_masks_frame0"], dtype=bool)
        has_scale = np.asarray(arrays["tgt_segment_has_scale_frame0"], dtype=bool)
        scale_mask = np.any(masks[has_scale], axis=0) if np.any(has_scale) else np.zeros(frame0_points.shape[:2], dtype=bool)
        anchor_points = flatten_points(frame0_points, mask=scale_mask)
        _add_cloud(
            server,
            f"/{prefix}/{pair['pair_name']}/scale_anchor_segments_orange",
            anchor_points,
            solid_colors(len(anchor_points), COLOR_ORANGE),
            point_size=point_size * 2.0,
            max_points=max_points,
            x_offset=x_offset,
            seed=6,
        )


def _select_pair(pairs, pair_index):
    if pair_index < 0:
        pair_index = 0
    if pair_index >= len(pairs):
        pair_index = len(pairs) - 1
    return pairs[pair_index]


def parse_args(argv=None):
    parser = argparse.ArgumentParser("LASER alignment debug Viser viewer")
    parser.add_argument("--debug_dir", default=None, help="Single alignment debug trace directory.")
    parser.add_argument("--baseline_debug_dir", default=None, help="Baseline/depth debug trace directory.")
    parser.add_argument("--geometry_debug_dir", default=None, help="Geometry debug trace directory.")
    parser.add_argument("--pair_index", type=int, default=0, help="Pair index to visualize.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max_points", type=int, default=200000)
    parser.add_argument("--point_size", type=float, default=0.003)
    parser.add_argument("--compare_offset", type=float, default=2.5)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import viser

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-z")

    if args.baseline_debug_dir and args.geometry_debug_dir:
        baseline_pair = _select_pair(load_debug_pairs(args.baseline_debug_dir), args.pair_index)
        geometry_pair = _select_pair(load_debug_pairs(args.geometry_debug_dir), args.pair_index)
        _add_pair_layers(
            server,
            baseline_pair,
            prefix="baseline",
            x_offset=-args.compare_offset,
            max_points=args.max_points,
            point_size=args.point_size,
        )
        _add_pair_layers(
            server,
            geometry_pair,
            prefix="geometry",
            x_offset=args.compare_offset,
            max_points=args.max_points,
            point_size=args.point_size,
        )
        server.gui.add_markdown(
            f"Loaded baseline `{baseline_pair['pair_name']}` and geometry `{geometry_pair['pair_name']}`."
        )
    elif args.debug_dir:
        pair = _select_pair(load_debug_pairs(args.debug_dir), args.pair_index)
        _add_pair_layers(
            server,
            pair,
            prefix="run",
            x_offset=0.0,
            max_points=args.max_points,
            point_size=args.point_size,
        )
        server.gui.add_markdown(f"Loaded `{pair['pair_name']}` from `{args.debug_dir}`.")
    else:
        raise ValueError("Provide --debug_dir or both --baseline_debug_dir and --geometry_debug_dir.")

    print(f"Alignment debug viewer running on port {args.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
