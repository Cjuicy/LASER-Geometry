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
AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}
WORLD_POSE_KEYS = (
    "src_camera_poses_overlap",
    "tgt_camera_poses_before_overlap",
    "tgt_camera_poses_after_overlap",
)


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


def _clamp_index(index, size):
    return max(0, min(int(index), size - 1))


def _select_frame(values, frame_index):
    values = np.asarray(values)
    if frame_index is None or values.ndim < 3:
        return values
    return values[_clamp_index(frame_index, values.shape[0])]


def _points_to_world(points, poses):
    points = np.asarray(points, dtype=np.float32)
    poses = np.asarray(poses, dtype=np.float32)
    squeeze_frame = points.ndim == 3
    if squeeze_frame:
        points = points[None]
    if poses.ndim == 2:
        poses = poses[None]
    if poses.shape[0] != points.shape[0]:
        raise ValueError(f"Pose count {poses.shape[0]} does not match point frame count {points.shape[0]}.")

    ones = np.ones(points.shape[:-1] + (1,), dtype=np.float32)
    homogeneous = np.concatenate([points, ones], axis=-1)
    world = np.einsum("tij,thwj->thwi", poses, homogeneous)[..., :3]
    return world[0] if squeeze_frame else world


def _has_world_poses(pair):
    arrays = pair["arrays"]
    return all(key in arrays.files for key in WORLD_POSE_KEYS)


def _resolve_coordinate_space(pairs, requested_space):
    if requested_space == "local":
        return "local"
    if all(_has_world_poses(pair) for pair in pairs):
        return "world"
    if requested_space == "world":
        raise ValueError("World coordinate visualization requires a newly generated debug trace with camera poses.")
    return "local"


def _prepare_points(arrays, points_key, pose_key, *, frame_index, coordinate_space):
    points = np.asarray(arrays[points_key], dtype=np.float32)
    if coordinate_space == "world":
        points = _points_to_world(points, arrays[pose_key])
    return _select_frame(points, frame_index)


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


def _offset_points(points, offset, axis="x"):
    points = np.asarray(points, dtype=np.float32).copy()
    if points.size:
        points[:, AXIS_TO_INDEX[axis]] += float(offset)
    return points


def _add_cloud(server, name, points, colors, *, point_size, max_points, x_offset=0.0, offset_axis="x", seed=0):
    points, colors = sample_points(points, colors, max_points=max_points, seed=seed)
    points = _offset_points(points, x_offset, axis=offset_axis)
    if points.shape[0] == 0:
        return None
    return server.scene.add_point_cloud(
        name=name,
        points=points,
        colors=colors,
        point_size=point_size,
        point_shape="rounded",
    )


def _add_pair_layers(
    server,
    pair,
    *,
    prefix,
    x_offset,
    offset_axis,
    frame_index,
    layer_mode,
    coordinate_space,
    max_points,
    point_size,
):
    arrays = pair["arrays"]
    src = flatten_points(
        _prepare_points(
            arrays,
            "src_points_overlap",
            "src_camera_poses_overlap",
            frame_index=frame_index,
            coordinate_space=coordinate_space,
        )
    )
    selected_refine_points = _prepare_points(
        arrays,
        "tgt_points_after_refine_overlap",
        "tgt_camera_poses_after_overlap",
        frame_index=frame_index,
        coordinate_space=coordinate_space,
    )
    tgt_after_refine = flatten_points(selected_refine_points)

    _add_cloud(
        server,
        f"/{prefix}/{pair['pair_name']}/src_overlap_grey",
        src,
        solid_colors(len(src), COLOR_GREY),
        point_size=point_size,
        max_points=max_points,
        x_offset=x_offset,
        offset_axis=offset_axis,
        seed=1,
    )

    if layer_mode == "process":
        tgt_before = flatten_points(
            _prepare_points(
                arrays,
                "tgt_points_before_overlap",
                "tgt_camera_poses_before_overlap",
                frame_index=frame_index,
                coordinate_space=coordinate_space,
            )
        )
        tgt_after_sim3 = flatten_points(
            _prepare_points(
                arrays,
                "tgt_points_after_sim3_overlap",
                "tgt_camera_poses_after_overlap",
                frame_index=frame_index,
                coordinate_space=coordinate_space,
            )
        )
        _add_cloud(
            server,
            f"/{prefix}/{pair['pair_name']}/target_before_red",
            tgt_before,
            solid_colors(len(tgt_before), COLOR_RED),
            point_size=point_size,
            max_points=max_points,
            x_offset=x_offset,
            offset_axis=offset_axis,
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
            offset_axis=offset_axis,
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
        offset_axis=offset_axis,
        seed=4,
    )

    if "mutual_conf_mask" in arrays:
        mutual_points = flatten_points(selected_refine_points, mask=_select_frame(arrays["mutual_conf_mask"], frame_index))
        _add_cloud(
            server,
            f"/{prefix}/{pair['pair_name']}/mutual_conf_cyan",
            mutual_points,
            solid_colors(len(mutual_points), COLOR_CYAN),
            point_size=point_size * 1.5,
            max_points=max_points,
            x_offset=x_offset,
            offset_axis=offset_axis,
            seed=5,
        )

    if "tgt_segment_masks_frame0" in arrays and "tgt_segment_has_scale_frame0" in arrays:
        num_frames = np.asarray(arrays["tgt_points_after_refine_overlap"]).shape[0]
        selected_frame = None if frame_index is None else _clamp_index(frame_index, num_frames)
        if selected_frame in (None, 0):
            frame0_points = np.asarray(arrays["tgt_points_after_refine_overlap"])[0]
            if coordinate_space == "world":
                frame0_points = _points_to_world(frame0_points, np.asarray(arrays["tgt_camera_poses_after_overlap"])[0])
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
                offset_axis=offset_axis,
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
    parser.add_argument("--frame_index", type=int, default=0, help="Overlap frame index to visualize by default.")
    parser.add_argument("--all_frames", action="store_true", help="Show all overlap frames, matching the original debug view.")
    parser.add_argument(
        "--layer_mode",
        choices=("key", "process"),
        default="key",
        help="key shows the clearest alignment layers; process restores before/sim3/refine layers.",
    )
    parser.add_argument(
        "--coordinate_space",
        choices=("auto", "world", "local"),
        default="auto",
        help="auto uses world coordinates when pose traces exist, otherwise falls back to local camera coordinates.",
    )
    parser.add_argument(
        "--compare_axis",
        choices=tuple(AXIS_TO_INDEX),
        default="x",
        help="Axis used to separate baseline and geometry views in compare mode.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import viser

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-z")
    frame_index = None if args.all_frames else args.frame_index
    frame_label = "all overlap frames" if frame_index is None else f"frame `{frame_index}`"

    if args.baseline_debug_dir and args.geometry_debug_dir:
        baseline_pair = _select_pair(load_debug_pairs(args.baseline_debug_dir), args.pair_index)
        geometry_pair = _select_pair(load_debug_pairs(args.geometry_debug_dir), args.pair_index)
        coordinate_space = _resolve_coordinate_space([baseline_pair, geometry_pair], args.coordinate_space)
        _add_pair_layers(
            server,
            baseline_pair,
            prefix="baseline",
            x_offset=-args.compare_offset,
            offset_axis=args.compare_axis,
            frame_index=frame_index,
            layer_mode=args.layer_mode,
            coordinate_space=coordinate_space,
            max_points=args.max_points,
            point_size=args.point_size,
        )
        _add_pair_layers(
            server,
            geometry_pair,
            prefix="geometry",
            x_offset=args.compare_offset,
            offset_axis=args.compare_axis,
            frame_index=frame_index,
            layer_mode=args.layer_mode,
            coordinate_space=coordinate_space,
            max_points=args.max_points,
            point_size=args.point_size,
        )
        server.gui.add_markdown(
            f"Loaded baseline `{baseline_pair['pair_name']}` and geometry `{geometry_pair['pair_name']}` "
            f"with `{args.compare_axis}` offset, showing {frame_label} in `{args.layer_mode}` mode "
            f"and `{coordinate_space}` coordinates."
        )
    elif args.debug_dir:
        pair = _select_pair(load_debug_pairs(args.debug_dir), args.pair_index)
        coordinate_space = _resolve_coordinate_space([pair], args.coordinate_space)
        _add_pair_layers(
            server,
            pair,
            prefix="run",
            x_offset=0.0,
            offset_axis=args.compare_axis,
            frame_index=frame_index,
            layer_mode=args.layer_mode,
            coordinate_space=coordinate_space,
            max_points=args.max_points,
            point_size=args.point_size,
        )
        server.gui.add_markdown(
            f"Loaded `{pair['pair_name']}` from `{args.debug_dir}`, showing {frame_label} "
            f"in `{args.layer_mode}` mode and `{coordinate_space}` coordinates."
        )
    else:
        raise ValueError("Provide --debug_dir or both --baseline_debug_dir and --geometry_debug_dir.")

    print(f"Alignment debug viewer running on port {args.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
