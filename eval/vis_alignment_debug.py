"""Interactive Viser viewer for optional LASER alignment debug traces."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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


def _load_debug_metadata(debug_dir):
    path = Path(debug_dir) / "meta.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing alignment debug metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _pair_sampled_image_range(pair_name, metadata):
    pair_number = int(pair_name.rsplit("_", 1)[1])
    window_size = int(metadata["window_size"])
    overlap = int(metadata["overlap"])
    start = pair_number * (window_size - overlap)
    return start, start + overlap


def _load_pair_rgb(pair_name, sampled_paths, metadata, *, image_loader, expected_shape):
    start, stop = _pair_sampled_image_range(pair_name, metadata)
    paths = sampled_paths[start:stop]
    if len(paths) != stop - start:
        raise ValueError(f"Not enough sampled images for {pair_name}: need [{start}:{stop}].")

    images = image_loader(paths)
    if hasattr(images, "detach"):
        images = images.detach().cpu().numpy()
    images = np.asarray(images, dtype=np.float32)
    rgb = np.clip(images.transpose(0, 2, 3, 1) * 255.0, 0, 255).astype(np.uint8)
    if rgb.shape[:3] != tuple(expected_shape):
        raise ValueError(f"RGB shape {rgb.shape[:3]} does not match point-map shape {expected_shape}.")
    return rgb


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


def _build_rgb_cloud(points, rgb, mask, *, frame_index):
    points = _select_frame(points, frame_index)
    rgb = _select_frame(rgb, frame_index)
    mask = _select_frame(mask, frame_index)
    if points.shape[:-1] != rgb.shape[:-1] or points.shape[:-1] != mask.shape:
        raise ValueError("Point, RGB, and confidence-mask shapes must match.")
    return flatten_points(points, mask=mask), np.asarray(rgb, dtype=np.uint8)[mask].reshape(-1, 3)


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


def _source_camera_spec(pose, *, look_distance=4.0):
    pose = np.asarray(pose, dtype=np.float32)
    position = pose[:3, 3]
    return {
        "position": position,
        "look_at": position + pose[:3, 2] * float(look_distance),
        "up_direction": -pose[:3, 1],
        "fov": np.deg2rad(60.0),
    }


def _overview_camera_spec(points):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] == 0:
        raise ValueError("Cannot initialize an overview camera from an empty point cloud.")
    lower = np.percentile(points, 1.0, axis=0)
    upper = np.percentile(points, 99.0, axis=0)
    center = (lower + upper) * 0.5
    extent = max(float(np.max(upper - lower)), 1.0)
    direction = np.array([1.0, -1.0, -0.8], dtype=np.float32)
    direction /= np.linalg.norm(direction)
    return {
        "position": center + direction * extent * 1.5,
        "look_at": center,
        "up_direction": np.array([0.0, 0.0, -1.0], dtype=np.float32),
        "fov": np.deg2rad(50.0),
    }


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


def _prepare_pair_rgb_cloud(pair, rgb, *, frame_index, coordinate_space):
    arrays = pair["arrays"]
    points = np.asarray(arrays["tgt_points_after_refine_overlap"], dtype=np.float32)
    if coordinate_space == "world":
        points = _points_to_world(points, arrays["tgt_camera_poses_after_overlap"])
    return _build_rgb_cloud(
        points,
        rgb,
        arrays["mutual_conf_mask"],
        frame_index=frame_index,
    )


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


def _aggregate_rgb_clouds(clouds, *, max_points, seed):
    points = np.concatenate([cloud[0] for cloud in clouds], axis=0)
    colors = np.concatenate([cloud[1] for cloud in clouds], axis=0)
    return sample_points(points, colors, max_points=max_points, seed=seed)


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


def _load_rgb_by_pair(pairs, sampled_paths, metadata, *, image_loader):
    rgb_by_pair = {}
    for pair in pairs:
        point_shape = np.asarray(pair["arrays"]["tgt_points_after_refine_overlap"]).shape[:3]
        rgb_by_pair[pair["pair_name"]] = _load_pair_rgb(
            pair["pair_name"],
            sampled_paths,
            metadata,
            image_loader=image_loader,
            expected_shape=point_shape,
        )
    return rgb_by_pair


def _add_rgb_pair_layer(
    server,
    pair,
    rgb,
    *,
    prefix,
    x_offset,
    offset_axis,
    frame_index,
    coordinate_space,
    max_points,
    point_size,
):
    points, colors = _prepare_pair_rgb_cloud(
        pair,
        rgb,
        frame_index=frame_index,
        coordinate_space=coordinate_space,
    )
    return _add_cloud(
        server,
        f"/{prefix}/{pair['pair_name']}/target_after_refine_rgb_mutual",
        points,
        colors,
        point_size=point_size,
        max_points=max_points,
        x_offset=x_offset,
        offset_axis=offset_axis,
        seed=7,
    )


def _add_all_pairs_rgb_layer(
    server,
    pairs,
    rgb_by_pair,
    *,
    prefix,
    x_offset,
    offset_axis,
    coordinate_space,
    max_points,
    point_size,
):
    clouds = [
        _prepare_pair_rgb_cloud(
            pair,
            rgb_by_pair[pair["pair_name"]],
            frame_index=None,
            coordinate_space=coordinate_space,
        )
        for pair in pairs
    ]
    points, colors = _aggregate_rgb_clouds(clouds, max_points=max_points, seed=7)
    handle = _add_cloud(
        server,
        f"/{prefix}/all_pairs/target_after_refine_rgb_mutual",
        points,
        colors,
        point_size=point_size,
        max_points=max_points,
        x_offset=x_offset,
        offset_axis=offset_axis,
        seed=7,
    )
    return handle, _offset_points(points, x_offset, axis=offset_axis)


def _register_camera_spec(server, spec):
    @server.on_client_connect
    def _(client):
        with client.atomic():
            client.camera.position = spec["position"]
            client.camera.look_at = spec["look_at"]
            client.camera.up_direction = spec["up_direction"]
            client.camera.fov = spec["fov"]


def _register_source_camera(server, pair, frame_index):
    poses = np.asarray(pair["arrays"]["tgt_camera_poses_after_overlap"], dtype=np.float32)
    pose = poses[_clamp_index(0 if frame_index is None else frame_index, poses.shape[0])]
    _register_camera_spec(server, _source_camera_spec(pose))


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
        choices=("key", "process", "rgb"),
        default="key",
        help="key shows alignment layers; process restores all stages; rgb shows high-confidence scene colors.",
    )
    parser.add_argument("--image_dir", default=None, help="Original image directory used by RGB mode.")
    parser.add_argument("--sample_interval", type=int, default=1, help="Image sampling interval used for inference.")
    parser.add_argument(
        "--camera_view",
        choices=("default", "source"),
        default="default",
        help="Initial camera view for the Viser client.",
    )
    parser.add_argument("--all_pairs", action="store_true", help="Aggregate all recorded pairs in RGB mode.")
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


def _validate_args(args):
    if args.sample_interval <= 0:
        raise ValueError("--sample_interval must be positive.")
    if args.layer_mode == "rgb" and not args.image_dir:
        raise ValueError("--image_dir is required for RGB mode.")
    if args.all_pairs and args.layer_mode != "rgb":
        raise ValueError("--all_pairs is only supported in RGB mode.")
    if args.all_pairs and args.coordinate_space == "local":
        raise ValueError("--all_pairs requires world coordinates.")
    return args


def main(argv=None):
    args = _validate_args(parse_args(argv))
    import viser

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-z")
    frame_index = None if args.all_frames else args.frame_index
    frame_label = "all overlap frames" if frame_index is None else f"frame `{frame_index}`"

    if args.baseline_debug_dir and args.geometry_debug_dir:
        baseline_pairs = load_debug_pairs(args.baseline_debug_dir)
        geometry_pairs = load_debug_pairs(args.geometry_debug_dir)
        if args.layer_mode == "rgb":
            if args.all_pairs:
                selected_baseline = baseline_pairs
                selected_geometry = geometry_pairs
                selected_frame_index = None
            else:
                selected_baseline = [_select_pair(baseline_pairs, args.pair_index)]
                selected_geometry = [_select_pair(geometry_pairs, args.pair_index)]
                selected_frame_index = frame_index

            baseline_names = [pair["pair_name"] for pair in selected_baseline]
            geometry_names = [pair["pair_name"] for pair in selected_geometry]
            if baseline_names != geometry_names:
                raise ValueError("Baseline and geometry debug traces must contain matching pair names.")

            coordinate_space = _resolve_coordinate_space(
                selected_baseline + selected_geometry,
                args.coordinate_space,
            )
            from utils.image_sequence import list_image_paths
            from utils.load_fn import load_and_preprocess_images

            sampled_paths = list_image_paths(args.image_dir, sample_interval=args.sample_interval)
            metadata = _load_debug_metadata(args.baseline_debug_dir)
            rgb_by_pair = _load_rgb_by_pair(
                selected_baseline,
                sampled_paths,
                metadata,
                image_loader=load_and_preprocess_images,
            )

            if args.all_pairs:
                _, baseline_overview_points = _add_all_pairs_rgb_layer(
                    server,
                    selected_baseline,
                    rgb_by_pair,
                    prefix="baseline",
                    x_offset=-args.compare_offset,
                    offset_axis=args.compare_axis,
                    coordinate_space=coordinate_space,
                    max_points=args.max_points,
                    point_size=args.point_size,
                )
                _, geometry_overview_points = _add_all_pairs_rgb_layer(
                    server,
                    selected_geometry,
                    rgb_by_pair,
                    prefix="geometry",
                    x_offset=args.compare_offset,
                    offset_axis=args.compare_axis,
                    coordinate_space=coordinate_space,
                    max_points=args.max_points,
                    point_size=args.point_size,
                )
                scope_label = f"all {len(selected_baseline)} alignment pairs"
            else:
                baseline_pair = selected_baseline[0]
                geometry_pair = selected_geometry[0]
                pair_rgb = rgb_by_pair[baseline_pair["pair_name"]]
                _add_rgb_pair_layer(
                    server,
                    baseline_pair,
                    pair_rgb,
                    prefix="baseline",
                    x_offset=-args.compare_offset,
                    offset_axis=args.compare_axis,
                    frame_index=selected_frame_index,
                    coordinate_space=coordinate_space,
                    max_points=args.max_points,
                    point_size=args.point_size,
                )
                _add_rgb_pair_layer(
                    server,
                    geometry_pair,
                    pair_rgb,
                    prefix="geometry",
                    x_offset=args.compare_offset,
                    offset_axis=args.compare_axis,
                    frame_index=selected_frame_index,
                    coordinate_space=coordinate_space,
                    max_points=args.max_points,
                    point_size=args.point_size,
                )
                scope_label = f"pair `{baseline_pair['pair_name']}` ({frame_label})"

            if args.camera_view == "source":
                if args.all_pairs:
                    overview_points = np.concatenate(
                        [baseline_overview_points, geometry_overview_points],
                        axis=0,
                    )
                    _register_camera_spec(server, _overview_camera_spec(overview_points))
                else:
                    _register_source_camera(server, selected_geometry[0], selected_frame_index)
            server.gui.add_markdown(
                f"Loaded baseline and geometry {scope_label} in RGB mutual-confidence mode, "
                f"with `{args.compare_axis}` offset, `{coordinate_space}` coordinates, and a "
                f"per-method cap of `{args.max_points}` points."
            )
        else:
            baseline_pair = _select_pair(baseline_pairs, args.pair_index)
            geometry_pair = _select_pair(geometry_pairs, args.pair_index)
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
        pairs = load_debug_pairs(args.debug_dir)
        if args.layer_mode == "rgb":
            selected_pairs = pairs if args.all_pairs else [_select_pair(pairs, args.pair_index)]
            selected_frame_index = None if args.all_pairs else frame_index
            coordinate_space = _resolve_coordinate_space(selected_pairs, args.coordinate_space)
            from utils.image_sequence import list_image_paths
            from utils.load_fn import load_and_preprocess_images

            sampled_paths = list_image_paths(args.image_dir, sample_interval=args.sample_interval)
            metadata = _load_debug_metadata(args.debug_dir)
            rgb_by_pair = _load_rgb_by_pair(
                selected_pairs,
                sampled_paths,
                metadata,
                image_loader=load_and_preprocess_images,
            )
            if args.all_pairs:
                _, overview_points = _add_all_pairs_rgb_layer(
                    server,
                    selected_pairs,
                    rgb_by_pair,
                    prefix="run",
                    x_offset=0.0,
                    offset_axis=args.compare_axis,
                    coordinate_space=coordinate_space,
                    max_points=args.max_points,
                    point_size=args.point_size,
                )
                scope_label = f"all {len(selected_pairs)} alignment pairs"
            else:
                pair = selected_pairs[0]
                _add_rgb_pair_layer(
                    server,
                    pair,
                    rgb_by_pair[pair["pair_name"]],
                    prefix="run",
                    x_offset=0.0,
                    offset_axis=args.compare_axis,
                    frame_index=selected_frame_index,
                    coordinate_space=coordinate_space,
                    max_points=args.max_points,
                    point_size=args.point_size,
                )
                scope_label = f"pair `{pair['pair_name']}` ({frame_label})"
            if args.camera_view == "source":
                if args.all_pairs:
                    _register_camera_spec(server, _overview_camera_spec(overview_points))
                else:
                    _register_source_camera(server, selected_pairs[0], selected_frame_index)
            server.gui.add_markdown(
                f"Loaded {scope_label} in RGB mutual-confidence mode with `{coordinate_space}` "
                f"coordinates and a cap of `{args.max_points}` points."
            )
        else:
            pair = _select_pair(pairs, args.pair_index)
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
