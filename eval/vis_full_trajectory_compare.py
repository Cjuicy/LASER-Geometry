"""Compare complete LASER baseline and geometry trajectories in Viser.

This viewer is deliberately standalone: it reads the files already written by
``eval.save_func.save_for_viser`` and never imports or changes the inference
pipeline.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from scipy.spatial.transform import Rotation


_TRAILING_NUMBER = re.compile(r"(\d+)(?=\D*$)")
_BASELINE_COLOR = (65, 145, 255)
_GEOMETRY_COLOR = (60, 220, 125)


def numeric_paths(directory: Path | str, pattern: str) -> list[Path]:
    """Return matching files ordered by their final numeric component."""

    def key(path: Path) -> int:
        match = _TRAILING_NUMBER.search(path.stem)
        if match is None:
            raise ValueError(f"No numeric frame index in {path.name}")
        return int(match.group(1))

    return sorted(Path(directory).glob(pattern), key=key)


def load_tum_poses(path: Path | str) -> np.ndarray:
    """Load timestamp/translation/quaternion-wxyz rows as camera-to-world poses."""
    rows = np.atleast_2d(np.loadtxt(path, dtype=np.float32))
    if rows.shape[1] != 8:
        raise ValueError(f"Expected 8 columns in {path}, got {rows.shape[1]}")

    quat_wxyz = rows[:, 4:8]
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], len(rows), axis=0)
    poses[:, :3, :3] = Rotation.from_quat(quat_xyzw).as_matrix().astype(np.float32)
    poses[:, :3, 3] = rows[:, 1:4]
    return poses


def load_run(directory: Path | str) -> dict[str, object]:
    """Load and strictly validate one standard Viser output directory."""
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {directory}")

    rgb_paths = numeric_paths(directory, "frame_*.png")
    depth_paths = numeric_paths(directory, "frame_*.npy")
    conf_paths = numeric_paths(directory, "conf_*.npy")
    poses = load_tum_poses(directory / "pred_traj.txt")
    intrinsics = np.asarray(
        np.loadtxt(directory / "pred_intrinsics.txt", dtype=np.float32)
    ).reshape(-1, 3, 3)

    counts = {
        "rgb": len(rgb_paths),
        "depth": len(depth_paths),
        "confidence": len(conf_paths),
        "intrinsics": len(intrinsics),
        "poses": len(poses),
    }
    if not counts["rgb"] or len(set(counts.values())) != 1:
        raise ValueError(f"Mismatched or empty frame counts in {directory}: {counts}")

    return {
        "directory": directory,
        "rgb_paths": rgb_paths,
        "depth_paths": depth_paths,
        "conf_paths": conf_paths,
        "intrinsics": intrinsics,
        "poses": poses,
        "num_frames": counts["rgb"],
    }


def align_geometry_to_baseline(
    baseline_poses: np.ndarray, geometry_poses: np.ndarray
) -> np.ndarray:
    """Put geometry in the baseline frame using one rigid transform at frame 0."""
    baseline_poses = np.asarray(baseline_poses, dtype=np.float32)
    geometry_poses = np.asarray(geometry_poses, dtype=np.float32)
    if not len(baseline_poses) or not len(geometry_poses):
        raise ValueError("Cannot align empty trajectories")
    alignment = baseline_poses[0] @ np.linalg.inv(geometry_poses[0])
    return (alignment[None] @ geometry_poses).astype(np.float32)


def unproject_depth(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Unproject a depth image into camera-coordinate XYZ values."""
    depth = np.asarray(depth, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    if depth.ndim != 2 or intrinsic.shape != (3, 3):
        raise ValueError("depth must be HxW and intrinsic must be 3x3")

    height, width = depth.shape
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    x = (u - intrinsic[0, 2]) * depth / intrinsic[0, 0]
    y = (v - intrinsic[1, 2]) * depth / intrinsic[1, 1]
    return np.stack((x, y, depth), axis=-1)


def sample_cloud(
    points: np.ndarray, colors: np.ndarray, max_points: int, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically cap a point cloud."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors).reshape(-1, 3)
    if len(points) != len(colors):
        raise ValueError("Point and color counts differ")
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    if len(points) <= max_points:
        return points, colors
    indices = np.random.default_rng(seed).choice(len(points), max_points, replace=False)
    indices.sort()
    return points[indices], colors[indices]


def confidence_mask(confidence: np.ndarray, quantile: float) -> np.ndarray:
    """Select finite confidence values at or above a per-frame quantile."""
    confidence = np.asarray(confidence, dtype=np.float32)
    finite = np.isfinite(confidence)
    if not finite.any():
        return np.zeros_like(confidence, dtype=bool)
    threshold = np.quantile(confidence[finite], quantile)
    return finite & (confidence >= threshold)


def frame_cloud(
    run: dict[str, object],
    poses: np.ndarray,
    frame_index: int,
    *,
    conf_quantile: float,
    pixel_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one confidence-filtered RGB point cloud in world coordinates."""
    if not 0 <= frame_index < int(run["num_frames"]):
        raise IndexError(f"Frame {frame_index} is outside this run")

    rgb = np.asarray(iio.imread(run["rgb_paths"][frame_index]))
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=-1)
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    depth = np.asarray(np.load(run["depth_paths"][frame_index]), dtype=np.float32)
    confidence = np.asarray(
        np.load(run["conf_paths"][frame_index]), dtype=np.float32
    )
    if rgb.shape[:2] != depth.shape or confidence.shape != depth.shape:
        raise ValueError(
            f"Frame {frame_index} shapes differ: rgb={rgb.shape[:2]}, "
            f"depth={depth.shape}, confidence={confidence.shape}"
        )

    points_camera = unproject_depth(depth, run["intrinsics"][frame_index])
    valid = (
        np.isfinite(depth)
        & (depth > 0)
        & confidence_mask(confidence, conf_quantile)
    )
    stride_slice = np.s_[::pixel_stride, ::pixel_stride]
    points_camera = points_camera[stride_slice]
    rgb = rgb[stride_slice]
    valid = valid[stride_slice]

    pose = np.asarray(poses[frame_index], dtype=np.float32)
    points_world = points_camera @ pose[:3, :3].T + pose[:3, 3]
    return points_world[valid].astype(np.float32), rgb[valid].astype(np.uint8)


def overview_cloud(
    run: dict[str, object],
    poses: np.ndarray,
    *,
    conf_quantile: float,
    pixel_stride: int,
    frame_stride: int,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream all selected frames into one deterministic global reservoir."""
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    rng = np.random.default_rng(seed)
    kept_points = np.empty((0, 3), dtype=np.float32)
    kept_colors = np.empty((0, 3), dtype=np.uint8)
    kept_priorities = np.empty((0,), dtype=np.float64)
    total_frames = int(run["num_frames"])
    for frame_index in range(0, total_frames, frame_stride):
        points, colors = frame_cloud(
            run,
            poses,
            frame_index,
            conf_quantile=conf_quantile,
            pixel_stride=pixel_stride,
        )
        if not len(points):
            continue

        priorities = rng.random(len(points))
        kept_points = np.concatenate((kept_points, points))
        kept_colors = np.concatenate((kept_colors, colors))
        kept_priorities = np.concatenate((kept_priorities, priorities))
        if len(kept_points) > max_points:
            indices = np.argpartition(kept_priorities, max_points - 1)[:max_points]
            kept_points = kept_points[indices]
            kept_colors = kept_colors[indices]
            kept_priorities = kept_priorities[indices]

    if len(kept_priorities):
        order = np.argsort(kept_priorities)
        kept_points = kept_points[order]
        kept_colors = kept_colors[order]
    return kept_points, kept_colors


def overview_camera(points: np.ndarray) -> dict[str, np.ndarray | float]:
    """Choose a stable camera from robust scene bounds."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    finite_points = points[np.isfinite(points).all(axis=1)]
    if not len(finite_points):
        finite_points = np.zeros((1, 3), dtype=np.float32)
    low, high = np.percentile(finite_points, [2, 98], axis=0)
    center = ((low + high) * 0.5).astype(np.float32)
    extent = max(float(np.linalg.norm(high - low)), 1.0)
    direction = np.array([0.8, -0.55, -0.75], dtype=np.float32)
    direction /= np.linalg.norm(direction)
    return {
        "look_at": center,
        "position": center + direction * extent * 1.35,
        "up_direction": np.array([0.0, -1.0, 0.0], dtype=np.float32),
        "fov": 0.9,
    }


def _add_trajectory(server, name: str, poses: np.ndarray, color) -> list[object]:
    centers = np.asarray(poses[:, :3, 3], dtype=np.float32)
    handles: list[object] = []
    if len(centers) > 1:
        handles.append(
            server.scene.add_spline_catmull_rom(
                f"/{name}/trajectory",
                centers,
                line_width=3.0,
                color=color,
                segments=max(100, min(4000, len(centers) * 2)),
            )
        )
    handles.append(
        server.scene.add_point_cloud(
            f"/{name}/start",
            centers[:1],
            np.asarray(color, dtype=np.uint8),
            point_size=0.04,
            point_shape="circle",
        )
    )
    return handles


def build_viewer(args: argparse.Namespace):
    """Load the outputs and construct the Viser scene."""
    import viser

    baseline = load_run(args.baseline_dir)
    geometry = load_run(args.geometry_dir)
    if baseline["num_frames"] != geometry["num_frames"]:
        raise ValueError(
            "Baseline and geometry frame counts differ: "
            f"{baseline['num_frames']} vs {geometry['num_frames']}"
        )

    baseline_poses = baseline["poses"]
    geometry_poses = align_geometry_to_baseline(baseline_poses, geometry["poses"])
    print(
        f"Loading complete trajectories: {baseline['num_frames']} frames per method"
    )
    print("Building baseline overview cloud...")
    baseline_points, baseline_colors = overview_cloud(
        baseline,
        baseline_poses,
        conf_quantile=args.conf_quantile,
        pixel_stride=args.pixel_stride,
        frame_stride=args.frame_stride,
        max_points=args.max_points,
        seed=17,
    )
    print("Building geometry overview cloud...")
    geometry_points, geometry_colors = overview_cloud(
        geometry,
        geometry_poses,
        conf_quantile=args.conf_quantile,
        pixel_stride=args.pixel_stride,
        frame_stride=args.frame_stride,
        max_points=args.max_points,
        seed=29,
    )

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-y")
    baseline_trajectory = _add_trajectory(
        server, "baseline", baseline_poses, _BASELINE_COLOR
    )
    geometry_trajectory = _add_trajectory(
        server, "geometry", geometry_poses, _GEOMETRY_COLOR
    )
    baseline_overview = server.scene.add_point_cloud(
        "/baseline/overview_rgb",
        baseline_points,
        baseline_colors,
        point_size=args.point_size,
        point_shape="rounded",
    )
    geometry_overview = server.scene.add_point_cloud(
        "/geometry/overview_rgb",
        geometry_points,
        geometry_colors,
        point_size=args.point_size,
        point_shape="rounded",
    )

    server.gui.add_markdown(
        "**Full trajectory comparison**  \n"
        f"Frames: `{baseline['num_frames']}`  \n"
        "Blue: LASER baseline; green: LASER-Geometry  \n"
        "Alignment: geometry frame 0 to baseline frame 0  \n"
        f"Confidence quantile: `{args.conf_quantile:.2f}`  \n"
        f"Overview cap: `{args.max_points:,}` points/method"
    )
    with server.gui.add_folder("Display"):
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=int(baseline["num_frames"]) - 1,
            step=1,
            initial_value=0,
        )
        show_baseline = server.gui.add_checkbox("Baseline", initial_value=True)
        show_geometry = server.gui.add_checkbox("Geometry", initial_value=True)
        show_overview = server.gui.add_checkbox("Overview clouds", initial_value=True)
        show_detail = server.gui.add_checkbox("Current-frame detail", initial_value=True)

    detail_handles: dict[str, object | None] = {"baseline": None, "geometry": None}

    def render_detail(frame_index: int) -> None:
        for key in tuple(detail_handles):
            handle = detail_handles[key]
            if handle is not None:
                handle.remove()
                detail_handles[key] = None
        if not show_detail.value:
            return

        baseline_detail = frame_cloud(
            baseline,
            baseline_poses,
            frame_index,
            conf_quantile=args.conf_quantile,
            pixel_stride=args.detail_pixel_stride,
        )
        geometry_detail = frame_cloud(
            geometry,
            geometry_poses,
            frame_index,
            conf_quantile=args.conf_quantile,
            pixel_stride=args.detail_pixel_stride,
        )
        with server.atomic():
            detail_handles["baseline"] = server.scene.add_point_cloud(
                "/baseline/current_frame_rgb",
                baseline_detail[0],
                baseline_detail[1],
                point_size=args.detail_point_size,
                point_shape="rounded",
                visible=show_baseline.value,
            )
            detail_handles["geometry"] = server.scene.add_point_cloud(
                "/geometry/current_frame_rgb",
                geometry_detail[0],
                geometry_detail[1],
                point_size=args.detail_point_size,
                point_shape="rounded",
                visible=show_geometry.value,
            )

    def update_visibility() -> None:
        baseline_overview.visible = show_baseline.value and show_overview.value
        geometry_overview.visible = show_geometry.value and show_overview.value
        for handle in baseline_trajectory:
            handle.visible = show_baseline.value
        for handle in geometry_trajectory:
            handle.visible = show_geometry.value
        if detail_handles["baseline"] is not None:
            detail_handles["baseline"].visible = show_baseline.value
        if detail_handles["geometry"] is not None:
            detail_handles["geometry"].visible = show_geometry.value

    @frame_slider.on_update
    def _(_) -> None:
        render_detail(int(frame_slider.value))

    @show_baseline.on_update
    def _(_) -> None:
        update_visibility()

    @show_geometry.on_update
    def _(_) -> None:
        update_visibility()

    @show_overview.on_update
    def _(_) -> None:
        update_visibility()

    @show_detail.on_update
    def _(_) -> None:
        render_detail(int(frame_slider.value))

    bounds_parts = [baseline_poses[:, :3, 3], geometry_poses[:, :3, 3]]
    if len(baseline_points):
        bounds_parts.append(baseline_points)
    if len(geometry_points):
        bounds_parts.append(geometry_points)
    camera_spec = overview_camera(np.concatenate(bounds_parts))

    @server.on_client_connect
    def _(client) -> None:
        with client.atomic():
            client.camera.position = camera_spec["position"]
            client.camera.look_at = camera_spec["look_at"]
            client.camera.up_direction = camera_spec["up_direction"]
            client.camera.fov = camera_spec["fov"]

    render_detail(0)
    print(
        f"Viewer ready: http://127.0.0.1:{args.port}/ "
        f"(baseline={len(baseline_points):,}, geometry={len(geometry_points):,} points)"
    )
    return server


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare complete LASER baseline and geometry trajectories."
    )
    parser.add_argument("--baseline_dir", type=Path, required=True)
    parser.add_argument("--geometry_dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--max_points", type=int, default=200_000)
    parser.add_argument("--pixel_stride", type=int, default=4)
    parser.add_argument("--detail_pixel_stride", type=int, default=2)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--conf_quantile", type=float, default=0.7)
    parser.add_argument("--point_size", type=float, default=0.006)
    parser.add_argument("--detail_point_size", type=float, default=0.01)
    args = parser.parse_args(argv)
    if not 0.0 <= args.conf_quantile <= 1.0:
        parser.error("--conf_quantile must be within [0, 1]")
    for name in ("max_points", "pixel_stride", "detail_pixel_stride", "frame_stride"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    build_viewer(args)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Viewer stopped.")


if __name__ == "__main__":
    main()
