"""
中文：
单独读取一帧 depth / confidence / intrinsic / optional RGB，
调用 geometry.py 计算 normal / edge，
可选调用 segment_geometry_felzenszwalb_rag 生成 geometry segment，
最后保存可视化图片。

English:
Independently load one frame of depth / confidence / intrinsic / optional RGB,
call geometry.py to compute normals and edges,
optionally call segment_geometry_felzenszwalb_rag to generate geometry segments,
and save visualization images.
"""

import os
import argparse
import numpy as np
import cv2

from inference_engine.utils.geometry import build_geometry_info_np
from inference_engine.utils.geometry_segmentation import segment_geometry_felzenszwalb_rag


def _normalize_to_uint8(arr):
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.uint8)

    min_val = np.nanmin(arr[finite])
    max_val = np.nanmax(arr[finite])
    norm = (arr - min_val) / (max_val - min_val + 1e-8)
    norm = np.where(finite, norm, 0.0)
    return np.clip(norm * 255.0, 0, 255).astype(np.uint8)


def save_depth_vis(depth, path):
    """
    中文：保存深度图的彩色可视化。
    English: Save color visualization of the depth map.
    """
    depth_u8 = _normalize_to_uint8(depth)
    color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    cv2.imwrite(str(path), color)


def save_conf_vis(conf, path):
    """
    中文：保存置信度图的可视化。
    English: Save visualization of the confidence map.
    """
    conf_u8 = _normalize_to_uint8(conf)
    color = cv2.applyColorMap(conf_u8, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(str(path), color)


def save_normal_vis(normal, path):
    """
    中文：保存 normal map。
    English: Save normal map visualization.
    """
    normal_vis = np.nan_to_num((normal + 1.0) * 0.5, nan=0.0, posinf=1.0, neginf=0.0)
    normal_vis = np.clip(normal_vis * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(normal_vis, cv2.COLOR_RGB2BGR))


def save_edge_vis(edge, path):
    """
    中文：保存边缘图。
    English: Save edge map visualization.
    """
    edge_u8 = _normalize_to_uint8(edge)
    cv2.imwrite(str(path), edge_u8)


def save_segment_vis(labels, path):
    """
    中文：保存 segment label 的随机彩色可视化。
    English: Save random-color visualization of segment labels.
    """
    labels = np.asarray(labels)
    unique_labels, inverse = np.unique(labels.reshape(-1), return_inverse=True)
    rng = np.random.default_rng(0)
    colors = rng.integers(0, 255, size=(len(unique_labels), 3), dtype=np.uint8)
    color_img = colors[inverse].reshape((*labels.shape, 3))
    cv2.imwrite(str(path), cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))


def save_segment_overlay(rgb, labels, path):
    """
    中文：将 segment 边界叠加到 RGB 图像上。
    English: Overlay segment boundaries on the RGB image.
    """
    labels = np.asarray(labels)
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]

    overlay = np.asarray(rgb).copy()
    if overlay.dtype != np.uint8:
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    overlay[boundary] = np.array([255, 0, 0], dtype=np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser()

    # 中文：输入 depth 文件，推荐 npy。
    # English: Input depth file, preferably npy.
    parser.add_argument("--depth", required=True)

    # 中文：输入 confidence 文件，可选。
    # English: Input confidence file, optional.
    parser.add_argument("--conf", default=None)

    # 中文：输入 intrinsic 文件。
    # English: Input intrinsic file.
    parser.add_argument("--intrinsic", required=True)

    # 中文：输入 RGB 图片，可选，用于 overlay。
    # English: Input RGB image, optional, used for overlay.
    parser.add_argument("--rgb", default=None)

    # 中文：输出文件夹。
    # English: Output directory.
    parser.add_argument("--out_dir", required=True)

    # 中文：normal 估计方法。
    # English: Normal estimation method.
    parser.add_argument("--normal_method", default="cross", choices=["cross", "sobel"])

    # 中文：是否额外可视化 geometry segment。
    # English: Whether to additionally visualize geometry segments.
    parser.add_argument("--vis_segment", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 中文：读取 depth / intrinsic。
    # English: Load depth and intrinsic.
    depth = np.load(args.depth)
    intrinsic = np.load(args.intrinsic)

    # 中文：读取 confidence，可选。
    # English: Load confidence, optional.
    conf = None
    if args.conf is not None:
        conf = np.load(args.conf)

    # 中文：读取 RGB，可选。
    # English: Load RGB image, optional.
    rgb = None
    if args.rgb is not None:
        rgb = cv2.imread(args.rgb)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # 中文：构建几何信息。
    # English: Build geometry information.
    geometry_info = build_geometry_info_np(
        depth=depth,
        conf=conf,
        intrinsic=intrinsic,
        points=None,
        normal_method=args.normal_method,
    )

    # 中文：保存 depth 可视化。
    # English: Save depth visualization.
    save_depth_vis(depth, os.path.join(args.out_dir, "depth.png"))

    # 中文：保存 confidence 可视化。
    # English: Save confidence visualization.
    if conf is not None:
        save_conf_vis(conf, os.path.join(args.out_dir, "confidence.png"))

    # 中文：保存 normal 可视化。
    # English: Save normal visualization.
    save_normal_vis(
        geometry_info["normal"],
        os.path.join(args.out_dir, f"normal_{args.normal_method}.png"),
    )

    # 中文：保存 depth edge。
    # English: Save depth edge.
    save_edge_vis(
        geometry_info["depth_edge"],
        os.path.join(args.out_dir, "depth_edge.png"),
    )

    # 中文：保存 normal edge。
    # English: Save normal edge.
    save_edge_vis(
        geometry_info["normal_edge"],
        os.path.join(args.out_dir, "normal_edge.png"),
    )

    # 中文：可选保存 geometry segment。
    # English: Optionally save geometry segment.
    if args.vis_segment:
        labels = segment_geometry_felzenszwalb_rag(
            depth,
            conf_map=conf,
            intrinsic=intrinsic,
            point_map=None,
            normal_method=args.normal_method,
        )

        save_segment_vis(
            labels,
            os.path.join(args.out_dir, "geometry_segment.png"),
        )

        if rgb is not None:
            save_segment_overlay(
                rgb,
                labels,
                os.path.join(args.out_dir, "geometry_segment_overlay.png"),
            )


if __name__ == "__main__":
    main()
