# Geometry and registration primitives.
# Responsibilities:
# 1. Register cameras or point clouds with SE3/Sim3-style transforms.
# 2. Apply scale/rotation/translation transforms to poses and points.
# 3. Build local geometric features such as normals, edges, and validity masks.
# 中文职责：
# 1. 提供几何/配准基础函数，包括相机位姿或点云的 SE3/Sim3 对齐。
# 2. 提供 scale、rotation、translation 在 pose/point 上的应用工具。
# 3. 提供 normal、edge、valid mask 等局部几何特征构建函数，供 geometry segmentation 使用。
import torch
import numpy as np


def homogenize_points(
        points,
):
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def homogenize_points_np(
        points,
):
    """Convert batched points (xyz) to (xyz1)."""
    return np.concatenate([points, np.ones_like(points[..., :1])], axis=-1)


def register_camera_poses_kabsch(src_cam_poses: np.ndarray, tgt_cam_poses: np.ndarray, scale=1.0):
    assert src_cam_poses.shape == tgt_cam_poses.shape
    src_cam_pos = src_cam_poses[:, :3, 3]
    src_cam_view = src_cam_poses[:, :3, :3] @ np.array([0., 0., -1.])
    src_cam_view_norm = src_cam_view / np.linalg.norm(src_cam_view, axis=-1, keepdims=True)

    tgt_cam_pos = tgt_cam_poses[:, :3, 3]
    tgt_cam_view = tgt_cam_poses[:, :3, :3] @ np.array([0., 0., -1.])
    tgt_cam_view_norm = tgt_cam_view / np.linalg.norm(tgt_cam_view, axis=-1, keepdims=True)

    src_centroid = np.mean(src_cam_pos, axis=0)
    tgt_centroid = np.mean(tgt_cam_pos, axis=0)

    # H = src_centered.T @ tgt_centered
    H = src_cam_view_norm.T @ tgt_cam_view_norm
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Fix improper rotation (reflection)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = tgt_centroid - scale * R @ src_centroid
    ret_se3 = np.eye(4)
    ret_se3[:3, :3] = scale * R
    ret_se3[:3, 3] = t
    return ret_se3


def register_camera_poses_kabsch_pytorch(
        src_cam_poses: torch.Tensor,
        tgt_cam_poses: torch.Tensor,
        scale=1.0
):
    assert src_cam_poses.shape == tgt_cam_poses.shape
    device = src_cam_poses.device

    src_cam_pos = src_cam_poses[:, :3, 3]
    tgt_cam_pos = tgt_cam_poses[:, :3, 3]

    view_direction = torch.tensor([0., 0., -1.], device=device)
    up_direction = torch.tensor([0., 1., 0.], device=device)

    src_cam_view = src_cam_poses[:, :3, :3] @ view_direction
    src_cam_view_norm = src_cam_view / torch.norm(src_cam_view, dim=-1, keepdim=True)
    src_cam_up = src_cam_poses[:, :3, :3] @ up_direction
    src_cam_up_norm = src_cam_up / torch.norm(src_cam_up, dim=-1, keepdim=True)

    tgt_cam_view = tgt_cam_poses[:, :3, :3] @ view_direction
    tgt_cam_view_norm = tgt_cam_view / torch.norm(tgt_cam_view, dim=-1, keepdim=True)
    tgt_cam_up = tgt_cam_poses[:, :3, :3] @ up_direction
    tgt_cam_up_norm = tgt_cam_up / torch.norm(tgt_cam_up, dim=-1, keepdim=True)

    src_corr = torch.vstack([scale * src_cam_pos,
                             scale * src_cam_pos + src_cam_view_norm,
                             scale * src_cam_pos + src_cam_up_norm])
    tgt_corr = torch.vstack([tgt_cam_pos,
                             tgt_cam_pos + tgt_cam_view_norm,
                             tgt_cam_pos + tgt_cam_up_norm])
    src_centroid = scale * src_cam_pos.mean(dim=0)
    tgt_centroid = tgt_cam_pos.mean(dim=0)
    src_corr_centered = src_corr - src_centroid
    tgt_corr_centered = tgt_corr - tgt_centroid

    # H = src_cam_view_norm.T @ tgt_cam_view_norm
    H = src_corr_centered.T @ tgt_corr_centered
    U, _, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T

    # Fix improper rotation (reflection)
    if torch.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = tgt_centroid - R @ src_centroid
    return R, t


def register_point_clouds_kabsch_pytorch(
        src_pcd: torch.Tensor,
        tgt_pcd: torch.Tensor,
        scale=1.0
):
    assert src_pcd.shape == tgt_pcd.shape

    src_corr = scale * src_pcd
    tgt_corr = tgt_pcd
    src_centroid = scale * src_pcd.mean(dim=0)
    tgt_centroid = tgt_pcd.mean(dim=0)
    src_corr_centered = src_corr - src_centroid
    tgt_corr_centered = tgt_corr - tgt_centroid

    # H = src_cam_view_norm.T @ tgt_cam_view_norm
    H = src_corr_centered.T @ tgt_corr_centered
    U, _, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T

    # Fix improper rotation (reflection)
    if torch.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # t = tgt_centroid - scale * R @ src_centroid
    t = tgt_centroid - R @ src_centroid
    return R, t


def apply_scale_with_so3(poses, R, scale):
    """
    Apply scale to camera poses in a rotated basis.

    Args:
        poses: (N, 4, 4) camera-to-world matrices
        R: (3, 3) rotation matrix (SO3)
        scale: float scalar

    Returns:
        poses_scaled: (N, 4, 4) scaled camera-to-world matrices
    """
    device = poses.device
    S = torch.eye(4, device=device)
    S[:3, :3] = scale * torch.eye(3, device=device)

    R_h = torch.eye(4, device=device)
    R_h[:3, :3] = R
    S_rot = R_h.T @ S @ R_h

    poses_scaled = S_rot @ poses
    return poses_scaled


def apply_sim3_to_pose(poses, scale, R, t):
    ret_pose = torch.eye(4, device=poses.device).repeat(poses.shape[0], 1, 1)
    R_c = poses[:, :3, :3]
    t_c = poses[:, :3, 3]

    R_new = R @ R_c
    t_new = scale * (R @ t_c.T).T + t
    ret_pose[:, :3, :3] = R_new
    ret_pose[:, :3, 3] = t_new

    return ret_pose


def closed_form_inverse_sim3(s, R, t):
    R_inv = R.T
    s_inv = 1.0 / s
    t_inv = -s_inv * (R_inv @ t)
    return s_inv, R_inv, t_inv


def accumulate_sim3(S1, S2):
    s1, R1, t1 = S1
    s2, R2, t2 = S2

    s = s1 * s2
    R = R1 @ R2
    t = s1 * R1 @ t2 + t1
    return s, R, t


# ============================================================
# Geometry feature extraction for geometry-aware segmentation
# 用于几何感知分割的几何特征提取函数
# ============================================================

def depth_to_local_points_np(depth, intrinsic, eps=1e-8):
    """
    中文：
    将单张深度图反投影到相机坐标系下的局部 3D 点图。
    这个函数用于适配 VGGT-Ω 这类只输出 depth / camera、不输出 pointmap 的模型。
    这里生成的 local_points 只是中间几何计算结果，不要求模型原生输出 pointmap。

    English:
    Back-project a single depth map into a local camera-coordinate 3D point map.
    This function is used to support models such as VGGT-Ω, which output depth and camera
    but do not directly output pointmaps.
    The generated local_points are only intermediate geometry features, not required model outputs.

    Args:
        depth: np.ndarray, shape [H, W]
            中文：单帧深度图。
            English: Single-frame depth map.

        intrinsic: np.ndarray, shape [3, 3]
            中文：相机内参矩阵。
            English: Camera intrinsic matrix.

    Returns:
        points: np.ndarray, shape [H, W, 3]
            中文：相机坐标系下的局部 3D 点图。
            English: Local 3D point map in the camera coordinate system.
    """
    # 中文：获取图像高度和宽度。
    # English: Get image height and width.
    H, W = depth.shape

    # 中文：构建像素坐标网格。
    # English: Build pixel coordinate grid.
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    # 中文：读取相机内参。
    # English: Read camera intrinsics.
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    # 中文：根据针孔相机模型，将 depth 反投影为 3D 点。
    # English: Back-project depth into 3D points using the pinhole camera model.
    X = (x - cx) / (fx + eps) * depth
    Y = (y - cy) / (fy + eps) * depth
    Z = depth

    # 中文：返回局部 3D 点图。
    # English: Return local 3D point map.
    return np.stack([X, Y, Z], axis=-1)


def compute_normals_cross_np(points, eps=1e-8):
    """
    中文：
    使用局部横向和纵向 3D 点差分，通过叉乘估计表面法向。
    这是第一版最简单、最快的 normal 估计方法。

    English:
    Estimate surface normals using the cross product of local horizontal and vertical
    3D point differences.
    This is the simplest and fastest normal estimation method for the first version.

    Args:
        points: np.ndarray, shape [H, W, 3]
            中文：相机坐标系下的局部 3D 点图。
            English: Local 3D point map in camera coordinates.

    Returns:
        normals: np.ndarray, shape [H, W, 3]
            中文：每个像素对应的单位法向量。
            English: Unit normal vector for each pixel.
    """
    # 中文：计算横向切向量。
    # English: Compute horizontal tangent vectors.
    dx = points[:, 2:, :] - points[:, :-2, :]

    # 中文：计算纵向切向量。
    # English: Compute vertical tangent vectors.
    dy = points[2:, :, :] - points[:-2, :, :]

    # 中文：对齐 dx 和 dy 的有效内部区域。
    # English: Align valid inner regions of dx and dy.
    dx = dx[1:-1, :, :]
    dy = dy[:, 1:-1, :]

    # 中文：通过叉乘得到局部表面法向。
    # English: Compute local surface normals by cross product.
    normals_inner = np.cross(dx, dy)

    # 中文：单位化法向量，避免尺度影响。
    # English: Normalize normals to unit length to remove scale influence.
    normals_inner = normals_inner / (
        np.linalg.norm(normals_inner, axis=-1, keepdims=True) + eps
    )

    # 中文：边界区域暂时填 0。
    # English: Fill boundary pixels with zeros for now.
    normals = np.zeros_like(points)
    normals[1:-1, 1:-1, :] = normals_inner

    return normals


def compute_normals_sobel_np(points, eps=1e-8):
    """
    Estimate normals with Sobel-style spatial derivatives on the 3D point map.

    This is a smoothed variant of the cross-product estimator. It keeps the same
    output contract as compute_normals_cross_np while being less sensitive to
    single-pixel depth noise.
    """
    try:
        import cv2

        points_f = points.astype(np.float32, copy=False)
        dx = np.stack(
            [cv2.Sobel(points_f[..., c], cv2.CV_32F, 1, 0, ksize=3) for c in range(3)],
            axis=-1,
        )
        dy = np.stack(
            [cv2.Sobel(points_f[..., c], cv2.CV_32F, 0, 1, ksize=3) for c in range(3)],
            axis=-1,
        )
    except Exception:
        dx = np.gradient(points, axis=1)
        dy = np.gradient(points, axis=0)

    normals = np.cross(dx, dy)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / (norm + eps)
    invalid = norm[..., 0] <= eps
    if np.any(invalid):
        fallback = compute_normals_cross_np(points, eps=eps)
        normals[invalid] = fallback[invalid]
    return normals


def compute_normals_pca_np(points, depth=None, conf=None, window_size=5, eps=1e-8):
    """
    中文：
    使用 PCA 局部平面拟合估计法向。
    对每个像素周围 kxk 邻域的 3D points 做 PCA，
    最小特征值对应的特征向量就是局部平面的法向。

    第一阶段可以先预留，不一定马上实现。
    它通常比 cross normal 更稳定，但计算更慢，也需要处理边缘处前景/背景混合问题。

    English:
    Estimate normals using local PCA plane fitting.
    For each pixel, we collect 3D points in a kxk neighborhood and run PCA.
    The eigenvector corresponding to the smallest eigenvalue is used as the local surface normal.

    This can be reserved for later stages.
    It is usually more stable than cross-product normals, but slower and needs to handle
    foreground-background mixing near object boundaries.

    Args:
        points: np.ndarray, shape [H, W, 3]
            中文：局部 3D 点图。
            English: Local 3D point map.

        depth: Optional[np.ndarray], shape [H, W]
            中文：可选深度图，用于过滤深度突变邻域。
            English: Optional depth map for filtering depth-discontinuous neighbors.

        conf: Optional[np.ndarray], shape [H, W]
            中文：可选置信度图，用于过滤低置信邻域点。
            English: Optional confidence map for filtering low-confidence neighbors.

    Returns:
        normals: np.ndarray, shape [H, W, 3]
            中文：PCA 估计得到的法向图。
            English: Normal map estimated by PCA.
    """
    raise NotImplementedError("normal_method='pca' is reserved for a later ablation.")


def compute_depth_edge_np(depth, eps=1e-8):
    """
    中文：
    计算深度图的边缘强度，也就是 depth gradient magnitude。
    它用于判断哪里存在明显的深度断裂，例如物体边界、遮挡边界、前景/背景分界。

    English:
    Compute edge strength of the depth map, i.e., depth gradient magnitude.
    It is used to detect depth discontinuities such as object boundaries,
    occlusion boundaries, and foreground-background transitions.

    Args:
        depth: np.ndarray, shape [H, W]
            中文：输入深度图。
            English: Input depth map.

    Returns:
        depth_edge: np.ndarray, shape [H, W]
            中文：归一化后的深度边缘图。
            English: Normalized depth edge map.
    """
    # 中文：计算 x / y 方向深度梯度。
    # English: Compute depth gradients along x and y directions.
    dy, dx = np.gradient(depth)

    # 中文：计算梯度幅值。
    # English: Compute gradient magnitude.
    edge = np.sqrt(dx ** 2 + dy ** 2)

    # 中文：归一化到 [0, 1]，方便后续可视化和阈值判断。
    # English: Normalize to [0, 1] for visualization and thresholding.
    edge = edge - np.nanmin(edge)
    edge = edge / (np.nanmax(edge) + eps)

    return edge


def compute_normal_edge_np(normals, eps=1e-8):
    """
    中文：
    计算法向变化边缘。
    如果相邻像素的 normal 差异很大，说明这里可能是几何转折、平面边界或曲面变化区域。

    English:
    Compute normal variation edges.
    If neighboring pixels have very different normals, the area may correspond to
    geometric corners, plane boundaries, or surface curvature changes.

    Args:
        normals: np.ndarray, shape [H, W, 3]
            中文：单位法向图。
            English: Unit normal map.

    Returns:
        normal_edge: np.ndarray, shape [H, W]
            中文：归一化后的法向边缘图。
            English: Normalized normal edge map.
    """
    # 中文：初始化法向边缘图。
    # English: Initialize normal edge map.
    H, W, _ = normals.shape
    edge = np.zeros((H, W), dtype=np.float32)

    # 中文：计算横向相邻 normal 的差异，使用 1 - dot(n1, n2)。
    # English: Compute horizontal normal difference using 1 - dot(n1, n2).
    dot_x = np.sum(normals[:, :-1, :] * normals[:, 1:, :], axis=-1)
    edge[:, :-1] += 1.0 - dot_x

    # 中文：计算纵向相邻 normal 的差异。
    # English: Compute vertical normal difference.
    dot_y = np.sum(normals[:-1, :, :] * normals[1:, :, :], axis=-1)
    edge[:-1, :] += 1.0 - dot_y

    # 中文：归一化到 [0, 1]。
    # English: Normalize to [0, 1].
    edge = edge - np.nanmin(edge)
    edge = edge / (np.nanmax(edge) + eps)

    return edge


def build_geometry_info_np(
    depth,
    conf=None,
    intrinsic=None,
    points=None,
    normal_method="cross",
):
    """
    中文：
    构建几何感知分割需要的所有几何信息。
    这是 geometry.py 的主入口函数。

    如果模型提供 pointmap，则直接使用 pointmap。
    如果模型不提供 pointmap，则使用 depth + intrinsic 临时生成 local_points。
    这样可以同时适配 VGGT / Pi3 / VGGT-Ω。

    English:
    Build all geometry information needed by geometry-aware segmentation.
    This is the main entry function of geometry.py.

    If the model provides pointmaps, we directly use them.
    If pointmaps are not available, we generate temporary local_points using depth and intrinsics.
    This makes the method compatible with VGGT, Pi3, and VGGT-Ω.

    Args:
        depth: np.ndarray, shape [H, W]
            中文：单帧深度图。
            English: Single-frame depth map.

        conf: Optional[np.ndarray], shape [H, W]
            中文：可选置信度图。
            English: Optional confidence map.

        intrinsic: Optional[np.ndarray], shape [3, 3]
            中文：可选相机内参。没有 pointmap 时必须提供。
            English: Optional camera intrinsics. Required when pointmap is unavailable.

        points: Optional[np.ndarray], shape [H, W, 3]
            中文：可选局部 3D 点图。
            English: Optional local 3D point map.

        normal_method: str
            中文：normal 估计方法，支持 cross / sobel / pca。
            English: Normal estimation method, supporting cross / sobel / pca.

    Returns:
        geometry_info: dict
            中文：包含 points、normal、depth_edge、normal_edge、conf_edge、valid_mask 等。
            English: Contains points, normals, depth edges, normal edges, confidence edges, and valid masks.
    """
    # 中文：如果没有提供 points，则通过 depth + intrinsic 生成临时局部点图。
    # English: If points are not provided, generate temporary local point map from depth and intrinsics.
    if points is None:
        if intrinsic is None:
            raise ValueError("intrinsic is required when points is None")
        points = depth_to_local_points_np(depth, intrinsic)

    # 中文：根据指定方法计算 normal。
    # English: Compute normals according to the selected method.
    if normal_method == "cross":
        normal = compute_normals_cross_np(points)
    elif normal_method == "sobel":
        normal = compute_normals_sobel_np(points)
    elif normal_method == "pca":
        normal = compute_normals_pca_np(points, depth=depth, conf=conf)
    else:
        raise ValueError(f"Unknown normal_method: {normal_method}")

    # 中文：计算深度边缘。
    # English: Compute depth edge map.
    depth_edge = compute_depth_edge_np(depth)

    # 中文：计算法向边缘。
    # English: Compute normal edge map.
    normal_edge = compute_normal_edge_np(normal)

    # 中文：如果有置信度，则计算置信度边缘；否则置为 None。
    # English: If confidence is available, compute confidence edge; otherwise set it to None.
    conf_edge = None
    if conf is not None:
        conf_edge = compute_depth_edge_np(conf)

    # 中文：构建有效 mask，排除 NaN、inf、非正深度区域。
    # English: Build valid mask by excluding NaN, inf, and non-positive depth regions.
    valid_mask = np.isfinite(depth) & (depth > 0)

    geometry_info = {
        "points": points,
        "normal": normal,
        "depth_edge": depth_edge,
        "normal_edge": normal_edge,
        "conf_edge": conf_edge,
        "valid_mask": valid_mask,
    }

    return geometry_info
