# 1️⃣ 导入必要的包
import numpy as np                              # numpy数组计算
from skimage.segmentation import felzenszwalb   # 初始图像分割算法

from .geometry import build_geometry_info_np    # 构建normal等几何特征


# 2️⃣ 小辅助函数，用来兼容单帧输入和 batch 输入
def _select_batch_item(value, batch_idx, single_frame_ndim):
    # 如果 value is None 或 batch_idx is None，直接返回原值。
    if value is None or batch_idx is None:
        return value

    # 如果传进来的是一批数据，就取当前 batch/frame 对应的那一个
    value = np.asarray(value)
    if value.ndim == single_frame_ndim + 1:
        return value[batch_idx]
    return value

# 3️⃣ 对每个初始 segment 区域计算几何统计量 （初始分割标签图； 深度图； 几何信息字典； 可选置信度图）
def compute_region_geometry_descriptors(labels, depth, geometry_info, conf=None):
    """
    Compute region-level geometry descriptors used by geometry-aware merging.
    """
    descriptors = {}
    normals = geometry_info["normal"]

    # 核心循环
    for label_id in np.unique(labels):
        mask = labels == label_id
        if mask.sum() == 0:
            continue

        area = int(mask.sum())                              # 区域像素数量
        mean_depth = float(np.nanmean(depth[mask]))         # 区域平均深度
        region_normals = normals[mask]                      # 区域平均法线，并做归一化
        mean_normal = np.nanmean(region_normals, axis=0)
        mean_normal = mean_normal / (np.linalg.norm(mean_normal) + 1e-8)
        normal_variance = float(np.nanmean(np.linalg.norm(region_normals - mean_normal, axis=-1)))  # 区域内部法线变化程度

        if conf is not None:
            mean_conf = float(np.nanmean(conf[mask]))
        else:
            mean_conf = 1.0

        ys, xs = np.where(mask)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))  # 区域外接框

        # 最终返回 descriptors :这个 descriptor 后面会用于判断两个相邻区域是否应该合并
        descriptors[label_id] = {
            "area": area,
            "mean_depth": mean_depth,
            "mean_normal": mean_normal,
            "normal_variance": normal_variance,
            "mean_conf": mean_conf,
            "bbox": bbox,
        }

    return descriptors


# 4️⃣ 这个函数判断两个相邻区域是否应该合并（看三个条件）
def should_merge_geometry(
    desc_a,
    desc_b,
    depth_thresh,
    normal_cos_thresh,
    conf_thresh=None,
):
    """
    Decide whether two adjacent regions should merge under geometry-aware rules.
    """

    depth_diff = abs(desc_a["mean_depth"] - desc_b["mean_depth"])
    normal_sim = float(np.dot(desc_a["mean_normal"], desc_b["mean_normal"]))

    # 1️⃣ 平均深度差不能太大
    if depth_diff > depth_thresh:
        return False

    # 2️⃣ 平均法线方向要足够接近
    if normal_sim < normal_cos_thresh:
        return False
    # 3️⃣ 两个区域的平均置信度都不能太低
    if conf_thresh is not None:
        if min(desc_a["mean_conf"], desc_b["mean_conf"]) < conf_thresh:
            return False

    # 一句话：深度接近、法线接近、置信度足够，才允许合并
    return True

# 5️⃣ 几何感知的区域合并函数。它接收初始 labels，然后根据区域级 depth/normal 统计做二次合并。
def merge_regions_geometry(
    labels,
    depth,
    geometry_info,
    conf=None,
    depth_thresh=None,
    normal_thresh_deg=20.0,
    conf_thresh=None,
):
    """
    Merge adjacent regions using depth and normal statistics.
    """
    # 1️⃣ 计算每个区域的 descriptor
    descriptors = compute_region_geometry_descriptors(labels, depth, geometry_info, conf=conf)
    # 2️⃣ 把 normal 角度阈值转成 cosine 阈值
    normal_cos_thresh = np.cos(np.deg2rad(normal_thresh_deg))

    # 3️⃣ 如果没有显式传入 depth_thresh，就根据当前 depth range 自动生成
    if depth_thresh is None:
        depth_range = np.nanmax(depth) - np.nanmin(depth)
        depth_thresh = 0.05 * depth_range

    # 4️⃣ 初始化 union-find（这里用并查集管理哪些 label 最终属于同一个合并区域）
    parent = {int(label_id): int(label_id) for label_id in descriptors}

    def find(label_id):
        label_id = int(label_id)
        while parent[label_id] != label_id:
            parent[label_id] = parent[parent[label_id]]
            label_id = parent[label_id]
        return label_id

    def union(a, b):
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    # 5️⃣ 找相邻区域 pair
    right_pairs = np.stack(
        [labels[:, :-1].reshape(-1), labels[:, 1:].reshape(-1)],
        axis=-1,
    )
    down_pairs = np.stack(
        [labels[:-1, :].reshape(-1), labels[1:, :].reshape(-1)],
        axis=-1,
    )
    adjacent_pairs = np.concatenate([right_pairs, down_pairs], axis=0)
    adjacent_pairs = adjacent_pairs[adjacent_pairs[:, 0] != adjacent_pairs[:, 1]]

    # 6️⃣ 逐个判断相邻区域能不能合并
    if adjacent_pairs.size > 0:
        adjacent_pairs = np.unique(np.sort(adjacent_pairs.astype(np.int64), axis=1), axis=0)

    for label_a, label_b in adjacent_pairs:
        if label_a not in descriptors or label_b not in descriptors:
            continue
        if should_merge_geometry(
            descriptors[label_a],
            descriptors[label_b],
            depth_thresh=depth_thresh,
            normal_cos_thresh=normal_cos_thresh,
            conf_thresh=conf_thresh,
        ):
            union(label_a, label_b)

    # 7️⃣ 根据 union-find 结果重新编号
    roots = np.array([find(label_id) for label_id in labels.reshape(-1)], dtype=np.int64)
    _, relabeled = np.unique(roots, return_inverse=True)
    return relabeled.reshape(labels.shape).astype(np.intp, copy=False)


# 6️⃣ 本文件的主入口，也就是 lsa.py 里 geometry 分支最终调用的函数。
def segment_geometry_felzenszwalb_rag(
    depth_map,                      # 当前帧深度图
    conf_map=None,                  # 当前帧置信度图
    intrinsic=None,                 # 相机内参
    point_map=None,                 # 当前帧3D点图
    top_conf_percentile=None,       # 高置信区域选择阈值
    depth_merge_thresh=0.1,         # 深度合并阈值比例
    normal_thresh_deg=20.0,         # 法线合并角度阈值
    seg_scale=200,                  # Felzenszwalb初始分割参数
    seg_sigma=1.0,
    seg_min_size=300,
    normal_method="cross",          # 法线估计方法，比如cross，sobel
    batch_idx=None,                 # 兼容 batch输入
):
    """
    Generate geometry-aware segmentation labels from depth, normals, and confidence.
    """
    # 1️⃣ 取当前 batch/frame 的输入
    conf_map = _select_batch_item(conf_map, batch_idx, depth_map.ndim)
    point_map = _select_batch_item(point_map, batch_idx, depth_map.ndim + 1)
    intrinsic = _select_batch_item(intrinsic, batch_idx, 2)

    # 2️⃣ 构建几何信息（这里会得到 normal 等几何量，与depth segmentation 最大的区别）
    geometry_info = build_geometry_info_np(
        depth=depth_map,
        conf=conf_map,
        intrinsic=intrinsic,
        points=point_map,
        normal_method=normal_method,
    )

    # 3️⃣ 构造用于初始分割的几何特征图（Felzenszwalb 不再只看 depth，而是看 depth + normal 的组合特征）
    depth_norm = depth_map - np.nanmin(depth_map)
    depth_norm = depth_norm / (np.nanmax(depth_norm) + 1e-8)
    normals = geometry_info["normal"]
    geom_img = np.concatenate(
        [
            depth_norm[..., None],
            normals,
        ],
        axis=-1,
    )

    # 4️⃣ Felzenszwalb 初始分割：得到初始 labels
    labels = felzenszwalb(
        geom_img,
        scale=seg_scale,
        sigma=seg_sigma,
        min_size=seg_min_size,
    )

    # 5️⃣ 根据高置信区域计算 depth merge threshold（如果高置信区域存在，就只用高置信区域的 depth range 来计算）
    if conf_map is not None and top_conf_percentile is not None:
        conf_thresh = np.quantile(conf_map, top_conf_percentile)
        high_conf_mask = conf_map >= conf_thresh
        if high_conf_mask.sum() > 0:
            depth_range = np.nanmax(depth_map[high_conf_mask]) - np.nanmin(depth_map[high_conf_mask])
            depth_thresh = depth_merge_thresh * depth_range
        else:
            depth_thresh = None
    else:
        depth_thresh = None

    # 6️⃣ 几何规则二次合并
    return merge_regions_geometry(
        labels,
        depth_map,
        geometry_info,
        conf=conf_map,
        depth_thresh=depth_thresh,
        normal_thresh_deg=normal_thresh_deg,
    )
