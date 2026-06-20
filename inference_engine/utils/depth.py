import os
import numpy as np
from skimage.segmentation import felzenszwalb
from concurrent.futures import ThreadPoolExecutor, as_completed

from .fast_seg import fast_graph_segmentation
from ._segmentation_cy import merge_regions
from pi3.utils.graph import Vertex


def align_depth_irls(
        src_depth,
        tgt_depth,
        mask=None,
        iters=10,
        eps=1e-8,
        stop_tol=0.05,
        clamp_min=1e-6
):
    if mask is not None:
        src_depth = src_depth[mask]
        tgt_depth = tgt_depth[mask]

    num = np.nanmean(tgt_depth)
    den = np.nanmean(src_depth)
    s_d = np.maximum(num / den, clamp_min)

    for _ in range(iters):
        d_res = s_d * src_depth - tgt_depth
        res = abs(d_res) + eps
        w = 1.0 / res

        num = (w * src_depth * tgt_depth).sum()
        den = (w * src_depth ** 2).sum()
        s_d_new = np.maximum(num / den, clamp_min)
        converged = abs(s_d_new - s_d) < stop_tol
        s_d = s_d_new

        if converged:
            break  # early stopping

    return s_d


def segment_depth_felzenszwalb_rag(
        depth_map,
        depth_merge_thresh,
        conf_map=None,
        top_conf_percentile=None,
        seg_scale=300,
        seg_sigma=1.1,
        seg_min_size=500,
        batch_idx=None
):
    seg_mask = felzenszwalb(depth_map, scale=seg_scale, sigma=seg_sigma, min_size=seg_min_size)
    # depth_img = gray2rgb(depth_map)
    # rag = graph.rag_mean_color(depth_img, seg_mask, mode='distance')
    #
    # seg_mask_merged = graph.cut_threshold(seg_mask, rag, merge_thresh)
    if conf_map is not None and top_conf_percentile is not None:
        conf_map = conf_map[batch_idx]
        conf_thresh = np.quantile(conf_map.reshape(-1), top_conf_percentile, method='nearest')
        conf_depth = depth_map[conf_map >= conf_thresh]
    else:
        conf_depth = depth_map
    merge_thresh = depth_merge_thresh * (np.max(conf_depth) - np.min(conf_depth))

    seg_mask_merged = merge_regions(seg_mask, depth_map, merge_thresh)
    return seg_mask_merged


def segment_depth_graph_fast(
        depth_map,
        depth_merge_thresh,
        conf_map=None,
        top_conf_percentile=None,
        batch_idx=None
):
    if conf_map is not None and top_conf_percentile is not None:
        conf_map = conf_map[batch_idx]
        conf_thresh = np.quantile(conf_map.reshape(-1), top_conf_percentile, method='nearest')
        conf_depth = depth_map[conf_map >= conf_thresh]
    else:
        conf_depth = depth_map
    merge_thresh = depth_merge_thresh * (np.max(conf_depth) - np.min(conf_depth))
    return fast_graph_segmentation(depth_map, merge_thresh)


def pairwise_intersection_ratio(mask1, mask2):
    """
    Highest pairwise intersection ratio for assigning correspondence
    """
    N, H, W = mask1.shape
    M = mask2.shape[0]

    mask1_f = mask1.reshape(N, -1).astype(np.float32)
    mask2_f = mask2.reshape(M, -1).astype(np.float32)
    inter = mask1_f @ mask2_f.T  # pairwise intersection - N, M

    area1 = mask1_f.sum(axis=-1, keepdims=True)  # N, 1
    area2 = mask2_f.sum(axis=-1, keepdims=True).T  # 1, M
    area1 = np.maximum(area1, 1)
    area2 = np.maximum(area2, 1)

    ratios1 = inter / area1
    ratios2 = inter / area2
    min_rel_inter = np.minimum(ratios1, ratios2)
    max_rel_inter = np.maximum(ratios1, ratios2)

    return min_rel_inter, max_rel_inter  # N, M


def pairwise_iou(mask1, mask2):
    N, H, W = mask1.shape
    M = mask2.shape[0]

    mask1_f = mask1.reshape(N, -1).astype(np.float32)
    mask2_f = mask2.reshape(M, -1).astype(np.float32)
    inter = mask1_f @ mask2_f.T  # pairwise intersection - N, M

    area1 = mask1_f.sum(axis=-1, keepdims=True)  # N, 1
    area2 = mask2_f.sum(axis=-1, keepdims=True).T  # 1, M
    union = area1 + area2 - inter

    iou = inter / np.maximum(union, 1)
    return iou


def match_segmentation_seq(labels, iou_thresh=0.4):
    def get_seg_vertices(seg):
        seg_ids = np.unique(seg)
        masks = seg[None, :, :] == seg_ids[:, None, None]
        seg_vertices_ = [Vertex(data=m, default_cache={'iou': [], 'scale': []}) for m in masks]
        return seg_vertices_  # , masks

    # root = get_seg_vertices(labels[0])
    sp_graph = [get_seg_vertices(labels[0])]

    for seg_map in labels[1:]:
        seg_vertices = get_seg_vertices(seg_map)
        connect_bipartite_sp_graphs(sp_graph[-1], seg_vertices, iou_thresh=iou_thresh)
        sp_graph.append(seg_vertices)
        # prev_mask = cur_mask

    # for v in root:
    #     v.cut_edge_threshold(inter_thresh)
    return sp_graph


def connect_bipartite_sp_graphs(graph1, graph2, iou_thresh=0.3):
    masks1 = np.stack([v.data for v in graph1])
    masks2 = np.stack([v.data for v in graph2])

    iou = pairwise_iou(masks1, masks2)
    matchable = iou >= iou_thresh
    graph1_indices, graph2_indices = np.nonzero(matchable)

    for v1, v2 in zip(graph1_indices, graph2_indices):
        graph1[v1].add_edge(graph2[v2], iou[v1, v2])


def _edge_scale_worker(
        src_depth,
        tgt_depth,
        src_vertex
):
    src_mask = src_vertex.data
    for tgt_v, tgt_iou in zip(src_vertex.connectivity, src_vertex.edge_weights):
        tgt_mask = tgt_v.data
        inter_mask = src_mask & tgt_mask
        tgt2src_s = align_depth_irls(tgt_depth, src_depth, inter_mask)
        tgt_v.cache['iou'].append(tgt_iou)
        tgt_v.cache['scale'].append(tgt2src_s)


def assign_overlap_window_depth_scale(
        src_depth_overlap,
        tgt_depth_overlap,
        src_sp_graphs_overlap,
        tgt_sp_graphs_overlap,
        iou_thresh=0.4,
        n_jobs=1
):
    for src_sp_graph, tgt_sp_graph in zip(src_sp_graphs_overlap, tgt_sp_graphs_overlap):
        connect_bipartite_sp_graphs(src_sp_graph, tgt_sp_graph, iou_thresh=iou_thresh)

    for idx, src_graph in enumerate(src_sp_graphs_overlap):
        n_jobs = min(os.cpu_count(), len(src_graph)) if n_jobs is None else n_jobs
        if n_jobs == 1:
            for src_v in src_graph:
                _edge_scale_worker(src_depth_overlap[idx], tgt_depth_overlap[idx], src_v)
        else:
            with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                promises = [
                    ex.submit(_edge_scale_worker, src_depth_overlap[idx], tgt_depth_overlap[idx], src_v)
                    for src_v in src_graph
                ]
                for promise in as_completed(promises):
                    promise.result()

        # for src_v in src_graph:
        #     src_mask = src_v.data
        #     for tgt_v, tgt_iou in zip(src_v.connectivity, src_v.edge_weights):
        #         tgt_mask = tgt_v.data
        #         inter_mask = src_mask & tgt_mask
        #         tgt2src_s = align_depth_irls(tgt_depth_overlap[idx], src_depth_overlap[idx], inter_mask)
        #         tgt_v.cache['iou'].append(tgt_iou)
        #         tgt_v.cache['scale'].append(tgt2src_s)

        # tgt_graph = tgt_sp_graphs_overlap[idx]
        # for v in tgt_graph:
        #     v.propagate_cache(merge_scale_cache)



# 中文：导入几何特征构建函数，用于 geometry-aware segmentation。
# English: Import geometry feature builder for geometry-aware segmentation.
from .geometry import build_geometry_info_np


def compute_region_geometry_descriptors(labels, depth, geometry_info, conf=None):
    """
    中文：
    统计每个 segment 的区域级几何描述子。
    这些描述子用于判断相邻区域是否应该合并。

    English:
    Compute region-level geometry descriptors for each segment.
    These descriptors are used to decide whether neighboring regions should be merged.

    Returns:
        descriptors: dict
            中文：key 是 label_id，value 是该区域的几何统计信息。
            English: key is label_id, value contains geometry statistics of that region.
    """
    descriptors = {}

    # 中文：取出 normal 信息。
    # English: Extract normal information.
    normals = geometry_info["normal"]

    # 中文：遍历所有 segment label。
    # English: Iterate over all segment labels.
    for label_id in np.unique(labels):
        # 中文：当前区域 mask。
        # English: Current region mask.
        mask = labels == label_id

        # 中文：跳过空区域。
        # English: Skip empty regions.
        if mask.sum() == 0:
            continue

        # 中文：区域面积。
        # English: Region area.
        area = int(mask.sum())

        # 中文：平均深度。
        # English: Mean depth.
        mean_depth = float(np.nanmean(depth[mask]))

        # 中文：平均法向。
        # English: Mean normal.
        region_normals = normals[mask]
        mean_normal = np.nanmean(region_normals, axis=0)
        mean_normal = mean_normal / (np.linalg.norm(mean_normal) + 1e-8)

        # 中文：法向方差，表示区域内部几何是否平滑。
        # English: Normal variance, indicating whether the region is geometrically smooth.
        normal_variance = float(np.nanmean(np.linalg.norm(region_normals - mean_normal, axis=-1)))

        # 中文：平均置信度。
        # English: Mean confidence.
        if conf is not None:
            mean_conf = float(np.nanmean(conf[mask]))
        else:
            mean_conf = 1.0

        # 中文：区域 bbox。
        # English: Region bounding box.
        ys, xs = np.where(mask)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

        descriptors[label_id] = {
            "area": area,
            "mean_depth": mean_depth,
            "mean_normal": mean_normal,
            "normal_variance": normal_variance,
            "mean_conf": mean_conf,
            "bbox": bbox,
        }

    return descriptors


def should_merge_geometry(
    desc_a,
    desc_b,
    depth_thresh,
    normal_cos_thresh,
    conf_thresh=None,
):
    """
    中文：
    判断两个相邻区域是否应该合并。
    第一版只使用三个条件：
    1. 平均深度接近
    2. 平均法向接近
    3. 置信度足够可靠，可选

    English:
    Decide whether two neighboring regions should be merged.
    The first version uses three conditions:
    1. Mean depth is close
    2. Mean normal is similar
    3. Confidence is reliable, optional
    """
    # 中文：深度差异。
    # English: Depth difference.
    depth_diff = abs(desc_a["mean_depth"] - desc_b["mean_depth"])

    # 中文：法向相似度，越接近 1 表示越相似。
    # English: Normal similarity; closer to 1 means more similar.
    normal_sim = float(np.dot(desc_a["mean_normal"], desc_b["mean_normal"]))

    # 中文：深度差过大，不合并。
    # English: Do not merge if depth difference is too large.
    if depth_diff > depth_thresh:
        return False

    # 中文：法向差异过大，不合并。
    # English: Do not merge if normal difference is too large.
    if normal_sim < normal_cos_thresh:
        return False

    # 中文：如果设置了置信度阈值，则低置信区域不轻易合并。
    # English: If confidence threshold is set, do not easily merge low-confidence regions.
    if conf_thresh is not None:
        if min(desc_a["mean_conf"], desc_b["mean_conf"]) < conf_thresh:
            return False

    return True


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
    中文：
    使用几何信息合并相邻区域。
    这是原始 merge_regions 的 geometry-aware 版本。

    English:
    Merge neighboring regions using geometry information.
    This is the geometry-aware version of the original merge_regions function.
    """
    # 中文：计算区域描述子。
    # English: Compute region descriptors.
    descriptors = compute_region_geometry_descriptors(labels, depth, geometry_info, conf=conf)

    # 中文：将角度阈值转成 cos 阈值。
    # English: Convert angle threshold to cosine threshold.
    normal_cos_thresh = np.cos(np.deg2rad(normal_thresh_deg))

    # 中文：如果没有给 depth_thresh，则用深度范围的固定比例。
    # English: If depth_thresh is not provided, use a fixed ratio of depth range.
    if depth_thresh is None:
        depth_range = np.nanmax(depth) - np.nanmin(depth)
        depth_thresh = 0.05 * depth_range

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

    roots = np.array([find(label_id) for label_id in labels.reshape(-1)], dtype=np.int64)
    _, relabeled = np.unique(roots, return_inverse=True)
    return relabeled.reshape(labels.shape).astype(np.intp, copy=False)


def segment_geometry_felzenszwalb_rag(
    depth_map,
    conf_map=None,
    intrinsic=None,
    point_map=None,
    top_conf_percentile=None,
    depth_merge_thresh=0.1,
    normal_thresh_deg=20.0,
    seg_scale=200,
    seg_sigma=1.0,
    seg_min_size=300,
    normal_method="cross",
):
    """
    中文：
    基于 depth、normal 和 confidence 生成几何感知分割区域。
    这个函数用于替代 LASER 原始的 depth-only segmentation。
    但注意：它只改变 segment labels 的来源，不改变后面的 scale estimation 和 scale propagation。

    English:
    Generate geometry-aware segments based on depth, normals, and confidence.
    This function replaces the original depth-only segmentation in LASER.
    Note: it only changes the source of segment labels, while keeping scale estimation
    and scale propagation unchanged.

    Returns:
        labels: np.ndarray, shape [H, W]
            中文：几何分割标签图。
            English: Geometry segment label map.
    """
    # 中文：构建几何信息，包括 local points、normal、depth edge、normal edge 等。
    # English: Build geometry information, including local points, normals, depth edges, and normal edges.
    geometry_info = build_geometry_info_np(
        depth=depth_map,
        conf=conf_map,
        intrinsic=intrinsic,
        points=point_map,
        normal_method=normal_method,
    )

    # 中文：归一化 depth，作为 Felzenszwalb 输入的一个通道。
    # English: Normalize depth as one channel of the Felzenszwalb input.
    depth_norm = depth_map - np.nanmin(depth_map)
    depth_norm = depth_norm / (np.nanmax(depth_norm) + 1e-8)

    # 中文：取出 normal，作为几何通道。
    # English: Extract normals as geometry channels.
    normals = geometry_info["normal"]

    # 中文：构建多通道几何图像。第一版使用 depth + normal。
    # English: Build multi-channel geometry image. The first version uses depth + normals.
    geom_img = np.concatenate(
        [
            depth_norm[..., None],
            normals,
        ],
        axis=-1,
    )

    # 中文：使用 Felzenszwalb 做初始过分割。
    # English: Use Felzenszwalb for initial over-segmentation.
    labels = felzenszwalb(
        geom_img,
        scale=seg_scale,
        sigma=seg_sigma,
        min_size=seg_min_size,
    )

    # 中文：如果有 high-confidence depth range，可以用它辅助确定 depth merge threshold。
    # English: If high-confidence depth range is available, use it to help determine depth merge threshold.
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

    # 中文：基于 depth / normal / confidence 合并区域。
    # English: Merge regions based on depth, normals, and confidence.
    labels = merge_regions_geometry(
        labels,
        depth_map,
        geometry_info,
        conf=conf_map,
        depth_thresh=depth_thresh,
        normal_thresh_deg=normal_thresh_deg,
    )

    return labels
