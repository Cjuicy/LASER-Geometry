# 给 lsa.py / streaming_window_engine.py 提供底层工具。
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from skimage.segmentation import felzenszwalb

from pi3.utils.graph import Vertex

from ._segmentation_cy import merge_regions
from .fast_seg import fast_graph_segmentation

SCALE_ANCHOR_MODES = ("depth_irls", "conf_weighted_irls")

# 1️⃣ 估计深度尺度s_d
def align_depth_irls(
        src_depth,
        tgt_depth,
        mask=None,
        iters=10,
        eps=1e-8,
        stop_tol=0.05,
        clamp_min=1e-6
):
    # 1️⃣ 如果传送了mask，就只在mask区域内估计尺度
    if mask is not None:
        src_depth = src_depth[mask]
        tgt_depth = tgt_depth[mask]

    # 2️⃣ 初值使用均值比例
    num = np.nanmean(tgt_depth)
    den = np.nanmean(src_depth)
    s_d = np.maximum(num / den, clamp_min)

    # 3️⃣ 使用IRLS迭代，每一轮先算残差
    for _ in range(iters):
        d_res = s_d * src_depth - tgt_depth
        res = abs(d_res) + eps
        # 残差越大，权重越小（这样异常点对尺度估计的影响会被压低）
        w = 1.0 / res

        num = (w * src_depth * tgt_depth).sum()
        den = (w * src_depth ** 2).sum()
        s_d_new = np.maximum(num / den, clamp_min)
        converged = abs(s_d_new - s_d) < stop_tol
        s_d = s_d_new

        if converged:
            break

    return s_d


def _normalize_confidence_for_weights(conf, eps=1e-6):
    """Map arbitrary confidence logits/scores to positive IRLS weights."""
    conf = np.asarray(conf, dtype=np.float64)
    finite = np.isfinite(conf)
    if not np.any(finite):
        return np.ones_like(conf, dtype=np.float64)

    valid_conf = conf[finite]
    conf_min = valid_conf.min()
    conf_span = valid_conf.max() - conf_min
    if conf_span <= eps:
        return np.ones_like(conf, dtype=np.float64)

    weights = (conf - conf_min) / conf_span
    weights[~finite] = 0.0
    return np.clip(weights, eps, 1.0)


def align_depth_irls_conf_weighted(
        src_depth,
        tgt_depth,
        src_conf=None,
        tgt_conf=None,
        mask=None,
        iters=10,
        eps=1e-8,
        stop_tol=0.05,
        clamp_min=1e-6,
        conf_eps=1e-6
):
    """Estimate depth scale with confidence-weighted IRLS.

    This is the M1 experimental anchor estimator. The original align_depth_irls
    path is intentionally left unchanged and remains the default.
    """
    if mask is not None:
        src_depth = src_depth[mask]
        tgt_depth = tgt_depth[mask]
        if src_conf is not None:
            src_conf = src_conf[mask]
        if tgt_conf is not None:
            tgt_conf = tgt_conf[mask]

    src_depth = np.asarray(src_depth, dtype=np.float64)
    tgt_depth = np.asarray(tgt_depth, dtype=np.float64)

    valid = np.isfinite(src_depth) & np.isfinite(tgt_depth) & (np.abs(src_depth) > eps)
    if src_conf is not None:
        src_conf = np.asarray(src_conf, dtype=np.float64)
        valid &= np.isfinite(src_conf)
    if tgt_conf is not None:
        tgt_conf = np.asarray(tgt_conf, dtype=np.float64)
        valid &= np.isfinite(tgt_conf)

    if not np.any(valid):
        return 1.0

    src_depth = src_depth[valid]
    tgt_depth = tgt_depth[valid]

    if src_conf is None:
        src_w = np.ones_like(src_depth, dtype=np.float64)
    else:
        src_w = _normalize_confidence_for_weights(src_conf[valid], eps=conf_eps)

    if tgt_conf is None:
        tgt_w = np.ones_like(tgt_depth, dtype=np.float64)
    else:
        tgt_w = _normalize_confidence_for_weights(tgt_conf[valid], eps=conf_eps)

    conf_w = np.sqrt(src_w * tgt_w)
    conf_w = np.clip(conf_w, conf_eps, 1.0)

    den = (conf_w * src_depth ** 2).sum()
    if den <= eps:
        return 1.0

    num = (conf_w * src_depth * tgt_depth).sum()
    s_d = np.maximum(num / den, clamp_min)

    for _ in range(iters):
        d_res = s_d * src_depth - tgt_depth
        res = np.abs(d_res) + eps
        w = conf_w / res

        den = (w * src_depth ** 2).sum()
        if den <= eps:
            break
        num = (w * src_depth * tgt_depth).sum()
        s_d_new = np.maximum(num / den, clamp_min)
        converged = abs(s_d_new - s_d) < stop_tol
        s_d = s_d_new

        if converged:
            break

    return s_d


def _validate_scale_anchor_mode(scale_anchor_mode):
    if scale_anchor_mode not in SCALE_ANCHOR_MODES:
        raise ValueError(
            f"Unknown scale_anchor_mode: {scale_anchor_mode}. "
            f"Expected one of {SCALE_ANCHOR_MODES}."
        )


def _estimate_depth_scale(
        src_depth,
        tgt_depth,
        mask,
        src_conf=None,
        tgt_conf=None,
        scale_anchor_mode="depth_irls"
):
    _validate_scale_anchor_mode(scale_anchor_mode)
    if scale_anchor_mode == "depth_irls":
        return align_depth_irls(src_depth, tgt_depth, mask)
    return align_depth_irls_conf_weighted(
        src_depth,
        tgt_depth,
        src_conf=src_conf,
        tgt_conf=tgt_conf,
        mask=mask,
    )

# 2️⃣ 原始 LASER 的 depth segmentation 分支之一： 深度分割
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
    # 1️⃣ 用 skimage.segmentation.felzenszwalb 在 depth map 上做初始过分割，得到的是每个像素的 segment label
    seg_mask = felzenszwalb(depth_map, scale=seg_scale, sigma=seg_sigma, min_size=seg_min_size)
    # 2️⃣ 如果有 confidence map，就只用高置信区域的 depth range 来决定 merge threshold
    if conf_map is not None and top_conf_percentile is not None:
        conf_map = conf_map[batch_idx]
        conf_thresh = np.quantile(conf_map.reshape(-1), top_conf_percentile, method='nearest')
        conf_depth = depth_map[conf_map >= conf_thresh]
    else:
        conf_depth = depth_map
    # 3️⃣ 根据深度范围计算合并阈值
    merge_thresh = depth_merge_thresh * (np.max(conf_depth) - np.min(conf_depth))

    # 4️⃣ 调用 Cython 实现的 merge_regions()，就是把 Felzenszwalb 初始分割中深度相近的区域继续合并
    return merge_regions(seg_mask, depth_map, merge_thresh)


# 3️⃣ 上一个目标类似，也是 depth-based segmentation，但走的是：跳过 felzenszwalb + merge_regions 这条路径，直接用更快的 graph segmentation 实现。
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


# 4️⃣ 判断“两个 segment 是否大面积互相覆盖”。不过在当前这段代码里，真正用于 graph 连边的是下面的 pairwise_iou()
def pairwise_intersection_ratio(mask1, mask2):
    """
    Highest pairwise intersection ratio for assigning correspondence.
    """
    N, H, W = mask1.shape
    M = mask2.shape[0]

    # 1️⃣ 输入两组 mask
    mask1_f = mask1.reshape(N, -1).astype(np.float32)
    mask2_f = mask2.reshape(M, -1).astype(np.float32)
    # 2️⃣ 它先把 mask 展平，然后矩阵乘法算所有 mask pair 的交集面积
    inter = mask1_f @ mask2_f.T

    area1 = mask1_f.sum(axis=-1, keepdims=True)
    area2 = mask2_f.sum(axis=-1, keepdims=True).T
    area1 = np.maximum(area1, 1)
    area2 = np.maximum(area2, 1)

    ratios1 = inter / area1
    ratios2 = inter / area2
    min_rel_inter = np.minimum(ratios1, ratios2)
    max_rel_inter = np.maximum(ratios1, ratios2)

    # 3️⃣ 返回两个比例的最小值和最大值
    return min_rel_inter, max_rel_inter


# 5️⃣ 计算两组 mask 的 IoU，于判断相邻帧、相邻窗口的 segment 是否对应
def pairwise_iou(mask1, mask2):
    N, H, W = mask1.shape
    M = mask2.shape[0]

    mask1_f = mask1.reshape(N, -1).astype(np.float32)
    mask2_f = mask2.reshape(M, -1).astype(np.float32)
    inter = mask1_f @ mask2_f.T

    area1 = mask1_f.sum(axis=-1, keepdims=True)
    area2 = mask2_f.sum(axis=-1, keepdims=True).T
    union = area1 + area2 - inter

    return inter / np.maximum(union, 1)


# 6️⃣ 把一整段序列的 segmentation labels 转成 segment graph。
def match_segmentation_seq(labels, iou_thresh=0.4):
    # 内部定义函数
    def get_seg_vertices(seg):
        # 每个 segment id 会变成一个 bool mask，然后包装成一个 Vertex
        seg_ids = np.unique(seg)
        masks = seg[None, :, :] == seg_ids[:, None, None]
        # 这里的 cache 很重要，后面 scale refinement 会往里面写 iou， scale
        return [Vertex(data=m, default_cache={'iou': [], 'scale': []}) for m in masks]

    sp_graph = [get_seg_vertices(labels[0])]

    for seg_map in labels[1:]:
        seg_vertices = get_seg_vertices(seg_map)
        # 然后从第二帧开始，逐帧把上一帧 graph 和当前帧 graph 连起来
        connect_bipartite_sp_graphs(sp_graph[-1], seg_vertices, iou_thresh=iou_thresh)
        sp_graph.append(seg_vertices)

    return sp_graph

# 7️⃣ 这个函数负责给两帧 segment graph 建边。
def connect_bipartite_sp_graphs(graph1, graph2, iou_thresh=0.3):
    # 1️⃣ 先取出所有 vertex mask
    masks1 = np.stack([v.data for v in graph1])
    masks2 = np.stack([v.data for v in graph2])

    # 2️⃣ 然后计算两两 IoU
    iou = pairwise_iou(masks1, masks2)
    # 3️⃣ 只保留大于阈值的 pair
    matchable = iou >= iou_thresh
    graph1_indices, graph2_indices = np.nonzero(matchable)

    for v1, v2 in zip(graph1_indices, graph2_indices):
        # 3️⃣ 最后给对应 vertex 加边
        graph1[v1].add_edge(graph2[v2], iou[v1, v2])


# 8️⃣ 对一个 source segment 做尺度估计的 worker
def _edge_scale_worker(
        src_depth,
        tgt_depth,
        src_vertex,
        src_conf=None,
        tgt_conf=None,
        scale_anchor_mode="depth_irls"
):
    src_mask = src_vertex.data
    # 1️⃣ 对 src_vertex 的每条连接边
    for tgt_v, tgt_iou in zip(src_vertex.connectivity, src_vertex.edge_weights):
        tgt_mask = tgt_v.data
        # 2️⃣ 取 source mask 和 target mask 的交集
        inter_mask = src_mask & tgt_mask
        # 3️⃣ 估计 target 到 source 的 scale
        tgt2src_s = _estimate_depth_scale(
            tgt_depth,
            src_depth,
            inter_mask,
            src_conf=tgt_conf,
            tgt_conf=src_conf,
            scale_anchor_mode=scale_anchor_mode,
        )
        # 4️⃣ 把这个 scale 写入 target vertex 的 cache
        tgt_v.cache['iou'].append(tgt_iou)
        tgt_v.cache['scale'].append(tgt2src_s)


# 9️⃣ overlap 区域尺度初始化的入口函数
def assign_overlap_window_depth_scale(
        src_depth_overlap,
        tgt_depth_overlap,
        src_sp_graphs_overlap,
        tgt_sp_graphs_overlap,
        src_conf_overlap=None,
        tgt_conf_overlap=None,
        scale_anchor_mode="depth_irls",
        iou_thresh=0.4,
        n_jobs=1
):
    _validate_scale_anchor_mode(scale_anchor_mode)

    # 1️⃣ 对每一对 overlap 帧的 graph 建边
    for src_sp_graph, tgt_sp_graph in zip(src_sp_graphs_overlap, tgt_sp_graphs_overlap):
        connect_bipartite_sp_graphs(src_sp_graph, tgt_sp_graph, iou_thresh=iou_thresh)

    for idx, src_graph in enumerate(src_sp_graphs_overlap):
        src_conf = None if src_conf_overlap is None else src_conf_overlap[idx]
        tgt_conf = None if tgt_conf_overlap is None else tgt_conf_overlap[idx]
        n_jobs = min(os.cpu_count(), len(src_graph)) if n_jobs is None else n_jobs
        if n_jobs == 1:
            for src_v in src_graph:
                # 2️⃣ 对每个 source segment 计算它连接到 target segment 的 scale
                _edge_scale_worker(
                    src_depth_overlap[idx],
                    tgt_depth_overlap[idx],
                    src_v,
                    src_conf=src_conf,
                    tgt_conf=tgt_conf,
                    scale_anchor_mode=scale_anchor_mode,
                )
        else:
            with ThreadPoolExecutor(max_workers=n_jobs) as ex:
                promises = [
                    ex.submit(
                        _edge_scale_worker,
                        src_depth_overlap[idx],
                        tgt_depth_overlap[idx],
                        src_v,
                        src_conf,
                        tgt_conf,
                        scale_anchor_mode,
                    )
                    for src_v in src_graph
                ]
                for promise in as_completed(promises):
                    promise.result()
