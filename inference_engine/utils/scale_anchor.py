# Scale-anchor estimators for overlap segment alignment.
# Responsibilities:
# 1. Estimate depth scale between matched overlap segments.
# 2. Provide the original IRLS and M1 confidence-weighted IRLS estimators.
# 3. Write segment-level scale anchors into target graph caches for LSA.
# 中文职责：
# 1. 负责相邻窗口 overlap 内、已经匹配上的 segment 之间的尺度估计。
# 2. 提供原始 depth_irls 和 M1 conf_weighted_irls 两种 scale anchor estimator。
# 3. 把估计出的 segment-level scale anchor 写入 target graph cache，供 lsa.py 后续传播。
# 4. 不负责生成 segment graph，也不负责选择 depth/geometry segmentation 模式。


# 此文件的说明
"""
source overlap depth + target overlap depth + segment graph correspondence
  -> 估计 target segment 到 source segment 的 scale
  -> 写入 target vertex.cache["scale"]
在相邻窗口 overlap 区域里，根据已经匹配上的 segment，估计 target window 应该乘多少 depth scale，作为后续 LSA 传播的 anchor。

整体数据流
src/tgt overlap depth
src/tgt overlap segment graphs
src/tgt overlap confidence
        ↓
connect_bipartite_sp_graphs
根据 IoU 找 source segment -> target segment 对应关系
        ↓
_edge_scale_worker
对每个对应 segment 交集区域估计 target-to-source scale
        ↓
_estimate_depth_scale
根据 scale_anchor_mode 选择 depth_irls 或 conf_weighted_irls
        ↓
target vertex.cache["scale"].append(scale)
target vertex.cache["iou"].append(iou)
        ↓
lsa.py 后续把这些 scale anchor 沿 target graph 传播


segment_graph.py 负责“谁对应谁”
scale_anchor.py 负责“对应上之后 scale 是多少”
lsa.py 负责“把这些 scale 传播成整张 scale mask”
"""
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .segment_graph import connect_bipartite_sp_graphs

# 1️⃣ 估计 depth scale 的两种方法：原始 IRLS 和 M1 confidence-weighted IRLS
SCALE_ANCHOR_MODES = ("depth_irls", "conf_weighted_irls")


# 2️⃣ 原始 scale estimator： 估计一个标量 s_d 让 s_d * src_depth ≈ tgt_depth
def align_depth_irls(
        src_depth,
        tgt_depth,
        mask=None,
        iters=10,
        eps=1e-8,
        stop_tol=0.05,
        clamp_min=1e-6
):
    # 1️⃣如果传了 mask，也就是只在某个 segment 交集区域内估计 scale。
    if mask is not None:
        src_depth = src_depth[mask]
        tgt_depth = tgt_depth[mask]

    # 2️⃣ 先用最小二乘法估计一个初始 scale：s_d = mean(tgt_depth) / mean(src_depth)
    num = np.nanmean(tgt_depth)
    den = np.nanmean(src_depth)
    s_d = np.maximum(num / den, clamp_min)

    # 3️⃣ 进入 IRLS 循环，迭代更新 s_d ： 残差越大，权重越小。也就是说，明显不符合当前 scale 的点会被压低影响。
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
            break
    
    # 函数返回的是把 src_depth 对齐到 tgt_depth 的 scale
    return s_d

# 3️⃣ 这个函数把任意 confidence map 转成 [eps, 1.0] 范围内的权重。 ：核心逻辑： weights = (conf - conf_min) / conf_span
def _normalize_confidence_for_weights(conf, eps=1e-6):
    """Map arbitrary confidence logits/scores to positive IRLS weights."""
    conf = np.asarray(conf, dtype=np.float64)
    finite = np.isfinite(conf)
    # 如果 confidence 全是无效值：不做额外加权
    if not np.any(finite):
        return np.ones_like(conf, dtype=np.float64)

    valid_conf = conf[finite]
    conf_min = valid_conf.min()
    conf_span = valid_conf.max() - conf_min
    # 如果 confidence 全是同一个值：不做额外加权
    if conf_span <= eps:
        return np.ones_like(conf, dtype=np.float64)

    weights = (conf - conf_min) / conf_span
    weights[~finite] = 0.0
    return np.clip(weights, eps, 1.0)


# 🌟4️⃣ M1核心函数：confidence-weighted IRLS。它和 align_depth_irls 的目标一样，仍然估计sd，但多用了src_conf 和 tgt_conf 来给每个点加权。也就是说，置信度高的点对 scale 的影响更大，置信度低的点对 scale 的影响更小。
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
    # 1️⃣ 应用mask
    if mask is not None:
        src_depth = src_depth[mask]
        tgt_depth = tgt_depth[mask]
        if src_conf is not None:
            src_conf = src_conf[mask]
        if tgt_conf is not None:
            tgt_conf = tgt_conf[mask]

    src_depth = np.asarray(src_depth, dtype=np.float64)
    tgt_depth = np.asarray(tgt_depth, dtype=np.float64)

    # 2️⃣ 过滤无效值
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

    # 3️⃣ 把 source / target confidence 都变成权重：
    if src_conf is None:
        src_w = np.ones_like(src_depth, dtype=np.float64)
    else:
        src_w = _normalize_confidence_for_weights(src_conf[valid], eps=conf_eps)

    if tgt_conf is None:
        tgt_w = np.ones_like(tgt_depth, dtype=np.float64)
    else:
        tgt_w = _normalize_confidence_for_weights(tgt_conf[valid], eps=conf_eps)

    # 🌟4️⃣ 合成一份 mutual confidence 权重（这里就是有VGGT-Long对齐的思想的）
    conf_w = np.sqrt(src_w * tgt_w)
    conf_w = np.clip(conf_w, conf_eps, 1.0)

    # 5️⃣ 用 confidence-weighted least squares 给初值
    den = (conf_w * src_depth ** 2).sum()
    if den <= eps:
        return 1.0

    num = (conf_w * src_depth * tgt_depth).sum()
    s_d = np.maximum(num / den, clamp_min)

    # 6️⃣ 进入 IRLS 循环，迭代更新 s_d ： 残差越大，权重越小。也就是说，明显不符合当前 scale 的点会被压低影响。
    for _ in range(iters):
        d_res = s_d * src_depth - tgt_depth
        res = np.abs(d_res) + eps
        w = conf_w / res                                # 最终权重 = confidence 权重 / residual

        den = (w * src_depth ** 2).sum()
        if den <= eps:
            break
        num = (w * src_depth * tgt_depth).sum()
        s_d_new = np.maximum(num / den, clamp_min)
        converged = abs(s_d_new - s_d) < stop_tol
        s_d = s_d_new

        if converged:
            break
    
    # 高置信 + 残差小     -> 权重大   ；  低置信 或 残差大   -> 权重小
    return s_d


# 5️⃣ 只是保护函数，避免传错模式
def _validate_scale_anchor_mode(scale_anchor_mode):
    if scale_anchor_mode not in SCALE_ANCHOR_MODES:
        raise ValueError(
            f"Unknown scale_anchor_mode: {scale_anchor_mode}. "
            f"Expected one of {SCALE_ANCHOR_MODES}."
        )


# 6️⃣ estimator 的分发器，M1 的开关就在这里生效
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


# 7️⃣ 函数是 overlap scale assignment 的核心 worker。
def _edge_scale_worker(
        src_depth,                              # 重要
        tgt_depth,                              # 重要
        src_vertex,                             # ‼️重要：已经通过 graph 边连接到了若干个 target vertex，每条边表示：这个 source segment 和这个 target segment 在 overlap 区域里有足够 IoU，可以认为是对应关系
        src_conf=None,
        tgt_conf=None,
        scale_anchor_mode="depth_irls"
):
    src_mask = src_vertex.data
    for tgt_v, tgt_iou in zip(src_vertex.connectivity, src_vertex.edge_weights):
        tgt_mask = tgt_v.data
        # 取两个segment mask 交集，只在这个交集区域估计scale
        inter_mask = src_mask & tgt_mask
        tgt2src_s = _estimate_depth_scale(      # 这个要关心一下：tgt2src_s * tgt_depth ≈ src_depth ：也就是 target segment 应该乘多少 scale 才能对齐 source segment。
            tgt_depth,
            src_depth,
            inter_mask,
            src_conf=tgt_conf,
            tgt_conf=src_conf,
            scale_anchor_mode=scale_anchor_mode,
        )
        # 最后写入 target vertex cache：后面 lsa.py 会从 target graph 的 cache 里读这些 scale，并沿 graph 传播
        tgt_v.cache['iou'].append(tgt_iou)
        tgt_v.cache['scale'].append(tgt2src_s)


# 8️⃣ 整个文件的主入口
def assign_overlap_window_depth_scale(
        src_depth_overlap,              # 接收overlap区域
        tgt_depth_overlap,
        src_sp_graphs_overlap,
        tgt_sp_graphs_overlap,
        src_conf_overlap=None,
        tgt_conf_overlap=None,
        scale_anchor_mode="depth_irls",
        iou_thresh=0.4,
        n_jobs=1
):
    # 1️⃣ 检查 estimator 模式
    _validate_scale_anchor_mode(scale_anchor_mode)

    # 2️⃣ 对每一对 overlap 帧的 source graph 和 target graph 建边
    for src_sp_graph, tgt_sp_graph in zip(src_sp_graphs_overlap, tgt_sp_graphs_overlap):
        connect_bipartite_sp_graphs(src_sp_graph, tgt_sp_graph, iou_thresh=iou_thresh)

    # 3️⃣ 逐帧处理 overlap
    for idx, src_graph in enumerate(src_sp_graphs_overlap):
        # 1️⃣ 取当前 overlap 帧的 confidence
        src_conf = None if src_conf_overlap is None else src_conf_overlap[idx]
        tgt_conf = None if tgt_conf_overlap is None else tgt_conf_overlap[idx]
        n_jobs = min(os.cpu_count(), len(src_graph)) if n_jobs is None else n_jobs
        if n_jobs == 1:     # 顺序跑
            for src_v in src_graph:
                # 2️⃣ 对当前 source graph 里的每个 source segment 估计 scale
                _edge_scale_worker(
                    src_depth_overlap[idx],
                    tgt_depth_overlap[idx],
                    src_v,
                    src_conf=src_conf,
                    tgt_conf=tgt_conf,
                    scale_anchor_mode=scale_anchor_mode,
                )
        else:
            with ThreadPoolExecutor(max_workers=n_jobs) as ex:      # 线程池跑
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
