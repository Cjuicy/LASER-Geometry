# Segment graph utilities shared by depth and geometry segmentation.
# Responsibilities:
# 1. Convert per-frame segmentation labels into Vertex graphs.
# 2. Connect segment vertices across frames/windows using mask overlap.
# 3. Keep graph construction independent from how labels were produced.
# 中文职责：
# 1. 负责把逐帧 segmentation labels 转换成 Vertex-based segment graph。
# 2. 负责根据 mask IoU 给相邻帧或相邻窗口的 segment 建边。
# 3. 只关心 labels/masks/graph，不关心 labels 是 depth 分割还是 geometry 分割产生的。


# 整体数据流
"""
segmentation labels
  -> 每帧 labels 转成多个 bool masks
  -> 每个 mask 包装成 Vertex
  -> 相邻帧/相邻窗口的 Vertex 根据 IoU 建边
  -> 输出 segment graph

  lsa.py 和 scale_anchor.py 会利用这个graph
  segment graph
  -> overlap 内找到 source segment 和 target segment 对应关系
  -> scale_anchor 估计 target segment scale
  -> 写入 target Vertex cache
  -> lsa 沿 graph 传播 scale
"""

import numpy as np

from pi3.utils.graph import Vertex


# 1️⃣ 计算两个 segment mask 的 pairwise intersection ratio（交集比例），用于判断 segment 是否对应。（当前主流程没有使用这个方法）
def pairwise_intersection_ratio(mask1, mask2):
    """
    Highest pairwise intersection ratio for assigning correspondence.
    """
    N, H, W = mask1.shape
    M = mask2.shape[0]

    # 1️⃣ 把每个mask 展平成一行，然后矩阵乘法，得到每个mask之间的交集像素数量（第i个mask1 和 第j个mask2的交集像素数量）
    mask1_f = mask1.reshape(N, -1).astype(np.float32)
    mask2_f = mask2.reshape(M, -1).astype(np.float32)
    inter = mask1_f @ mask2_f.T

    area1 = mask1_f.sum(axis=-1, keepdims=True)
    area2 = mask2_f.sum(axis=-1, keepdims=True).T
    area1 = np.maximum(area1, 1)
    area2 = np.maximum(area2, 1)

    # 2️⃣ 计算交集占各自面积的比例，mask1有多少比例被mask2覆盖，mask2有多少比例被mask1覆盖
    ratios1 = inter / area1
    ratios2 = inter / area2
    min_rel_inter = np.minimum(ratios1, ratios2)
    max_rel_inter = np.maximum(ratios1, ratios2)

    # 3️⃣ 返回两个 segment mask 的 pairwise intersection ratio（交集比例），用于判断 segment 是否对应。
    return min_rel_inter, max_rel_inter


# 2️⃣ 当前真正建立边的函数
def pairwise_iou(mask1, mask2):
    N, H, W = mask1.shape
    M = mask2.shape[0]

    # 1️⃣ 展平计算交集
    mask1_f = mask1.reshape(N, -1).astype(np.float32)
    mask2_f = mask2.reshape(M, -1).astype(np.float32)
    inter = mask1_f @ mask2_f.T

    # 2️⃣ 计算并集
    area1 = mask1_f.sum(axis=-1, keepdims=True)
    area2 = mask2_f.sum(axis=-1, keepdims=True).T
    union = area1 + area2 - inter

    # 3️⃣ 返回 iou[i, j] 表示 graph1 的第 i 个 segment 和 graph2 的第 j 个 segment 的 IoU。
    return inter / np.maximum(union, 1)


# 3️⃣ 把一整段序列的 segmentation labels 转成 graph
def match_segmentation_seq(labels, iou_thresh=0.4):
    # 1️⃣ 内部函数：把单帧 segmentation labels 转成 Vertex graph
    def get_seg_vertices(seg):
        # 找出当前帧所有的 segment id
        seg_ids = np.unique(seg)
        # 每个 segment id 变成一个bool mask
        masks = seg[None, :, :] == seg_ids[:, None, None]
        # 每个mask 包装成Vertex，每个Vertex自带cache
        return [
            Vertex(
                data=mask,
                vid=int(seg_id),
                default_cache={'iou': [], 'scale': []},
            )
            for seg_id, mask in zip(seg_ids, masks)
        ]

    # 2️⃣ 先处理第一帧
    sp_graph = [get_seg_vertices(labels[0])]

    # 3️⃣ 接着处理后续每一帧：第 t-1 帧的 segments 和第 t 帧的 segments 根据 IoU 建边，建完边后，把第 t 帧加入 graph 序列。
    for seg_map in labels[1:]:
        seg_vertices = get_seg_vertices(seg_map)
        connect_bipartite_sp_graphs(sp_graph[-1], seg_vertices, iou_thresh=iou_thresh)
        sp_graph.append(seg_vertices)

    return sp_graph


# 4️⃣ 给两组 segment graph 建边，基于 mask IoU
def connect_bipartite_sp_graphs(graph1, graph2, iou_thresh=0.3):
    # 1️⃣ 取出所有mask
    masks1 = np.stack([v.data for v in graph1])
    masks2 = np.stack([v.data for v in graph2])

    # 2️⃣ 计算两组mask的IoU
    iou = pairwise_iou(masks1, masks2)
    
    # 3️⃣ 筛选出IoU大于阈值的 pair
    matchable = iou >= iou_thresh
    graph1_indices, graph2_indices = np.nonzero(matchable)

    # 最后建立边
    for v1, v2 in zip(graph1_indices, graph2_indices):
        graph1[v1].add_edge(graph2[v2], iou[v1, v2])
