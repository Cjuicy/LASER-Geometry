# Local Scale Alignment / segment-level scale refinement orchestration.
# Responsibilities:
# 1. Build depth or geometry segment graphs from per-frame segmentation labels.
# 2. Cut adjacent windows to their overlap region and initialize scale anchors there.
# 3. Propagate scale anchors through the target graph and return a dense scale mask.
# 中文职责：
# 1. 作为 Local Scale Alignment 的编排层，统一调度 depth/geometry segment graph。
# 2. 负责切出相邻窗口的 overlap 区域，并在 overlap 内初始化 segment scale anchor。
# 3. 负责把 overlap 得到的尺度锚点沿当前窗口 graph 传播，最终生成稠密 scale mask。
# 4. 不负责具体的 depth 分割、geometry 分割，也不负责具体 scale estimator 的实现。
import torch
import numpy as np


from .depth import (
    segment_depth_felzenszwalb_rag,
    segment_depth_felzenszwalb_rag_stages,
)
from .geometry_segmentation import (
    segment_geometry_felzenszwalb_rag,
    segment_geometry_felzenszwalb_rag_stages,
)
from .scale_anchor import assign_overlap_window_depth_scale
from .segment_graph import match_segmentation_seq

# batched_image_op_wrapper 用在 depth segmentation 分支里，负责对一个 batch/序列里的多帧逐帧执行图像操作。
from .batch_threading import batched_image_op_wrapper, ordered_batch_apply

# 1️⃣ 外部调用 segment-level scale refinement 的入口函数
def refine_segment_scales(
        src_pcd,                # 前一个窗口的点云
        tgt_pcd,                # 当前窗口的点云
        src_sp_graphs,          # 前一个窗口的 segment graph
        tgt_sp_graphs,          # 当前窗口的 segment graph
        overlap,                # 两个窗口的重叠帧数量
        corr_iou_thresh=0.4,    # 判断 segment 是否对应的 IoU 阈值
        src_conf=None,
        tgt_conf=None,
        scale_anchor_mode="depth_irls",
        trace=None,
):
    # 1️⃣ 只取点云最后一维作为 depth。也就是说 refinement 本质上是在 depth map 上估计尺度修正。
    """Estimate a segment-level scale mask from two windows and their graphs."""
    src_depth = src_pcd[..., -1]
    tgt_depth = tgt_pcd[..., -1]

    # 2️⃣ 根据前后窗口 overlap 区域的 segment 对应关系，算出当前窗口每一帧，每个像素应该乘的 scale mask
    align_kwargs = {}
    if trace is not None:
        align_kwargs["trace"] = trace
    tgt_scale_mask = align_adjacent_windows_depth_segments(
        src_depth,
        tgt_depth,
        src_sp_graphs,
        tgt_sp_graphs,
        overlap,
        corr_iou_thresh,
        src_conf=src_conf,
        tgt_conf=tgt_conf,
        scale_anchor_mode=scale_anchor_mode,
        **align_kwargs,
    )

    # 3️⃣ 将 [N, H, W] 的 numpy mask 转成 torch tensor，并扩成 [N, H, W, 1]，方便在 streaming_window_engine.py 里直接乘到 local_points 上。
    return torch.from_numpy(tgt_scale_mask[..., None])


# 2️⃣ 兼容函数，老的调用 refine_depth_segments() 现在会直接调用新的 refine_segment_scales()
def refine_depth_segments(*args, **kwargs):
    """Backward-compatible alias for older call sites."""
    return refine_segment_scales(*args, **kwargs)

# 🌟 3️⃣ 核心尺度传播函数，接收depth 和 segment graph，而不是完整的点云
def align_adjacent_windows_depth_segments(
        src_depth,  # N, H, W
        tgt_depth,  # N, H, W
        src_sp_graphs,
        tgt_sp_graphs,
        overlap,
        corr_iou_thresh=0.4,
        src_conf=None,
        tgt_conf=None,
        scale_anchor_mode="depth_irls",
        trace=None,
):
    """
    src_depth: previous window depth map
    tgt_depth: current window depth map
    src_sp_graphs: previous window superpixel graph (nested list of Vertex)
    tgt_sp_graphs: current window superpixel graph
    overlap: window overlap size
    corr_iou_thresh: IoU threshold for superpixels to be considered as corresponding

    Return:
        depth_scale_mask: N, H, W for current window pcd

    整体目标：
    前一个窗口 overlap depth + 当前窗口 overlap depth
            ↓
    找到对应 segment，估计 scale
            ↓
    把 scale 沿当前窗口的 segment graph 传播
            ↓
    生成当前窗口完整的 depth_scale_mask
    """

    # 1️⃣ 内部函数： 定义了 scale 如何从一个 segment 节点传播到另一个节点。
    def _propagate_scale_cache(parent, child, edge_wt):
        if len(parent.cache['scale']) > 0:                      # 如果 parent.cache['scale'] 里已经有 scale 值，就根据 parent.cache['iou'] 做加权平均
            iou_wts = np.asarray(parent.cache['iou'])
            prop_scale = np.dot(np.asarray(parent.cache['scale']), iou_wts / np.sum(iou_wts))
            child.cache['iou'].append(edge_wt)
            child.cache['scale'].append(prop_scale)             # # 然后把传播后的 scale 和当前边的权重 edge_wt 存到 child 节点里。
            if trace is not None:
                parent_frame, parent_segment = vertex_locations[id(parent)]
                child_frame, child_segment = vertex_locations[id(child)]
                trace.record_propagation(
                    parent_frame,
                    parent_segment,
                    child_frame,
                    child_segment,
                    edge_wt,
                    prop_scale,
                )

    # 2️⃣ 内部函数： 把某个 segment 的 scale cache 转成像素级 mask。（没有可靠对应关系的区域不会被强行改尺度，默认保持原值）
    def _get_scale_mask(mask, cache):
        mask = mask.astype(np.float32)
        if len(cache['scale']) > 0:             # 如果当前 segment 有 scale cache，就按 IoU 权重求平均 scale
            iou_wts = np.asarray(cache['iou'])
            mu_scale = np.dot(np.asarray(cache['scale']), iou_wts / np.sum(iou_wts))
        else:                                   # 如果没有 scale cache，就默认
            mu_scale = 1.0
        return mask * mu_scale                  # 然后返回 mask * mu_scale，得到当前 segment 的像素级 scale mask。

    # 3️⃣ overlap区域的 segment 对应关系，估计当前窗口每个 segment 的 scale
    src_depth_overlap = src_depth[-overlap:]
    tgt_depth_overlap = tgt_depth[:overlap]
    src_sp_graphs_overlap = src_sp_graphs[-overlap:]
    tgt_sp_graphs_overlap = tgt_sp_graphs[:overlap]
    src_conf_overlap = None if src_conf is None else src_conf[-overlap:]
    tgt_conf_overlap = None if tgt_conf is None else tgt_conf[:overlap]

    # 4️⃣ 清掉 source overlap graph 里的旧边，避免上一次窗口对齐残留的边影响这一次 scale assignment。
    for sp_graph in src_sp_graphs_overlap:
        for v in sp_graph:
            v.remove_all_edges()

    # 5️⃣ 初始化 overlap 尺度： scale refinement 起点，只会比较前后窗口 overlap 区域的 segment 对应关系，根据depth 差异估计初始scale，并把 scale 存到当前窗口的 segment cache 里。
    # sptial scale initilaization （初始化scale 只发生在 overlap区域）
    assign_overlap_window_depth_scale(
        src_depth_overlap,
        tgt_depth_overlap,
        src_sp_graphs_overlap,
        tgt_sp_graphs_overlap,
        src_conf_overlap=src_conf_overlap,
        tgt_conf_overlap=tgt_conf_overlap,
        scale_anchor_mode=scale_anchor_mode,
        iou_thresh=corr_iou_thresh,      # 控制 segment 对应关系的严格程度，IoU不够高的segment 不回被当作可靠对象
        trace=trace,
    )
    if trace is not None:
        trace.capture_direct_anchors(tgt_sp_graphs)

    vertex_locations = {
        id(vertex): (
            frame_idx,
            int(vertex.vid) if vertex.vid is not None else segment_idx,
        )
        for frame_idx, graph in enumerate(tgt_sp_graphs)
        for segment_idx, vertex in enumerate(graph)
    }
    # 6️⃣ 沿当前窗口 graph 传播 scale
    # temporal scale propagation （现在重叠部分找到尺度锚点，再通过 segment graph 把尺度修正扩散到整个窗口）
    for tgt_graph_layer in tgt_sp_graphs:
        for v in tgt_graph_layer:
            v.propagate_data_once(_propagate_scale_cache)

    # 7️⃣ 生成最终mask
    mask_seq = []
    for sp_graph in tgt_sp_graphs:      # 逐帧生成 scale mask （每个 segment节点 根据自己的 mask区域 和 cache scale 生成一张局部 mask，然后所有 segment mask加起来，得到一张完整的[h,w] scale map）
        mask_frame = sp_graph[0].data_cache_op(_get_scale_mask)
        for v in sp_graph[1:]:
            mask_frame += v.data_cache_op(_get_scale_mask)
        mask_seq.append(mask_frame)
    # 8️⃣ 得到 [N, H, W] 的当前窗口尺度修正 mask
    if trace is not None:
        trace.capture_segment_states(tgt_sp_graphs)
    return np.stack(mask_seq)


# 4️⃣ 把每帧的 segmentation labels 转换成跨帧 segment graph
def build_sp_graph_from_labels(labels, corr_iou_thresh=0.3):
    return match_segmentation_seq(labels, iou_thresh=corr_iou_thresh)

# 🌟5️⃣ 原始 LASER 的 depth segmentation 分支
def build_depth_sp_graph(
    depth,
    depth_merge_thresh=0.1,
    conf_map=None,
    top_conf_percentile=None,
    corr_iou_thresh=0.3,
    segmentation_trace=None,
):
    # 对序列里的每一帧 depth map 调用 segment_depth_felzenszwalb_rag()，生成 depth-based segmentation labels，把 labels 转成 segment graph
    if segmentation_trace is None:
        labels = batched_image_op_wrapper(
            depth,
            segment_depth_felzenszwalb_rag,
            depth_merge_thresh=depth_merge_thresh,
            conf_map=conf_map,
            top_conf_percentile=top_conf_percentile,
        )
    else:
        stages = ordered_batch_apply(
            depth,
            segment_depth_felzenszwalb_rag_stages,
            depth_merge_thresh=depth_merge_thresh,
            conf_map=conf_map,
            top_conf_percentile=top_conf_percentile,
        )
        labels = np.stack([stage.merged_labels for stage in stages], axis=0)
        segmentation_trace.update(
            initial_labels=np.stack([stage.initial_labels for stage in stages], axis=0),
            merged_labels=labels,
            confidence_thresholds=np.asarray(
                [stage.confidence_threshold for stage in stages], dtype=np.float32
            ),
            high_confidence_masks=np.stack(
                [stage.high_confidence_mask for stage in stages], axis=0
            ),
        )
    return build_sp_graph_from_labels(labels, corr_iou_thresh=corr_iou_thresh)

# 🌟6️⃣ 新增 geometry-aware segmentation 分支
def build_geometry_sp_graph(
    depth,
    depth_merge_thresh=0.1,
    conf_map=None,
    top_conf_percentile=None,
    corr_iou_thresh=0.3,

    # 🌟 其他的一些参数选项（可能需要对应帧的置信度图，点云图，内参，或者共享的内参）
    point_map=None,
    intrinsic=None,
    normal_method="cross",
    segmentation_trace=None,
):
    op_kwargs = {
        "depth_merge_thresh": depth_merge_thresh,
        "conf_map": conf_map,
        "top_conf_percentile": top_conf_percentile,
        "point_map": point_map,
        "intrinsic": intrinsic,
        "normal_method": normal_method,
    }
    if segmentation_trace is None:
        labels = batched_image_op_wrapper(
            depth,
            segment_geometry_felzenszwalb_rag,
            **op_kwargs,
        )
    else:
        stages = ordered_batch_apply(
            depth,
            segment_geometry_felzenszwalb_rag_stages,
            **op_kwargs,
        )
        labels = np.stack([stage.merged_labels for stage in stages], axis=0)
        segmentation_trace.update(
            initial_labels=np.stack([stage.initial_labels for stage in stages], axis=0),
            merged_labels=labels,
            confidence_thresholds=np.asarray(
                [stage.confidence_threshold for stage in stages], dtype=np.float32
            ),
            high_confidence_masks=np.stack(
                [stage.high_confidence_mask for stage in stages], axis=0
            ),
        )
    return build_sp_graph_from_labels(labels, corr_iou_thresh=corr_iou_thresh)


# 7️⃣ 兼容函数，旧代码仍然可以调用 make_sp_graph(..., segment_mode="geometry")
def make_sp_graph(
    depth,
    depth_merge_thresh=0.1,
    conf_map=None,
    top_conf_percentile=None,
    corr_iou_thresh=0.3,
    point_map=None,
    intrinsic=None,
    segment_mode="depth",
    normal_method="cross",
):
    """Backward-compatible graph builder; new code should call explicit builders."""
    if segment_mode == "depth":
        return build_depth_sp_graph(
            depth,
            depth_merge_thresh=depth_merge_thresh,
            conf_map=conf_map,
            top_conf_percentile=top_conf_percentile,
            corr_iou_thresh=corr_iou_thresh,
        )
    elif segment_mode == "geometry":
        return build_geometry_sp_graph(
            depth,
            depth_merge_thresh=depth_merge_thresh,
            conf_map=conf_map,
            top_conf_percentile=top_conf_percentile,
            corr_iou_thresh=corr_iou_thresh,
            point_map=point_map,
            intrinsic=intrinsic,
            normal_method=normal_method,
        )
    else:
        raise ValueError(f"Unknown segment_mode: {segment_mode}")
