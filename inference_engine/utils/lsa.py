import torch
import numpy as np

from .depth import (
    segment_depth_felzenszwalb_rag,
    segment_geometry_felzenszwalb_rag,
    match_segmentation_seq,
    assign_overlap_window_depth_scale,
)
from .batch_threading import batched_image_op_wrapper


def refine_depth_segments(
        src_pcd,
        tgt_pcd,
        src_sp_graphs,
        tgt_sp_graphs,
        overlap,
        corr_iou_thresh=0.4
):
    """
    src_pcd: previous window pcd
    tgt_pcd: current window pcd
    src_sp_graphs: previous window superpixel graph
    overlap: window overlap size
    conf_mask: confidence mask
    depth_merge_thresh: percentage confident depth range to be considered as smooth change
    corr_iou_thresh: IoU threshold for superpixels to be considered as corresponding
    """
    src_depth = src_pcd[..., -1]
    tgt_depth = tgt_pcd[..., -1]

    tgt_scale_mask = align_adjacent_windows_depth_segments(
        src_depth,
        tgt_depth,
        src_sp_graphs,
        tgt_sp_graphs,
        overlap,
        corr_iou_thresh
    )

    return torch.from_numpy(tgt_scale_mask[..., None])


def align_adjacent_windows_depth_segments(
        src_depth,  # N, H, W
        tgt_depth,  # N, H, W
        src_sp_graphs,
        tgt_sp_graphs,
        overlap,
        corr_iou_thresh=0.4
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
    """

    def _propagate_scale_cache(parent, child, edge_wt):
        if len(parent.cache['scale']) > 0:
            iou_wts = np.asarray(parent.cache['iou'])
            prop_scale = np.dot(np.asarray(parent.cache['scale']), iou_wts / np.sum(iou_wts))
            child.cache['iou'].append(edge_wt)
            child.cache['scale'].append(prop_scale)

    def _get_scale_mask(mask, cache):
        mask = mask.astype(np.float32)
        if len(cache['scale']) > 0:
            iou_wts = np.asarray(cache['iou'])
            mu_scale = np.dot(np.asarray(cache['scale']), iou_wts / np.sum(iou_wts))
        else:
            mu_scale = 1.0
        return mask * mu_scale

    src_depth_overlap = src_depth[-overlap:]
    tgt_depth_overlap = tgt_depth[:overlap]
    src_sp_graphs_overlap = src_sp_graphs[-overlap:]
    tgt_sp_graphs_overlap = tgt_sp_graphs[:overlap]

    for sp_graph in src_sp_graphs_overlap:
        for v in sp_graph:
            v.remove_all_edges()

    # sptial scale initilaization
    assign_overlap_window_depth_scale(
        src_depth_overlap,
        tgt_depth_overlap,
        src_sp_graphs_overlap,
        tgt_sp_graphs_overlap,
        iou_thresh=corr_iou_thresh
    )
    # temporal scale propagation
    for tgt_graph_layer in tgt_sp_graphs:
        for v in tgt_graph_layer:
            v.propagate_data_once(_propagate_scale_cache)

    mask_seq = []
    for sp_graph in tgt_sp_graphs:
        mask_frame = sp_graph[0].data_cache_op(_get_scale_mask)
        for v in sp_graph[1:]:
            mask_frame += v.data_cache_op(_get_scale_mask)
        mask_seq.append(mask_frame)

    return np.stack(mask_seq)

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
    """
    中文：
    根据每帧的分割标签构建 LASER 使用的 segment graph。
    第一阶段只切换 segment 的生成方式：
        - depth: 使用 LASER 原始 depth segmentation
        - geometry: 使用新增 geometry-aware segmentation

    注意：
    这里不修改 LASER 原始的 scale estimation 和 scale propagation。
    后面的 match_segmentation_seq、assign_overlap_window_depth_scale、
    align_adjacent_windows_depth_segments 都保持原始逻辑。

    English:
    Build the segment graph used by LASER from per-frame segmentation labels.
    In the first stage, we only switch the source of segment labels:
        - depth: use original LASER depth segmentation
        - geometry: use new geometry-aware segmentation

    Note:
    This function does not modify the original LASER scale estimation or scale propagation.
    The following match_segmentation_seq, assign_overlap_window_depth_scale,
    and align_adjacent_windows_depth_segments remain unchanged.
    """

    if segment_mode == "depth":
        # 中文：原始 LASER 分支，保持 baseline 完全不变。
        # English: Original LASER branch, keeping the baseline fully unchanged.
        labels = batched_image_op_wrapper(
            depth,
            segment_depth_felzenszwalb_rag,
            depth_merge_thresh=depth_merge_thresh,
            conf_map=conf_map,
            top_conf_percentile=top_conf_percentile,
        )

    elif segment_mode == "geometry":
        # 中文：新增 geometry 分支，只替换 labels 的来源。
        # English: New geometry branch; only replace the source of labels.
        labels_list = []

        # 中文：逐帧做 geometry segmentation。第一版用 for-loop，方便处理 intrinsic / point_map。
        # English: Run geometry segmentation frame by frame. The first version uses a for-loop
        # for easier handling of intrinsics and pointmaps.
        for i in range(depth.shape[0]):
            # 中文：取当前帧深度。
            # English: Get current-frame depth.
            depth_i = depth[i]

            # 中文：取当前帧置信度。
            # English: Get current-frame confidence.
            conf_i = conf_map[i] if conf_map is not None else None

            # 中文：如果提供了 point_map，则取当前帧 point_map。
            # English: If point_map is provided, get current-frame point_map.
            point_i = point_map[i] if point_map is not None else None

            # 中文：处理 intrinsic，支持 [N, 3, 3] 或 [3, 3]。
            # English: Handle intrinsics, supporting [N, 3, 3] or [3, 3].
            if intrinsic is not None:
                intrinsic_i = intrinsic[i] if intrinsic.ndim == 3 else intrinsic
            else:
                intrinsic_i = None

            # 中文：生成当前帧 geometry labels。
            # English: Generate geometry labels for the current frame.
            labels_i = segment_geometry_felzenszwalb_rag(
                depth_i,
                conf_map=conf_i,
                intrinsic=intrinsic_i,
                point_map=point_i,
                top_conf_percentile=top_conf_percentile,
                depth_merge_thresh=depth_merge_thresh,
                normal_method=normal_method,
            )

            labels_list.append(labels_i)

        labels = np.stack(labels_list, axis=0)

    else:
        raise ValueError(f"Unknown segment_mode: {segment_mode}")

    # 中文：从 labels 构建 segment graph，这一步继续复用 LASER 原始逻辑。
    # English: Build segment graph from labels, reusing the original LASER logic.
    sp_graph = match_segmentation_seq(labels, iou_thresh=corr_iou_thresh)

    return sp_graph
