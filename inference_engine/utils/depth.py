# Depth-specific segmentation utilities.
# Responsibilities:
# 1. Build depth-only segmentation labels from a depth map.
# 2. Keep original LASER depth segmentation behavior isolated from geometry and scale-anchor experiments.
# 中文职责：
# 1. 只负责 depth-only segmentation，也就是原始 LASER 的深度分割路径。
# 2. 根据 depth map 生成 depth-based segmentation labels。
# 3. 不负责 geometry-aware segmentation、segment graph 连边、overlap scale anchor 或 M1 实验。
import numpy as np
from skimage.segmentation import felzenszwalb

from ._segmentation_cy import merge_regions
from .fast_seg import fast_graph_segmentation


# 1️⃣ 原始 LASER 的 depth segmentation 分支之一：深度分割
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


# 2️⃣ depth-based segmentation 的快速实现：跳过 felzenszwalb + merge_regions，直接使用 graph segmentation。
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
