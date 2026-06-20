# Geometry-Aware LASER M0 中文说明

## 1. M0 的目标

M0 的目标不是重写 LASER，而是做一个最小、可控、能做实验对比的第一版：

```text
只替换 segment labels 的来源。
不改 LASER 原来的 scale estimation。
不改 LASER 原来的 scale propagation。
不改 loop closure。
```

原始 LASER 的流程是：

```text
model prediction
-> depth map
-> depth-based segmentation
-> make_sp_graph
-> LASER segment-wise scale estimation
-> LASER scale propagation
-> refined local_points / depth
```

M0 geometry 模式的流程是：

```text
model prediction
-> depth + confidence + camera / point map
-> geometry.py 提取几何信息
-> geometry-aware segmentation
-> make_sp_graph
-> LASER segment-wise scale estimation
-> LASER scale propagation
-> refined local_points / depth
```

所以 M0 的核心原则是：

```text
depth layer -> geometry segment
```

后面的 LASER 对齐、尺度估计、尺度传播先完整继承。

## 2. M0 改了哪些文件

### `inference_engine/utils/geometry.py`

负责从 depth / point map 中提取几何信息：

- `depth_to_local_points_np`
- `compute_normals_cross_np`
- `compute_normals_sobel_np`
- `compute_depth_edge_np`
- `compute_normal_edge_np`
- `build_geometry_info_np`

这个文件只做几何特征，不做分割、不做尺度估计、不做 loop closure。

### `inference_engine/utils/depth.py`

新增 geometry-aware segmentation：

- 统计每个 segment 的区域几何描述子
- 判断相邻区域是否应该合并
- 用 union-find 合并相邻几何一致的区域
- 生成 `segment_geometry_felzenszwalb_rag`

第一版使用：

```text
depth + normal -> Felzenszwalb over-segmentation
-> mean depth + mean normal + confidence 判断相邻区域是否合并
```

### `inference_engine/utils/lsa.py`

`make_sp_graph` 增加：

```python
segment_mode="depth" | "geometry"
normal_method="cross" | "sobel"
```

`depth` 模式保持原始 LASER。

`geometry` 模式只替换 labels 的来源，然后仍然调用原来的：

```text
match_segmentation_seq
assign_overlap_window_depth_scale
align_adjacent_windows_depth_segments
refine_depth_segments
```

### `demo.py` / `demo_lc.py`

新增命令行参数：

```bash
--segment_mode depth
--segment_mode geometry
--normal_method cross
--normal_method sobel
```

当前建议先用：

```bash
--segment_mode geometry --normal_method cross
```

## 3. `cross` 方法到底做了什么

`cross` 指的是用局部 3D 点的横向、纵向差分做叉乘，估计每个像素的表面法向。

输入是局部点图：

```text
points: [H, W, 3]
```

也就是说每个像素都有一个相机坐标系下的 3D 点：

```text
P(x, y) = [X, Y, Z]
```

如果模型已经输出了 `local_points`，就直接用它。

如果没有 point map，就用 depth + intrinsic 反投影：

```text
X = (u - cx) / fx * depth
Y = (v - cy) / fy * depth
Z = depth
```

### 第一步：取横向切向量

对每个内部像素，取左右两个 3D 点的差：

```text
dx = P(x + 1, y) - P(x - 1, y)
```

代码里对应：

```python
dx = points[:, 2:, :] - points[:, :-2, :]
```

它表示这个像素附近沿图像横向的 3D 几何变化。

### 第二步：取纵向切向量

取上下两个 3D 点的差：

```text
dy = P(x, y + 1) - P(x, y - 1)
```

代码里对应：

```python
dy = points[2:, :, :] - points[:-2, :, :]
```

它表示这个像素附近沿图像纵向的 3D 几何变化。

### 第三步：叉乘得到法向

局部表面可以近似成一个小平面。

这个小平面上有两个方向：

```text
dx: 横向切线
dy: 纵向切线
```

那么这个平面的法向就是：

```text
normal = dx x dy
```

代码里对应：

```python
normals_inner = np.cross(dx, dy)
```

### 第四步：归一化

叉乘结果的长度会受深度尺度、像素间距影响，所以要变成单位向量：

```text
normal = normal / ||normal||
```

代码里对应：

```python
normals_inner = normals_inner / (
    np.linalg.norm(normals_inner, axis=-1, keepdims=True) + eps
)
```

归一化后，我们更关注方向，而不是向量长度。

## 4. `cross` normal 对 segmentation 有什么用

原始 LASER 的 depth segmentation 主要看深度层。

这样会有一个问题：

```text
两个区域深度接近，但几何朝向不同，可能被错误合并。
两个区域深度变化明显，但其实属于同一个倾斜平面，也可能被切得太碎。
```

加入 normal 后，M0 可以同时参考：

```text
1. mean depth 是否接近
2. mean normal 是否接近
3. confidence 是否可靠
```

对于桌面、墙面、地面这类结构，normal 能提供比单纯 depth 更强的几何约束。

## 5. 当前 desk smoke test 结果

你在云端已经跑了两组 desk：

### depth baseline

命令参数：

```text
--segment_mode depth
--depth_refine
--sample_interval 5
--window_size 20
--overlap 5
```

终端结果：

```text
Found 123 images.
Total Windows: 8
Inference sec: 12.20
Peak GPU memory usage: 6.46 GB
Steady State Avg: 1938.41 ms
```

### geometry + cross

命令参数：

```text
--segment_mode geometry
--normal_method cross
--depth_refine
--sample_interval 5
--window_size 20
--overlap 5
```

终端结果：

```text
Found 123 images.
Total Windows: 8
Inference sec: 107.53
Peak GPU memory usage: 6.46 GB
Steady State Avg: 14118.49 ms
```

结论：

```text
两组都跑通了。
显存基本一致，约 6.46 GB。
geometry + cross 明显更慢，主要慢在逐帧 geometry segmentation 和 region merge。
```

终端里出现的 warning：

```text
Got image with third dimension of 4.
This image will be interpreted as a multichannel 2d image.
```

这是因为 geometry segmentation 输入给 Felzenszwalb 的图像是：

```text
[depth, normal_x, normal_y, normal_z]
```

也就是 4 通道。这正是 M0 设计里想要的 multichannel 几何输入，不是运行失败。

## 6. 下一步建议

第一阶段建议按这个顺序看结果：

```text
1. 先看 outputs/viser/desk_depth
2. 再看 outputs/viser/desk_geometry
3. 对比相机轨迹是否更稳定
4. 对比重建点云是否减少局部尺度错位
5. 再用 quick_vis_geometry.py 抽单帧看 normal / segment 是否合理
```

如果 geometry 结果质量没有明显改善，下一步优先调：

```text
seg_scale
seg_sigma
seg_min_size
depth_merge_thresh
normal_thresh_deg
```

如果质量有改善但太慢，下一步再优化：

```text
1. 减少逐帧 Python 循环
2. 缓存 geometry_info
3. 降低 segmentation 分辨率
4. 并行化 geometry segmentation
```
