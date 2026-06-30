# 分割深度与双方法详情对照设计

## 1. 目标

扩展现有 LASER depth/geometry 流水线报告，使用户能够同时观察：

1. 实际送入分割流程的深度图。
2. depth 或 geometry 初始分割与该深度图的对应关系。
3. 区域融合后分割与该深度图的对应关系。
4. 点击任意阶段后，Baseline 与 Geometry 的同帧、同阶段并排对照。

现有报告的一行十张卡片结构保持不变。

## 2. 范围

本次包含：

- 非回环 `StreamingWindowEngine` 的 debug pipeline trace。
- depth 与 geometry 两种分割模式。
- 每窗口、每帧真实 segmentation input depth 的保存。
- 初始分割和融合后分割的“深度图 + 分割图”组合卡片。
- 所有五个阶段的 Baseline/Geometry 双方法详情弹窗。
- 两种方法共享的逐帧深度色标。
- 对旧 trace 的明确版本错误。

本次不包含：

- 修改模型、分割、merge、IoU、IRLS 或传播算法。
- 使用最终 refine 后 depth 替代分割输入 depth。
- 额外增加独立 depth 阶段卡片。
- 将每行十张图扩展为十二张图。
- loop closure 路径。

## 3. 深度数据语义

新增数组命名为 `segmentation_depths`。

它必须等于构建当前窗口 segment graph 时传入的：

```python
local_points_np[..., -1]
```

具体时点为：

1. 模型完成当前窗口预测。
2. 使用固定参考内参重新反投影。
3. 对后续窗口完成相邻窗口 coarse Sim3 尺度修正。
4. 尚未执行 `refine_segment_scales`。
5. depth/geometry graph builder 使用该 depth 进行初始分割和 merge。

因此页面必须标注为“分割输入深度（refine 前）”，不能笼统标成“最终深度”。

第一窗口没有前一窗口 Sim3，但仍保存实际送入 graph builder 的 depth。

## 4. Trace Schema

pipeline trace schema 从 v1 升级为 v2。

每个 `window_XXXX.npz` 新增：

```text
segmentation_depths: float32 [F, H, W]
```

要求：

- dtype 固定为 `float32`，保存实际数值，不提前归一化。
- shape 必须与 `initial_labels`、`merged_labels` 和 `high_confidence_masks` 一致。
- depth 与 geometry 分支分别保存自身运行时的输入 depth。
- 只在 pipeline debug 启用时创建，不增加普通推理输出字段。
- `AlignmentDebugRecorder` 使用现有 compressed NPZ 保存，不改变 labels 的紧凑整数存储。

metadata 中 `schema_version` 更新为 `2`。

报告生成器要求两种方法均为 schema v2。旧 v1 trace 缺少 `segmentation_depths` 时直接报错：

```text
Pipeline trace v2 with segmentation_depths is required; rerun both methods.
```

不允许从 `outputs/viser/frame_XXXX.npy` 回填，因为该文件是聚合后的最终结果，与实际分割输入时点不同。

## 5. 深度可视化

### 5.1 色标

对每个 window/local frame，同时读取：

- Baseline segmentation depth。
- Geometry segmentation depth。

将两者有效像素合并，计算共享的 `p02` 与 `p98`：

```text
shared_min = percentile(combined_valid_depth, 2)
shared_max = percentile(combined_valid_depth, 98)
```

两种方法使用完全相同的范围归一化和同一个 Turbo colormap。

这样颜色差异来自 depth 数值差异，而不是各自独立归一化。无有限值时生成全黑 depth 图并在详情中标记；范围退化时使用常量色并保存真实数值范围。

详情数据包含：

- depth min
- depth p02
- depth p50
- depth p98
- depth max
- shared color minimum
- shared color maximum

### 5.2 初始分割卡片

原来的单幅初始分割图改为一张横向组合图：

```text
[ 分割输入深度热力图 | 初始分割边界图 ]
```

右侧保持现有逻辑：

- 全图初始边界可见。
- 高置信区域明亮。
- 能形成直接 overlap anchor 的 merged segment 所覆盖的初始区域使用强调边界。

### 5.3 融合后分割卡片

同样改为：

```text
[ 分割输入深度热力图 | 融合后分割边界图 ]
```

右侧继续显示直接 anchor 强调、segment 数和 merge ratio。

### 5.4 其他阶段

置信度、overlap 锚点和窗口内传播卡片保持现有图像语义。它们在详情弹窗中参与双方法并排对照，但不额外嵌入 depth 子图。

## 6. 一行十卡片布局

阶段顺序保持：

```text
Depth:    confidence | initial | merged | overlap | propagation
Geometry: confidence | initial | merged | overlap | propagation
```

初始和融合卡片内部变宽为双子图，但仍各自算一张卡片。页面依旧每行十张卡片，窄屏继续使用行内横向滚动。

## 7. 双方法详情弹窗

点击任意 stage card 后，不再只显示被点击的方法，而是根据：

- row index
- stage name

查找同一行中 depth 与 geometry 的同阶段资源。

弹窗布局：

```text
┌──────────────── Baseline / Depth ────────────────┬──────────────── Geometry ────────────────┐
│ 完整尺寸阶段图                                   │ 完整尺寸阶段图                           │
│ 该方法的结构化详情                               │ 该方法的结构化详情                       │
└──────────────────────────────────────────────────┴──────────────────────────────────────────┘
```

规则：

- 桌面端左右并排。
- 窄屏自动改为上下排列。
- 两侧图片使用 `object-fit: contain`，不裁剪。
- 标题显示 stage、全局帧、窗口编号和 local frame。
- 两侧详情分别显示，不混成一份 JSON。
- 点击 depth 或 geometry 卡片得到完全相同的双方法弹窗。
- 关闭行为和原 modal 一致。

初始和融合阶段的每一侧本身是“depth heatmap + segmentation”组合，因此详情弹窗整体形成四个语义子图，但仍保持两列方法对照。

## 8. 组件边界

### 8.1 运行时记录

`build_depth_sp_graph` 和 `build_geometry_sp_graph` 已经收到用于分割的 depth。它们在 `segmentation_trace` 非空时增加：

```python
segmentation_trace["segmentation_depths"] = np.asarray(depth, dtype=np.float32).copy()
```

正常路径不复制 depth。

### 8.2 Recorder

`AlignmentDebugRecorder.record_pipeline_window` 继续接受纯数值 payload，并把 `segmentation_depths` 原样保存为 float32。

### 8.3 报告渲染器

新增独立纯函数职责：

- 计算两个 depth map 的共享显示范围。
- 将一个 depth map 按指定范围着色。
- 将 depth heatmap 与 segmentation 图横向拼接。

现有 `render_segmentation_stage` 继续只负责分割图。组合由上层 renderer 完成，避免把深度归一化逻辑耦合进边界绘制。

### 8.4 HTML

manifest 保持每行十个 stage entry。每个 initial/merged entry 的 details 增加 depth statistics。JavaScript 按 stage name 查找配对 entry，并填充两个 modal panel。

## 9. 校验与错误处理

报告生成前新增检查：

- 两种方法的 `schema_version == 2`。
- 每个窗口都包含 `segmentation_depths`。
- depth shape 与 labels shape 一致。
- Baseline 与 Geometry 当前帧 depth shape 一致。
- shared range 只使用有限值。

任何一项不满足都在渲染资源前失败，避免生成部分更新、部分旧格式的报告。

旧报告目录可以被新的完整生成过程覆盖。生成器仍需确保所有 manifest asset 都存在。

## 10. 测试策略

### 10.1 Trace

- depth graph builder 在 debug trace 中保存输入 depth 的 float32 副本。
- geometry graph builder保存输入 depth 的 float32 副本。
- 修改原 depth 数组后，trace 中的 depth 不随之变化。
- 普通 graph builder 路径不创建 trace 数组。
- window NPZ 可在 `allow_pickle=False` 下读出 depth。

### 10.2 色标与组合图

- 两张范围不同的 depth map 使用同一 shared p02/p98。
- 同一个 depth 值在两种方法中得到相同颜色。
- NaN/Inf 不参与 percentile。
- initial/merged 组合图宽度为单幅分割图的两倍，高度不变。
- depth details 保存实际统计值与共享范围。

### 10.3 HTML

- 每行仍严格包含十个 stage card。
- modal 包含 Baseline 和 Geometry 两个 panel。
- JavaScript 使用 stage name 配对，而不是依赖脆弱的固定相邻下标。
- 点击任意方法的同一 stage 使用同一配对结果。
- 窄屏 CSS 将 modal 两列改为一列。

### 10.4 兼容与回归

- v1 trace 给出明确 rerun 错误。
- v2 合成 trace 能生成完整报告。
- 所有已有 segmentation、LSA、streaming、debug 与报告测试继续通过。

## 11. 重新运行要求

当前已下载的 `0cba9c2` trace 为 v1，不包含真实 segmentation input depth。实现完成并推送后，必须在云端重新运行 baseline 与 geometry，再生成并下载新报告。

这是保证深度图与分割阶段严格对应所必需的，不使用近似回填绕过。
