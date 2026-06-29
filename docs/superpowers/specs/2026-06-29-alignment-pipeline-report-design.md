# Depth 与 Geometry 对齐流水线可视化设计

## 1. 目标

为 LASER baseline 的 depth 分割路径和新增的 geometry 分割路径生成一份可纵向滚动的静态 HTML 报告，逐窗口、逐帧对照两种方法的真实处理过程。

每个目标窗口帧占一行，一行固定显示十张图：

| Baseline / Depth | Baseline / Depth | Baseline / Depth | Baseline / Depth | Baseline / Depth | Geometry | Geometry | Geometry | Geometry | Geometry |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 置信度筛选 | 初始分割 | 融合后分割 | overlap 锚点 | 窗口内传播 | 置信度筛选 | 初始分割 | 融合后分割 | overlap 锚点 | 窗口内传播 |

该报告的首要用途是解释两种分割方法如何改变 segment 数量、IoU 对应、尺度锚点和传播路径，而不是只比较最终 ATE。

## 2. 范围

本次包含：

- 非回环 `StreamingWindowEngine`。
- `segment_mode=depth` 与 `segment_mode=geometry` 的并排对照。
- 原始 `scale_anchor_mode=depth_irls`。
- 可调整的 `--top_conf_percentile` 运行参数。
- 初始分割、融合后分割、overlap IoU 锚点和窗口内传播的真实运行数据。
- 完整处理序列，默认不跳过任何已采样帧。
- 静态 HTML 报告、懒加载缩略图和点击放大。

本次不包含：

- M1 `conf_weighted_irls` 的效果分析或算法修改。
- loop closure 路径。
- 新的分割、融合、IoU、IRLS 或传播策略。
- 3D Viser 场景替代。
- 在 HTML 中拖动阈值并把结果伪装成本次真实运行结果。

## 3. 当前算法事实

报告必须如实表达下列现有行为：

1. `--top_conf_percentile 0.3` 表示保留置信度最高的 30%，引擎内部使用的 quantile 为 `0.7`。
2. 相邻窗口 Sim(3) 粗配准使用 source 和 target 都通过阈值的 mutual confidence mask。
3. depth 和 geometry 分割均在整张图上执行。高置信区域目前用于估计 merge depth range，并没有删除低置信像素。
4. 默认 `depth_irls` 尺度锚点在匹配 segment 的完整像素交集上估计，不只使用高置信像素。
5. overlap segment 对应由 IoU 阈值筛选；当前 refinement 默认阈值为 `0.4`。
6. target 窗口内部的 segment graph 使用相邻帧 IoU 建边，当前默认阈值为 `0.3`。
7. 没有直接锚点且没有收到传播尺度的 segment 保持 scale `1.0`。

可视化不能把第 2、3、4 点混为一个“高置信像素专属对齐”步骤。

## 4. 置信度参数

置信度保留比例不能硬编码为 30%。

- 运行入口继续使用 `--top_conf_percentile`。
- 默认值保持 `0.3`，即保留最高 30%。
- 参数必须允许用户在每次运行时调整，例如 `0.2` 或 `0.4`。
- 调试 metadata 同时记录：
  - 用户输入的保留比例。
  - 引擎使用的 quantile。
  - 每帧分割 merge-threshold 计算对应的数值阈值。
  - overlap 注册使用的 source/target 数值阈值。
- baseline 与 geometry 报告输入必须使用相同的保留比例、窗口大小、overlap 和采样间隔。不同则报告生成器直接报错，避免不公平对照。

HTML 只展示记录数据对应的运行阈值。若要改变真实阈值，应以新的 `--top_conf_percentile` 重新运行两种方法。

## 5. 五阶段图像语义

### 5.1 置信度筛选

- 以 RGB 为底图。
- 高置信区域保留正常亮度，未通过阈值的区域压暗。
- 显示本帧阈值、实际保留像素比例和运行配置中的目标保留比例。
- 对 overlap 帧额外标出实际 mutual confidence mask，说明它参与 Sim(3) 粗配准。
- 对非 overlap 帧注明该高置信区域只用于本帧 merge depth range 估计，不参与相邻窗口 Sim(3)。

### 5.2 初始分割

- 展示 Felzenszwalb 的原始 labels，尚未执行区域融合。
- depth 路径的输入特征是 depth。
- geometry 路径的输入特征是归一化 depth 与 normal。
- 高置信区域保持明亮，低置信区域压暗，但全图分割边界仍以弱线显示。
- 最终能够形成有效 overlap 锚点的 merged segment 所覆盖的初始区域使用强调线，其余区域使用弱线。
- 显示初始 segment 数量。

### 5.3 融合后分割

- 展示真实用于构建 segment graph 的最终 labels。
- depth 路径显示 depth merge 后的区域。
- geometry 路径显示 depth 与 normal 规则 merge 后的区域。
- 高置信区域保持明亮，低置信区域压暗，但不暗示低置信像素已被删除。
- 有直接 overlap anchor 的 target segment 使用强调边界。
- 显示融合后 segment 数量和融合比例。

### 5.4 Overlap 锚点

- 单张结果图内部左右分屏：左侧为上一窗口 source，右侧为当前窗口 target。
- 两侧均使用融合后 segment labels。
- source 边界使用青色，target 边界使用品红色。
- 只显示通过实际 IoU 阈值的匹配边。
- 连线透明度或线宽反映 IoU 大小。
- 图中标注 source segment ID、target segment ID、IoU 和 `depth_irls` scale。
- 多对多匹配必须保留，不能只画每个 target 的最佳匹配。
- 无匹配时明确显示 `No accepted IoU match`。

缩略图可以简化文字，但点击放大后必须展示全部匹配线和匹配表。

### 5.5 窗口内传播

- merged segment 以最终 scale 着色：蓝色 `< 1`、白色 `= 1`、红色 `> 1`。
- 每个 segment 标记来源类型：
  - `A`：overlap 中直接获得 anchor。
  - `P`：没有直接 anchor，但从 target 窗口前序帧传播获得尺度。
  - `I`：没有可靠尺度来源，保持 `1.0`。
- 当前帧与前一 target 帧采用小型双帧构图，绘制实际参与传播的时序 IoU 边。
- 传播边标注 IoU；点击放大后显示 parent frame、parent segment、child frame、child segment、edge IoU 和传播 scale。
- 第一帧没有前序传播边时只显示 anchor/identity 状态。

## 6. 页面布局

采用已确认的方案 A：十图连续时间线。

- 页面顶部显示场景、窗口大小、overlap、采样间隔、置信度保留比例、quantile、IoU 阈值和 scale anchor mode。
- 两级 sticky header：上层为 `BASELINE / DEPTH` 与 `GEOMETRY`，下层为五个阶段名。
- 每个 target-window frame 对应一行十张缩略图。
- 窄屏只允许该行横向滚动，不破坏页面纵向时间顺序。
- 点击任意图打开大图 modal，并显示该图相关的数值详情。
- 图片使用 `loading="lazy"`，避免一次载入完整序列。
- overlap 行使用浅色背景和 `O` 标记；非 overlap 行使用 `N` 标记。
- 每个窗口形成独立 section，显示 target window 的全局帧范围。
- overlap 帧会在相邻窗口 section 中有意重复，因为同一全局帧在 source 和 target 窗口中的预测、分割和对齐角色不同。
- 第一个窗口没有前一窗口：第 4 图显示 `First window / no overlap alignment`，第 5 图显示当前 identity 状态。

## 7. 输出结构

报告使用一个入口文件和独立资源目录：

```text
pipeline_report/
├── index.html
├── data.json
└── assets/
    ├── window_0000/
    │   ├── frame_0000_depth_conf.webp
    │   ├── frame_0000_depth_initial.webp
    │   ├── ...
    │   └── frame_0000_geometry_propagation.webp
    └── window_0001/
        └── ...
```

`index.html` 内嵌页面所需的轻量 manifest，因此可直接打开。`data.json` 保留同一份结构化信息，方便后续分析和重新构建页面。大图资源保持外置，避免单个 HTML 文件过大。

## 8. 运行时记录架构

### 8.1 原则

- 调试关闭时不保存新数据，不改变原有计算路径和输出。
- 调试记录只读取已经产生的 labels、graph、cache 和 scale，不重新决定算法结果。
- 现有 `pair_XXXX.npz` 与 Viser 调试格式保持兼容。
- 新的完整流水线数据放在 scene 调试目录的 `pipeline/` 子目录。

### 8.2 分割阶段接口

depth 和 geometry 单帧分割各自增加一个显式 stages 接口，返回：

```text
SegmentationStages
├── initial_labels
├── merged_labels
├── confidence_threshold
└── high_confidence_mask
```

原有公共分割函数仍只返回 `merged_labels`，内部复用 stages 接口，确保原调用方和 baseline 行为不变。

batch helper 只负责保持帧顺序，不理解 depth 或 geometry 语义。

### 8.3 尺度锚点与传播记录

`refine_segment_scales` 增加一个默认关闭的 trace sink：

1. overlap scale assignment 完成后，快照记录 target segment 的直接 anchor cache。
2. 在现有 `_propagate_scale_cache` 回调执行时，额外记录 parent/child vertex 位置、edge IoU 和传播 scale。
3. 传播结束后按以下规则分类：
   - 直接 anchor 快照中存在 scale：`A`。
   - 快照中不存在、传播后 cache 存在 scale：`P`。
   - 传播后仍无 scale：`I`。
4. trace sink 不参与尺度加权，也不修改传播顺序。

### 8.4 每窗口 trace schema

每种方法分别写入：

```text
outputs/debug_alignment/<scene>/pipeline/
├── meta.json
├── window_0000.npz
├── window_0001.npz
└── ...
```

`meta.json` 至少包含：

- schema version
- scene name
- segment mode
- normal method
- scale anchor mode
- window size
- overlap
- sample interval（由 `demo.py` 在运行时传给 recorder 并写入）
- confidence retained fraction
- confidence quantile
- graph IoU threshold
- overlap anchor IoU threshold

`window_XXXX.npz` 使用无 pickle 的数值数组，至少包含：

- global frame indices
- per-frame confidence thresholds
- per-frame high-confidence masks
- initial labels
- merged labels
- overlap mutual confidence masks
- accepted match frame/source ID/target ID/IoU/scale arrays
- segment frame/ID/final scale/source-role arrays
- propagation parent frame/segment/child frame/segment/IoU/scale arrays

labels 优先保存为 `uint16`；若 label 数超过范围则保存为 `uint32`。文件使用压缩 NPZ，不重复保存 RGB。

## 9. 离线报告生成器

新增独立命令：

```bash
python eval/build_alignment_pipeline_report.py \
  --baseline_debug_dir outputs/debug_alignment/<depth_scene> \
  --geometry_debug_dir outputs/debug_alignment/<geometry_scene> \
  --image_dir data/08/image_2 \
  --sample_interval 10 \
  --out_dir outputs/pipeline_report/kitti08_s10
```

职责：

1. 读取并验证两套 metadata。
2. 使用项目现有的自然排序规则和 metadata 中的 sample interval 得到采样 RGB 路径；若命令行显式传入 `--sample_interval`，其值必须与 metadata 一致。
3. 按窗口编号和 local frame 恢复全局帧映射。
4. 渲染十种 WebP 资源。
5. 生成 `data.json` 和 `index.html`。

默认生成全部已记录窗口和帧。可以提供可选的窗口范围和 frame step 仅用于快速预览，但默认值不得跳帧。

## 10. 校验与错误处理

报告生成前必须检查：

- 两个 debug 目录都存在完整 `pipeline/meta.json`。
- segment mode 分别为 depth 和 geometry。
- scale anchor mode 均为 `depth_irls`；否则明确警告当前报告超出本次设计范围。
- window size、overlap、置信度保留比例、quantile、IoU 阈值和采样间隔一致。
- 窗口编号与每窗口 frame 数一致。
- RGB 数量足以覆盖最大的 global frame index。
- label、confidence mask 和 RGB 空间尺寸兼容。

输入不一致时应在生成前失败并报告具体字段，不能生成表面上可比较但配置不同的页面。

单帧允许出现以下正常空状态：

- 无有效 overlap match。
- 无直接 anchor。
- 无传播尺度，全部保持 identity。
- 第一窗口没有 source window。

这些状态应显示说明图，而不是缺图或抛出异常。

## 11. 测试策略

### 11.1 分割阶段

- depth stages 的 `merged_labels` 与原 depth 分割函数结果完全一致。
- geometry stages 的 `merged_labels` 与原 geometry 分割函数结果完全一致。
- 参数改变时 high-confidence mask 和记录阈值随之变化，不固定为 30%。
- batch stages 保持帧顺序。

### 11.2 锚点与传播

- 合成两帧 graph 验证 direct anchor 被标记为 `A`。
- 无 direct anchor、由父节点传播的 segment 被标记为 `P`。
- 无 cache 的 segment 被标记为 `I` 且 scale 为 `1.0`。
- trace sink 关闭前后输出 scale mask 数值完全一致。
- 多对多 overlap 匹配均被记录。

### 11.3 记录器

- 调试关闭时 recorder 为 no-op。
- window NPZ 不需要 `allow_pickle=True` 即可读取。
- metadata 正确保存用户输入 retained fraction 和内部 quantile。
- 原有 pair trace 文件与 viewer 测试继续通过。

### 11.4 报告生成

- 合成两个小型 debug 目录，生成两帧报告。
- 每一行 DOM 严格包含十张阶段图。
- stage 顺序与方法分组固定。
- overlap 与 non-overlap 行标记正确。
- 第一窗口、无 match 和 identity 状态都有资源。
- baseline/geometry 参数不一致时生成器拒绝运行。
- 所有 HTML 引用资源存在。

### 11.5 回归

- 运行现有 streaming engine、depth、geometry、LSA、alignment debug 和 CLI 测试。
- 调试关闭时 baseline 与 geometry 的数值路径不发生变化。

## 12. 实施边界

本功能只增加观测能力。任何在可视化中发现的算法问题，例如 geometry 过分割、IoU 边不稳定、尺度 anchor 异常或传播扩散错误，都应在后续独立实验中修改并单独评估，不能夹带在本次报告实现中。
