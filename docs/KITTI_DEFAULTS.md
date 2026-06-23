# KITTI 默认参数记录

本文档记录上游 `neu-vi/LASER` 源码中与 KITTI 相关的默认运行参数，供后续批量实验对齐使用。

## 信息来源

- 论文链接：`http://arxiv.org/abs/2512.13680`
- 项目主页：`https://neu-vi.github.io/LASER/`
- 上游仓库：`https://github.com/neu-vi/LASER/tree/main`
- README 中的 KITTI evaluation 命令
- `eval_launch.py` 中 `streaming_pi3` / `streaming_pi3_lc` 的 engine 构造参数
- `eval/eval_metadata.py` 中 `kitti` / `kitti_odometry` 的数据路径和序列设置

## 参数总表

| 场景 | 上游 model | 数据集参数 | window_size | overlap | top_conf_percentile | 其他关键参数 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| KITTI video depth / cropped depth selection | `streaming_pi3` | `kitti` | 20 | 5 | 0.5 | `--no_crop`, `--flow_loss_weight 0`, `--translation_weight 1e-3` |
| KITTI Odometry 非 LC | `streaming_pi3` | `kitti_odometry` | 20 | 5 | 0.5 | `pose_eval_stride=1`, `depth_refine=True` |
| KITTI Odometry LC | `streaming_pi3_lc` | `kitti_odometry` | 75 | 30 | 0.3 | 使用 `configs/loop_config.yaml` |
| README demo 示例 | `demo.py` | 任意图像目录 | 30 | 10 | 0.3 | `--sample_interval 1`, `--depth_refine` |

说明：

- `streaming_pi3` 的 `window_size=20, overlap=5, top_conf_percentile=0.5` 写在 `eval_launch.py`。
- `streaming_pi3_lc` 的 `window_size=75, overlap=30, top_conf_percentile=0.3` 写在 `eval_launch.py`。
- `demo.py` 示例中的 `window_size=30, overlap=10` 是 README 的可视化 demo 示例，不是 evaluation 默认值。
- `depth_refine=True` 是 `StreamingWindowEngine` / `StreamingWindowEngineLC` 构造函数默认值。显式命令行 demo 需要加 `--depth_refine`。

## 上游 KITTI 命令

README 中公开的 KITTI video depth 命令：

```bash
export PYTHONPATH="./":$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 eval_launch.py \
  --mode=eval_pose \
  --model=streaming_pi3 \
  --eval_dataset=kitti \
  --output_dir="outputs/video_depth/kitti_depth" \
  --no_crop \
  --flow_loss_weight 0 \
  --translation_weight 1e-3
```

对应 depth metric：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 depth_metric.py \
  --eval_dataset=kitti \
  --result_dir="outputs/video_depth/kitti_depth" \
  --output_dir="outputs/video_depth"
```

README 中注释掉的 KITTI Odometry LC 命令：

```bash
export PYTHONPATH="./":$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12345 eval_launch.py \
  --mode=eval_pose \
  --model=streaming_pi3_lc \
  --eval_dataset=kitti_odometry \
  --output_dir="outputs/cam_pose/kitti_odometry_pose"
```

## KITTI Odometry 数据路径

`eval/eval_metadata.py` 中默认路径：

```text
data/KITTI_Odometry/dataset/sequences/{seq}/image_2
data/KITTI_Odometry/dataset/poses/{seq}.txt
```

默认序列：

```text
00, 01, 02, 03, 04, 05, 06, 07, 08, 09, 10
```

## 当前实验建议

为了对比 `depth` 与 `geometry`，建议非 LC 先固定为上游 evaluation 默认窗口：

```text
window_size=20
overlap=5
sample_interval=1
top_conf_percentile=0.5
```

如果用 `demo.py` / `demo_vggt.py` 跑单序列可视化与 ATE，建议实验名包含：

```text
{seq}_{model}_{segment_mode}_s1_w20_o5
```

例如：

```text
kitti09_pi3_depth_s1_w20_o5
kitti09_pi3_geometry_s1_w20_o5
kitti09_vggt_depth_s1_w20_o5
kitti09_vggt_geometry_s1_w20_o5
```

LC 对比建议固定为：

```text
window_size=75
overlap=30
sample_interval=1
top_conf_percentile=0.3
```

实验名示例：

```text
kitti09_pi3_depth_lc_s1_w75_o30
kitti09_pi3_geometry_lc_s1_w75_o30
kitti09_vggt_depth_lc_s1_w75_o30
kitti09_vggt_geometry_lc_s1_w75_o30
```
