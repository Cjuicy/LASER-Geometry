# LASER 可视化运行命令

本文档记录原版 Viser、完整双轨迹对比，以及带 Ground Truth 和 ATE 的
KITTI 09 长轨迹对比命令。

## 1. 环境准备

```bash
cd ~/autodl-tmp/LASER-Geometry-main
conda activate vggt-dem
export PYTHONPATH="./:${PYTHONPATH}"
```

本地仓库路径：

```bash
cd /Users/cjuicy/WorkSpace/py-workspace/LASER/LASER-Geometry
conda activate vggt-dem
export PYTHONPATH="./:${PYTHONPATH}"
```

## 2. 原版单结果 Viser

查看 LASER baseline：

```bash
python viser/visualizer_monst3r.py \
  --data outputs/viser/kitti09_pi3_depth_s10_w30_o10_world_debug \
  --conf_thre 0.1 \
  --fg_conf_thre 0.0 \
  --point_size 0.001
```

查看 LASER-Geometry：

```bash
python viser/visualizer_monst3r.py \
  --data outputs/viser/kitti09_pi3_geometry_s10_w30_o10_world_debug \
  --conf_thre 0.1 \
  --fg_conf_thre 0.0 \
  --point_size 0.001
```

原版脚本使用 Viser 默认端口，通常为 `8080`。它内部默认最多读取 100
帧，因此适合查看单帧点云和原有播放效果，不适合观察完整 KITTI 09
轨迹。

## 3. 完整双轨迹对比（不使用真值）

```bash
python eval/vis_full_trajectory_compare.py \
  --baseline_dir outputs/viser/kitti09_pi3_depth_s10_w30_o10_world_debug \
  --geometry_dir outputs/viser/kitti09_pi3_geometry_s10_w30_o10_world_debug \
  --port 8100 \
  --frame_stride 1 \
  --pixel_stride 6 \
  --detail_pixel_stride 2 \
  --max_points 200000 \
  --conf_quantile 0.7
```

访问：`http://127.0.0.1:8100/`

该模式不读取真值。LASER 保持保存坐标，LASER-Geometry 仅通过首帧刚体
变换进入 LASER 坐标系。

## 4. KITTI 09 长轨迹：GT、Baseline、Geometry

这是当前推荐的 KITTI 09 对比命令：

```bash
python eval/vis_full_trajectory_compare.py \
  --baseline_dir outputs/viser/kitti09_pi3_depth_s10_w30_o10_world_debug \
  --geometry_dir outputs/viser/kitti09_pi3_geometry_s10_w30_o10_world_debug \
  --gt_traj data/dataset/poses/09.txt \
  --gt_format kitti \
  --gt_stride 10 \
  --port 8101 \
  --frame_stride 1 \
  --pixel_stride 6 \
  --detail_pixel_stride 2 \
  --max_points 200000 \
  --conf_quantile 0.7
```

访问：`http://127.0.0.1:8101/`

当前数据对应关系：

```text
KITTI 09 Ground Truth: 1591 帧
gt_stride: 10
预测轨迹: 160 帧
```

脚本分别完成：

```text
LASER baseline  --独立 Sim(3)--> Ground Truth
LASER-Geometry  --独立 Sim(3)--> Ground Truth
```

当前结果：

```text
LASER baseline ATE: 21.901049
LASER-Geometry ATE: 19.353710
```

## 5. 页面控制

快速对比按钮：

- `GT vs Baseline`
- `GT vs Geometry`
- `Baseline vs Geometry`

独立显示开关：

- `Ground Truth`
- `LASER baseline`
- `LASER-Geometry`
- `Overview clouds`
- `Current-frame detail`

三个轨迹开关可以任意组合，因此也支持单独查看任意一条轨迹，或者三条
轨迹同时显示。

逐帧播放控制：

- `Play`：从当前帧开始自动播放，末帧之后回到第 0 帧；
- `Pause`：暂停播放并保留当前帧；
- `Playback FPS`：调节播放速度，默认 2 FPS；
- `Frame`：播放中仍可手动拖动，之后会从选择的新帧继续播放。

滚轮缩放后如果无法看到全貌，点击：

```text
Fit full trajectory
```

只分析轨迹偏差时，建议关闭 `Overview clouds` 和
`Current-frame detail`，避免点云遮挡轨迹线。

## 6. 资源不足时

增加 `pixel_stride` 并降低 `max_points`：

```bash
--pixel_stride 10 \
--detail_pixel_stride 4 \
--max_points 100000
```

`frame_stride=1` 表示全部 160 个采样帧都参与概览点云。增大
`frame_stride` 只会减少概览点云来源，不会缩短三条完整轨迹。

## 7. 云端同步最新代码

```bash
cd ~/autodl-tmp/LASER-Geometry-main
git stash push -m "autodl-before-viser-update"
git pull --ff-only origin main
git log -1 --oneline
```

确认最新提交信息包含轨迹播放控件后，再启动本地 Viser。
