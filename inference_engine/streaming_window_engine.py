# 1️⃣ import与常量
import torch
import torch.nn as nn

import os
import threading    # 支持后台线程、队列、临时缓存目录
import queue
import pathlib
import gc
import tempfile
import shutil
import glob
from collections import defaultdict
import time

from . import VanillaEngine
from .inference_utils import (              # 包含了一系列的核心几何函数
    dict_to_device,
    register_adjacent_windows,              # 相邻滑窗对齐
    estimate_pseudo_depth_and_intrinsics,   # 从Pi3输出估计伪深度和内参
    unproject_depth_to_local_points,        # 用深度和内参反投影成局部点云
    apply_sim3_to_pose,                     # 把Sim3变换作用到相机位姿
    build_depth_sp_graph,                   # 原始 LASER depth graph 构建
    build_geometry_sp_graph,                # 新增 geometry-aware graph 构建
    refine_segment_scales,                  # segment-level 尺度细化
    sliding_window_t,
    sliding_window_l
)
from .alignment_debug import AlignmentDebugRecorder, summarize_graph_layer
from .utils.geometry import homogenize_points

STOP_SIGNAL = object()                      # 线程队列里的停止标记


# 2️⃣ 整个StreamingWindowEngine的核心类，继承自VanillaEngine，重写了forward方法，并新增了滑窗处理、线程管理、缓存管理等功能。
class StreamingWindowEngine(VanillaEngine):
    def __init__(
            self,
            delegate: nn.Module,
            inference_device: str,
            dtype: torch.dtype,
            intermediate_device: str = 'cuda',
            process_device: str = 'cpu',
            top_conf_percentile: float = 0.5,
            window_size: int = 20,
            overlap: int = 5,
            depth_refine=True,
            cache_root: str = './cache',
            benchmark_latency=True,

            # ================================
            # 中文：新增分割模式配置。
            # English: New segmentation mode configuration.
            segment_mode: str = 'depth',                # 分割方法
            normal_method: str = 'cross',               # 法线估计方法
            scale_anchor_mode: str = 'depth_irls',      # 尺度锚点估计方法
            # 新增的更好的可视化内容的部分(记录功能的入口)
            debug_alignment: bool = False,              # 是否开启对齐调试记录（默认False）
            debug_alignment_path: str | None = None,    # 调试文件保存根目录
            debug_alignment_scene: str | None = None,   # 当前scene的子目录名
            # ================================
    ):
        # 参数保护
        if segment_mode not in ('depth', 'geometry'):
            raise ValueError(f"Unknown segment_mode: {segment_mode}")
        if scale_anchor_mode not in ('depth_irls', 'conf_weighted_irls'):
            raise ValueError(f"Unknown scale_anchor_mode: {scale_anchor_mode}")
        if segment_mode == 'geometry' and not depth_refine:
            raise ValueError("segment_mode='geometry' only affects segment refinement; enable depth_refine.")

        # 1️⃣ 继承VanillaEngine，把真实模型delegate放到inference_device上
        super().__init__(
            delegate=delegate.to(inference_device)
        )
        # 2️⃣ 保存滑动窗口参数
        self.window_size = window_size                  # 每个窗口多少帧
        self.overlap = overlap                          # 响铃窗口共享多少帧
        self.intermediate_device = intermediate_device
        self.top_conf_percentile = 1 - top_conf_percentile if top_conf_percentile is not None else 0.0      # 反转逻辑，就是提取置信度最高的top_conf_percentile比例的点

        # 3️⃣ 保存推理和处理设备
        self.inference_device = inference_device        # 模型向前传播的设备，通常是CUDA
        self.process_device = process_device            # 后处理用的设备，默认CPU
        self.dtype = dtype                              # autocast的半精度类型
        self.depth_refine = depth_refine                # 是否启用segment-level refinement，启用后会触发LASER的深度分割细化流程

        # 4️⃣ 新增的配置
        # ================================
        # 中文：segment_mode 控制使用原始 depth segmentation 还是新增 geometry segmentation。
        # English: segment_mode controls whether to use original depth segmentation or new geometry segmentation.
        self.segment_mode = segment_mode            # depth：原始LASER depth segmentation   geometry：新增geometry-aware segmentation

        # 中文：normal_method 控制 geometry.py 中 normal 的估计方法。
        # English: normal_method controls the normal estimation method in geometry.py.
        self.normal_method = normal_method          # 控制geometry segmentation 里的normal估计方法（这里面后面会有各种方法，需要钻研的地方）

        # 中文：scale_anchor_mode 控制 segment 对齐时的初始尺度锚点估计方式。
        # English: scale_anchor_mode controls how overlap segment scale anchors are estimated.
        self.scale_anchor_mode = scale_anchor_mode  # depth_irls：原始 LASER；conf_weighted_irls：M1 置信度加权实验

        self.debug_alignment = bool(debug_alignment)
        self.alignment_debug_recorder = AlignmentDebugRecorder(
            enabled=self.debug_alignment,
            root_dir=debug_alignment_path,
            scene_name=debug_alignment_scene,
        )
        # ================================

        # 5️⃣ 初始化缓存和线程状态
        os.makedirs(cache_root, exist_ok=True)
        self.cache_dir = cache_root                 # 总缓存目录
        self.temp_cache_dir = None
        self.cache_id = 0
        self.inference_queue = queue.Queue()        # 主线程把窗口送给模型推理线程
        self.registration_queue = queue.Queue()     # 推理线程把预测结果送给配准线程

        self.prev_window_cache = None               # 保存上一个窗口结果
        self.anchor_sp_graph = None                 # 保存上一个窗口的 segment graph，用于下一窗口refinement

        self._inference_thread = None
        self._registration_thread = None

        self.running = False

        self.benchmark_latency = benchmark_latency
        self.latencies = []                        # 记录每个窗口的耗时
        self.warmup_steps = 2

        
    # 3️⃣ 切换cache目录
    def set_cache_dir(self, cache_dir):             # 如果engine正在运行，不能切换cache目录，否则会直接报错，避免推理中途改路径导致缓存错乱
        if self.running:
            raise RuntimeError('Cannot change cache directory while running')
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir

    # 4️⃣ 动态设置是否开启 depth refinement（同样只能在未运行状态下修改）
    def set_depth_refine(self, flag):
        if self.running:
            raise RuntimeError('Cannot change depth refinement mode while running')
        self.depth_refine = flag

    # 🌟辅助：原始 LASER depth segmentation 路径，只使用 depth/conf/top_conf，保持 baseline 输入干净。
    def _build_depth_segment_graph(self, local_points, conf):
        local_points_np = local_points.cpu().numpy()
        return build_depth_sp_graph(
            local_points_np[..., -1],
            conf_map=conf.cpu().numpy(),
            top_conf_percentile=self.top_conf_percentile,
        )

    # 🌟辅助：新增 geometry segmentation 路径，只有这里会传入 point_map/intrinsic/normal_method。
    def _build_geometry_segment_graph(self, local_points, conf, ref_intrinsic):
        local_points_np = local_points.cpu().numpy()
        return build_geometry_sp_graph(
            local_points_np[..., -1],
            conf_map=conf.cpu().numpy(),
            top_conf_percentile=self.top_conf_percentile,
            point_map=local_points_np,
            intrinsic=ref_intrinsic.cpu().numpy() if hasattr(ref_intrinsic, "cpu") else ref_intrinsic,
            normal_method=self.normal_method,
        )

    # 🌟 可视化调试：保存Sim3前后，refine后，segment mask/scale/IoU的内容
    def _record_alignment_debug_pair(
        self,
        *,                              # *表示后面参数必须用关键字传入
        pair_index,
        sim3_scale,
        sim3_R,
        sim3_t,
        prev_local_points,
        cur_local_points_before,
        cur_local_points_after_sim3,
        cur_local_points_after_refine,
        prev_conf,
        cur_conf,
        mutual_conf_mask,
        tgt_sp_graph,
    ):
        # 1️⃣ 开关保护
        if not self.debug_alignment:
            return

        # 2️⃣ 保存核心对齐数据
        payload = {
            "sim3_scale": sim3_scale,                                               # 当前窗口对齐到前一个窗口时估计出来的 Sim3 变换
            "sim3_R": sim3_R,
            "sim3_t": sim3_t,
            "src_points_overlap": prev_local_points,                                # 前一个窗口最后 overlap 帧的点云，也就是 source
            "tgt_points_before_overlap": cur_local_points_before,                   # 当前窗口最开始 overlap 帧，在做 Sim3 之前的点云。
            "tgt_points_after_sim3_overlap": cur_local_points_after_sim3,           # 当前窗口做完 Sim3 粗对齐之后的点云
            "tgt_points_after_refine_overlap": cur_local_points_after_refine,       # 如果开启 segment refinement，这是 refinement 之后的点云；如果没开启，基本就是 Sim3 后的状态
            "src_conf_overlap": prev_conf,                                          # 前窗和当前窗 overlap 区域的置信度
            "tgt_conf_overlap": cur_conf,
            "mutual_conf_mask": mutual_conf_mask,                                   # 两边都高置信的区域。这个 mask 是真正参与相邻窗口注册的区域。
        }

        # 3️⃣ 如果有 segment graph，就额外记录 segment 信息 主要用于检查 segment refinement 的可靠性：哪些区域有尺度锚点，哪些没有；尺度是否异常；IoU 是否太低。
        if tgt_sp_graph:
            graph_summary = summarize_graph_layer(tgt_sp_graph[0])
            payload.update(
                {
                    "tgt_segment_masks_frame0": graph_summary["masks"],             # 第 0 帧每个 segment 的 mask。
                    "tgt_segment_has_scale_frame0": graph_summary["has_scale"],     # 这个 segment 有没有拿到 scale anchor。
                    "tgt_segment_mean_iou_frame0": graph_summary["mean_iou"],       # 这个 segment 和 source segment 匹配时的平均 IoU。
                    "tgt_segment_mean_scale_frame0": graph_summary["mean_scale"],   # 这个 segment 最终估计到的平均 scale。
                }
            )

        # 4️⃣ 真正写文件 这里调用 AlignmentDebugRecorder 把数据写出去。
        self.alignment_debug_recorder.record_pair(
            pair_index=pair_index,
            payload=payload,
            metadata={
                "segment_mode": self.segment_mode,
                "normal_method": self.normal_method,
                "scale_anchor_mode": self.scale_anchor_mode,
                "window_size": self.window_size,
                "overlap": self.overlap,
                "top_conf_percentile": self.top_conf_percentile,
            },
        )

    # 5️⃣ 把当前窗口的 prev_window_cache 保存到磁盘，文件名为 window_cache_{cache_id}.pt
    def _save_cache(self):
        torch.save(self.prev_window_cache, self.temp_cache_dir / f'window_cache_{self.cache_id}.pt')
        self.cache_id += 1

    # 6️⃣ 更新两个关键状态
    def _update_cache(self, new_window_cache, sp_graph=None):
        self.prev_window_cache = new_window_cache           # 当前窗口处理完成之后，会成为下一个窗口的“前一窗口”
        self.anchor_sp_graph = sp_graph
        gc.collect()

    # 7️⃣ 一次完整推理结束后清理状态：不回删除最终输出结果
    def _reset_state(self):
        self.cache_id = 0                           # 归零
        self.inference_queue = queue.Queue()        # 重新创建两个 queue
        self.registration_queue = queue.Queue()

        self.prev_window_cache = None               # 清空上一个窗口缓存
        self.anchor_sp_graph = None

        self._inference_thread = None               # 清空线程对象
        self._registration_thread = None

        self.latencies = []                         # 清空latency记录

        gc.collect()                                # 清理内存，避免残留的缓存占用显存

    # 8️⃣ 第一个后台线程：模型推理线程（该线程只负责向前推理，不负责窗口对齐）
    @torch.no_grad()
    def _model_inference_worker(self):
        while True:
            # 1️⃣ 从inference_queue里取出一个窗口
            sample_window = self.inference_queue.get()
            # 2️⃣ 如果是STOP_SIGNAL就退出线程
            if sample_window is STOP_SIGNAL:
                return

            t_start = time.perf_counter()

            # 3️⃣ 在inference_device上执行模型推理，使用autocast半精度
            with torch.autocast(self.inference_device, dtype=self.dtype):
                prediction_window = self.delegate(sample_window)

            inference_duration = time.perf_counter() - t_start

            # 4️⃣ 把预测结果转到process_device上
            processed_window = dict_to_device(prediction_window, self.process_device)
            # 5️⃣ 把结果和推理耗时一起送进 registration_queue，等待配准线程处理
            self.registration_queue.put((processed_window, inference_duration))
            # 6️⃣ 如果是CUDA，调用 empty_cache() 清理显存，避免显存占用过高
            if self.inference_device == 'cuda':
                torch.cuda.empty_cache()

    # 9️⃣ 第二个后台线程：窗口配准和深度细化线程
    def _registration_worker(self):
        ref_intrinsic = None
        tgt_sp_graph = None

        while True:
            # 1️⃣ 从registration_queue里取出预测结果和推理耗时（分两种情况 第一个窗口：初始化内参和anchor  后续窗口：和前一个窗口overlap帧做Sim3对齐，再可选做 segment depth refinement）
            item = self.registration_queue.get()
            if item is STOP_SIGNAL:
                return

            working_window, inference_duration = item
            t_start = time.perf_counter()

            # 2️⃣ 每个窗口都先去掉batch纬度，然后计算当前窗口 overlap部分的 置信度阈值，生成mask
            for key in working_window.keys():
                if isinstance(working_window[key], torch.Tensor):
                    working_window[key] = working_window[key].squeeze(0)

            # 3️⃣ 得到target mask，取当前窗口 overlap部分的置信度大于阈值的点（只选择高置信区域参与）
            # camera pose registration
            conf_thresh = torch.quantile(working_window['conf'][:self.overlap], self.top_conf_percentile,
                                         interpolation='nearest')
            tgt_mask = working_window['conf'][:self.overlap] >= conf_thresh

            # 4️⃣ 后续窗口的处理逻辑
            if self.prev_window_cache is not None:
                # fixed intrinsic enforce
                # 1️⃣ 固定内参重新反投影
                working_window['local_points'] = unproject_depth_to_local_points(
                    working_window.pop('local_points')[..., -1],
                    ref_intrinsic
                )
                # 2️⃣ 构造mutual conf mask（前一个窗口末尾overlap 和当前窗口开头 overlap 帧都高置信度区域才参与注册）
                prev_conf_thresh = torch.quantile(self.prev_window_cache['conf'][-self.overlap:],
                                                  self.top_conf_percentile, interpolation='nearest')
                conf_mask = (self.prev_window_cache['conf'][-self.overlap:] >= prev_conf_thresh) & tgt_mask

                # metric depth align
                prev_local_points = self.prev_window_cache['local_points'][-self.overlap:]
                cur_local_points = working_window['local_points'][:self.overlap]
                cur_local_points_before_sim3 = cur_local_points.clone()

                # 3️⃣ 相邻窗口配准
                s_d, R, t = register_adjacent_windows(
                    prev_local_points,
                    cur_local_points,
                    self.prev_window_cache['camera_poses'][-self.overlap:],
                    working_window['camera_poses'][:self.overlap],
                    conf_mask
                )

                # 4️⃣ 把Sim3应用到当前窗口（先修正当前窗口局部点云尺度，再修正相机位姿，式当前窗口接到全局轨迹上）
                working_window['local_points'] = s_d * working_window.pop('local_points')
                working_window['camera_poses'] = apply_sim3_to_pose(working_window.pop('camera_poses'), s_d, R, t)
                cur_local_points_after_sim3 = working_window['local_points'][:self.overlap].clone()

                # 🌟5️⃣ 如果开启depth_refine,进入 segment-level depth refinement
                if self.depth_refine:
                    # 1️⃣ 先把当前窗口点云转成numpy
                    tgt_pcd = working_window['local_points'].cpu().numpy()

                    # 🌟2️⃣ 分支1:保持原版本 LASER baseline
                    if self.segment_mode == 'depth':
                        # 构建当前窗口的 depth segment graph
                        tgt_sp_graph = self._build_depth_segment_graph(
                            working_window['local_points'],             # 传入窗口的局部点云（传入后实际上是拿高度获取深度）
                            working_window['conf'],                     # 传入窗口的置信度图（用于过滤低置信度点）
                        )
                        working_window['local_points'] = working_window['local_points'] * refine_segment_scales(
                            self.prev_window_cache['local_points'].cpu().numpy(),   # 传入前一个窗口的局部点云（用于和当前窗口做segment-level对齐）
                            tgt_pcd,                                                # 当前的点云
                            self.anchor_sp_graph,                                   # 前一个窗口的segment graph（用于和当前窗口做segment-level对齐）
                            tgt_sp_graph,                                           # 当前窗口的segment graph（用于和前一个窗口做segment-level对齐）
                            self.overlap,                                           # 窗口重叠帧数（用于segment-level对齐时的IoU计算）
                            src_conf=self.prev_window_cache['conf'].cpu().numpy(),
                            tgt_conf=working_window['conf'].cpu().numpy(),
                            scale_anchor_mode=self.scale_anchor_mode
                        )
                    # 🌟3️⃣ 分支2:新增 geometry segmentation 先构建 geometry-aware graph，再独立进入 refinement。
                    elif self.segment_mode == 'geometry':
                        # 构建当前窗口的 Geometry segment graph
                        tgt_sp_graph = self._build_geometry_segment_graph(
                            working_window['local_points'],                         # 传入窗口的局部点云（传入后实际上是拿高度获取深度）
                            working_window['conf'],                                 # 传入窗口的置信度图（用于过滤低置信度点）
                            ref_intrinsic,                                          # 传入参考内参（用于计算法向量，法向量是geometry segmentation的关键特征）（用来做一些获取几何特征的特殊计算）
                        )
                        working_window['local_points'] = working_window['local_points'] * refine_segment_scales(
                            self.prev_window_cache['local_points'].cpu().numpy(),   # 一样，和基线论文一样的方法来精细微调
                            tgt_pcd,
                            self.anchor_sp_graph,
                            tgt_sp_graph,
                            self.overlap,
                            src_conf=self.prev_window_cache['conf'].cpu().numpy(),
                            tgt_conf=working_window['conf'].cpu().numpy(),
                            scale_anchor_mode=self.scale_anchor_mode
                        )
                    else:
                        raise ValueError(f'Unknown segment_mode: {self.segment_mode}')
                self._record_alignment_debug_pair(
                    pair_index=self.cache_id,
                    sim3_scale=s_d,
                    sim3_R=R,
                    sim3_t=t,
                    prev_local_points=prev_local_points,
                    cur_local_points_before=cur_local_points_before_sim3,
                    cur_local_points_after_sim3=cur_local_points_after_sim3,
                    cur_local_points_after_refine=working_window['local_points'][:self.overlap],
                    prev_conf=self.prev_window_cache['conf'][-self.overlap:],
                    cur_conf=working_window['conf'][:self.overlap],
                    mutual_conf_mask=conf_mask,
                    tgt_sp_graph=tgt_sp_graph,
                )
            # 5️⃣第一个窗口分支（没有前窗，所以不做相邻的窗口配准）
            else:
                # 1️⃣ 初始化：从第一个窗口估计参考内参，后续所有窗口都是用 ref_intrinsic 重新反投影。
                _, intrinsic_ = estimate_pseudo_depth_and_intrinsics(working_window['local_points'])
                ref_intrinsic = intrinsic_[0]
                working_window['local_points'] = unproject_depth_to_local_points(       # 然后同样把深度反投影成局部点云
                    working_window.pop('local_points')[..., -1],
                    ref_intrinsic
                )

                # 2️⃣ 如果 depth_refine=True，第一个窗口会先构建 anchor graph，供第二个窗口做 refinement。
                if self.depth_refine:
                    if self.segment_mode == 'depth':
                        tgt_sp_graph = self._build_depth_segment_graph(
                            working_window['local_points'],
                            working_window['conf'],
                        )
                    elif self.segment_mode == 'geometry':
                        tgt_sp_graph = self._build_geometry_segment_graph(
                            working_window['local_points'],
                            working_window['conf'],
                            ref_intrinsic,
                        )
                    else:
                        raise ValueError(f'Unknown segment_mode: {self.segment_mode}')

            # 6️⃣ 窗口处理完成之后，更新缓存状态，并保存到磁盘
            self._update_cache(working_window, tgt_sp_graph)
            self._save_cache()

            # 记录每个窗口的总耗时（推理+配准），用于benchmark
            reg_duration = time.perf_counter() - t_start
            total_process_time = inference_duration + reg_duration
            self.latencies.append(total_process_time)                   # latency包含模型推理时间和 registration/refinement时间

    # 1️⃣0️⃣ 启动一次 streaming inference，创建两个后台线程，分别处理模型推理和窗口配准
    def begin(self):
        # 1️⃣ 检查engine 是否已经运行
        if self.running:
            raise RuntimeError('Cannot start a running inference engine')

        # 2️⃣ 在 cache_dir 下创建一个临时目录，用于存放每个窗口的缓存文件，线程结束后会删除该目录
        self.temp_cache_dir = pathlib.Path(tempfile.mkdtemp(dir=self.cache_dir))
        # 3️⃣ 创建两个后台线程，分别处理模型推理和窗口配准
        self._inference_thread = threading.Thread(target=self._model_inference_worker, daemon=True)
        self._registration_thread = threading.Thread(target=self._registration_worker, daemon=True)
        # 4️⃣ 启动两个线程
        self._inference_thread.start()
        self._registration_thread.start()

        self.running = True

    # 1️⃣1️⃣ 重写forward方法，把输入窗口送入inference_queue，等待后台线程处理
    def forward(self, sample, **kwargs):
        self.inference_queue.put(sample)

    # 1️⃣2️⃣ 结束一次 streaming inference，发送停止信号给两个后台线程，并等待它们结束
    def end(self):
        if not self.running:
            raise RuntimeError('Cannot terminate a stopped inference engine')

        # 1️⃣ 给 inference_queue 发送 STOP_SIGNAL（推理队列）
        self.inference_queue.put(STOP_SIGNAL)
        # 2️⃣ 等待推理线程结束
        self._inference_thread.join()
        # 3️⃣ 给 registration_queue 发送 STOP_SIGNAL（配准队列）
        self.registration_queue.put(STOP_SIGNAL)
        # 4️⃣ 等待配准线程结束
        self._registration_thread.join()

        # 5️⃣ 打印latency summary
        if self.benchmark_latency:
            if self.latencies:
                print("\n" + "=" * 50)
                print("        INFERENCE PERFORMANCE SUMMARY        ")
                print("=" * 50)

                # Print list of all times
                latencies_ms = [t * 1000 for t in self.latencies]
                print(f"Raw Latencies (ms): {latencies_ms}")

                if len(self.latencies) > self.warmup_steps + 1:
                    steady_times = self.latencies[self.warmup_steps:-1]
                    avg_steady = sum(steady_times) / len(steady_times)
                    print("-" * 50)
                    print(f"Total Windows:     {len(self.latencies)}")
                    print(f"Warmup Windows:    {self.warmup_steps}")
                    print(f"Steady State Avg:  {avg_steady * 1000:.2f} ms")
                else:
                    avg_all = sum(self.latencies) / len(self.latencies)
                    print(f"Average (All):     {avg_all * 1000:.2f} ms")
                print("=" * 50 + "\n")

        # 6️⃣ 清理状态，删除临时缓存目录
        self._reset_state()
        # 7️⃣ 标记engine不再运行
        self.running = False

    # 1️⃣3️⃣ 把图像切成滑框
    def img_sliding_window(self, imgs):
        if isinstance(imgs, torch.Tensor):
            if len(imgs.shape) == 5:
                return sliding_window_t(imgs, self.window_size, self.overlap, dim=1)
            return sliding_window_t(imgs, self.window_size, self.overlap, dim=0)
        elif isinstance(imgs, list):
            return sliding_window_l(imgs, self.window_size, self.overlap)

    # 1️⃣4️⃣ 解析单个窗口的缓存文件，返回一个字典，去掉overlap部分（为了聚合窗口时，避免重复帧）
    @staticmethod
    def parse_cache_file(cache_file, overlap=0):
        window_cache = torch.load(cache_file, map_location='cpu', weights_only=False)
        for key in window_cache.keys():
            if isinstance(window_cache[key], torch.Tensor):
                window_cache[key] = window_cache[key][overlap:]

        return window_cache

    # 1️⃣5️⃣ 把所有窗口cache合成完整序列
    @staticmethod
    def aggregate_caches(parsed_caches):
        aggregated_cache = defaultdict(list)
        # 1️⃣ 遍历所有 parsed cache， 把同名字段收集到list
        for cache in parsed_caches:
            for k, v in cache.items():
                # 2️⃣ 跳过已有的points，因为后面会重新计算全局点
                if k == 'points':
                    continue
                aggregated_cache[k].append(v)

        # 3️⃣ 对tensor字段沿着时间维concat，并恢复batch纬
        for k in list(aggregated_cache.keys()):
            if isinstance(aggregated_cache[k][0], torch.Tensor):
                aggregated_cache[k] = torch.concat(aggregated_cache.pop(k), dim=0)[None]

        # 4️⃣ 根据相机位姿和局部点云重新生成全局点云（把每帧的 local points 通过对应的 camera pose 变换到全局坐标系）
        aggregated_cache['points'] = torch.einsum(
            'bnij, bnhwj -> bnhwi',
            aggregated_cache['camera_poses'],
            homogenize_points(aggregated_cache['local_points'])
        )[..., :3]
        return aggregated_cache

    # 1️⃣6️⃣ 最终结果汇总函数，demo中推理结束后调用它
    def parse_inference_cache_summary(self, remove_cache=True):
        # 1️⃣ 找到临时目录中所有 window_cache_*.pt 文件
        assert self.temp_cache_dir is not None
        # 2️⃣ 按编号排序
        cache_files = sorted(glob.glob(str(self.temp_cache_dir / 'window_cache_*.pt')),
                             key=lambda p: int(p.split('_')[-1].split('.')[0]))

        # 3️⃣ 第一个窗口完整读取
        parsed_caches = [self.parse_cache_file(cache_files[0])]
        # 4️⃣ 后续窗口读取时丢掉overlap帧
        for cache_fname in cache_files[1:]:
            parsed_caches.append(self.parse_cache_file(cache_fname, overlap=self.overlap))

        # 5️⃣ 调用 aggregate_caches() 合并，而后调用父类的 _post_process_pred() 进行最终处理（补齐字段）
        ret_dict = StreamingWindowEngine._post_process_pred(self.aggregate_caches(parsed_caches))

        # 6️⃣ 删除临时缓存目录
        if remove_cache:
            shutil.rmtree(self.temp_cache_dir)

        # 7️⃣ 返回最终结果字典
        return ret_dict
