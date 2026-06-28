# 1️⃣ Imports 与 全局设备配置
import torch

from pi3.models.pi3 import Pi3                              # 主模型
from inference_engine import StreamingWindowEngine          # 流式滑动窗口推理与窗口间配准的核心引擎
from utils.load_fn import load_and_preprocess_images        # 图像序列读取和预处理
from utils.checkpoint import load_checkpoint_state_dict
from utils.image_sequence import list_image_paths
from eval.save_func import save_for_viser                   # 把最终结果保存成可视化格式

import argparse
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"     # device优先使用CUDA
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
dtype = (
    torch.bfloat16
    if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    else torch.float16
)

# 2️⃣ ArgumentParser 配置 定义命令行参数
def get_args_parser():
    parser = argparse.ArgumentParser('Streaming Pi3 Demo')
    parser.add_argument('--model_ckpt', default=None, type=str, help='checkpoint to load model')    # 本地checkpoint路径
    parser.add_argument('--data_path', type=str, help='sequence data path')                         # 图像序列路径
    parser.add_argument('--scene_name', default=None, type=str, help='scene_name')                  # 输出场景名
    parser.add_argument('--cache_path', default='./inference_cache', type=str,                      # 中间推理cache输出目录
                        help='output inference cache')
    parser.add_argument('--output_path', default='./viser_results', type=str,                       # viser 结果保存目录
                        help='output visualization results')
    parser.add_argument('--sample_interval', default=1, type=int, help='sequence sample interval')  # 图像采样间隔
    parser.add_argument('--window_size', default=10, type=int, help='sliding window size')          # 滑窗长度
    parser.add_argument('--overlap', default=5, type=int, help='sliding window overlap size')       # 相邻窗口重叠帧数
    parser.add_argument(
        '--top_conf_percentile',
        default=0.3,
        type=float,
        help='fraction of highest-confidence overlap points retained by the shared confidence gate',
    )
    parser.add_argument('--depth_refine', action='store_true', help='enable depth refine')          # 是否开启 LASER depth segment refinement

    # =================================
    # 中文：分割模式，depth 表示原始 LASER，geometry 表示新增几何分割。
    # English: Segmentation mode. "depth" means original LASER, "geometry" means new geometry segmentation.
    parser.add_argument(
        '--segment_mode',
        default='depth',                            # depth： 原始 LASER depth segmentation
        choices=['depth', 'geometry'],              # geometry： 新增geometry-aware segmentation
        type=str,
        help='segmentation mode: depth or geometry'
    )

    # 中文：normal 估计方法，第一版默认 cross。
    # English: Normal estimation method. The first version uses "cross" by default.
    parser.add_argument(
        '--normal_method',                          # 服务于geometry分割模式
        default='cross',                            # cross： cross product method, sobel： sobel filter method
        choices=['cross', 'sobel'],
        type=str,
        help='normal estimation method'
    )
    parser.add_argument(
        '--scale_anchor_mode',
        default='depth_irls',
        choices=['depth_irls', 'conf_weighted_irls'],
        type=str,
        help='segment scale anchor estimator: original depth_irls or M1 confidence-weighted IRLS'
    )
    parser.add_argument(
        '--debug_alignment',
        action='store_true',
        help='save optional alignment debug traces without changing normal outputs'
    )
    parser.add_argument(
        '--debug_alignment_path',
        default='outputs/debug_alignment',
        type=str,
        help='root directory for optional alignment debug traces'
    )
    # =================================

    return parser

# 3️⃣ 加载模型，并将模型送入到 StreamingWindowEngine 中，返回一个流式推理引擎对象
def load_model(args):
    # model 加载pi3模型
    if args.model_ckpt:
        model = Pi3().to(device)
        print('Loading checkpoint: ', args.model_ckpt)
        ckpt = load_checkpoint_state_dict(args.model_ckpt, map_location=device)
        print(model.load_state_dict(ckpt, strict=True))
        del ckpt
    else:
        model = Pi3.from_pretrained("yyfz233/Pi3").to(device)

    # 把模型包进StreamingWindowEngine中，返回一个流式推理引擎对象
    return StreamingWindowEngine(
        model,
        inference_device=device,
        dtype=dtype,
        window_size=args.window_size,
        overlap=args.overlap,
        cache_root=args.cache_path,
        depth_refine=args.depth_refine,         # ⚠️只有开启depth_refine，make_sp_graph() 才会被调用，进而触发 LASER depth segmentation refinement
        top_conf_percentile=args.top_conf_percentile,

        # 中文：新增 geometry segmentation 配置。
        # English: New geometry segmentation configuration.
        segment_mode=args.segment_mode,
        normal_method=args.normal_method,
        scale_anchor_mode=args.scale_anchor_mode,
        debug_alignment=args.debug_alignment,
        debug_alignment_path=args.debug_alignment_path,
        debug_alignment_scene=args.scene_name,
    )
# 4️⃣ 实际跑推理主流程
def run_model(image_names, scene_name, output_path):
    # 1️⃣ 把图像路径切分成多个滑窗，由window_size 和 overlap决定
    image_name_windows = model.img_sliding_window(image_names)

    # 2️⃣ 创建CUDA event计时
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    # 3️⃣ 启动engine内部推理线程 和 registration线程
    model.begin()

    start_ev.record()
    # 4️⃣ 遍历每个窗口
    for sample in tqdm(image_name_windows, 'Window inference'):
        imgs = load_and_preprocess_images(sample).to(device)
        model(imgs)
    # 5️⃣ 发送停止信号，等待后台线程处理完所有窗口
    model.end()
    end_ev.record()
    duration = start_ev.elapsed_time(end_ev)

    # 6️⃣ 把每个窗口保存的cache聚合成完整序列结果
    save_dict = model.parse_inference_cache_summary()
    for key in save_dict.keys():
        if isinstance(save_dict[key], torch.Tensor):
            save_dict[key] = save_dict[key].cpu().numpy().squeeze(0)

    # 7️⃣ tensor转numpy后，保存成viser可视化格式
    save_for_viser(save_dict, scene_name, output_path, inverse_extrinsic=False)

    torch.cuda.synchronize()  # make sure the event timestamps are set
    gpu_mem_usage = torch.cuda.max_memory_allocated()

    summary_text = f"""
    Summary:
        Inference sec: {duration / 1000}
        Peak GPU memory usage (GB): {gpu_mem_usage / (1024 ** 3)}
    """
    print(summary_text)

    # save_cache_to_viser(model.cache_dir, scene_name, output_path, overlap)

# 5️⃣ 把数据路径转成图像列表
def run_dynamic_scene(args):
    data_path = args.data_path
    scene_name = data_path.split('/')[-1] if args.scene_name is None else args.scene_name

    img_names = list_image_paths(data_path, sample_interval=args.sample_interval)
    print(f'Found {len(img_names)} images.')
    run_model(img_names, scene_name, args.output_path)

# 6️⃣ main函数入口
if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    model = load_model(args)

    model.eval()
    run_dynamic_scene(args)
