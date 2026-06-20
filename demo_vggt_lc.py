import argparse
import glob
import os
import shutil
import time
from pathlib import Path

import torch
from tqdm import tqdm

from eval.save_func import save_for_viser
from inference_engine import StreamingWindowEngineLC
from utils.checkpoint import load_checkpoint_state_dict
from utils.image_sequence import list_image_paths
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = (
    torch.bfloat16
    if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
    else torch.float16
)


def get_args_parser():
    parser = argparse.ArgumentParser("Loop-closure Streaming VGGT Demo")
    parser.add_argument("--config_path", default=None, type=str, help="loop closure config")
    parser.add_argument(
        "--model_ckpt",
        default="./weights/model.pt",
        type=str,
        help="local VGGT checkpoint to load",
    )
    parser.add_argument("--data_path", type=str, help="sequence data path")
    parser.add_argument("--scene_name", default=None, type=str, help="scene_name")
    parser.add_argument(
        "--cache_path",
        default="./inference_cache",
        type=str,
        help="output inference cache",
    )
    parser.add_argument(
        "--output_path",
        default="./viser_results",
        type=str,
        help="output visualization results",
    )
    parser.add_argument("--sample_interval", default=1, type=int, help="sequence sample interval")
    parser.add_argument("--window_size", default=10, type=int, help="sliding window size")
    parser.add_argument("--overlap", default=5, type=int, help="sliding window overlap size")
    parser.add_argument("--depth_refine", action="store_true", help="enable depth refine")
    parser.add_argument(
        "--segment_mode",
        default="depth",
        choices=["depth", "geometry"],
        type=str,
        help="segmentation mode: depth or geometry",
    )
    parser.add_argument(
        "--normal_method",
        default="cross",
        choices=["cross", "sobel"],
        type=str,
        help="normal estimation method",
    )
    return parser


def load_model(args):
    checkpoint_path = Path(args.model_ckpt)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"VGGT checkpoint not found: {checkpoint_path}. "
            "Pass --model_ckpt with the cloud/local model.pt path."
        )

    base_model = VGGT()
    print("Loading checkpoint: ", checkpoint_path)
    ckpt = load_checkpoint_state_dict(str(checkpoint_path), map_location="cpu")
    print(base_model.load_state_dict(ckpt, strict=True))
    del ckpt
    base_model = base_model.to(device)

    engine = StreamingWindowEngineLC(
        base_model,
        inference_device=device,
        dtype=dtype,
        window_size=args.window_size,
        overlap=args.overlap,
        cache_root=args.cache_path,
        depth_refine=args.depth_refine,
        segment_mode=args.segment_mode,
        normal_method=args.normal_method,
    )
    return base_model, engine


def _start_timer():
    if device == "cuda":
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        return start_ev, end_ev
    return time.perf_counter(), None


def _finish_timer(timer):
    start_ev, end_ev = timer
    if device == "cuda":
        end_ev.record()
        torch.cuda.synchronize()
        return start_ev.elapsed_time(end_ev) / 1000
    return time.perf_counter() - start_ev


def run_model(engine, image_names):
    image_name_windows = engine.img_sliding_window(image_names)

    engine.begin()
    timer = _start_timer()
    with torch.inference_mode():
        for sample in tqdm(image_name_windows, "Window inference"):
            imgs = load_and_preprocess_images(sample).to(device)
            engine(imgs)
    engine.end()
    duration = _finish_timer(timer)

    gpu_mem_usage = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    summary_text = f"""
    Summary:
        Inference sec: {duration}
        Peak GPU memory usage (GB): {gpu_mem_usage / (1024 ** 3)}
    """
    print(summary_text)


def run_dynamic_scene(args, engine):
    data_path = args.data_path
    scene_name = data_path.split("/")[-2] if args.scene_name is None else args.scene_name

    img_names = list_image_paths(data_path, sample_interval=args.sample_interval)
    print(f"Found {len(img_names)} images.")
    run_model(engine, img_names)
    return scene_name


def run_loop_closure(args, base_model, engine, scene_name):
    from loop_closure.loop_closure import LoopClosureEngine
    from loop_closure.utils.config_utils import load_config

    config = load_config(args.config_path)
    cache_path = Path(args.cache_path)
    cache_path_lc = cache_path.parent / f"{cache_path.name}_lc"
    lc_engine = LoopClosureEngine(
        config,
        args.data_path,
        cache_path_lc,
        base_model,
        args.window_size,
        args.overlap,
        args.sample_interval,
    )

    cache_files = sorted(
        glob.glob(str(engine.temp_cache_dir / "window_cache_*.pt")),
        key=lambda p: int(p.split("_")[-1].split(".")[0]),
    )
    raw_predictions = [StreamingWindowEngineLC.parse_cache_file(cache_fname) for cache_fname in cache_files]
    sim3_list_lc = lc_engine.run(raw_predictions)
    sim3_list_lc.insert(0, raw_predictions[0]["sim3"])

    os.makedirs(str(cache_path_lc), exist_ok=True)
    for idx, (pred, sim3_lc) in enumerate(zip(raw_predictions, sim3_list_lc)):
        pred["sim3"] = sim3_lc
        torch.save(pred, str(cache_path_lc / f"window_cache_{idx}.pt"))

    cache_files_lc = sorted(
        glob.glob(str(cache_path_lc / "window_cache_*.pt")),
        key=lambda p: int(p.split("_")[-1].split(".")[0]),
    )
    parsed_caches = [StreamingWindowEngineLC.parse_cache_file(cache_files_lc[0])]
    for cache_fname in cache_files_lc[1:]:
        parsed_caches.append(StreamingWindowEngineLC.parse_cache_file(cache_fname, overlap=args.overlap))

    ret_dict = StreamingWindowEngineLC._post_process_pred(
        StreamingWindowEngineLC.aggregate_caches(parsed_caches)
    )
    shutil.rmtree(cache_path_lc)
    for key in ret_dict.keys():
        if isinstance(ret_dict[key], torch.Tensor):
            ret_dict[key] = ret_dict[key].cpu().numpy().squeeze(0)

    save_for_viser(ret_dict, scene_name, args.output_path, inverse_extrinsic=False)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    base_model, model = load_model(args)
    model.eval()
    scene_name = run_dynamic_scene(args, model)
    run_loop_closure(args, base_model, model, scene_name)
