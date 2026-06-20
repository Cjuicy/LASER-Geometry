import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.vo_eval import load_traj, eval_metrics, plot_trajectory


def main():
    """
    中文：
    轻量本地轨迹评估脚本。
    不重新跑模型，不依赖完整 dataset pipeline。
    只读取预测轨迹和 GT 轨迹，计算 ATE / RPE，并保存轨迹图。

    English:
    Lightweight local trajectory evaluation script.
    It does not rerun the model and does not depend on the full dataset pipeline.
    It only reads predicted and GT trajectories, computes ATE / RPE, and saves trajectory plots.
    """
    parser = argparse.ArgumentParser()

    # 中文：预测轨迹文件。
    # English: Predicted trajectory file.
    parser.add_argument("--pred", required=True)

    # 中文：GT 轨迹文件。
    # English: Ground-truth trajectory file.
    parser.add_argument("--gt", required=True)

    # 中文：预测轨迹格式。
    # English: Predicted trajectory format.
    parser.add_argument("--pred_format", default="tum")

    # 中文：GT 轨迹格式。
    # English: GT trajectory format.
    parser.add_argument("--gt_format", default="tum")

    # 中文：输出文件夹。
    # English: Output directory.
    parser.add_argument("--out_dir", required=True)

    # 中文：序列名字。
    # English: Sequence name.
    parser.add_argument("--seq", default="quick_test")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 中文：读取预测轨迹。
    # English: Load predicted trajectory.
    pred_traj = load_traj(args.pred, traj_format=args.pred_format)

    # 中文：读取 GT 轨迹。
    # English: Load GT trajectory.
    gt_traj = load_traj(args.gt, traj_format=args.gt_format)

    metric_file = os.path.join(args.out_dir, f"{args.seq}_metrics.txt")
    plot_file = os.path.join(args.out_dir, f"{args.seq}_trajectory.png")

    # 中文：计算 ATE / RPE。
    # English: Compute ATE / RPE.
    ate, rpe_trans, rpe_rot = eval_metrics(
        pred_traj,
        gt_traj,
        seq=args.seq,
        filename=metric_file,
    )

    # 中文：保存轨迹图。
    # English: Save trajectory plot.
    plot_trajectory(
        pred_traj,
        gt_traj,
        title=args.seq,
        filename=plot_file,
    )

    print("ATE:", ate)
    print("RPE trans:", rpe_trans)
    print("RPE rot:", rpe_rot)


if __name__ == "__main__":
    main()
