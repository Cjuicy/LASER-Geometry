# Cloud Run Notes

These commands assume the repository code is cloned without `data/` and `weights/`.
Place datasets and checkpoints on the cloud machine after cloning.

## Environment

```bash
conda activate vggt
pip install -r requirements.txt
python setup.py build_ext --inplace
```

Loop closure needs the optional retrieval stack:

```bash
pip install faiss-gpu-cu12 numpy==1.26.4
```

## Expected Local Assets

```text
weights/
  model.safetensors
  model.pt
  dino_salad.ckpt
  dinov2_vitb14_pretrain.pth
  ORBvoc.txt.tar.gz

data/
  09/
    image_2/
    image_3/
    poses.txt
    times.txt
    calib.txt
  rgbd_dataset_freiburg1_desk.tgz
```

`data/` and `weights/` are ignored by Git on purpose.

## Verify Utilities

```bash
python -m pytest tests/test_geometry_segmentation.py tests/test_eval_utilities.py -q
python demo.py --help
python demo_lc.py --help
```

## Desk Smoke Test

Extract the TUM RGB-D desk sequence, then point `demo.py` at its RGB folder.

```bash
tar -xzf data/rgbd_dataset_freiburg1_desk.tgz -C data/

python demo.py \
  --model_ckpt weights/model.safetensors \
  --data_path data/rgbd_dataset_freiburg1_desk/rgb \
  --scene_name desk_depth \
  --output_path outputs/viser \
  --cache_path cache/desk_depth \
  --sample_interval 5 \
  --window_size 20 \
  --overlap 5 \
  --depth_refine \
  --segment_mode depth

python demo.py \
  --model_ckpt weights/model.safetensors \
  --data_path data/rgbd_dataset_freiburg1_desk/rgb \
  --scene_name desk_geometry \
  --output_path outputs/viser \
  --cache_path cache/desk_geometry \
  --sample_interval 5 \
  --window_size 20 \
  --overlap 5 \
  --depth_refine \
  --segment_mode geometry \
  --normal_method cross
```

## KITTI09 Smoke Test

Use a larger `sample_interval` first. Drop it to `1` only after smoke tests pass.

```bash
python demo.py \
  --model_ckpt weights/model.safetensors \
  --data_path data/09/image_2 \
  --scene_name kitti09_depth_s10 \
  --output_path outputs/viser \
  --cache_path cache/kitti09_depth_s10 \
  --sample_interval 10 \
  --window_size 30 \
  --overlap 10 \
  --depth_refine \
  --segment_mode depth

python demo.py \
  --model_ckpt weights/model.safetensors \
  --data_path data/09/image_2 \
  --scene_name kitti09_geometry_s10 \
  --output_path outputs/viser \
  --cache_path cache/kitti09_geometry_s10 \
  --sample_interval 10 \
  --window_size 30 \
  --overlap 10 \
  --depth_refine \
  --segment_mode geometry \
  --normal_method cross
```

## KITTI09 Quick Evaluation

Predictions saved by `save_for_viser` are in TUM format. KITTI GT can be read with
`--gt_format kitti`.

```bash
python eval/quick_eval_local.py \
  --pred outputs/viser/kitti09_geometry_s10/pred_traj.txt \
  --gt data/09/poses.txt \
  --pred_format tum \
  --gt_format kitti \
  --out_dir outputs/eval/kitti09_geometry_s10 \
  --seq kitti09_geometry_s10
```

## Geometry A1 Baseline-Parameter Experiment

This is the non-LC, non-debug KITTI 00 `s1-w75-o30` run for the Geometry A1
baseline-parameter check. The only variable changed versus legacy Geometry is
the Felzenszwalb setting: `200/1.0/300` becomes `300/1.1/500`.

```bash
cd ~/autodl-tmp/LASER-Geometry-main
conda activate vggt

SCENE=kitti00_pi3_geometry_a1_parammatch_s1_w75_o30

python demo.py \
  --model_ckpt weights/model.safetensors \
  --data_path data/00/image_2 \
  --scene_name "$SCENE" \
  --output_path outputs/viser \
  --cache_path cache/"$SCENE" \
  --sample_interval 1 \
  --window_size 75 \
  --overlap 30 \
  --top_conf_percentile 0.3 \
  --depth_refine \
  --segment_mode geometry \
  --geometry_seg_profile baseline_params \
  --normal_method cross \
  --scale_anchor_mode depth_irls
```

Do not pass `--debug_alignment` here. This run only needs the normal Viser
output plus the later evaluation output.

If KITTI ground truth is stored at `data/poses/00.txt`, use that file. If not,
fall back to `data/dataset/poses/00.txt`. Fail clearly if neither path exists.

```bash
if [ -f data/poses/00.txt ]; then
  GT=data/poses/00.txt
elif [ -f data/dataset/poses/00.txt ]; then
  GT=data/dataset/poses/00.txt
else
  echo "Missing KITTI GT: expected data/poses/00.txt or data/dataset/poses/00.txt" >&2
  exit 1
fi

python eval/quick_eval_local.py \
  --pred outputs/viser/"$SCENE"/pred_traj.txt \
  --gt "$GT" \
  --pred_format tum \
  --gt_format kitti \
  --gt_stride 1 \
  --out_dir outputs/eval/"$SCENE" \
  --seq "$SCENE"
```

After the run, inspect the metrics file and the output directories:

```bash
cat outputs/eval/"$SCENE"/"${SCENE}_metrics.txt"
ls -la outputs/viser/"$SCENE"
ls -la outputs/eval/"$SCENE"
```

## Depth vs Geometry Pipeline Report

Run both methods with exactly the same sampling, window, overlap, confidence,
and scale-anchor settings. `--top_conf_percentile 0.3` means the run requests
the highest-confidence 30%; the report also shows the actual retained pixel
ratio after quantile ties.

The report requires pipeline trace schema v2. V2 records the exact depth map
used to build each segmentation graph, after coarse Sim3 alignment and before
segment-scale refinement. Existing v1 traces must be regenerated; the report
intentionally rejects them instead of substituting the final Viser depth.

The following KITTI 08 example records every processed frame:

```bash
python demo.py \
  --model_ckpt weights/model.safetensors \
  --data_path data/08/image_2 \
  --scene_name kitti08_depth_pipeline_s10 \
  --output_path outputs/viser \
  --cache_path cache/kitti08_depth_pipeline_s10 \
  --sample_interval 10 \
  --window_size 30 \
  --overlap 10 \
  --top_conf_percentile 0.3 \
  --depth_refine \
  --segment_mode depth \
  --scale_anchor_mode depth_irls \
  --debug_alignment \
  --debug_alignment_path outputs/debug_alignment

python demo.py \
  --model_ckpt weights/model.safetensors \
  --data_path data/08/image_2 \
  --scene_name kitti08_geometry_pipeline_s10 \
  --output_path outputs/viser \
  --cache_path cache/kitti08_geometry_pipeline_s10 \
  --sample_interval 10 \
  --window_size 30 \
  --overlap 10 \
  --top_conf_percentile 0.3 \
  --depth_refine \
  --segment_mode geometry \
  --normal_method cross \
  --scale_anchor_mode depth_irls \
  --debug_alignment \
  --debug_alignment_path outputs/debug_alignment
```

Build the ten-image-per-frame report:

```bash
python eval/build_alignment_pipeline_report.py \
  --baseline_debug_dir outputs/debug_alignment/kitti08_depth_pipeline_s10 \
  --geometry_debug_dir outputs/debug_alignment/kitti08_geometry_pipeline_s10 \
  --image_dir data/08/image_2 \
  --sample_interval 10 \
  --out_dir outputs/pipeline_report/kitti08_s10
```

Each row still contains ten stage cards: five for Baseline/Depth and five for
Geometry. The initial-segmentation and merged-segmentation cards show the input
depth heatmap beside the segmentation result. Both methods use the same p02-p98
depth color range for that frame. Clicking any card opens the matching Depth
and Geometry stage together for direct comparison.

The builder also writes `player.html`. It compares the previous and current
canonical global frames for Baseline/Depth and Geometry, with Initial/Merged
stage selection, playback speed, seeking, and frame controls. Open
`/player.html` from the same server.

If schema-v2 traces already exist, code updates to the report or player only
require rerunning `eval/build_alignment_pipeline_report.py`; model inference
does not need to run again.

`--sample_interval` on the report command is only a consistency check; the
recorded runtime metadata remains authoritative. The default report contains
all recorded windows and frames. For a quick preview, add for example
`--window_stop 2 --frame_step 5`.

Open `outputs/pipeline_report/kitti08_s10/index.html` directly, or serve it on
the cloud machine:

```bash
python -m http.server 8080 \
  --bind 0.0.0.0 \
  --directory outputs/pipeline_report/kitti08_s10
```

Then visit the forwarded port root, for example `http://127.0.0.1:8080/`.

## GitHub Upload

After checking the run commands, initialize and upload only code:

```bash
git init
git status --short
git add .
git status --short
git commit -m "feat: add geometry-aware LASER segmentation"
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

Before committing, confirm `git status --short` does not show `data/` or `weights/`.
