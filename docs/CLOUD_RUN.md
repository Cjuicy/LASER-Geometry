# Cloud Run Notes

These commands assume the repository code is cloned without `data/` and `weights/`.
Place datasets and checkpoints on the cloud machine after cloning.

## Environment

```bash
conda activate vggt-dem
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
