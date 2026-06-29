import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from eval.build_alignment_pipeline_report import (
    build_report,
    render_confidence_stage,
    render_overlap_stage,
    render_propagation_stage,
    render_segmentation_stage,
)


def _rgb():
    image = np.full((8, 12, 3), 160, dtype=np.uint8)
    image[:, :6, 0] = 220
    return image


def _labels():
    return np.tile(np.array([0] * 6 + [1] * 6, dtype=np.int32), (8, 1))


def test_confidence_stage_dims_rejected_pixels():
    mask = np.zeros((8, 12), dtype=bool)
    mask[:, 6:] = True

    rendered = render_confidence_stage(_rgb(), mask, mutual_mask=None)

    assert rendered[:, :6].mean() < rendered[:, 6:].mean()


def test_segmentation_stage_draws_boundaries_and_emphasizes_anchors():
    mask = np.zeros((8, 12), dtype=bool)
    mask[:, 6:] = True

    rendered, details = render_segmentation_stage(
        _rgb(),
        _labels(),
        mask,
        merged_labels=_labels(),
        anchor_ids=np.array([1], dtype=np.int32),
    )

    assert rendered.shape == _rgb().shape
    assert not np.array_equal(rendered, _rgb())
    assert details["segment_count"] == 2
    assert details["anchor_segment_count"] == 1


def test_overlap_stage_renders_source_target_and_all_matches():
    matches = [
        {"src_segment": 0, "tgt_segment": 0, "iou": 0.67, "scale": 1.04},
        {"src_segment": 1, "tgt_segment": 1, "iou": 0.51, "scale": 0.97},
    ]

    image, details = render_overlap_stage(_rgb(), _labels(), _labels(), matches)

    assert image.shape[1] == _rgb().shape[1] * 2
    assert len(details["matches"]) == 2


def test_propagation_stage_reports_a_p_i_roles():
    current_labels = _labels().copy()
    current_labels[:2, :2] = 2
    states = [
        {"segment": 0, "role": "A", "scale": 0.95},
        {"segment": 1, "role": "P", "scale": 1.05},
        {"segment": 2, "role": "I", "scale": 1.0},
    ]
    edges = [
        {"parent_segment": 0, "child_segment": 0, "iou": 0.7, "scale": 0.95},
        {"parent_segment": 1, "child_segment": 1, "iou": 0.6, "scale": 1.05},
    ]

    image, details = render_propagation_stage(
        _rgb(),
        _rgb(),
        _labels(),
        current_labels,
        states,
        edges,
    )

    assert image.shape[1] == _rgb().shape[1] * 2
    assert {item["role"] for item in details["segments"]} == {"A", "P", "I"}


def _metadata(segment_mode):
    return {
        "schema_version": 1,
        "segment_mode": segment_mode,
        "normal_method": "cross",
        "scale_anchor_mode": "depth_irls",
        "window_size": 2,
        "overlap": 1,
        "sample_interval": 1,
        "confidence_retained_fraction": 0.3,
        "confidence_quantile": 0.7,
        "graph_iou_threshold": 0.3,
        "anchor_iou_threshold": 0.4,
    }


def _write_window(path, window_index, global_indices, with_alignment):
    initial = np.stack([_labels(), _labels()], axis=0).astype(np.uint16)
    if path.parent.parent.name == "geometry":
        initial[:, :2, :2] = 2
    merged = np.stack([_labels(), _labels()], axis=0).astype(np.uint16)
    if with_alignment:
        mutual = np.ones((1, 8, 12), dtype=bool)
        match_frame = np.array([0], dtype=np.int32)
        match_src = np.array([0], dtype=np.int32)
        match_tgt = np.array([0], dtype=np.int32)
        match_iou = np.array([0.75], dtype=np.float32)
        match_scale = np.array([1.05], dtype=np.float32)
        prop_parent_frame = np.array([0], dtype=np.int32)
        prop_parent_segment = np.array([0], dtype=np.int32)
        prop_child_frame = np.array([1], dtype=np.int32)
        prop_child_segment = np.array([0], dtype=np.int32)
        prop_iou = np.array([0.7], dtype=np.float32)
        prop_scale = np.array([1.05], dtype=np.float32)
        segment_role = np.array([2, 0, 1, 0], dtype=np.uint8)
        segment_scale = np.array([1.05, 1.0, 1.05, 1.0], dtype=np.float32)
    else:
        mutual = np.empty((0, 8, 12), dtype=bool)
        match_frame = match_src = match_tgt = np.empty(0, dtype=np.int32)
        match_iou = match_scale = np.empty(0, dtype=np.float32)
        prop_parent_frame = prop_parent_segment = np.empty(0, dtype=np.int32)
        prop_child_frame = prop_child_segment = np.empty(0, dtype=np.int32)
        prop_iou = prop_scale = np.empty(0, dtype=np.float32)
        segment_role = np.zeros(4, dtype=np.uint8)
        segment_scale = np.ones(4, dtype=np.float32)

    np.savez_compressed(
        path,
        global_frame_indices=np.asarray(global_indices, dtype=np.int32),
        confidence_thresholds=np.array([0.6, 0.65], dtype=np.float32),
        high_confidence_masks=np.ones((2, 8, 12), dtype=bool),
        initial_labels=initial,
        merged_labels=merged,
        mutual_confidence_masks=mutual,
        match_frame=match_frame,
        match_src_segment=match_src,
        match_tgt_segment=match_tgt,
        match_iou=match_iou,
        match_scale=match_scale,
        prop_parent_frame=prop_parent_frame,
        prop_parent_segment=prop_parent_segment,
        prop_child_frame=prop_child_frame,
        prop_child_segment=prop_child_segment,
        prop_iou=prop_iou,
        prop_scale=prop_scale,
        segment_frame=np.array([0, 0, 1, 1], dtype=np.int32),
        segment_id=np.array([0, 1, 0, 1], dtype=np.int32),
        segment_role=segment_role,
        segment_scale=segment_scale,
    )


def _write_run(tmp_path, name, mode):
    root = tmp_path / name
    pipeline = root / "pipeline"
    pipeline.mkdir(parents=True)
    (pipeline / "meta.json").write_text(
        json.dumps(_metadata(mode)),
        encoding="utf-8",
    )
    _write_window(pipeline / "window_0000.npz", 0, [0, 1], False)
    _write_window(pipeline / "window_0001.npz", 1, [1, 2], True)
    return root


def _write_images(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for index in range(3):
        image = _rgb().copy()
        image[:, :, 1] = 40 + index * 50
        cv2.imwrite(str(image_dir / f"{index:06d}.png"), image)
    return image_dir


def test_report_rejects_mismatched_confidence_fraction(tmp_path):
    baseline = _write_run(tmp_path, "depth", "depth")
    geometry = _write_run(tmp_path, "geometry", "geometry")
    meta_path = geometry / "pipeline" / "meta.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["confidence_retained_fraction"] = 0.4
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="confidence_retained_fraction"):
        build_report(baseline, geometry, _write_images(tmp_path), tmp_path / "report")


def test_report_writes_ten_assets_per_row_and_single_entry_html(tmp_path):
    baseline = _write_run(tmp_path, "depth", "depth")
    geometry = _write_run(tmp_path, "geometry", "geometry")
    out_dir = tmp_path / "report"

    build_report(baseline, geometry, _write_images(tmp_path), out_dir)

    manifest = json.loads((out_dir / "data.json").read_text(encoding="utf-8"))
    assert len(manifest["rows"]) == 4
    assert all(len(row["stages"]) == 10 for row in manifest["rows"])
    html = (out_dir / "index.html").read_text(encoding="utf-8")
    assert 'loading="lazy"' in html
    assert 'class="pipeline-row"' in html
    for row in manifest["rows"]:
        for stage in row["stages"]:
            assert (out_dir / stage["asset"]).is_file()
