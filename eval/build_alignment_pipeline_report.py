"""Build a static, scrollable depth-vs-geometry alignment pipeline report."""

import argparse
import html
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.image_sequence import list_image_paths


ROLE_CODES = {0: "I", 1: "P", 2: "A"}
SOURCE_BOUNDARY_BGR = (255, 220, 40)
TARGET_BOUNDARY_BGR = (210, 80, 255)
LOW_CONF_BRIGHTNESS = 0.22
COMPARABLE_FIELDS = (
    "window_size",
    "overlap",
    "sample_interval",
    "confidence_retained_fraction",
    "confidence_quantile",
    "graph_iou_threshold",
    "anchor_iou_threshold",
    "scale_anchor_mode",
)
STAGE_ORDER = (
    ("depth", "confidence"),
    ("depth", "initial"),
    ("depth", "merged"),
    ("depth", "overlap"),
    ("depth", "propagation"),
    ("geometry", "confidence"),
    ("geometry", "initial"),
    ("geometry", "merged"),
    ("geometry", "overlap"),
    ("geometry", "propagation"),
)
STAGE_TITLES = {
    "confidence": "1 置信度筛选",
    "initial": "2 初始分割",
    "merged": "3 融合后分割",
    "overlap": "4 Overlap 锚点",
    "propagation": "5 窗口内传播",
}
PLAYBACK_METHODS = ("depth", "geometry")
PLAYBACK_STAGES = ("initial", "merged")


def _as_bool_mask(mask, shape):
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"mask shape {mask.shape} does not match image shape {shape}")
    return mask


def _segment_boundaries(labels):
    labels = np.asarray(labels)
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    return boundary


def _mask_boundaries(mask):
    mask_u8 = np.asarray(mask, dtype=np.uint8) * 255
    return cv2.Canny(mask_u8, 50, 150) > 0


def _deterministic_colors(labels):
    ids = np.unique(labels)
    colors = {}
    for label_id in ids:
        seed = int(label_id) * 2654435761 % (2**32)
        rng = np.random.default_rng(seed)
        colors[int(label_id)] = tuple(int(x) for x in rng.integers(45, 225, 3))
    return colors


def _dim_confidence(rgb_bgr, high_mask):
    rgb_bgr = np.asarray(rgb_bgr, dtype=np.uint8)
    high_mask = _as_bool_mask(high_mask, rgb_bgr.shape[:2])
    result = (rgb_bgr.astype(np.float32) * LOW_CONF_BRIGHTNESS).astype(np.uint8)
    result[high_mask] = rgb_bgr[high_mask]
    return result


def _draw_badge(image, text, origin=(6, 16), color=(255, 255, 255)):
    image = image.copy()
    cv2.putText(
        image,
        str(text),
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        str(text),
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        1,
        cv2.LINE_AA,
    )
    return image


def _segment_centroid(labels, segment_id):
    ys, xs = np.where(np.asarray(labels) == segment_id)
    if xs.size == 0:
        return None
    return int(np.mean(xs)), int(np.mean(ys))


def _scale_color(scale):
    delta = float(np.clip((scale - 1.0) / 0.2, -1.0, 1.0))
    white = np.array([245, 245, 245], dtype=np.float32)
    endpoint = (
        np.array([220, 95, 65], dtype=np.float32)
        if delta < 0
        else np.array([65, 85, 220], dtype=np.float32)
    )
    color = white * (1.0 - abs(delta)) + endpoint * abs(delta)
    return tuple(int(x) for x in color)


def compute_shared_depth_range(*depth_maps):
    valid_chunks = []
    for depth in depth_maps:
        depth = np.asarray(depth, dtype=np.float32)
        valid = depth[np.isfinite(depth)]
        if valid.size:
            valid_chunks.append(valid)
    if not valid_chunks:
        return 0.0, 1.0

    values = np.concatenate(valid_chunks)
    lo, hi = np.percentile(values, (2.0, 98.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(values))
        hi = float(np.max(values))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def colorize_depth(depth, display_range):
    depth = np.asarray(depth, dtype=np.float32)
    lo, hi = display_range
    finite = np.isfinite(depth)
    normalized = np.zeros(depth.shape, dtype=np.float32)
    normalized[finite] = np.clip((depth[finite] - lo) / (hi - lo), 0.0, 1.0)
    image = cv2.applyColorMap(
        (normalized * 255).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    image[~finite] = 0

    valid = depth[finite]
    details = {
        "valid_pixels": int(valid.size),
        "depth_min": float(np.min(valid)) if valid.size else None,
        "depth_p02": float(np.percentile(valid, 2)) if valid.size else None,
        "depth_p50": float(np.percentile(valid, 50)) if valid.size else None,
        "depth_p98": float(np.percentile(valid, 98)) if valid.size else None,
        "depth_max": float(np.max(valid)) if valid.size else None,
        "shared_color_min": float(lo),
        "shared_color_max": float(hi),
    }
    return image, details


def compose_depth_segmentation(depth_color, segmentation):
    depth_color = np.asarray(depth_color, dtype=np.uint8)
    segmentation = np.asarray(segmentation, dtype=np.uint8)
    if depth_color.shape != segmentation.shape:
        raise ValueError("depth and segmentation images must have the same shape")
    return np.concatenate([depth_color, segmentation], axis=1)


def render_confidence_stage(rgb_bgr, high_mask, mutual_mask=None):
    result = _dim_confidence(rgb_bgr, high_mask)
    result[_mask_boundaries(high_mask)] = (255, 255, 255)
    if mutual_mask is not None:
        mutual_mask = _as_bool_mask(mutual_mask, result.shape[:2])
        result[_mask_boundaries(mutual_mask)] = (255, 255, 0)
    return result


def render_segmentation_stage(
    rgb_bgr,
    labels,
    high_mask,
    *,
    merged_labels=None,
    anchor_ids=None,
):
    labels = np.asarray(labels)
    merged_labels = labels if merged_labels is None else np.asarray(merged_labels)
    anchor_ids = np.asarray([] if anchor_ids is None else anchor_ids, dtype=np.int64)
    result = _dim_confidence(rgb_bgr, high_mask)

    colors = _deterministic_colors(labels)
    tint = np.zeros_like(result)
    for label_id, color in colors.items():
        tint[labels == label_id] = color
    high_mask = np.asarray(high_mask, dtype=bool)
    result[high_mask] = cv2.addWeighted(
        result,
        0.78,
        tint,
        0.22,
        0.0,
    )[high_mask]

    boundaries = _segment_boundaries(labels)
    result[boundaries] = (105, 105, 105)
    anchor_mask = np.isin(merged_labels, anchor_ids)
    emphasized = boundaries & anchor_mask
    emphasized = cv2.dilate(emphasized.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
    result[emphasized] = (0, 230, 255)
    return result, {
        "segment_count": int(np.unique(labels).size),
        "anchor_segment_count": int(np.unique(merged_labels[anchor_mask]).size)
        if np.any(anchor_mask)
        else 0,
    }


def render_overlap_stage(rgb_bgr, src_labels, tgt_labels, matches):
    src = np.asarray(rgb_bgr, dtype=np.uint8).copy()
    tgt = np.asarray(rgb_bgr, dtype=np.uint8).copy()
    src[_segment_boundaries(src_labels)] = SOURCE_BOUNDARY_BGR
    tgt[_segment_boundaries(tgt_labels)] = TARGET_BOUNDARY_BGR
    combined = np.concatenate([src, tgt], axis=1)
    width = src.shape[1]

    details = {"matches": []}
    for match in matches:
        src_id = int(match["src_segment"])
        tgt_id = int(match["tgt_segment"])
        src_center = _segment_centroid(src_labels, src_id)
        tgt_center = _segment_centroid(tgt_labels, tgt_id)
        if src_center is None or tgt_center is None:
            continue
        tgt_center = (tgt_center[0] + width, tgt_center[1])
        iou = float(match["iou"])
        thickness = max(1, int(round(1 + iou * 3)))
        cv2.line(combined, src_center, tgt_center, (80, 220, 120), thickness, cv2.LINE_AA)
        cv2.circle(combined, src_center, 3, SOURCE_BOUNDARY_BGR, -1)
        cv2.circle(combined, tgt_center, 3, TARGET_BOUNDARY_BGR, -1)
        details["matches"].append(
            {
                "src_segment": src_id,
                "tgt_segment": tgt_id,
                "iou": iou,
                "scale": float(match["scale"]),
            }
        )

    if not details["matches"]:
        combined = _draw_badge(combined, "No accepted IoU match", origin=(8, 20))
    else:
        combined = _draw_badge(combined, f"accepted matches: {len(details['matches'])}")
    return combined, details


def render_propagation_stage(
    previous_rgb,
    current_rgb,
    previous_labels,
    current_labels,
    segment_states,
    edges,
):
    current = np.asarray(current_rgb, dtype=np.uint8).copy()
    current_labels = np.asarray(current_labels)
    state_details = []
    for state in segment_states:
        segment_id = int(state["segment"])
        role = str(state["role"])
        scale = float(state["scale"])
        mask = current_labels == segment_id
        if np.any(mask):
            color = np.asarray(_scale_color(scale), dtype=np.float32)
            current[mask] = np.clip(
                current[mask].astype(np.float32) * 0.55 + color * 0.45,
                0,
                255,
            ).astype(np.uint8)
            center = _segment_centroid(current_labels, segment_id)
            if center is not None:
                current = _draw_badge(current, role, origin=(center[0], center[1]), color=(255, 255, 255))
        state_details.append({"segment": segment_id, "role": role, "scale": scale})
    current[_segment_boundaries(current_labels)] = (225, 225, 225)

    if previous_rgb is None or previous_labels is None:
        previous = np.full_like(current, 28)
        previous = _draw_badge(previous, "No previous target frame", origin=(6, 20))
        previous_labels = np.zeros(current_labels.shape, dtype=np.int32)
    else:
        previous = np.asarray(previous_rgb, dtype=np.uint8).copy()
        previous[_segment_boundaries(previous_labels)] = (150, 150, 150)

    combined = np.concatenate([previous, current], axis=1)
    width = previous.shape[1]
    edge_details = []
    for edge in edges:
        parent_id = int(edge["parent_segment"])
        child_id = int(edge["child_segment"])
        parent_center = _segment_centroid(previous_labels, parent_id)
        child_center = _segment_centroid(current_labels, child_id)
        if parent_center is None or child_center is None:
            continue
        child_center = (child_center[0] + width, child_center[1])
        iou = float(edge["iou"])
        cv2.arrowedLine(
            combined,
            parent_center,
            child_center,
            (255, 200, 80),
            max(1, int(round(1 + iou * 2))),
            cv2.LINE_AA,
            tipLength=0.12,
        )
        edge_details.append(
            {
                "parent_segment": parent_id,
                "child_segment": child_id,
                "iou": iou,
                "scale": float(edge["scale"]),
            }
        )
    return combined, {"segments": state_details, "edges": edge_details}


def _placeholder(rgb_bgr, text):
    image = np.concatenate([rgb_bgr, rgb_bgr], axis=1)
    image = (image.astype(np.float32) * 0.18).astype(np.uint8)
    return _draw_badge(image, text, origin=(8, 22)), {"message": text}


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _load_pipeline_run(debug_dir):
    debug_dir = Path(debug_dir)
    pipeline_dir = debug_dir / "pipeline"
    meta_path = pipeline_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing pipeline metadata: {meta_path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    windows = []
    for path in sorted(pipeline_dir.glob("window_*.npz")):
        with np.load(path, allow_pickle=False) as arrays:
            windows.append(
                {
                    "name": path.stem,
                    "path": path,
                    "arrays": {key: arrays[key] for key in arrays.files},
                }
            )
    if not windows:
        raise FileNotFoundError(f"No window_*.npz files found in {pipeline_dir}")
    return {"debug_dir": debug_dir, "metadata": metadata, "windows": windows}


def validate_comparable_runs(baseline_meta, geometry_meta):
    if baseline_meta.get("segment_mode") != "depth":
        raise ValueError("baseline segment_mode must be depth")
    if geometry_meta.get("segment_mode") != "geometry":
        raise ValueError("geometry segment_mode must be geometry")
    if baseline_meta.get("scale_anchor_mode") != "depth_irls":
        raise ValueError("this report requires scale_anchor_mode=depth_irls")
    if geometry_meta.get("scale_anchor_mode") != "depth_irls":
        raise ValueError("this report requires scale_anchor_mode=depth_irls")

    for field in COMPARABLE_FIELDS:
        if field not in baseline_meta or field not in geometry_meta:
            raise ValueError(f"Missing comparable metadata field: {field}")
        left = baseline_meta[field]
        right = geometry_meta[field]
        equal = np.isclose(left, right) if isinstance(left, float) else left == right
        if not equal:
            raise ValueError(f"Mismatched {field}: {left!r} vs {right!r}")


def _records_for_frame(arrays, frame_idx, prefix):
    if prefix == "match":
        selected = arrays["match_frame"] == frame_idx
        return [
            {
                "src_segment": int(src),
                "tgt_segment": int(tgt),
                "iou": float(iou),
                "scale": float(scale),
            }
            for src, tgt, iou, scale in zip(
                arrays["match_src_segment"][selected],
                arrays["match_tgt_segment"][selected],
                arrays["match_iou"][selected],
                arrays["match_scale"][selected],
            )
        ]
    if prefix == "segment":
        selected = arrays["segment_frame"] == frame_idx
        return [
            {
                "segment": int(segment),
                "role": ROLE_CODES[int(role)],
                "scale": float(scale),
            }
            for segment, role, scale in zip(
                arrays["segment_id"][selected],
                arrays["segment_role"][selected],
                arrays["segment_scale"][selected],
            )
        ]
    if prefix == "prop":
        selected = arrays["prop_child_frame"] == frame_idx
        return [
            {
                "parent_frame": int(parent_frame),
                "parent_segment": int(parent_segment),
                "child_frame": int(child_frame),
                "child_segment": int(child_segment),
                "iou": float(iou),
                "scale": float(scale),
            }
            for parent_frame, parent_segment, child_frame, child_segment, iou, scale in zip(
                arrays["prop_parent_frame"][selected],
                arrays["prop_parent_segment"][selected],
                arrays["prop_child_frame"][selected],
                arrays["prop_child_segment"][selected],
                arrays["prop_iou"][selected],
                arrays["prop_scale"][selected],
            )
        ]
    raise ValueError(f"Unknown record prefix: {prefix}")


def _write_webp(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_WEBP_QUALITY, 88]):
        raise RuntimeError(f"Failed to write report asset: {path}")


def _stage_asset_entry(method, stage, path, details):
    return {
        "method": method,
        "stage": stage,
        "title": STAGE_TITLES[stage],
        "asset": path.as_posix(),
        "details": _jsonable(details),
    }


def _render_method_stages(
    *,
    method,
    window_index,
    local_frame,
    rgb,
    previous_rgb,
    arrays,
    previous_window_arrays,
    overlap,
    assets_dir,
    shared_depth_range,
):
    high_mask = arrays["high_confidence_masks"][local_frame]
    mutual_mask = None
    if window_index > 0 and local_frame < len(arrays["mutual_confidence_masks"]):
        mutual_mask = arrays["mutual_confidence_masks"][local_frame]
    matches = _records_for_frame(arrays, local_frame, "match")
    states = _records_for_frame(arrays, local_frame, "segment")
    edges = _records_for_frame(arrays, local_frame, "prop")
    anchor_ids = np.asarray(
        [match["tgt_segment"] for match in matches],
        dtype=np.int32,
    )
    depth_color, depth_details = colorize_depth(
        arrays["segmentation_depths"][local_frame],
        shared_depth_range,
    )
    depth_color = _draw_badge(depth_color, "Segmentation depth")

    prefix = Path("assets") / f"window_{window_index:04d}"
    stages = []

    confidence_image = render_confidence_stage(rgb, high_mask, mutual_mask)
    confidence_path = prefix / f"frame_{local_frame:04d}_{method}_confidence.webp"
    _write_webp(assets_dir.parent / confidence_path, confidence_image)
    stages.append(
        _stage_asset_entry(
            method,
            "confidence",
            confidence_path,
            {
                "threshold": float(arrays["confidence_thresholds"][local_frame]),
                "actual_retained_ratio": float(np.mean(high_mask)),
                "mutual_registration_pixels": int(np.count_nonzero(mutual_mask))
                if mutual_mask is not None
                else 0,
            },
        )
    )

    initial_image, initial_details = render_segmentation_stage(
        rgb,
        arrays["initial_labels"][local_frame],
        high_mask,
        merged_labels=arrays["merged_labels"][local_frame],
        anchor_ids=anchor_ids,
    )
    initial_image = compose_depth_segmentation(
        depth_color,
        _draw_badge(initial_image, "Initial segments"),
    )
    initial_details["depth"] = depth_details
    initial_path = prefix / f"frame_{local_frame:04d}_{method}_initial.webp"
    _write_webp(assets_dir.parent / initial_path, initial_image)
    stages.append(_stage_asset_entry(method, "initial", initial_path, initial_details))

    merged_image, merged_details = render_segmentation_stage(
        rgb,
        arrays["merged_labels"][local_frame],
        high_mask,
        merged_labels=arrays["merged_labels"][local_frame],
        anchor_ids=anchor_ids,
    )
    merged_details["merge_ratio"] = float(
        1.0
        - merged_details["segment_count"]
        / max(initial_details["segment_count"], 1)
    )
    merged_image = compose_depth_segmentation(
        depth_color,
        _draw_badge(merged_image, "Merged segments"),
    )
    merged_details["depth"] = depth_details
    merged_path = prefix / f"frame_{local_frame:04d}_{method}_merged.webp"
    _write_webp(assets_dir.parent / merged_path, merged_image)
    stages.append(_stage_asset_entry(method, "merged", merged_path, merged_details))

    if window_index == 0:
        overlap_image, overlap_details = _placeholder(rgb, "First window / no overlap alignment")
    elif local_frame >= overlap:
        overlap_image, overlap_details = _placeholder(rgb, "Not an overlap frame")
    else:
        src_index = len(previous_window_arrays["merged_labels"]) - overlap + local_frame
        src_labels = previous_window_arrays["merged_labels"][src_index]
        overlap_image, overlap_details = render_overlap_stage(
            rgb,
            src_labels,
            arrays["merged_labels"][local_frame],
            matches,
        )
    overlap_path = prefix / f"frame_{local_frame:04d}_{method}_overlap.webp"
    _write_webp(assets_dir.parent / overlap_path, overlap_image)
    stages.append(_stage_asset_entry(method, "overlap", overlap_path, overlap_details))

    previous_labels = (
        arrays["merged_labels"][local_frame - 1] if local_frame > 0 else None
    )
    propagation_image, propagation_details = render_propagation_stage(
        previous_rgb,
        rgb,
        previous_labels,
        arrays["merged_labels"][local_frame],
        states,
        edges,
    )
    propagation_path = prefix / f"frame_{local_frame:04d}_{method}_propagation.webp"
    _write_webp(assets_dir.parent / propagation_path, propagation_image)
    stages.append(
        _stage_asset_entry(
            method,
            "propagation",
            propagation_path,
            propagation_details,
        )
    )
    return stages


def select_playback_row_indices(rows):
    selected = {}
    for row_index, row in enumerate(rows):
        global_frame = int(row["global_frame"])
        current_index = selected.get(global_frame)
        if current_index is None:
            selected[global_frame] = row_index
            continue
        current = rows[current_index]
        if current["is_overlap"] and not row["is_overlap"]:
            selected[global_frame] = row_index
    return [selected[global_frame] for global_frame in sorted(selected)]


def build_playback_manifest(manifest):
    frames = []
    for row_index in select_playback_row_indices(manifest["rows"]):
        row = manifest["rows"][row_index]
        lookup = {
            (stage["method"], stage["stage"]): stage for stage in row["stages"]
        }
        stages = {}
        for stage_name in PLAYBACK_STAGES:
            stages[stage_name] = {}
            for method in PLAYBACK_METHODS:
                stage = lookup.get((method, stage_name))
                if stage is None:
                    raise ValueError(
                        f"Missing {method} {stage_name} stage for "
                        f"global frame {row['global_frame']}"
                    )
                stages[stage_name][method] = {
                    "asset": stage["asset"],
                    "segment_count": int(stage["details"]["segment_count"]),
                }
        frames.append(
            {
                "global_frame": int(row["global_frame"]),
                "row_index": row_index,
                "stages": stages,
            }
        )
    return {"metadata": dict(manifest["metadata"]), "frames": frames}


def _build_html(manifest):
    row_html = []
    for row_index, row in enumerate(manifest["rows"]):
        stage_html = []
        for stage in row["stages"]:
            title = html.escape(stage["title"])
            asset = html.escape(stage["asset"])
            stage_name = html.escape(stage["stage"], quote=True)
            stage_html.append(
                f'<button class="stage-card" data-row="{row_index}" '
                f'data-stage-name="{stage_name}" '
                f'type="button"><img loading="lazy" src="{asset}" alt="{title}">'
                f'<span>{title}</span></button>'
            )
        badge = "O" if row["is_overlap"] else "N"
        row_html.append(
            '<section class="pipeline-row">'
            f'<div class="row-meta"><b>{badge}</b> W{row["window_index"]:04d} '
            f'L{row["local_frame"]:03d} / G{row["global_frame"]:05d}</div>'
            f'<div class="stage-grid">{"".join(stage_html)}</div>'
            '</section>'
        )

    manifest_json = json.dumps(manifest, ensure_ascii=False).replace("</", "<\\/")
    headers = "".join(
        f"<div>{html.escape(STAGE_TITLES[stage])}</div>"
        for _, stage in STAGE_ORDER
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>LASER Depth vs Geometry Pipeline</title>
  <style>
    :root {{ color-scheme: light; font-family: Arial, sans-serif; color: #17212b; background: #eef1f4; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; }}
    header {{ padding: 14px 18px; background: #fff; border-bottom: 1px solid #ccd3da; }}
    h1 {{ margin: 0 0 7px; font-size: 22px; letter-spacing: 0; }}
    .summary {{ margin: 0; color: #4d5a66; font-size: 13px; }}
    .report-nav {{ margin: 8px 0 0; }}
    .report-nav a {{ color: #245a8d; font-weight: 700; }}
    .sticky {{ position: sticky; top: 0; z-index: 5; background: #fff; border-bottom: 1px solid #aeb8c1; overflow-x: auto; }}
    .method-grid, .stage-header, .stage-grid {{ display: grid; grid-template-columns: repeat(10, minmax(170px, 1fr)); gap: 6px; min-width: 1760px; }}
    .method-grid {{ padding: 8px 12px 3px; }}
    .method-grid div {{ padding: 6px; font-weight: 700; }}
    .method-depth {{ grid-column: 1 / 6; color: #245a8d; border-left: 4px solid #245a8d; }}
    .method-geometry {{ grid-column: 6 / 11; color: #28734c; border-left: 4px solid #28734c; }}
    .stage-header {{ padding: 3px 12px 8px; font-size: 12px; font-weight: 700; }}
    .stage-header div {{ padding: 5px; border-bottom: 2px solid #d4dbe1; }}
    main {{ padding: 10px 12px 40px; }}
    .pipeline-row {{ margin: 0 0 10px; padding: 8px; background: #fff; border: 1px solid #d5dbe0; border-radius: 6px; overflow-x: auto; }}
    .row-meta {{ position: sticky; left: 0; width: max-content; margin-bottom: 6px; padding: 3px 7px; background: #f2f5f7; font-size: 12px; z-index: 2; }}
    .row-meta b {{ display: inline-block; min-width: 21px; text-align: center; color: #fff; background: #52616d; margin-right: 4px; }}
    .stage-card {{ display: block; min-width: 0; padding: 4px; text-align: left; color: inherit; background: #f8fafb; border: 1px solid #cbd3d9; border-radius: 4px; cursor: zoom-in; }}
    .stage-card img {{ display: block; width: 100%; aspect-ratio: 3 / 2; object-fit: contain; background: #18212a; }}
    .stage-card span {{ display: block; padding: 5px 2px 1px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 11px; }}
    dialog {{ width: min(1680px, 96vw); max-height: 94vh; padding: 0; border: 1px solid #7f8b95; border-radius: 6px; }}
    dialog::backdrop {{ background: rgba(0, 0, 0, .72); }}
    .modal-head {{ display: flex; justify-content: space-between; gap: 12px; padding: 10px 12px; border-bottom: 1px solid #d5dbe0; }}
    .modal-body {{ padding: 12px; overflow: auto; max-height: 82vh; }}
    .modal-compare {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 12px; }}
    .modal-panel {{ min-width: 0; }}
    .modal-panel h2 {{ margin: 0 0 8px; font-size: 15px; }}
    .modal-panel img {{ display: block; width: 100%; max-height: 62vh; object-fit: contain; margin: 0 auto 10px; background: #18212a; }}
    pre {{ white-space: pre-wrap; word-break: break-word; padding: 10px; background: #f2f5f7; font-size: 12px; }}
    button.close {{ border: 1px solid #aeb8c1; background: #fff; border-radius: 4px; padding: 4px 9px; cursor: pointer; }}
    @media (max-width: 760px) {{
      .modal-compare {{ grid-template-columns: 1fr; }}
      .modal-panel img {{ max-height: none; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>LASER Depth vs Geometry 对齐流水线</h1>
    <p class="summary">window={manifest['metadata']['window_size']} · overlap={manifest['metadata']['overlap']} · sample={manifest['metadata']['sample_interval']} · retained={manifest['metadata']['confidence_retained_fraction']:.3g} · quantile={manifest['metadata']['confidence_quantile']:.3g} · anchor=depth_irls</p>
    <p class="report-nav"><a href="player.html">播放分割序列</a></p>
  </header>
  <div class="sticky">
    <div class="method-grid"><div class="method-depth">BASELINE / DEPTH</div><div class="method-geometry">GEOMETRY</div></div>
    <div class="stage-header">{headers}</div>
  </div>
  <main>{''.join(row_html)}</main>
  <dialog id="detail-modal">
    <div class="modal-head"><b id="modal-title"></b><button class="close" type="button">关闭</button></div>
    <div class="modal-body">
      <div class="modal-compare">
        <section class="modal-panel">
          <h2>BASELINE / DEPTH</h2>
          <img id="modal-depth-image" alt="Baseline depth stage">
          <pre id="modal-depth-details"></pre>
        </section>
        <section class="modal-panel">
          <h2>GEOMETRY</h2>
          <img id="modal-geometry-image" alt="Geometry stage">
          <pre id="modal-geometry-details"></pre>
        </section>
      </div>
    </div>
  </dialog>
  <script id="pipeline-data" type="application/json">{manifest_json}</script>
  <script>
    const report = JSON.parse(document.getElementById('pipeline-data').textContent);
    const modal = document.getElementById('detail-modal');
    document.querySelectorAll('.stage-card').forEach((card) => {{
      card.addEventListener('click', () => {{
        const row = report.rows[Number(card.dataset.row)];
        const stageName = card.dataset.stageName;
        const depthStage = row.stages.find((candidate) => candidate.method === 'depth' && candidate.stage === stageName);
        const geometryStage = row.stages.find((candidate) => candidate.method === 'geometry' && candidate.stage === stageName);
        if (!depthStage || !geometryStage) return;
        document.getElementById('modal-title').textContent = `${{depthStage.title}} · G${{row.global_frame}}`;
        document.getElementById('modal-depth-image').src = depthStage.asset;
        document.getElementById('modal-geometry-image').src = geometryStage.asset;
        document.getElementById('modal-depth-details').textContent = JSON.stringify(depthStage.details, null, 2);
        document.getElementById('modal-geometry-details').textContent = JSON.stringify(geometryStage.details, null, 2);
        modal.showModal();
      }});
    }});
    document.querySelector('button.close').addEventListener('click', () => modal.close());
  </script>
</body>
</html>
"""


def _build_player_html(playback_manifest):
    playback_json = json.dumps(playback_manifest, ensure_ascii=False).replace(
        "</", "<\\/"
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>LASER Segmentation Playback</title>
  <style>
    :root {{ color-scheme: light; font-family: Arial, sans-serif; color: #17212b; background: #eef1f4; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; }}
    .toolbar {{ position: sticky; top: 0; z-index: 10; display: flex; flex-wrap: wrap; align-items: center; gap: 8px; padding: 10px 14px; background: #fff; border-bottom: 1px solid #b9c2ca; }}
    .toolbar a {{ color: #245a8d; font-weight: 700; }}
    .controls {{ display: flex; align-items: center; gap: 6px; }}
    button, select, input {{ font: inherit; }}
    button {{ min-height: 34px; padding: 5px 10px; border: 1px solid #aeb8c1; border-radius: 4px; background: #fff; cursor: pointer; }}
    button[aria-pressed="true"] {{ color: #fff; background: #245a8d; border-color: #245a8d; }}
    #timeline {{ flex: 1 1 260px; min-width: 160px; }}
    .status {{ min-width: 190px; font-variant-numeric: tabular-nums; }}
    main {{ padding: 12px; }}
    .playback-grid {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 10px; }}
    .panel {{ min-width: 0; padding: 8px; background: #fff; border: 1px solid #d5dbe0; border-radius: 6px; }}
    .panel.depth {{ border-top: 4px solid #245a8d; }}
    .panel.geometry {{ border-top: 4px solid #28734c; }}
    .panel h2 {{ margin: 0 0 7px; font-size: 14px; }}
    .panel img {{ display: block; width: 100%; min-height: 160px; object-fit: contain; background: #18212a; }}
    .panel-meta {{ margin-top: 6px; color: #4d5a66; font-size: 12px; }}
    .panel-error {{ display: none; margin-bottom: 10px; padding: 12px; color: #8b1e1e; background: #fff0f0; border: 1px solid #e0aaaa; }}
    @media (max-width: 760px) {{
      .playback-grid {{ grid-template-columns: 1fr; }}
      .panel img {{ min-height: 120px; }}
    }}
  </style>
</head>
<body>
  <header class="toolbar">
    <a href="index.html">返回完整报告</a>
    <div class="controls" aria-label="帧控制">
      <button id="previous-button" type="button" title="上一帧" aria-label="上一帧">|&lt;</button>
      <button id="play-button" type="button" title="播放" aria-label="播放">Play</button>
      <button id="next-button" type="button" title="下一帧" aria-label="下一帧">&gt;|</button>
    </div>
    <div class="controls" aria-label="分割阶段">
      <button type="button" data-stage="initial" aria-pressed="false">初始分割</button>
      <button type="button" data-stage="merged" aria-pressed="true">融合后分割</button>
    </div>
    <label>速度
      <select id="playback-speed">
        <option value="0.5">0.5 FPS</option>
        <option value="1">1 FPS</option>
        <option value="2" selected>2 FPS</option>
        <option value="4">4 FPS</option>
      </select>
    </label>
    <input id="timeline" type="range" min="0" value="0" step="1" aria-label="时间轴">
    <span id="playback-status" class="status"></span>
  </header>
  <main>
    <div id="playback-error" class="panel-error" role="alert"></div>
    <div id="playback-grid" class="playback-grid">
      <section class="panel depth"><h2 id="previous-depth-title">BASELINE / DEPTH · 前一帧</h2><img id="previous-depth-image" alt="Baseline previous frame"><div id="previous-depth-meta" class="panel-meta"></div></section>
      <section class="panel geometry"><h2 id="previous-geometry-title">GEOMETRY · 前一帧</h2><img id="previous-geometry-image" alt="Geometry previous frame"><div id="previous-geometry-meta" class="panel-meta"></div></section>
      <section class="panel depth"><h2 id="current-depth-title">BASELINE / DEPTH · 当前帧</h2><img id="current-depth-image" alt="Baseline current frame"><div id="current-depth-meta" class="panel-meta"></div></section>
      <section class="panel geometry"><h2 id="current-geometry-title">GEOMETRY · 当前帧</h2><img id="current-geometry-image" alt="Geometry current frame"><div id="current-geometry-meta" class="panel-meta"></div></section>
    </div>
  </main>
  <script id="playback-data" type="application/json">{playback_json}</script>
  <script>
    const playback = JSON.parse(document.getElementById('playback-data').textContent);
    const frames = playback.frames;
    let frameIndex = 0;
    let stageName = 'merged';
    let fps = 2;
    let timer = null;

    function stopPlayback() {{
      if (timer !== null) clearInterval(timer);
      timer = null;
      const playButton = document.getElementById('play-button');
      playButton.textContent = 'Play';
      playButton.title = '播放';
      playButton.setAttribute('aria-label', '播放');
    }}

    function resolveStage(frame, method) {{
      return frame?.stages?.[stageName]?.[method] || null;
    }}

    function renderPanel(prefix, frame, method, label, comparisonFrame = null) {{
      const image = document.getElementById(`${{prefix}}-${{method}}-image`);
      const title = document.getElementById(`${{prefix}}-${{method}}-title`);
      const meta = document.getElementById(`${{prefix}}-${{method}}-meta`);
      const methodLabel = method === 'depth' ? 'BASELINE / DEPTH' : 'GEOMETRY';
      if (!frame) {{
        image.removeAttribute('src');
        image.hidden = true;
        title.textContent = `${{methodLabel}} · ${{label}} · 无`;
        meta.textContent = '这是序列第一帧';
        return;
      }}
      const stage = resolveStage(frame, method);
      if (!stage) {{
        throw new Error(`缺少 ${{method}} ${{stageName}}，G${{frame.global_frame}}`);
      }}
      image.hidden = false;
      image.onerror = () => {{
        if (image.getAttribute('src') !== stage.asset) return;
        image.hidden = true;
        meta.textContent = `图片加载失败：${{stage.asset}}`;
      }};
      image.src = stage.asset;
      title.textContent = `${{methodLabel}} · ${{label}} · G${{String(frame.global_frame).padStart(5, '0')}}`;
      const comparisonStage = resolveStage(comparisonFrame, method);
      const delta = comparisonStage
        ? stage.segment_count - comparisonStage.segment_count
        : null;
      const deltaText = delta === null
        ? ''
        : ` · Δ vs 前一帧: ${{delta >= 0 ? '+' : ''}}${{delta}}`;
      meta.textContent = `segments: ${{stage.segment_count}}${{deltaText}}`;
    }}

    function preloadNextFrame() {{
      const frame = frames[frameIndex + 1];
      if (!frame) return;
      for (const method of ['depth', 'geometry']) {{
        const stage = resolveStage(frame, method);
        if (stage) new Image().src = stage.asset;
      }}
    }}

    function renderPlayback() {{
      const error = document.getElementById('playback-error');
      error.style.display = 'none';
      if (!frames.length) {{
        stopPlayback();
        error.textContent = '没有可播放的帧';
        error.style.display = 'block';
        return;
      }}
      frameIndex = Math.max(0, Math.min(frameIndex, frames.length - 1));
      const current = frames[frameIndex];
      const previous = frameIndex > 0 ? frames[frameIndex - 1] : null;
      try {{
        renderPanel('previous', previous, 'depth', '前一帧');
        renderPanel('previous', previous, 'geometry', '前一帧');
        renderPanel('current', current, 'depth', '当前帧', previous);
        renderPanel('current', current, 'geometry', '当前帧', previous);
      }} catch (renderError) {{
        stopPlayback();
        error.textContent = renderError.message;
        error.style.display = 'block';
      }}
      const timeline = document.getElementById('timeline');
      timeline.max = String(frames.length - 1);
      timeline.value = String(frameIndex);
      document.getElementById('playback-status').textContent =
        `G${{String(current.global_frame).padStart(5, '0')}} · ${{frameIndex + 1}}/${{frames.length}}`;
      preloadNextFrame();
    }}

    function stepFrame(delta) {{
      stopPlayback();
      frameIndex = Math.max(0, Math.min(frameIndex + delta, frames.length - 1));
      renderPlayback();
    }}

    function togglePlayback() {{
      if (timer !== null) {{
        stopPlayback();
        return;
      }}
      if (!frames.length) return;
      if (frameIndex === frames.length - 1) frameIndex = 0;
      const playButton = document.getElementById('play-button');
      playButton.textContent = 'Pause';
      playButton.title = '暂停';
      playButton.setAttribute('aria-label', '暂停');
      timer = setInterval(() => {{
        if (frameIndex >= frames.length - 1) {{
          stopPlayback();
          return;
        }}
        frameIndex += 1;
        renderPlayback();
      }}, 1000 / fps);
      renderPlayback();
    }}

    document.getElementById('previous-button').addEventListener('click', () => stepFrame(-1));
    document.getElementById('play-button').addEventListener('click', togglePlayback);
    document.getElementById('next-button').addEventListener('click', () => stepFrame(1));
    document.getElementById('timeline').addEventListener('input', (event) => {{
      stopPlayback();
      frameIndex = Number(event.target.value);
      renderPlayback();
    }});
    document.getElementById('playback-speed').addEventListener('change', (event) => {{
      const wasPlaying = timer !== null;
      stopPlayback();
      fps = Number(event.target.value);
      if (wasPlaying) togglePlayback();
    }});
    document.querySelectorAll('[data-stage]').forEach((button) => {{
      button.addEventListener('click', () => {{
        stageName = button.dataset.stage;
        document.querySelectorAll('[data-stage]').forEach((candidate) => {{
          candidate.setAttribute('aria-pressed', String(candidate === button));
        }});
        renderPlayback();
      }});
    }});
    document.addEventListener('keydown', (event) => {{
      if (event.target instanceof Element && event.target.matches('input, select, button, a')) return;
      if (event.key === 'ArrowLeft') stepFrame(-1);
      if (event.key === 'ArrowRight') stepFrame(1);
      if (event.code === 'Space') {{
        event.preventDefault();
        togglePlayback();
      }}
    }});
    document.addEventListener('visibilitychange', () => {{
      if (document.hidden) stopPlayback();
    }});
    renderPlayback();
  </script>
</body>
</html>
"""


def build_report(
    baseline_debug_dir,
    geometry_debug_dir,
    image_dir,
    out_dir,
    *,
    sample_interval=None,
    window_start=0,
    window_stop=None,
    frame_step=1,
):
    if frame_step <= 0:
        raise ValueError("frame_step must be positive")
    baseline = _load_pipeline_run(baseline_debug_dir)
    geometry = _load_pipeline_run(geometry_debug_dir)
    validate_comparable_runs(baseline["metadata"], geometry["metadata"])
    metadata = baseline["metadata"]
    if baseline["metadata"].get("schema_version") != 2 or geometry["metadata"].get(
        "schema_version"
    ) != 2:
        raise ValueError(
            "Pipeline trace v2 with segmentation_depths is required; rerun both methods."
        )
    if sample_interval is not None and sample_interval != metadata["sample_interval"]:
        raise ValueError(
            f"Mismatched sample_interval: CLI={sample_interval} metadata={metadata['sample_interval']}"
        )
    if len(baseline["windows"]) != len(geometry["windows"]):
        raise ValueError("Baseline and geometry window counts differ")

    image_paths = list_image_paths(image_dir, sample_interval=metadata["sample_interval"])
    out_dir = Path(out_dir)
    assets_dir = out_dir / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    stop = len(baseline["windows"]) if window_stop is None else min(window_stop, len(baseline["windows"]))
    rows = []
    for window_index in range(window_start, stop):
        baseline_window = baseline["windows"][window_index]
        geometry_window = geometry["windows"][window_index]
        if baseline_window["name"] != geometry_window["name"]:
            raise ValueError("Baseline and geometry window numbers differ")
        base_arrays = baseline_window["arrays"]
        geom_arrays = geometry_window["arrays"]
        for arrays in (base_arrays, geom_arrays):
            if "segmentation_depths" not in arrays:
                raise ValueError(
                    "Pipeline trace v2 with segmentation_depths is required; "
                    "rerun both methods."
                )
            if arrays["segmentation_depths"].shape != arrays["merged_labels"].shape:
                raise ValueError(
                    f"segmentation_depths shape differs from labels for {baseline_window['name']}"
                )
        np.testing.assert_array_equal(
            base_arrays["global_frame_indices"],
            geom_arrays["global_frame_indices"],
            err_msg=f"Global frame indices differ for {baseline_window['name']}",
        )
        if base_arrays["merged_labels"].shape != geom_arrays["merged_labels"].shape:
            raise ValueError(f"Label shapes differ for {baseline_window['name']}")

        previous_base = baseline["windows"][window_index - 1]["arrays"] if window_index > 0 else None
        previous_geom = geometry["windows"][window_index - 1]["arrays"] if window_index > 0 else None
        global_indices = base_arrays["global_frame_indices"]
        for local_frame in range(0, len(global_indices), frame_step):
            global_frame = int(global_indices[local_frame])
            if global_frame >= len(image_paths):
                raise ValueError(
                    f"RGB sequence has {len(image_paths)} sampled frames, need index {global_frame}"
                )
            rgb = cv2.imread(image_paths[global_frame], cv2.IMREAD_COLOR)
            if rgb is None:
                raise ValueError(f"Failed to read RGB image: {image_paths[global_frame]}")
            target_shape = base_arrays["merged_labels"].shape[1:]
            if rgb.shape[:2] != target_shape:
                rgb = cv2.resize(rgb, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
            previous_rgb = None
            if local_frame > 0:
                previous_global = int(global_indices[local_frame - 1])
                previous_rgb = cv2.imread(image_paths[previous_global], cv2.IMREAD_COLOR)
                if previous_rgb.shape[:2] != target_shape:
                    previous_rgb = cv2.resize(
                        previous_rgb,
                        (target_shape[1], target_shape[0]),
                        interpolation=cv2.INTER_AREA,
                    )

            shared_depth_range = compute_shared_depth_range(
                base_arrays["segmentation_depths"][local_frame],
                geom_arrays["segmentation_depths"][local_frame],
            )

            depth_stages = _render_method_stages(
                method="depth",
                window_index=window_index,
                local_frame=local_frame,
                rgb=rgb,
                previous_rgb=previous_rgb,
                arrays=base_arrays,
                previous_window_arrays=previous_base,
                overlap=metadata["overlap"],
                assets_dir=assets_dir,
                shared_depth_range=shared_depth_range,
            )
            geometry_stages = _render_method_stages(
                method="geometry",
                window_index=window_index,
                local_frame=local_frame,
                rgb=rgb,
                previous_rgb=previous_rgb,
                arrays=geom_arrays,
                previous_window_arrays=previous_geom,
                overlap=metadata["overlap"],
                assets_dir=assets_dir,
                shared_depth_range=shared_depth_range,
            )
            stages = depth_stages + geometry_stages
            if [(item["method"], item["stage"]) for item in stages] != list(STAGE_ORDER):
                raise RuntimeError("Internal stage order does not match the ten-image contract")
            rows.append(
                {
                    "window_index": window_index,
                    "local_frame": local_frame,
                    "global_frame": global_frame,
                    "is_overlap": bool(window_index > 0 and local_frame < metadata["overlap"]),
                    "stages": stages,
                }
            )

    manifest = {"metadata": metadata, "rows": rows}
    jsonable_manifest = _jsonable(manifest)
    (out_dir / "data.json").write_text(
        json.dumps(jsonable_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "index.html").write_text(
        _build_html(jsonable_manifest), encoding="utf-8"
    )
    playback_manifest = build_playback_manifest(jsonable_manifest)
    (out_dir / "player.html").write_text(
        _build_player_html(playback_manifest), encoding="utf-8"
    )
    return {"out_dir": out_dir, "row_count": len(rows)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a ten-image-per-frame LASER depth-vs-geometry pipeline report."
    )
    parser.add_argument("--baseline_debug_dir", required=True)
    parser.add_argument("--geometry_debug_dir", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sample_interval", type=int, default=None)
    parser.add_argument("--window_start", type=int, default=0)
    parser.add_argument("--window_stop", type=int, default=None)
    parser.add_argument("--frame_step", type=int, default=1)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = build_report(
        args.baseline_debug_dir,
        args.geometry_debug_dir,
        args.image_dir,
        args.out_dir,
        sample_interval=args.sample_interval,
        window_start=args.window_start,
        window_stop=args.window_stop,
        frame_step=args.frame_step,
    )
    print(f"Saved alignment pipeline report to: {result['out_dir'] / 'index.html'}")


if __name__ == "__main__":
    main()
