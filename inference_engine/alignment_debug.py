"""Optional alignment-debug trace serialization utilities.

This module is intentionally independent from the viewer and from the main
streaming control flow. When disabled, the recorder is a no-op.
"""

import json
from pathlib import Path

import numpy as np


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _jsonable(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


class AlignmentDebugRecorder:
    def __init__(self, *, enabled=False, root_dir=None, scene_name=None):
        self.enabled = bool(enabled)
        self.root_dir = None if root_dir is None else Path(root_dir)
        self.scene_name = scene_name or "alignment_debug"

    @property
    def scene_dir(self):
        if self.root_dir is None:
            return None
        return self.root_dir / self.scene_name

    def record_pair(self, *, pair_index, payload, metadata=None):
        if not self.enabled:
            return None
        if self.scene_dir is None:
            raise ValueError("root_dir is required when alignment debug recording is enabled.")

        self.scene_dir.mkdir(parents=True, exist_ok=True)
        if metadata:
            meta_path = self.scene_dir / "meta.json"
            meta_payload = {key: _jsonable(value) for key, value in metadata.items()}
            meta_path.write_text(
                json.dumps(meta_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )

        np_payload = {key: _to_numpy(value) for key, value in payload.items()}
        out_path = self.scene_dir / f"pair_{pair_index:04d}.npz"
        np.savez_compressed(out_path, **np_payload)
        return out_path


def summarize_graph_layer(graph_layer):
    masks = []
    has_scale = []
    mean_iou = []
    mean_scale = []

    for vertex in graph_layer:
        masks.append(np.asarray(vertex.data, dtype=bool))
        scales = np.asarray(vertex.cache.get("scale", []), dtype=np.float32)
        ious = np.asarray(vertex.cache.get("iou", []), dtype=np.float32)
        has_scale.append(scales.size > 0)
        mean_scale.append(float(scales.mean()) if scales.size else 1.0)
        mean_iou.append(float(ious.mean()) if ious.size else 0.0)

    if masks:
        masks_array = np.stack(masks, axis=0)
    else:
        masks_array = np.zeros((0, 0, 0), dtype=bool)

    return {
        "masks": masks_array,
        "has_scale": np.asarray(has_scale, dtype=bool),
        "mean_iou": np.asarray(mean_iou, dtype=np.float32),
        "mean_scale": np.asarray(mean_scale, dtype=np.float32),
    }
