from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class MatchRecord:
    frame: int
    src_segment: int
    tgt_segment: int
    iou: float
    scale: float


@dataclass(frozen=True)
class PropagationRecord:
    parent_frame: int
    parent_segment: int
    child_frame: int
    child_segment: int
    iou: float
    scale: float


@dataclass(frozen=True)
class SegmentState:
    frame: int
    segment: int
    role: str
    scale: float


def _mean_cached_scale(cache):
    scales = np.asarray(cache.get("scale", []), dtype=np.float64)
    if scales.size == 0:
        return 1.0
    weights = np.asarray(cache.get("iou", []), dtype=np.float64)
    return float(np.dot(scales, weights / weights.sum()))


def _vertex_segment_id(vertex, fallback):
    return int(vertex.vid) if vertex.vid is not None else int(fallback)


@dataclass
class ScaleTraceCollector:
    matches: list[MatchRecord] = field(default_factory=list)
    propagation_edges: list[PropagationRecord] = field(default_factory=list)
    segment_states: list[SegmentState] = field(default_factory=list)
    direct_anchor_keys: set[tuple[int, int]] = field(default_factory=set)

    def record_match(self, frame, src_segment, tgt_segment, iou, scale):
        self.matches.append(
            MatchRecord(
                frame=int(frame),
                src_segment=int(src_segment),
                tgt_segment=int(tgt_segment),
                iou=float(iou),
                scale=float(scale),
            )
        )

    def capture_direct_anchors(self, graphs):
        self.direct_anchor_keys = {
            (frame_idx, segment_idx)
            for frame_idx, graph in enumerate(graphs)
            for fallback, vertex in enumerate(graph)
            for segment_idx in [_vertex_segment_id(vertex, fallback)]
            if vertex.cache.get("scale")
        }

    def record_propagation(
        self,
        parent_frame,
        parent_segment,
        child_frame,
        child_segment,
        iou,
        scale,
    ):
        self.propagation_edges.append(
            PropagationRecord(
                parent_frame=int(parent_frame),
                parent_segment=int(parent_segment),
                child_frame=int(child_frame),
                child_segment=int(child_segment),
                iou=float(iou),
                scale=float(scale),
            )
        )

    def capture_segment_states(self, graphs):
        states = []
        for frame_idx, graph in enumerate(graphs):
            for fallback, vertex in enumerate(graph):
                segment_idx = _vertex_segment_id(vertex, fallback)
                key = (frame_idx, segment_idx)
                if key in self.direct_anchor_keys:
                    role = "A"
                elif vertex.cache.get("scale"):
                    role = "P"
                else:
                    role = "I"
                states.append(
                    SegmentState(
                        frame=frame_idx,
                        segment=segment_idx,
                        role=role,
                        scale=_mean_cached_scale(vertex.cache),
                    )
                )
        self.segment_states = states

    def to_arrays(self):
        role_codes = {"I": 0, "P": 1, "A": 2}
        return {
            "match_frame": np.asarray([x.frame for x in self.matches], dtype=np.int32),
            "match_src_segment": np.asarray(
                [x.src_segment for x in self.matches], dtype=np.int32
            ),
            "match_tgt_segment": np.asarray(
                [x.tgt_segment for x in self.matches], dtype=np.int32
            ),
            "match_iou": np.asarray([x.iou for x in self.matches], dtype=np.float32),
            "match_scale": np.asarray([x.scale for x in self.matches], dtype=np.float32),
            "prop_parent_frame": np.asarray(
                [x.parent_frame for x in self.propagation_edges], dtype=np.int32
            ),
            "prop_parent_segment": np.asarray(
                [x.parent_segment for x in self.propagation_edges], dtype=np.int32
            ),
            "prop_child_frame": np.asarray(
                [x.child_frame for x in self.propagation_edges], dtype=np.int32
            ),
            "prop_child_segment": np.asarray(
                [x.child_segment for x in self.propagation_edges], dtype=np.int32
            ),
            "prop_iou": np.asarray(
                [x.iou for x in self.propagation_edges], dtype=np.float32
            ),
            "prop_scale": np.asarray(
                [x.scale for x in self.propagation_edges], dtype=np.float32
            ),
            "segment_frame": np.asarray(
                [x.frame for x in self.segment_states], dtype=np.int32
            ),
            "segment_id": np.asarray(
                [x.segment for x in self.segment_states], dtype=np.int32
            ),
            "segment_role": np.asarray(
                [role_codes[x.role] for x in self.segment_states], dtype=np.uint8
            ),
            "segment_scale": np.asarray(
                [x.scale for x in self.segment_states], dtype=np.float32
            ),
        }
