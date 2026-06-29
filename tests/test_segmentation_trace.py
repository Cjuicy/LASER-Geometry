import numpy as np


def test_depth_stages_keep_original_merged_labels(monkeypatch):
    from inference_engine.utils import depth as depth_module

    initial = np.array([[0, 0], [1, 1]], dtype=np.intp)
    merged = np.array([[0, 0], [0, 0]], dtype=np.intp)
    monkeypatch.setattr(depth_module, "felzenszwalb", lambda *args, **kwargs: initial)
    monkeypatch.setattr(depth_module, "merge_regions", lambda *args, **kwargs: merged)

    depth = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    conf = np.array([[[0.1, 0.2], [0.8, 0.9]]], dtype=np.float32)

    stages = depth_module.segment_depth_felzenszwalb_rag_stages(
        depth,
        0.1,
        conf_map=conf,
        top_conf_percentile=0.5,
        batch_idx=0,
    )
    labels = depth_module.segment_depth_felzenszwalb_rag(
        depth,
        0.1,
        conf_map=conf,
        top_conf_percentile=0.5,
        batch_idx=0,
    )

    np.testing.assert_array_equal(stages.initial_labels, initial)
    np.testing.assert_array_equal(stages.merged_labels, merged)
    np.testing.assert_array_equal(labels, merged)
    assert np.isclose(stages.confidence_threshold, 0.8)
    assert stages.high_confidence_mask.sum() == 2


def test_geometry_stages_keep_original_merged_labels(monkeypatch):
    from inference_engine.utils import geometry_segmentation as geometry_module

    initial = np.array([[0, 0], [1, 1]], dtype=np.intp)
    merged = np.array([[0, 0], [0, 0]], dtype=np.intp)
    geometry_info = {
        "normal": np.dstack(
            [
                np.zeros((2, 2), dtype=np.float32),
                np.zeros((2, 2), dtype=np.float32),
                np.ones((2, 2), dtype=np.float32),
            ]
        )
    }
    monkeypatch.setattr(
        geometry_module,
        "build_geometry_info_np",
        lambda **kwargs: geometry_info,
    )
    monkeypatch.setattr(
        geometry_module,
        "felzenszwalb",
        lambda *args, **kwargs: initial,
    )
    monkeypatch.setattr(
        geometry_module,
        "merge_regions_geometry",
        lambda *args, **kwargs: merged,
    )

    depth = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    conf = np.array([[[0.1, 0.2], [0.8, 0.9]]], dtype=np.float32)

    stages = geometry_module.segment_geometry_felzenszwalb_rag_stages(
        depth,
        conf_map=conf,
        top_conf_percentile=0.5,
        intrinsic=np.eye(3, dtype=np.float32),
        batch_idx=0,
    )
    labels = geometry_module.segment_geometry_felzenszwalb_rag(
        depth,
        conf_map=conf,
        top_conf_percentile=0.5,
        intrinsic=np.eye(3, dtype=np.float32),
        batch_idx=0,
    )

    np.testing.assert_array_equal(stages.initial_labels, initial)
    np.testing.assert_array_equal(stages.merged_labels, merged)
    np.testing.assert_array_equal(labels, merged)
    assert np.isclose(stages.confidence_threshold, 0.5)
    assert stages.high_confidence_mask.sum() == 2


def test_confidence_retention_is_not_hard_coded(monkeypatch):
    from inference_engine.utils import depth as depth_module

    monkeypatch.setattr(
        depth_module,
        "felzenszwalb",
        lambda depth, **kwargs: np.zeros(depth.shape, dtype=np.intp),
    )
    monkeypatch.setattr(
        depth_module,
        "merge_regions",
        lambda labels, depth, threshold: labels,
    )
    depth = np.arange(10, dtype=np.float32).reshape(2, 5) + 1.0
    conf = np.arange(10, dtype=np.float32).reshape(1, 2, 5)

    top_20 = depth_module.segment_depth_felzenszwalb_rag_stages(
        depth,
        0.1,
        conf_map=conf,
        top_conf_percentile=0.8,
        batch_idx=0,
    )
    top_40 = depth_module.segment_depth_felzenszwalb_rag_stages(
        depth,
        0.1,
        conf_map=conf,
        top_conf_percentile=0.6,
        batch_idx=0,
    )

    assert top_20.confidence_threshold == 7.0
    assert top_40.confidence_threshold == 5.0
    assert top_20.high_confidence_mask.sum() == 3
    assert top_40.high_confidence_mask.sum() == 5
