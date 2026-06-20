from pathlib import Path

import pytest

from utils.image_sequence import list_image_paths


def test_list_image_paths_sorts_frames_before_sampling(tmp_path):
    for name in [
        "frame_10.png",
        "frame_2.png",
        "frame_1.png",
        "frame_3.JPG",
        "notes.txt",
    ]:
        (tmp_path / name).write_text("x")

    image_paths = list_image_paths(tmp_path, sample_interval=2)

    assert [Path(path).name for path in image_paths] == [
        "frame_1.png",
        "frame_3.JPG",
    ]


def test_list_image_paths_rejects_non_positive_sample_interval(tmp_path):
    with pytest.raises(ValueError, match="sample_interval"):
        list_image_paths(tmp_path, sample_interval=0)
