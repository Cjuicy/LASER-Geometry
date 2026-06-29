import subprocess
import sys
from pathlib import Path

import torch

import demo


def test_demo_exposes_optional_alignment_debug_flags():
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "demo.py", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Streaming Pi3 Demo" in result.stdout
    assert "--debug_alignment" in result.stdout
    assert "--debug_alignment_path" in result.stdout
    assert "--top_conf_percentile" in result.stdout


def test_demo_passes_sample_interval_to_alignment_debug_metadata(monkeypatch):
    captured = {}

    class FakePi3:
        @classmethod
        def from_pretrained(cls, name):
            return torch.nn.Identity()

    def fake_engine(model, **kwargs):
        captured.update(kwargs)
        return "engine"

    monkeypatch.setattr(demo, "Pi3", FakePi3)
    monkeypatch.setattr(demo, "StreamingWindowEngine", fake_engine)
    args = demo.get_args_parser().parse_args(
        [
            "--data_path",
            "images",
            "--sample_interval",
            "7",
            "--top_conf_percentile",
            "0.4",
        ]
    )

    assert demo.load_model(args) == "engine"
    assert captured["debug_sample_interval"] == 7
    assert captured["top_conf_percentile"] == 0.4
