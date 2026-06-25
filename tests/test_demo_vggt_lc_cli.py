import subprocess
import sys
from pathlib import Path


def test_demo_vggt_lc_exposes_expected_cli_options():
    repo_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "demo_vggt_lc.py", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Loop-closure Streaming VGGT Demo" in result.stdout
    assert "--config_path" in result.stdout
    assert "--model_ckpt" in result.stdout
    assert "--segment_mode" in result.stdout
    assert "--normal_method" in result.stdout
    assert "--scale_anchor_mode" in result.stdout
