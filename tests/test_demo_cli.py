import subprocess
import sys
from pathlib import Path


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
