import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("git") is None, reason="Git unavailable")
def test_env_credentials_ignored_and_blank_template_shareable(tmp_path):
    root = Path(__file__).resolve().parents[1]
    shutil.copyfile(root / ".gitignore", tmp_path / ".gitignore")
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    names = [
        ".env",
        ".env.local",
        ".env.production",
        "nested/.env",
        "nested/.env.local",
    ]
    for name in names:
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", name],
            cwd=tmp_path,
            capture_output=True,
        )
        assert result.returncode == 0, name
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", ".env.example"],
        cwd=tmp_path,
        capture_output=True,
    )
    assert result.returncode == 1
    template = (root / ".env.example").read_text()
    for line in template.splitlines():
        if line and not line.startswith("#"):
            assert line.partition("=")[2] == ""
