"""Export docs/assets/overview.drawio to the README's PNG (needs draw.io desktop)."""

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/assets/overview.drawio"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drawio", help="Path to the draw.io desktop executable")
    args = parser.parse_args()
    executable = args.drawio or shutil.which("drawio")
    if executable is None:
        macos = Path("/Applications/draw.io.app/Contents/MacOS/draw.io")
        if macos.exists():
            executable = str(macos)
    if executable is None:
        parser.error("Install draw.io desktop, or pass its executable with --drawio")
    command = [executable, "--export", "--format", "png", "--scale", "2"]
    command += ["--border", "20", "--output", str(SOURCE.with_suffix(".png"))]
    subprocess.run([*command, str(SOURCE)], check=True, cwd=ROOT, timeout=120)


if __name__ == "__main__":
    main()
