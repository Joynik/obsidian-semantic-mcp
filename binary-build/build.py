"""Build an OSM executable with the optional Nuitka build dependency."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = Path(__file__).resolve().parent / "output"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onefile", action="store_true", help="build a single executable")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "nuitka",
        "--mode=onefile" if args.onefile else "--mode=standalone",
        "--disable-cache=ccache",
        "--update-check=never",
        f"--output-dir={OUTPUT}",
        "--output-filename=osm",
        "--include-package=src",
        "--include-package-data=src.dashboard_ui",
        "--include-distribution-metadata=obsidian-semantic-mcp",
        f"--include-data-dir={ROOT / 'embedding-profiles'}=embedding-profiles",
        str(ROOT / "binary-build" / "osm.py"),
    ]
    environment = os.environ.copy()
    environment.setdefault("NUITKA_CACHE_DIR", str(ROOT / "binary-build" / "cache"))
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
