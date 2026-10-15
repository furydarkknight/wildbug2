"""
download_data.py
================
Downloads public bearing datasets for MachinaSense:
1. Case Western Reserve University (CWRU) 12 kHz drive-end .mat files (all 4 loads x 4 conditions)
2. NASA IMS run-to-failure dataset info / sample snapshots

Usage:
    python download_data.py
"""

import sys
import urllib.request
from pathlib import Path

CWRU_FILES = [
    # Normal baseline (0, 1, 2, 3 HP)
    97, 98, 99, 100,
    # Inner race fault (0, 1, 2, 3 HP)
    105, 106, 107, 108,
    # Outer race fault (0, 1, 2, 3 HP)
    118, 119, 130, 131,
    # Ball fault (0, 1, 2, 3 HP)
    122, 123, 124, 125,
]

CWRU_BASE_URL = "https://engineering.case.edu/sites/default/files"


def download_cwru(dest_dir: str | Path = "data/cwru") -> None:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"\n[CWRU Dataset] Downloading 16 .mat files into {dest}...")

    total_downloaded = 0
    for num in CWRU_FILES:
        target = dest / f"{num}.mat"
        if target.exists() and target.stat().st_size > 1000:
            print(f"  [OK] {num}.mat exists ({target.stat().st_size / 1024:.1f} KB)")
            continue

        url = f"{CWRU_BASE_URL}/{num}.mat"
        print(f"  [DOWNLOADING] {num}.mat from {url}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp, open(target, "wb") as f:
                f.write(resp.read())
            print(f"  [SAVED] {num}.mat ({target.stat().st_size / 1024:.1f} KB)")
            total_downloaded += 1
        except Exception as err:
            print(f"  [ERROR] Failed to download {num}.mat: {err}")

    print(f"[CWRU Dataset] Download check complete. ({total_downloaded} files newly downloaded)\n")


if __name__ == "__main__":
    download_cwru()
