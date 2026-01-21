#!/usr/bin/env python3
"""
Debug downloader for FSD50K split ZIP files from Zenodo.

Usage:
  python scripts/debug_fsd50k_download.py --out ./tmp
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import requests

FSD50K_PARTS = [
    "FSD50K.dev_audio.z01",
    "FSD50K.dev_audio.z02",
    "FSD50K.dev_audio.z03",
    "FSD50K.dev_audio.z04",
    "FSD50K.dev_audio.z05",
    "FSD50K.dev_audio.zip",
]

BASE_URL = "https://zenodo.org/record/4060432/files"


def download_part(session: requests.Session, fname: str, out_dir: Path) -> None:
    url = f"{BASE_URL}/{fname}?download=1"
    out_path = out_dir / fname
    print(f"\n==> GET {url}")
    headers = {"User-Agent": "soundkit-fsd50k-debug/1.0"}
    with session.get(url, stream=True, allow_redirects=True, headers=headers) as resp:
        print(f"status: {resp.status_code}")
        print(f"url (final): {resp.url}")
        ct = resp.headers.get("content-type", "")
        cl = resp.headers.get("content-length", "")
        cd = resp.headers.get("content-disposition", "")
        print(f"content-type: {ct}")
        print(f"content-length: {cl}")
        if cd:
            print(f"content-disposition: {cd}")

        # Peek first chunk for debugging without loading the whole body.
        iterator = resp.iter_content(chunk_size=1024 * 1024)
        first_chunk = next(iterator, b"")
        print(f"first 64 bytes: {first_chunk[:64]!r}")

        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code} for {url}")

        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            if first_chunk:
                f.write(first_chunk)
            for chunk in iterator:
                if chunk:
                    f.write(chunk)

    size = out_path.stat().st_size
    print(f"saved: {out_path} ({size} bytes)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Debug downloader for FSD50K parts.")
    ap.add_argument("--out", default="./tmp", help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out)
    session = requests.Session()

    try:
        for fname in FSD50K_PARTS:
            download_part(session, fname, out_dir)
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
