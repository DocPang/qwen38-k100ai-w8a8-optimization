#!/usr/bin/env python3
"""Verify the downloaded checkpoint against the hashes used for R054.

By default all listed files, including ~30 GiB of weights, are hashed. Use
--metadata-only for a quick preflight before a full verification.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "model_metadata" / "SHA256SUMS.quantized.txt"
METADATA_NAMES = {
    "config.json",
    "model.safetensors.index.json",
    "recipe.yaml",
    "README.md",
    "tokenizer_config.json",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_manifest() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw in MANIFEST.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        digest, name = raw.split(None, 1)
        name = name.strip()
        if name in seen:
            continue
        seen.add(name)
        rows.append((digest, name))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--metadata-only", action="store_true")
    args = ap.parse_args()
    root = Path(args.model_dir).expanduser().resolve()
    failures = 0
    checked = 0
    for expected, name in parse_manifest():
        if args.metadata_only and name not in METADATA_NAMES:
            continue
        path = root / name
        if not path.is_file():
            print(f"MISSING {name}")
            failures += 1
            continue
        got = sha256(path)
        checked += 1
        if got != expected:
            print(f"FAIL {name}\n  expected {expected}\n  got      {got}")
            failures += 1
        else:
            print(f"OK {name}")
    if failures:
        raise SystemExit(f"verification failed: {failures} file(s)")
    print(f"PASS checked={checked} metadata_only={args.metadata_only}")


if __name__ == "__main__":
    main()
