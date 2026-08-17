#!/usr/bin/env python3
"""Download the exact HuggingFace checkpoint used by the published results."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "Freaksterz/Qwen3.8-27B-SmoothQuant-W8A8-INT8"
REVISION = "417ede1e4524c8fdbb586ebdabc9cfc5d0760b3e"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-dir",
        default=str(Path.home() / "models" / "Qwen3.8-27B-SmoothQuant-W8A8-INT8"),
    )
    args = ap.parse_args()
    target = Path(args.model_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    print(f"repo={REPO_ID}")
    print(f"revision={REVISION}")
    print(f"target={target}")
    snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        local_dir=str(target),
    )
    print(target)


if __name__ == "__main__":
    main()
