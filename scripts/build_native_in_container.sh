#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-harbor.sourcefind.cn:5443/dcu/admin/base/custom@sha256:13ce550647063a7fe76e87fd173986175946e5046bd36980c4289c60a4bdd811}"
devices=(--device=/dev/kfd --device=/dev/dri)
[[ -e /dev/mkfd ]] && devices+=(--device=/dev/mkfd)
hyhal=()
[[ -d /opt/hyhal ]] && hyhal=(-v /opt/hyhal:/opt/hyhal:ro)
docker run --rm --ipc host --shm-size 8g \
  "${devices[@]}" "${hyhal[@]}" \
  -v "$ROOT_DIR":/workspace \
  -w /workspace \
  "$IMAGE" python3 scripts/build_native.py
