#!/usr/bin/env bash
set -euo pipefail
# Promotion-style profile. Same R054 runtime stack, but the historical promotion
# evidence used 0.92 memory utilization and no prefix cache.
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
export RESTART_POLICY="${RESTART_POLICY:-no}"
export CONTAINER_NAME="${CONTAINER_NAME:-qwen38-k100ai-r054-benchmark}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/serve_r054.sh"
