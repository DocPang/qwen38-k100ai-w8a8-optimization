#!/usr/bin/env bash
set -euo pipefail
command -v docker >/dev/null || { echo 'ERROR: docker not found' >&2; exit 2; }
[[ -e /dev/kfd ]] || { echo 'ERROR: /dev/kfd not found' >&2; exit 3; }
[[ -e /dev/dri ]] || { echo 'ERROR: /dev/dri not found' >&2; exit 4; }
echo 'Docker:'; docker --version
if command -v hy-smi >/dev/null 2>&1; then
  echo 'hy-smi:'; hy-smi --version 2>/dev/null || true
  hy-smi --showproductname 2>/dev/null || true
else
  echo 'WARNING: hy-smi not found; verify the accelerator is Hygon K100AI/gfx928 manually.' >&2
fi
if command -v rocminfo >/dev/null 2>&1; then
  rocminfo 2>/dev/null | grep -m1 -E 'gfx928|Name:' || true
fi
echo 'Host preflight complete.'
