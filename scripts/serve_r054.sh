#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MODEL_DIR:?Set MODEL_DIR to the pinned HuggingFace checkpoint directory}"

IMAGE="${IMAGE:-harbor.sourcefind.cn:5443/dcu/admin/base/custom@sha256:13ce550647063a7fe76e87fd173986175946e5046bd36980c4289c60a4bdd811}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MTP_CUTOFF="${MTP_CUTOFF:-41216}"
SPEC_TOKENS="${SPEC_TOKENS:-5}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-k100ai-r054}"
CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/.cache/r054}"
RESTART_POLICY="${RESTART_POLICY:-unless-stopped}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"

MODEL_DIR="$(cd "$MODEL_DIR" && pwd)"
PATCH_DIR="$ROOT_DIR/patches/r054"
NATIVE_SO="$ROOT_DIR/native_ext/k100_int8_gemv_v7.so"
EXPECTED_CONFIG_SHA="109472b03e1ba725a000052309b1497f5979bddcfcbc281017b6d72e63431d78"

if (( MTP_CUTOFF <= 0 || MTP_CUTOFF > MAX_MODEL_LEN )); then
  echo "ERROR: MTP_CUTOFF must be in 1..MAX_MODEL_LEN" >&2; exit 2
fi
if [[ ! -f "$NATIVE_SO" ]]; then
  echo "ERROR: missing $NATIVE_SO" >&2
  echo "Build it first: bash scripts/build_native_in_container.sh" >&2
  exit 3
fi
if command -v sha256sum >/dev/null 2>&1; then
  GOT_CONFIG_SHA="$(sha256sum "$MODEL_DIR/config.json" | awk '{print $1}')"
else
  GOT_CONFIG_SHA="$(shasum -a 256 "$MODEL_DIR/config.json" | awk '{print $1}')"
fi
if [[ "$GOT_CONFIG_SHA" != "$EXPECTED_CONFIG_SHA" ]]; then
  echo "ERROR: model config does not match the validated HuggingFace revision." >&2
  echo "expected=$EXPECTED_CONFIG_SHA got=$GOT_CONFIG_SHA" >&2
  exit 4
fi

mkdir -p "$CACHE_DIR"
devices=(--device=/dev/kfd --device=/dev/dri)
[[ -e /dev/mkfd ]] && devices+=(--device=/dev/mkfd)
hyhal=()
[[ -d /opt/hyhal ]] && hyhal=(-v /opt/hyhal:/opt/hyhal:ro)

prefix_args=()
[[ "$ENABLE_PREFIX_CACHING" == "1" ]] && prefix_args+=(--enable-prefix-caching)
served_args=(--served-model-name qwen3.8-27b-w8a8)
verify_m=$((SPEC_TOKENS + 1))
source_sha="$(python3 - <<PY
import hashlib
print(hashlib.sha256(open('$PATCH_DIR/candidate_head.py','rb').read()).hexdigest())
PY
)"
spec_cfg="$(printf '{\"model\":\"/models/qwen38-w8a8\",\"method\":\"qwen3_next_mtp\",\"num_speculative_tokens\":%s,\"quantization\":\"compressed-tensors\",\"max_model_len\":%s}' "$SPEC_TOKENS" "$MTP_CUTOFF")"

if command -v ss >/dev/null 2>&1 && ss -lnt 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)$PORT$"; then
  echo "ERROR: port $PORT is already in use" >&2; exit 5
fi
if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME")" == "true" ]]; then
    echo "ERROR: refusing to replace running container $CONTAINER_NAME" >&2; exit 6
  fi
  docker rm "$CONTAINER_NAME" >/dev/null
fi

docker run -d \
  --name "$CONTAINER_NAME" \
  --restart "$RESTART_POLICY" \
  --network host --ipc host --shm-size 16g \
  "${devices[@]}" "${hyhal[@]}" \
  -e TZ=Asia/Shanghai \
  -e HIP_VISIBLE_DEVICES="$GPU_ID" \
  -e PYTHONPATH=/opt/k100-q38-patch \
  -e K100_27B_SHAPE1_EXACT_ACCEPT=1 \
  -e K100_27B_VALID_COUNT_INT32=1 \
  -e K100_27B_PREFILL_BM32_W8=1 \
  -e K100_27B_PREFILL_GATE=512 \
  -e K100_27B_PREFILL_W8_Q3072_K=5120 \
  -e K100_27B_PREFILL_W8_Q2032_K=7168 \
  -e K100_27B_PREFILL_W8_Q1536_K=16384 \
  -e K100_27B_PREFILL_W8_Q1280_K=32768 \
  -e K100_27B_PREFILL_W8_Q1024_K=49152 \
  -e K100_27B_SC_LINEAR_A8W8=1 \
  -e K100_27B_SC_LINEAR_A8W8_MAX_M=6 \
  -e K100_27B_LM_HEAD_W8A8=0 \
  -e K100_27B_LM_HEAD_W8A8_MAX_M=5 \
  -e K100_27B_GDN_FUSED=1 \
  -e K100_27B_NATIVE_OUT_GEMV=1 \
  -e K100_27B_NATIVE_OUT_GEMV_SO=/opt/q38-release/native_ext/k100_int8_gemv_v7.so \
  -e K100_R304_DRAFTER_MAX_MODEL_LEN="$MTP_CUTOFF" \
  -e K100_R305_SPEC_CUTOFF="$MTP_CUTOFF" \
  -e K100_R308_DUAL_CUDAGRAPH=1 \
  -e K100_RMSNORM_INT8_FUSION=1 \
  -e K100_DRAFT_ONLY_LMHEAD_INT8=1 \
  -e K100_DRAFT_ONLY_LMHEAD_CHUNK=2048 \
  -e K100_DRAFT_BF16_VERIFY=1 \
  -e K100_DRAFT_FULL_BF16_ORACLE=0 \
  -e K100_DRAFT_ORACLE_LOG_RECALL=0 \
  -e K100_DRAFT_CANDIDATE_K=1024 \
  -e K100_R004D_SOURCE_SHA="$source_sha" \
  -e K100_Q38_R025_M1_COMPACT_TARGET=1 \
  -v "$MODEL_DIR":/models/qwen38-w8a8:ro \
  -v "$CACHE_DIR":/root/.cache/vllm \
  -v "$PATCH_DIR":/opt/k100-q38-patch:ro \
  -v "$ROOT_DIR":/opt/q38-release:ro \
  "$IMAGE" \
  vllm serve /models/qwen38-w8a8 \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code --dtype bfloat16 --tensor-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --quantization compressed-tensors \
    "${served_args[@]}" \
    --language-model-only --generation-config vllm --disable-custom-all-reduce \
    "${prefix_args[@]}" \
    --mamba-cache-dtype float16 --mamba-ssm-cache-dtype float16 \
    -cc.mode=3 -cc.inductor_compile_config '{"combo_kernels": false, "benchmark_combo_kernel": false}' \
    --cudagraph-capture-sizes 1 "$verify_m" \
    --max-num-seqs 32 --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --speculative-config "$spec_cfg"

echo "Started $CONTAINER_NAME"
echo "GPU=$GPU_ID PORT=$PORT mem_util=$GPU_MEMORY_UTILIZATION prefix_cache=$ENABLE_PREFIX_CACHING MTP5_cutoff=$MTP_CUTOFF"
echo "Follow logs: docker logs -f $CONTAINER_NAME"
