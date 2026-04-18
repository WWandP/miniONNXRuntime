#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build_local}"
MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/models/yolov8n.onnx}"
IMAGE_PATH="${IMAGE_PATH:-$ROOT_DIR/pic/bus.jpg}"
GPT_MODEL_DIR="${GPT_MODEL_DIR:-$ROOT_DIR/models/gpt2}"
GPT_MODEL_PATH="${GPT_MODEL_PATH:-$GPT_MODEL_DIR/model.kv_prefill.onnx}"
GPT_KV_PREFILL_MODEL_PATH="${GPT_KV_PREFILL_MODEL_PATH:-$GPT_MODEL_DIR/model.kv_prefill.onnx}"
GPT_KV_DECODE_MODEL_PATH="${GPT_KV_DECODE_MODEL_PATH:-$GPT_MODEL_DIR/model.kv_decode.onnx}"
GPT_PROMPT_FILE="${GPT_PROMPT_FILE:-$ROOT_DIR/examples/gpt2_tiny/story_prompt.txt}"
GPT_GENERATE="${GPT_GENERATE:-48}"
QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-$ROOT_DIR/models/qwen2_5_0_5b_instruct}"
QWEN_KV_PREFILL_MODEL_PATH="${QWEN_KV_PREFILL_MODEL_PATH:-$QWEN_MODEL_DIR/model.kv_prefill.onnx}"
QWEN_KV_DECODE_MODEL_PATH="${QWEN_KV_DECODE_MODEL_PATH:-$QWEN_MODEL_DIR/model.kv_decode.onnx}"
QWEN_PROMPT="${QWEN_PROMPT:-你好}"
QWEN_GENERATE="${QWEN_GENERATE:-8}"
CMAKE_BIN="${CMAKE_BIN:-cmake}"

usage() {
  cat <<EOF
usage:
  ./scripts/run_phase.sh build
  ./scripts/run_phase.sh test
  ./scripts/run_phase.sh phase1
  ./scripts/run_phase.sh phase2
  ./scripts/run_phase.sh phase3
  ./scripts/run_phase.sh phase4-opt
  ./scripts/run_phase.sh phase4-memory
  ./scripts/run_phase.sh phase5
  ./scripts/run_phase.sh phase6
  ./scripts/run_phase.sh phase6-kv
  ./scripts/run_phase.sh phase7
  ./scripts/run_phase.sh all

notes:
  - all currently runs phase1 -> phase5
  - phase6 / phase6-kv / phase7 are opt-in text-model phases (local model assets are required)

environment overrides:
  BUILD_DIR=/path/to/build
  MODEL_PATH=/path/to/model.onnx
  IMAGE_PATH=/path/to/image.jpg
  GPT_MODEL_DIR=/path/to/models/gpt2
  GPT_MODEL_PATH=/path/to/model.kv_prefill.onnx
  GPT_KV_PREFILL_MODEL_PATH=/path/to/model.kv_prefill.onnx
  GPT_KV_DECODE_MODEL_PATH=/path/to/model.kv_decode.onnx
  GPT_PROMPT_FILE=/path/to/prompt.txt
  GPT_GENERATE=48
  QWEN_MODEL_DIR=/path/to/models/qwen
  QWEN_KV_PREFILL_MODEL_PATH=/path/to/model.kv_prefill.onnx
  QWEN_KV_DECODE_MODEL_PATH=/path/to/model.kv_decode.onnx
  QWEN_PROMPT=你好
  QWEN_GENERATE=8
  CMAKE_BIN=/path/to/cmake
EOF
}

log() {
  printf '[run_phase] %s\n' "$1"
}

require_file() {
  [[ -f "$1" ]] || {
    printf '[run_phase][error] missing file: %s\n' "$1" >&2
    exit 1
  }
}

configure_build() {
  log "configuring build directory: $BUILD_DIR"
  "$CMAKE_BIN" -S "$ROOT_DIR" -B "$BUILD_DIR" -DMINIORT_BUILD_OPTIMIZER_TOOLS=ON
}

build_all() {
  configure_build
  log "building tools"
  "$CMAKE_BIN" --build "$BUILD_DIR" -j4
}

run_test() {
  build_all
  log "running tests"
  ctest --test-dir "$BUILD_DIR" --output-on-failure
}

run_phase1() {
  build_all
  require_file "$MODEL_PATH"
  "$BUILD_DIR/miniort_inspect" "$MODEL_PATH"
}

run_phase2() {
  build_all
  require_file "$MODEL_PATH"
  require_file "$IMAGE_PATH"
  "$BUILD_DIR/miniort_session_trace" "$MODEL_PATH" --image "$IMAGE_PATH" --max-nodes 8
}

run_phase3() {
  build_all
  require_file "$MODEL_PATH"
  require_file "$IMAGE_PATH"
  "$BUILD_DIR/miniort_run" "$MODEL_PATH" --image "$IMAGE_PATH"
}

run_phase4_opt() {
  build_all
  require_file "$MODEL_PATH"
  require_file "$IMAGE_PATH"
  "$BUILD_DIR/miniort_optimize_model" "$MODEL_PATH" --image "$IMAGE_PATH"
}

run_phase4_memory() {
  build_all
  require_file "$MODEL_PATH"
  require_file "$IMAGE_PATH"
  "$BUILD_DIR/miniort_memory_trace" "$MODEL_PATH" --image "$IMAGE_PATH"
}

run_phase5() {
  build_all
  require_file "$MODEL_PATH"
  require_file "$IMAGE_PATH"
  "$BUILD_DIR/miniort_compare_providers" "$MODEL_PATH" --image "$IMAGE_PATH" --repeat 1
}

run_phase6() {
  build_all
  require_file "$GPT_MODEL_PATH"
  require_file "$GPT_PROMPT_FILE"
  require_file "$GPT_MODEL_DIR/vocab.json"
  require_file "$GPT_MODEL_DIR/merges.txt"
  "$BUILD_DIR/miniort_run_gpt" \
    "$GPT_MODEL_PATH" \
    --prompt-file "$GPT_PROMPT_FILE" \
    --model-dir "$GPT_MODEL_DIR" \
    --generate "$GPT_GENERATE" \
    --top-k 5 \
    --strict
}

run_phase6_kv() {
  build_all
  require_file "$GPT_KV_PREFILL_MODEL_PATH"
  require_file "$GPT_KV_DECODE_MODEL_PATH"
  require_file "$GPT_PROMPT_FILE"
  require_file "$GPT_MODEL_DIR/vocab.json"
  require_file "$GPT_MODEL_DIR/merges.txt"
  "$BUILD_DIR/miniort_run_gpt" \
    --prompt-file "$GPT_PROMPT_FILE" \
    --model-dir "$GPT_MODEL_DIR" \
    --generate "$GPT_GENERATE" \
    --top-k 5 \
    --kv-cache \
    --kv-cache-prefill-model "$GPT_KV_PREFILL_MODEL_PATH" \
    --kv-cache-decode-model "$GPT_KV_DECODE_MODEL_PATH" \
    --strict
}

run_phase7() {
  build_all
  require_file "$QWEN_KV_PREFILL_MODEL_PATH"
  require_file "$QWEN_KV_DECODE_MODEL_PATH"
  require_file "$QWEN_MODEL_DIR/vocab.json"
  require_file "$QWEN_MODEL_DIR/merges.txt"
  "$BUILD_DIR/miniort_run_qwen" \
    --kv-cache \
    --kv-cache-prefill-model "$QWEN_KV_PREFILL_MODEL_PATH" \
    --kv-cache-decode-model "$QWEN_KV_DECODE_MODEL_PATH" \
    --model-dir "$QWEN_MODEL_DIR" \
    --prompt "$QWEN_PROMPT" \
    --generate "$QWEN_GENERATE" \
    --top-k 5 \
    --strict
}

run_all() {
  # Keep `all` stable for the default teaching flow.
  # Text-model phases are opt-in to avoid failures on missing local model assets.
  run_phase1
  run_phase2
  run_phase3
  run_phase4_opt
  run_phase4_memory
  run_phase5
}

main() {
  cd "$ROOT_DIR"

  if [[ $# -ne 1 ]]; then
    usage
    exit 1
  fi

  case "$1" in
    build)
      build_all
      ;;
    test)
      run_test
      ;;
    phase1)
      run_phase1
      ;;
    phase2)
      run_phase2
      ;;
    phase3)
      run_phase3
      ;;
    phase4-opt)
      run_phase4_opt
      ;;
    phase4-memory)
      run_phase4_memory
      ;;
    phase5)
      run_phase5
      ;;
    phase6)
      run_phase6
      ;;
    phase6-kv)
      run_phase6_kv
      ;;
    phase7)
      run_phase7
      ;;
    all)
      run_all
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      printf '[run_phase][error] unknown command: %s\n' "$1" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
