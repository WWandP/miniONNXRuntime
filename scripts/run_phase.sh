#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build_local}"
MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/models/yolov8n.onnx}"
IMAGE_PATH="${IMAGE_PATH:-$ROOT_DIR/pic/bus.jpg}"
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
  ./scripts/run_phase.sh all

environment overrides:
  BUILD_DIR=/path/to/build
  MODEL_PATH=/path/to/model.onnx
  IMAGE_PATH=/path/to/image.jpg
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

run_all() {
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
