#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_BIN="$HOME/.local/bin"
CONDA_PREFIX_DEFAULT="${CONDA_PREFIX:-$HOME/miniconda3}"

log() {
  printf '[setup] %s\n' "$1"
}

die() {
  printf '[setup][error] %s\n' "$1" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

ensure_path_contains_local_bin() {
  case ":$PATH:" in
    *":$LOCAL_BIN:"*) ;;
    *) export PATH="$LOCAL_BIN:$PATH" ;;
  esac
}

install_cmake() {
  if command -v cmake >/dev/null 2>&1; then
    log "cmake already available: $(cmake --version | head -n 1)"
    return
  fi

  log "installing cmake with pip --user"
  python3 -m pip install --user cmake
  ensure_path_contains_local_bin
  command -v cmake >/dev/null 2>&1 || die "cmake install finished but cmake is still not on PATH"
  log "cmake ready: $(cmake --version | head -n 1)"
}

install_protobuf_cpp() {
  if command -v conda >/dev/null 2>&1; then
    log "installing protobuf Python package and C++ runtime/tooling with conda-forge"
    conda install -y --solver libmamba -c conda-forge --override-channels protobuf libprotobuf
    return
  fi

  if command -v apt-get >/dev/null 2>&1; then
    log "conda not found, falling back to apt-get"
    if [[ "$(id -u)" -eq 0 ]]; then
      apt-get update
      apt-get install -y build-essential cmake libprotobuf-dev protobuf-compiler
    elif command -v sudo >/dev/null 2>&1; then
      sudo apt-get update
      sudo apt-get install -y build-essential cmake libprotobuf-dev protobuf-compiler
    else
      die "apt-get is available but sudo is missing; rerun as root or install cmake/libprotobuf-dev/protobuf-compiler manually"
    fi
    return
  fi

  die "neither conda nor apt-get is available; please install cmake, libprotobuf-dev and protobuf-compiler manually"
}

verify_protobuf_cpp() {
  local protoc_bin
  protoc_bin="$(command -v protoc || true)"
  [[ -n "$protoc_bin" ]] || die "protoc not found after installation"

  local header_path=""
  local library_path=""

  if [[ -f "$CONDA_PREFIX_DEFAULT/include/google/protobuf/message.h" ]]; then
    header_path="$CONDA_PREFIX_DEFAULT/include/google/protobuf/message.h"
  elif [[ -f "/usr/include/google/protobuf/message.h" ]]; then
    header_path="/usr/include/google/protobuf/message.h"
  elif [[ -f "/usr/local/include/google/protobuf/message.h" ]]; then
    header_path="/usr/local/include/google/protobuf/message.h"
  fi

  for candidate in \
    "$CONDA_PREFIX_DEFAULT/lib/libprotobuf.so" \
    "/usr/lib/x86_64-linux-gnu/libprotobuf.so" \
    "/usr/lib64/libprotobuf.so" \
    "/usr/local/lib/libprotobuf.so"
  do
    if [[ -f "$candidate" ]]; then
      library_path="$candidate"
      break
    fi
  done

  [[ -n "$header_path" ]] || die "protobuf C++ header not found in expected locations"
  [[ -n "$library_path" ]] || die "protobuf library not found in expected locations"

  log "protoc ready: $(protoc --version)"
  log "protobuf header: $header_path"
  log "protobuf library: $library_path"
}

main() {
  [[ "$(uname -s)" == "Linux" ]] || die "this script is intended for Linux only"

  require_cmd python3
  require_cmd g++

  ensure_path_contains_local_bin

  log "project root: $ROOT_DIR"
  log "using conda prefix: $CONDA_PREFIX_DEFAULT"

  install_cmake
  install_protobuf_cpp
  verify_protobuf_cpp

  cat <<EOF

[setup] environment is ready

next steps:
  cd "$ROOT_DIR"
  cmake -S . -B build_local
  cmake --build build_local -j4
  ctest --test-dir build_local --output-on-failure

optional smoke test:
  ./build_local/miniort_inspect models/yolov8n.onnx
  ./build_local/miniort_run models/yolov8n.onnx --image pic/bus.jpg
EOF
}

main "$@"
