#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$ROOT_DIR/models}"

YOLOV8N_ONNX_URL="${YOLOV8N_ONNX_URL:-https://huggingface.co/cabelo/yolov8/resolve/main/yolov8n.onnx}"

# This archive is expected to contain the minimal GPT-2 KV assets used by phase6-kv:
#   model.kv_prefill.onnx, model.kv_decode.onnx, vocab.json, merges.txt
GPT2_ZIP_FILE_ID="${GPT2_ZIP_FILE_ID:-18MEDHiReBKk1nXuJrvSYNNCID-kJ5wiG}"
GPT2_DIR="${GPT2_DIR:-$MODELS_DIR/gpt2}"
GPT2_SOURCE="${GPT2_SOURCE:-gdrive}"

# Qwen exported ONNX files are large. By default this script supports the current
# shared Google Drive folder, while also making manual/exported placement easy to
# verify through the status command.
QWEN_GDRIVE_FOLDER_URL="${QWEN_GDRIVE_FOLDER_URL:-https://drive.google.com/drive/folders/1Fa_pBaL6ZbxDt4bNij5c87_cH0sIX3XK?usp=drive_link}"
QWEN_DIR="${QWEN_DIR:-$MODELS_DIR/qwen2_5_0_5b_instruct}"

log() {
  printf '[download_models] %s\n' "$1"
}

warn() {
  printf '[download_models][warn] %s\n' "$1" >&2
}

die() {
  printf '[download_models][error] %s\n' "$1" >&2
  exit 1
}

has_cmd() {
  command -v "$1" >/dev/null 2>&1
}

download_url() {
  local url="$1"
  local output_path="$2"
  if [[ -f "$output_path" ]]; then
    log "skip existing $output_path"
    return
  fi
  has_cmd curl || die "missing required command: curl"
  curl -fL --retry 3 --retry-delay 2 "$url" -o "$output_path"
}

download_gdrive_file() {
  local file_id="$1"
  local output_path="$2"
  if [[ -f "$output_path" ]]; then
    log "skip existing $output_path"
    return
  fi
  if has_cmd gdown; then
    gdown "https://drive.google.com/uc?id=${file_id}" -O "$output_path"
    return
  fi
  die "gdown is required for Google Drive downloads; install with: python -m pip install gdown"
}

require_file_for_status() {
  local path="$1"
  if [[ -f "$path" ]]; then
    printf '  [ok]      %s\n' "${path#$ROOT_DIR/}"
    return 0
  fi
  printf '  [missing] %s\n' "${path#$ROOT_DIR/}"
  return 1
}

status_yolo() {
  printf 'YOLOv8n:\n'
  require_file_for_status "$MODELS_DIR/yolov8n.onnx"
}

status_gpt2() {
  local ok=0
  printf 'GPT-2 KV:\n'
  require_file_for_status "$GPT2_DIR/model.kv_prefill.onnx" || ok=1
  require_file_for_status "$GPT2_DIR/model.kv_decode.onnx" || ok=1
  require_file_for_status "$GPT2_DIR/vocab.json" || ok=1
  require_file_for_status "$GPT2_DIR/merges.txt" || ok=1
  return "$ok"
}

status_qwen() {
  local ok=0
  printf 'Qwen2.5-0.5B KV:\n'
  require_file_for_status "$QWEN_DIR/model.kv_prefill.onnx" || ok=1
  require_file_for_status "$QWEN_DIR/model.kv_decode.onnx" || ok=1
  require_file_for_status "$QWEN_DIR/vocab.json" || ok=1
  require_file_for_status "$QWEN_DIR/merges.txt" || ok=1
  require_file_for_status "$QWEN_DIR/tokenizer.json" || true
  require_file_for_status "$QWEN_DIR/tokenizer_config.json" || true
  return "$ok"
}

status_all() {
  local ok=0
  status_yolo || ok=1
  status_gpt2 || ok=1
  status_qwen || ok=1
  return "$ok"
}

download_yolo() {
  mkdir -p "$MODELS_DIR"
  log "downloading YOLOv8n ONNX model"
  download_url "$YOLOV8N_ONNX_URL" "$MODELS_DIR/yolov8n.onnx"
  status_yolo
}

download_gpt2_from_archive() {
  mkdir -p "$GPT2_DIR"
  if status_gpt2 >/dev/null 2>&1; then
    log "GPT-2 KV assets already exist"
    status_gpt2
    return
  fi

  local archive="$GPT2_DIR/gpt2_kv_assets.zip"
  log "downloading GPT-2 KV asset archive"
  download_gdrive_file "$GPT2_ZIP_FILE_ID" "$archive"

  has_cmd unzip || die "missing required command: unzip"
  log "extracting GPT-2 archive into $GPT2_DIR"
  unzip -o "$archive" -d "$GPT2_DIR/"
  rm -f "$archive"

  status_gpt2 || die "GPT-2 download finished but required files are still missing"
}

download_gpt2_from_hf_export() {
  log "delegating GPT-2 download/export to scripts/fetch_gpt2.sh"
  MODEL_DIR="$GPT2_DIR" EXPORT_KV_CACHE="${EXPORT_KV_CACHE:-1}" "$ROOT_DIR/scripts/fetch_gpt2.sh"
  status_gpt2 || die "GPT-2 export finished but required KV files are still missing"
}

download_gpt2() {
  case "$GPT2_SOURCE" in
    gdrive)
      download_gpt2_from_archive
      ;;
    hf-export)
      download_gpt2_from_hf_export
      ;;
    *)
      die "unknown GPT2_SOURCE=$GPT2_SOURCE; expected gdrive or hf-export"
      ;;
  esac
}

download_qwen() {
  mkdir -p "$QWEN_DIR"
  if status_qwen >/dev/null 2>&1; then
    log "Qwen KV assets already exist"
    status_qwen
    return
  fi

  if has_cmd gdown; then
    log "downloading Qwen folder via gdown"
    gdown --folder "$QWEN_GDRIVE_FOLDER_URL" --output "$QWEN_DIR" --remaining-ok
    status_qwen || die "Qwen download finished but required files are still missing"
    return
  fi

  warn "gdown is not installed; skipping automatic Qwen folder download."
  warn "Install it with: pip install gdown"
  warn "Then rerun: ./scripts/download_models.sh qwen"
  warn "Or place/export the required files under: $QWEN_DIR"
  status_qwen || return 1
}

usage() {
  cat <<EOF
usage:
  ./scripts/download_models.sh [status|yolo|gpt2|qwen|all]

targets:
  status  show required local model files
  yolo    download models/yolov8n.onnx
  gpt2    prepare GPT-2 KV assets for phase6-kv
  qwen    prepare Qwen2.5-0.5B KV assets for phase7
  all     run yolo + gpt2 + qwen

environment overrides:
  MODELS_DIR=/path/to/models
  YOLOV8N_ONNX_URL=https://...
  GPT2_DIR=/path/to/models/gpt2
  GPT2_SOURCE=gdrive|hf-export
  GPT2_ZIP_FILE_ID=...
  QWEN_DIR=/path/to/models/qwen2_5_0_5b_instruct
  QWEN_GDRIVE_FOLDER_URL=https://drive.google.com/drive/folders/...

notes:
  - GPT2_SOURCE=gdrive uses the repository's small shared archive of exported KV assets.
  - Google Drive downloads require gdown: python -m pip install gdown
  - GPT2_SOURCE=hf-export downloads Hugging Face GPT-2 files and exports ONNX locally.
  - Qwen ONNX files are large; status tells you exactly which files must exist.
EOF
}

main() {
  local target="${1:-all}"

  case "$target" in
    status)
      status_all || true
      ;;
    yolo)
      download_yolo
      ;;
    gpt2)
      download_gpt2
      ;;
    qwen)
      download_qwen
      ;;
    all)
      download_yolo
      download_gpt2
      download_qwen
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      printf '[download_models][error] unknown target: %s\n' "$target" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
