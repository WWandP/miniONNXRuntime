#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$ROOT_DIR/models}"

# Existing GPT-2 zip file id used by this repo.
GPT2_ZIP_FILE_ID="18MEDHiReBKk1nXuJrvSYNNCID-kJ5wiG"

# Existing additional model zip file id used by this repo.
ADDITIONAL_ZIP_FILE_ID="15kO6Yn8Spo90hYBVLTB3a0CauordZSav"

# User-provided Qwen folder link.
QWEN_GDRIVE_FOLDER_URL="https://drive.google.com/drive/folders/1Fa_pBaL6ZbxDt4bNij5c87_cH0sIX3XK?usp=drive_link"
QWEN_DIR="$MODELS_DIR/qwen2_5_0_5b_instruct"

log() {
  printf '[download_models] %s\n' "$1"
}

warn() {
  printf '[download_models][warn] %s\n' "$1" >&2
}

download_gdrive_file() {
  local file_id="$1"
  local output_path="$2"
  curl -fL "https://drive.google.com/uc?export=download&id=${file_id}" -o "$output_path"
}

ensure_qwen_folder() {
  mkdir -p "$QWEN_DIR"

  if command -v gdown >/dev/null 2>&1; then
    log "Downloading Qwen folder via gdown..."
    # --remaining-ok keeps existing files and continues downloading missing ones.
    gdown --folder "$QWEN_GDRIVE_FOLDER_URL" --output "$QWEN_DIR" --remaining-ok
    return 0
  fi

  warn "gdown is not installed; skipping automatic Qwen folder download."
  warn "Install gdown with: pip install gdown"
  warn "Then rerun this script, or download manually from:"
  warn "$QWEN_GDRIVE_FOLDER_URL"
  warn "And place files under: $QWEN_DIR"
}

main() {
  mkdir -p "$MODELS_DIR"

  log "Downloading GPT-2 model archive..."
  mkdir -p "$MODELS_DIR/gpt2"
  download_gdrive_file "$GPT2_ZIP_FILE_ID" "$MODELS_DIR/gpt2/gpt2_model.zip"

  log "Extracting GPT-2 model archive..."
  unzip -o "$MODELS_DIR/gpt2/gpt2_model.zip" -d "$MODELS_DIR/gpt2/"
  rm -f "$MODELS_DIR/gpt2/gpt2_model.zip"
  log "GPT-2 model downloaded to $MODELS_DIR/gpt2"

  log "Downloading additional model archive..."
  download_gdrive_file "$ADDITIONAL_ZIP_FILE_ID" "$MODELS_DIR/additional_model.zip"

  log "Extracting additional model archive..."
  unzip -o "$MODELS_DIR/additional_model.zip" -d "$MODELS_DIR/"
  rm -f "$MODELS_DIR/additional_model.zip"
  log "Additional model extracted to $MODELS_DIR"

  ensure_qwen_folder

  log "Done."
}

main "$@"
