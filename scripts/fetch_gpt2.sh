#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models/gpt2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HF_BASE_URL="${HF_BASE_URL:-https://huggingface.co/openai-community/gpt2/resolve/main}"

log() {
  printf '[fetch_gpt2] %s\n' "$1"
}

die() {
  printf '[fetch_gpt2][error] %s\n' "$1" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

download_file() {
  local file_name="$1"
  if [[ -f "$MODEL_DIR/$file_name" ]]; then
    log "skip existing $file_name"
    return
  fi
  log "downloading $file_name"
  curl -L --fail --output "$MODEL_DIR/$file_name" "$HF_BASE_URL/$file_name"
}

main() {
  require_cmd curl
  require_cmd "$PYTHON_BIN"

  mkdir -p "$MODEL_DIR"

  download_file config.json
  download_file generation_config.json
  download_file model.safetensors
  download_file tokenizer.json
  download_file tokenizer_config.json
  download_file vocab.json
  download_file merges.txt

  log "downloaded Hugging Face assets into $MODEL_DIR"

  if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import importlib.util
mods = ["torch", "transformers", "onnx"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
raise SystemExit(0 if not missing else 1)
PY
  then
    cat <<EOF

[fetch_gpt2] download finished, but ONNX export dependencies are missing

required Python packages:
  torch
  transformers
  onnx

optional package for model.sim.onnx:
  onnxsim

once they are installed, rerun:
  MODEL_DIR="$MODEL_DIR" PYTHON_BIN="$PYTHON_BIN" ./scripts/fetch_gpt2.sh
EOF
    exit 0
  fi

  log "exporting model.onnx"
  MODEL_DIR="$MODEL_DIR" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = Path(os.environ["MODEL_DIR"])
onnx_path = model_dir / "model.onnx"

tokenizer = AutoTokenizer.from_pretrained(model_dir.as_posix(), local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_dir.as_posix(), local_files_only=True)
model.eval()

class Gpt2LogitsOnly(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, input_ids):
        return self.inner(input_ids=input_ids, return_dict=True).logits

wrapper = Gpt2LogitsOnly(model)
example_ids = tokenizer.encode("Hello", add_special_tokens=False, return_tensors="pt")

with torch.no_grad():
    torch.onnx.export(
        wrapper,
        (example_ids,),
        onnx_path.as_posix(),
        input_names=["input_ids"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
    )
PY

  if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("onnxsim") is not None else 1)
PY
  then
    log "simplifying model.onnx into model.sim.onnx"
    MODEL_DIR="$MODEL_DIR" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

import onnx
from onnxsim import simplify

model_dir = Path(os.environ["MODEL_DIR"])
model_path = model_dir / "model.onnx"
sim_path = model_dir / "model.sim.onnx"

model = onnx.load(model_path.as_posix())
simplified, ok = simplify(model)
if not ok:
    raise RuntimeError("onnxsim failed to validate the simplified model")
onnx.save(simplified, sim_path.as_posix())
PY
  else
    cat <<EOF

[fetch_gpt2] model.onnx is ready
[fetch_gpt2] onnxsim is not installed, so model.sim.onnx was not generated

install it with:
  $PYTHON_BIN -m pip install onnxsim
EOF
  fi

  cat <<EOF

[fetch_gpt2] ready

local files:
  $MODEL_DIR/config.json
  $MODEL_DIR/model.safetensors
  $MODEL_DIR/model.onnx
  $MODEL_DIR/model.sim.onnx   (only if onnxsim is installed)
EOF
}

main "$@"
