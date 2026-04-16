#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models/tiny_gpt2_export}"
PYTHON_BIN="${PYTHON_BIN:-/Volumes/ww/miniconda3/envs/norm/bin/python}"
MODEL_ID="${MODEL_ID:-sshleifer/tiny-gpt2}"

log() {
  printf '[fetch_tiny_gpt2] %s\n' "$1"
}

die() {
  printf '[fetch_tiny_gpt2][error] %s\n' "$1" >&2
  exit 1
}

require_python_deps() {
  "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import importlib.util
mods = ["torch", "transformers", "onnx"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
raise SystemExit(0 if not missing else 1)
PY
}

[[ -x "$PYTHON_BIN" ]] || die "python not found or not executable: $PYTHON_BIN"
require_python_deps || die "python env missing one of torch/transformers/onnx: $PYTHON_BIN"

mkdir -p "$MODEL_DIR"
log "model_id=$MODEL_ID"
log "model_dir=$MODEL_DIR"
log "python=$PYTHON_BIN"

MODEL_ID="$MODEL_ID" MODEL_DIR="$MODEL_DIR" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = os.environ["MODEL_ID"]
model_dir = Path(os.environ["MODEL_DIR"])
model_dir.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
model.eval()

tokenizer.save_pretrained(model_dir)
model.save_pretrained(model_dir)

class Gpt2LogitsOnly(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, input_ids):
        return self.inner(input_ids=input_ids, return_dict=True).logits

wrapper = Gpt2LogitsOnly(model)
example_ids = tokenizer.encode("Hello", add_special_tokens=False, return_tensors="pt")
onnx_path = model_dir / "model.onnx"

with torch.no_grad():
    torch.onnx.utils._export(
        wrapper,
        (example_ids,),
        onnx_path.as_posix(),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}, "logits": {0: "batch", 1: "sequence"}},
        opset_version=17,
        do_constant_folding=True,
    )

print(f"exported {onnx_path}")
PY

log "done"
