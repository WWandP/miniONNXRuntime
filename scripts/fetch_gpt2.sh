#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models/gpt2}"
PYTHON_BIN="${PYTHON_BIN:-}"
HF_BASE_URL="${HF_BASE_URL:-https://huggingface.co/openai-community/gpt2/resolve/main}"
EXPORT_KV_CACHE="${EXPORT_KV_CACHE:-0}"

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

python_has_deps() {
  local candidate="$1"
  "$candidate" - <<'PY' >/dev/null 2>&1
import importlib.util
mods = ["torch", "transformers", "onnx"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
raise SystemExit(0 if not missing else 1)
PY
}

resolve_python() {
  if [[ -n "$PYTHON_BIN" ]]; then
    if python_has_deps "$PYTHON_BIN"; then
      printf '%s\n' "$PYTHON_BIN"
      return
    fi
    die "PYTHON_BIN=$PYTHON_BIN does not have torch/transformers/onnx"
  fi

  local candidates=()
  if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    candidates+=("$CONDA_PREFIX/bin/python")
  fi
  if command -v python >/dev/null 2>&1; then
    candidates+=("python")
  fi
  if command -v python3 >/dev/null 2>&1; then
    candidates+=("python3")
  fi

  local candidate
  for candidate in "${candidates[@]}"; do
    if python_has_deps "$candidate"; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  die "could not find a Python interpreter with torch/transformers/onnx installed"
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
  PYTHON_BIN="$(resolve_python)"
  log "using python: $PYTHON_BIN"

  mkdir -p "$MODEL_DIR"

  download_file config.json
  download_file generation_config.json
  download_file model.safetensors
  download_file tokenizer.json
  download_file tokenizer_config.json
  download_file vocab.json
  download_file merges.txt

  log "downloaded Hugging Face assets into $MODEL_DIR"

  if ! python_has_deps "$PYTHON_BIN"; then
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
    torch.onnx.utils._export(
        wrapper,
        (example_ids,),
        onnx_path.as_posix(),
        input_names=["input_ids"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
    )
PY

  if [[ "$EXPORT_KV_CACHE" == "1" ]]; then
    log "exporting kv-cache models"
    MODEL_DIR="$MODEL_DIR" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = Path(os.environ["MODEL_DIR"])
prefill_path = model_dir / "model.kv_prefill.onnx"
decode_path = model_dir / "model.kv_decode.onnx"

tokenizer = AutoTokenizer.from_pretrained(model_dir.as_posix(), local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_dir.as_posix(), local_files_only=True)
model.eval()

num_layers = int(model.config.n_layer)
num_heads = int(model.config.n_head)
head_dim = int(model.config.n_embd // model.config.n_head)

class Gpt2Prefill(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, input_ids):
        outputs = self.inner(input_ids=input_ids, use_cache=True, return_dict=True)
        flat = [outputs.logits]
        for key, value in outputs.past_key_values:
            flat.extend([key, value])
        return tuple(flat)


class Gpt2Decode(torch.nn.Module):
    def __init__(self, inner, num_layers):
        super().__init__()
        self.inner = inner
        self.num_layers = num_layers

    def forward(self, input_ids, *past_key_values):
        if len(past_key_values) != self.num_layers * 2:
            raise RuntimeError("unexpected past_key_values arity")
        past = []
        for i in range(self.num_layers):
            past.append((past_key_values[i * 2], past_key_values[i * 2 + 1]))
        outputs = self.inner(
            input_ids=input_ids,
            past_key_values=tuple(past),
            use_cache=True,
            return_dict=True,
        )
        flat = [outputs.logits]
        for key, value in outputs.past_key_values:
            flat.extend([key, value])
        return tuple(flat)


prefill_wrapper = Gpt2Prefill(model)
decode_wrapper = Gpt2Decode(model, num_layers)
prefill_wrapper.eval()
decode_wrapper.eval()
prefill_ids = tokenizer.encode("Hello", add_special_tokens=False, return_tensors="pt")
decode_ids = tokenizer.encode("!", add_special_tokens=False, return_tensors="pt")
past_shape = (1, num_heads, 2, head_dim)
past_inputs = tuple(torch.zeros(past_shape, dtype=torch.float32) for _ in range(num_layers * 2))

input_names = ["input_ids"]
output_names = ["logits"]
prefill_dynamic_axes = {"input_ids": {0: "batch", 1: "sequence"}, "logits": {0: "batch", 1: "sequence"}}
decode_dynamic_axes = {"input_ids": {0: "batch", 1: "sequence"}, "logits": {0: "batch", 1: "sequence"}}
for i in range(num_layers):
    input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
    output_names.extend([f"present.{i}.key", f"present.{i}.value"])
    decode_dynamic_axes[f"past_key_values.{i}.key"] = {0: "batch", 2: "past_sequence"}
    decode_dynamic_axes[f"past_key_values.{i}.value"] = {0: "batch", 2: "past_sequence"}
    prefill_dynamic_axes[f"present.{i}.key"] = {0: "batch", 2: "past_sequence"}
    prefill_dynamic_axes[f"present.{i}.value"] = {0: "batch", 2: "past_sequence"}
    decode_dynamic_axes[f"present.{i}.key"] = {0: "batch", 2: "past_sequence"}
    decode_dynamic_axes[f"present.{i}.value"] = {0: "batch", 2: "past_sequence"}

with torch.no_grad():
    torch.onnx.utils._export(
        prefill_wrapper,
        (prefill_ids,),
        prefill_path.as_posix(),
        input_names=["input_ids"],
        output_names=output_names,
        dynamic_axes=prefill_dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    torch.onnx.utils._export(
        decode_wrapper,
        (decode_ids, *past_inputs),
        decode_path.as_posix(),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=decode_dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
PY
  fi

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
  $MODEL_DIR/model.kv_prefill.onnx  (only if EXPORT_KV_CACHE=1 and torch/transformers are installed)
  $MODEL_DIR/model.kv_decode.onnx   (only if EXPORT_KV_CACHE=1 and torch/transformers are installed)
EOF
}

main "$@"
