#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Export local Qwen CausalLM checkpoint to KV-cache prefill/decode ONNX models.")
  parser.add_argument(
      "--model-dir",
      default="models/qwen2_5_0_5b_instruct",
      help="Local model directory (downloaded from ModelScope/HF).",
  )
  parser.add_argument(
      "--prefill-output",
      default="model.kv_prefill.onnx",
      help="Prefill ONNX filename (relative to --model-dir if not absolute).",
  )
  parser.add_argument(
      "--decode-output",
      default="model.kv_decode.onnx",
      help="Decode ONNX filename (relative to --model-dir if not absolute).",
  )
  parser.add_argument(
      "--prefill-prompt",
      default="你好",
      help="Prompt used to create prefill dummy input ids.",
  )
  parser.add_argument(
      "--decode-prompt",
      default="!",
      help="Prompt used to create decode dummy input ids (first token will be used).",
  )
  parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
  parser.add_argument(
      "--constant-folding",
      action="store_true",
      help="Enable constant folding during export (default: disabled for stability).",
  )
  parser.add_argument(
      "--external-data",
      action="store_true",
      help="Allow exporting external tensor data files.",
  )
  return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
  model_dir = Path(args.model_dir).resolve()
  prefill_path = Path(args.prefill_output)
  decode_path = Path(args.decode_output)
  if not prefill_path.is_absolute():
    prefill_path = model_dir / prefill_path
  if not decode_path.is_absolute():
    decode_path = model_dir / decode_path
  return model_dir, prefill_path, decode_path


def resolve_model_dims(model: torch.nn.Module) -> Tuple[int, int, int]:
  config = model.config
  if hasattr(config, "num_hidden_layers"):
    num_layers = int(config.num_hidden_layers)
  elif hasattr(config, "n_layer"):
    num_layers = int(config.n_layer)
  else:
    raise RuntimeError("cannot resolve number of layers from model config")

  if hasattr(config, "num_attention_heads"):
    num_attention_heads = int(config.num_attention_heads)
  elif hasattr(config, "n_head"):
    num_attention_heads = int(config.n_head)
  else:
    raise RuntimeError("cannot resolve number of attention heads from model config")

  num_kv_heads = int(getattr(config, "num_key_value_heads", num_attention_heads))
  if hasattr(config, "hidden_size"):
    hidden_size = int(config.hidden_size)
  elif hasattr(config, "n_embd"):
    hidden_size = int(config.n_embd)
  else:
    raise RuntimeError("cannot resolve hidden size from model config")

  head_dim = hidden_size // num_attention_heads
  if head_dim * num_attention_heads != hidden_size:
    raise RuntimeError("invalid head dimension derived from hidden_size / num_attention_heads")
  return num_layers, num_kv_heads, head_dim


class QwenPrefill(torch.nn.Module):
  def __init__(self, inner: torch.nn.Module):
    super().__init__()
    self.inner = inner

  def forward(self, input_ids: torch.Tensor):
    outputs = self.inner(input_ids=input_ids, use_cache=True, return_dict=True)
    flat = [outputs.logits]
    for key, value in outputs.past_key_values:
      flat.extend([key, value])
    return tuple(flat)


class QwenDecode(torch.nn.Module):
  def __init__(self, inner: torch.nn.Module, num_layers: int):
    super().__init__()
    self.inner = inner
    self.num_layers = num_layers

  def forward(self, input_ids: torch.Tensor, *past_key_values: torch.Tensor):
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


def export_legacy(wrapper: torch.nn.Module, inputs, output_path: Path, input_names, output_names, dynamic_axes,
                  opset: int, constant_folding: bool, external_data: bool) -> None:
  try:
    torch.onnx.utils._export(
        wrapper,
        inputs,
        output_path.as_posix(),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=constant_folding,
        use_external_data_format=external_data,
    )
  except TypeError:
    torch.onnx.utils._export(
        wrapper,
        inputs,
        output_path.as_posix(),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=constant_folding,
    )


def main() -> None:
  args = parse_args()
  model_dir, prefill_path, decode_path = resolve_paths(args)

  tokenizer = AutoTokenizer.from_pretrained(model_dir.as_posix(), local_files_only=True)
  model = AutoModelForCausalLM.from_pretrained(model_dir.as_posix(), local_files_only=True)
  model.eval()

  num_layers, num_kv_heads, head_dim = resolve_model_dims(model)

  prefill_ids = tokenizer.encode(args.prefill_prompt, return_tensors="pt")
  decode_ids = tokenizer.encode(args.decode_prompt, return_tensors="pt")
  if decode_ids.shape[-1] == 0:
    raise RuntimeError("--decode-prompt encoded to empty token ids")
  decode_ids = decode_ids[:, :1]

  past_shape = (1, num_kv_heads, 2, head_dim)
  past_inputs = tuple(torch.zeros(past_shape, dtype=torch.float32) for _ in range(num_layers * 2))

  prefill_wrapper = QwenPrefill(model).eval()
  decode_wrapper = QwenDecode(model, num_layers).eval()

  decode_input_names = ["input_ids"]
  output_names = ["logits"]
  prefill_dynamic_axes = {
      "input_ids": {0: "batch", 1: "sequence"},
      "logits": {0: "batch", 1: "sequence"},
  }
  decode_dynamic_axes = {
      "input_ids": {0: "batch", 1: "sequence"},
      "logits": {0: "batch", 1: "sequence"},
  }

  for i in range(num_layers):
    decode_input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
    output_names.extend([f"present.{i}.key", f"present.{i}.value"])
    decode_dynamic_axes[f"past_key_values.{i}.key"] = {0: "batch", 2: "past_sequence"}
    decode_dynamic_axes[f"past_key_values.{i}.value"] = {0: "batch", 2: "past_sequence"}
    prefill_dynamic_axes[f"present.{i}.key"] = {0: "batch", 2: "past_sequence"}
    prefill_dynamic_axes[f"present.{i}.value"] = {0: "batch", 2: "past_sequence"}
    decode_dynamic_axes[f"present.{i}.key"] = {0: "batch", 2: "past_sequence"}
    decode_dynamic_axes[f"present.{i}.value"] = {0: "batch", 2: "past_sequence"}

  with torch.no_grad():
    export_legacy(
        prefill_wrapper,
        (prefill_ids,),
        prefill_path,
        ["input_ids"],
        output_names,
        prefill_dynamic_axes,
        args.opset,
        args.constant_folding,
        args.external_data,
    )
    export_legacy(
        decode_wrapper,
        (decode_ids, *past_inputs),
        decode_path,
        decode_input_names,
        output_names,
        decode_dynamic_axes,
        args.opset,
        args.constant_folding,
        args.external_data,
    )

  print(f"exported prefill: {prefill_path}")
  print(f"exported decode:  {decode_path}")
  print(f"layers={num_layers} kv_heads={num_kv_heads} head_dim={head_dim}")


if __name__ == "__main__":
  main()
