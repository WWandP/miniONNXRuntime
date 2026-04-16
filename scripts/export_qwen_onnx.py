#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Export a local Qwen CausalLM checkpoint to logits-only ONNX.")
  parser.add_argument(
      "--model-dir",
      default="models/qwen2_5_0_5b_instruct",
      help="Local model directory (downloaded from ModelScope/HF).",
  )
  parser.add_argument(
      "--output",
      default="model.baseline.onnx",
      help="Output ONNX file name (relative to --model-dir if not absolute).",
  )
  parser.add_argument(
      "--prompt",
      default="你好",
      help="Sample prompt used to create dummy input ids.",
  )
  parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
  parser.add_argument(
      "--constant-folding",
      action="store_true",
      help="Enable constant folding during export (default: disabled for stability/size).",
  )
  parser.add_argument(
      "--external-data",
      action="store_true",
      help="Allow exporting external tensor data files (default: single-file ONNX).",
  )
  return parser.parse_args()


class LogitsOnly(torch.nn.Module):
  def __init__(self, inner: torch.nn.Module):
    super().__init__()
    self.inner = inner

  def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
    outputs = self.inner(input_ids=input_ids, use_cache=False, return_dict=True)
    return outputs.logits


def export_with_torch_onnx_export(wrapper: torch.nn.Module, example_ids: torch.Tensor, output_path: Path,
                                  opset: int, constant_folding: bool) -> None:
  dynamic_axes = {
      "input_ids": {0: "batch", 1: "sequence"},
      "logits": {0: "batch", 1: "sequence"},
  }

  try:
    torch.onnx.export(
        wrapper,
        (example_ids,),
        output_path.as_posix(),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=constant_folding,
        dynamo=False,
    )
    return
  except TypeError:
    torch.onnx.export(
        wrapper,
        (example_ids,),
        output_path.as_posix(),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=constant_folding,
    )


def export_with_legacy_export(wrapper: torch.nn.Module, example_ids: torch.Tensor, output_path: Path, opset: int,
                              constant_folding: bool, external_data: bool) -> None:
  dynamic_axes = {
      "input_ids": {0: "batch", 1: "sequence"},
      "logits": {0: "batch", 1: "sequence"},
  }
  torch.onnx.utils._export(
      wrapper,
      (example_ids,),
      output_path.as_posix(),
      input_names=["input_ids"],
      output_names=["logits"],
      dynamic_axes=dynamic_axes,
      opset_version=opset,
      do_constant_folding=constant_folding,
      use_external_data_format=external_data,
  )


def main() -> None:
  args = parse_args()

  model_dir = Path(args.model_dir).resolve()
  output_path = Path(args.output)
  if not output_path.is_absolute():
    output_path = model_dir / output_path

  tokenizer = AutoTokenizer.from_pretrained(model_dir.as_posix(), local_files_only=True)
  model = AutoModelForCausalLM.from_pretrained(model_dir.as_posix(), local_files_only=True)
  model.eval()

  wrapper = LogitsOnly(model)
  example_ids = tokenizer.encode(args.prompt, return_tensors="pt")

  with torch.no_grad():
    try:
      export_with_legacy_export(
          wrapper,
          example_ids,
          output_path,
          args.opset,
          args.constant_folding,
          args.external_data,
      )
    except Exception as primary_error:
      print(f"[warn] torch.onnx.utils._export failed, fallback to torch.onnx.export: {primary_error}")
      export_with_torch_onnx_export(wrapper, example_ids, output_path, args.opset, args.constant_folding)

  size_mb = output_path.stat().st_size / (1024 * 1024)
  print(f"exported: {output_path}")
  print(f"size_mb: {size_mb:.2f}")


if __name__ == "__main__":
  main()
