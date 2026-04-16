#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Iterable, List


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Quantize existing Qwen ONNX models to INT8 (dynamic quantization, CPU-friendly).")
  parser.add_argument(
      "--model-dir",
      default="models/qwen2_5_0_5b_instruct",
      help="Directory containing ONNX models.",
  )
  parser.add_argument(
      "--input",
      action="append",
      dest="inputs",
      default=[],
      help="Input ONNX file (absolute or relative to --model-dir). Repeatable. "
           "If omitted, script auto-detects model.baseline/model.kv_prefill/model.kv_decode.",
  )
  parser.add_argument(
      "--suffix",
      default=".int8.onnx",
      help="Output filename suffix appended after the input stem. Example: model.onnx -> model.int8.onnx",
  )
  parser.add_argument(
      "--per-channel",
      action="store_true",
      help="Enable per-channel weight quantization where supported.",
  )
  parser.add_argument(
      "--weight-type",
      choices=["qint8", "quint8"],
      default="qint8",
      help="Weight quantization type.",
  )
  parser.add_argument(
      "--ops",
      default="MatMul,Gemm",
      help="Comma-separated op types to quantize (default: MatMul,Gemm).",
  )
  parser.add_argument(
      "--force",
      action="store_true",
      help="Overwrite existing output files.",
  )
  return parser.parse_args()


def resolve_inputs(model_dir: Path, raw_inputs: List[str]) -> List[Path]:
  if raw_inputs:
    paths: List[Path] = []
    for raw in raw_inputs:
      p = Path(raw)
      if not p.is_absolute():
        p = model_dir / p
      paths.append(p.resolve())
    return paths

  defaults = [
      model_dir / "model.baseline.onnx",
      model_dir / "model.kv_prefill.onnx",
      model_dir / "model.kv_decode.onnx",
  ]
  return [p.resolve() for p in defaults if p.exists()]


def make_output_path(input_path: Path, suffix: str) -> Path:
  if not suffix:
    raise RuntimeError("--suffix must not be empty")
  stem = input_path.name[:-5] if input_path.name.endswith(".onnx") else input_path.name
  return input_path.with_name(f"{stem}{suffix}")


def to_weight_type(name: str):
  try:
    from onnxruntime.quantization import QuantType  # type: ignore
  except Exception as ex:
    raise RuntimeError("onnxruntime.quantization is required; install onnxruntime in your Python env") from ex
  if name == "qint8":
    return QuantType.QInt8
  if name == "quint8":
    return QuantType.QUInt8
  raise RuntimeError(f"unsupported weight type: {name}")


def parse_op_types(raw: str) -> List[str]:
  ops = [item.strip() for item in raw.split(",") if item.strip()]
  if not ops:
    raise RuntimeError("--ops produced an empty op list")
  return ops


def quantize_one(input_path: Path, output_path: Path, op_types: Iterable[str], per_channel: bool, weight_type) -> None:
  try:
    from onnxruntime.quantization import quantize_dynamic  # type: ignore
  except Exception as ex:
    raise RuntimeError("onnxruntime.quantization is required; install onnxruntime in your Python env") from ex

  quantize_dynamic(
      model_input=input_path.as_posix(),
      model_output=output_path.as_posix(),
      op_types_to_quantize=list(op_types),
      per_channel=per_channel,
      weight_type=weight_type,
  )


def format_size_mb(path: Path) -> float:
  return path.stat().st_size / (1024 * 1024)


def main() -> None:
  args = parse_args()
  model_dir = Path(args.model_dir).resolve()
  inputs = resolve_inputs(model_dir, args.inputs)
  if not inputs:
    raise RuntimeError("no input ONNX files found; pass --input or place default ONNX files in --model-dir")

  op_types = parse_op_types(args.ops)
  weight_type = to_weight_type(args.weight_type)

  print(f"model_dir: {model_dir}")
  print(f"ops_to_quantize: {','.join(op_types)}")
  print(f"per_channel: {args.per_channel}")
  print(f"weight_type: {args.weight_type}")
  print("")

  for input_path in inputs:
    if not input_path.exists():
      raise RuntimeError(f"input not found: {input_path}")
    if input_path.suffix != ".onnx":
      raise RuntimeError(f"input is not an ONNX file: {input_path}")

    output_path = make_output_path(input_path, args.suffix)
    if output_path.exists() and not args.force:
      raise RuntimeError(f"output exists (use --force to overwrite): {output_path}")

    print(f"[quantize] {input_path.name}")
    quantize_one(input_path, output_path, op_types, args.per_channel, weight_type)

    src_size = format_size_mb(input_path)
    dst_size = format_size_mb(output_path)
    ratio = (dst_size / src_size * 100.0) if src_size > 0.0 else 0.0
    print(f"  output: {output_path}")
    print(f"  size_mb: {src_size:.2f} -> {dst_size:.2f} ({ratio:.1f}%)")
    print("")


if __name__ == "__main__":
  main()
