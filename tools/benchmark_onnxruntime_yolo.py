#!/usr/bin/env python3
import argparse
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

if hasattr(ort, "preload_dlls"):
    ort.preload_dlls(directory="")


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct / 100.0
    lo = int(np.floor(index))
    hi = int(np.ceil(index))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - index) + ordered[hi] * (index - lo)


def prepare_input(image_path, shape):
    height = int(shape[2]) if len(shape) > 2 and isinstance(shape[2], int) and shape[2] > 0 else 640
    width = int(shape[3]) if len(shape) > 3 and isinstance(shape[3], int) and shape[3] > 0 else 640
    image = Image.open(image_path).convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))[None, :, :, :].copy()


def main():
    parser = argparse.ArgumentParser(description="Benchmark YOLOv8n ONNX with Python ONNX Runtime.")
    parser.add_argument("model")
    parser.add_argument("--image", default="pic/bus.jpg")
    parser.add_argument("--provider", action="append", help="Provider order. Can be passed multiple times.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--disable-graph-opt", action="store_true")
    args = parser.parse_args()

    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if args.disable_graph_opt
        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    if args.threads > 0:
        options.intra_op_num_threads = args.threads

    providers = args.provider if args.provider else ["CPUExecutionProvider"]
    session = ort.InferenceSession(args.model, sess_options=options, providers=providers)
    input_meta = session.get_inputs()[0]
    output_names = [output.name for output in session.get_outputs()]
    input_tensor = prepare_input(args.image, input_meta.shape)
    feeds = {input_meta.name: input_tensor}

    for _ in range(args.warmup):
        session.run(output_names, feeds)

    samples = []
    for _ in range(args.repeat):
        start = time.perf_counter()
        session.run(output_names, feeds)
        samples.append((time.perf_counter() - start) * 1000.0)

    print("onnxruntime_benchmark")
    print(f"  version={ort.__version__}")
    print(f"  available_providers={ort.get_available_providers()}")
    print(f"  requested_providers={providers}")
    print(f"  active_providers={session.get_providers()}")
    print(f"  input_name={input_meta.name}")
    print(f"  input_shape={list(input_tensor.shape)}")
    print(f"  warmup={args.warmup}")
    print(f"  repeat={args.repeat}")
    print(f"  threads={args.threads}")
    print(f"  graph_optimization={'disabled' if args.disable_graph_opt else 'all'}")
    print(
        "  latency_ms "
        f"mean={float(np.mean(samples)):.3f} "
        f"min={min(samples):.3f} "
        f"p50={percentile(samples, 50):.3f} "
        f"p95={percentile(samples, 95):.3f} "
        f"max={max(samples):.3f}"
    )


if __name__ == "__main__":
    main()
