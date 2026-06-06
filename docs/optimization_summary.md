# 2026-06-06 CUDA 相关优化

This page explains the CUDA-related MiniORT optimizations completed around
2026-06-06. It is written for repository readers who want to understand what was
optimized, what results to expect, and how to reproduce the main paths. Longer
experiment logs are kept separately as working notes and are not required for
normal use.

## What Changed

MiniORT is a teaching-oriented ONNX runtime. The optimization work focuses on
three representative model paths:

| Model path | What it demonstrates |
| --- | --- |
| YOLOv8n mixed CUDA | ExecutionProvider assignment, CUDA tensor residency, memory reuse, and vision-model operator coverage |
| GPT-2 KV-cache CUDA | Prefill/decode ONNX graphs, token generation loop, KV-cache reuse, and warmed benchmarking |
| Qwen2.5-0.5B KV-cache CUDA | Larger text-model execution, CUDA initializer preparation, CPU/CUDA boundary reduction, and repeated decode optimization |

## Results

These numbers were measured locally on an NVIDIA GeForce RTX 4090. The ORT CPU
reference uses ONNX Runtime 1.24.4 with `CPUExecutionProvider`; the ORT CUDA
reference uses ONNX Runtime 1.23.2 with `CUDAExecutionProvider`. They are useful
as a project result reference, not as a universal benchmark claim.

| Path | MiniORT CUDA | ORT CUDA reference | ORT CPU reference |
| --- | --- | --- | --- |
| YOLOv8n mixed CUDA | `repeat=50` mean `4.96 ms`, p50 `4.99 ms` | mean `1.86 ms`, p50 `1.86 ms` | mean `12.87 ms`, p50 `12.84 ms` |
| GPT-2 KV CUDA | `generate=96` about `227.40 tokens/s` (generation mean `422.16 ms`, prefill mean `11.13 ms`) | about `438.65 tokens/s` (generation mean `218.85 ms`, prefill mean `1.80 ms`) | about `83.46 tokens/s` (generation mean `1150.24 ms`, prefill mean `16.54 ms`) |
| Qwen2.5-0.5B KV CUDA | `generate=8` about `61.65 tokens/s` (generation mean `129.77 ms`, prefill mean `15.16 ms`) | about `171.36 tokens/s` (generation mean `46.68 ms`, prefill mean `4.75 ms`) | about `15.45 tokens/s` (generation mean `517.85 ms`, prefill mean `52.98 ms`) |

## Optimization Themes

### 1. Keep Data on the Device

The early CUDA path copied tensors between CPU and GPU inside many individual
operators. The current path lets runtime tensors carry CUDA buffers, so later
CUDA operators can consume device-resident values directly.

This matters most when a graph has many adjacent CUDA-capable operators. It
also makes mixed CPU/CUDA boundaries explicit and easier to diagnose.

### 2. Reuse CUDA Buffers

Repeated `cudaMalloc` / `cudaFree` calls were a major overhead in the original
mixed CUDA path. MiniORT now keeps a simple CUDA buffer pool and reuses device
allocations across operators.

For YOLOv8n, this was one of the first changes that made the CUDA path behave
like a runtime instead of a sequence of isolated kernel demos.

### 3. Move One-Time CUDA Costs Out of the Hot Path

For Qwen, the first measured prefill originally included lazy CUDA initializer
upload and CUDA/cuBLAS setup. MiniORT now has an explicit CUDA initializer
prepare step and a tiny CUDA/cuBLAS warmup.

This does not remove the cold startup cost. It makes the timing boundary clear:
model/session preparation is separate from warm generation.

### 4. Reduce CPU/CUDA Boundary Churn

Some exported text-model patterns are made of small primitive ONNX ops. For
Qwen RMSNorm-like chains, MiniORT added narrow CUDA coverage for primitives such
as `Sqrt`, `Pow(x, 2)`, and last-dimension `ReduceMean`.

For Qwen broadcast-heavy paths, MiniORT also added narrow tail-dimension CUDA
broadcast support for `Add` and `Mul`, avoiding unnecessary CPU fallback for
common hidden-size vector patterns.

### 5. Avoid Repeating Decode Constants

Decode runs execute the same graph many times. Recomputing thousands of
`Constant` nodes each step became visible in longer Qwen generation traces.

MiniORT now reuses existing Constant outputs only when they are safe for the
current graph. The graph-scoped check avoids accidentally reusing same-named
intermediate tensors between prefill and decode graphs.

## Reproduce the Main Paths

Prepare local assets:

```bash
./scripts/download_models.sh status
./scripts/download_models.sh yolo
./scripts/download_models.sh gpt2
./scripts/download_models.sh qwen
```

Build CPU/default tools:

```bash
cmake -S . -B build_local -DMINIORT_BUILD_OPTIMIZER_TOOLS=ON
cmake --build build_local -j4
```

Build CUDA tools:

```bash
cmake -S . -B build_cuda_release \
  -DCMAKE_BUILD_TYPE=Release \
  -DMINIORT_BUILD_CUDA_EP=ON \
  -DMINIORT_BUILD_OPTIMIZER_TOOLS=ON
cmake --build build_cuda_release -j$(nproc)
```

Run examples:

```bash
# YOLOv8n provider comparison
./build_cuda_release/miniort_compare_providers \
  models/yolov8n.onnx \
  --image pic/bus.jpg \
  --warmup 3 \
  --repeat 20 \
  --planned-memory-reuse

# GPT-2 KV-cache benchmark
./build_cuda_release/miniort_benchmark_gpt2_kv \
  --generate 96 \
  --warmup 1 \
  --repeat 3 \
  --graph-opt

# Qwen KV-cache benchmark
./build_cuda_release/miniort_benchmark_qwen_kv \
  --generate 8 \
  --warmup 1 \
  --repeat 3 \
  --shared-context
```

Run ORT CPU references:

```bash
# YOLOv8n ORT CPU
python3 tools/benchmark_onnxruntime_yolo.py \
  models/yolov8n.onnx \
  --image pic/bus.jpg \
  --warmup 5 \
  --repeat 50

# GPT-2 KV-cache ORT CPU
python3 tools/benchmark_onnxruntime_gpt2_kv.py \
  --generate 96 \
  --warmup 1 \
  --repeat 3

# Qwen KV-cache ORT CPU
python3 tools/benchmark_onnxruntime_gpt2_kv.py \
  --prefill-model models/qwen2_5_0_5b_instruct/model.kv_prefill.onnx \
  --decode-model models/qwen2_5_0_5b_instruct/model.kv_decode.onnx \
  --tokens 108386 \
  --generate 8 \
  --warmup 1 \
  --repeat 3
```

Run ORT CUDA references:

```bash
PY=/path/to/python-with-onnxruntime-gpu

# YOLOv8n ORT CUDA
$PY tools/benchmark_onnxruntime_yolo.py \
  models/yolov8n.onnx \
  --image pic/bus.jpg \
  --provider CUDAExecutionProvider \
  --provider CPUExecutionProvider \
  --warmup 5 \
  --repeat 50

# GPT-2 KV-cache ORT CUDA
$PY tools/benchmark_onnxruntime_gpt2_kv.py \
  --generate 96 \
  --warmup 1 \
  --repeat 3 \
  --provider CUDAExecutionProvider \
  --provider CPUExecutionProvider

# Qwen KV-cache ORT CUDA
$PY tools/benchmark_onnxruntime_gpt2_kv.py \
  --prefill-model models/qwen2_5_0_5b_instruct/model.kv_prefill.onnx \
  --decode-model models/qwen2_5_0_5b_instruct/model.kv_decode.onnx \
  --tokens 108386 \
  --generate 8 \
  --warmup 1 \
  --repeat 3 \
  --provider CUDAExecutionProvider \
  --provider CPUExecutionProvider
```

## Interpretation Boundary

Good wording:

```text
MiniORT demonstrates core runtime optimization ideas on selected ONNX workloads:
provider assignment, device residency, memory reuse, graph optimization, and
KV-cache text generation.
```

Avoid overclaiming:

```text
MiniORT is faster than ONNX Runtime in general.
```

The current results are workload-specific, local-machine measurements. They are
best read as an educational runtime optimization story rather than a broad
production benchmark.
