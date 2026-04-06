# miniONNXRuntime

A teaching-oriented mini implementation of ONNX Runtime.
It uses `yolov8n.onnx` to show how a model is parsed, optimized, executed, and memory-optimized.

![miniONNXRuntime showcase](./assets/readme_showcase.png)

Chinese version: [README.md](./README.md)

## Environment

Build requirements:

- CMake 3.20+
- A C++20-capable compiler
- Protobuf
  - `protoc` is required
  - CMake first tries `find_package(Protobuf CONFIG QUIET)` and then falls back to CMake's built-in `FindProtobuf`

The repository already includes `third_party/onnx`, so no extra ONNX source download is needed.

### Ubuntu / Debian

On Linux, install the basic packages first:

```bash
sudo apt update
sudo apt install -y build-essential cmake git libprotobuf-dev protobuf-compiler
```

If you already have `cmake` and only want the rest of the dependencies:

```bash
sudo apt install -y build-essential git libprotobuf-dev protobuf-compiler
```

If you only need to install `cmake`:

```bash
sudo apt install -y cmake
```

### macOS

On macOS, install the Homebrew dependencies:

```bash
brew install cmake protobuf git
```

If you already have `cmake`, you can install only the remaining packages:

```bash
brew install protobuf git
```

## What It Shows

- Parse ONNX graph
- Optimize graph structure
- Run CPU / Apple provider kernels
- Trace tensor memory and buffer reuse

## Current Status

- ONNX model parsing and internal graph construction
- `Session` / `ExecutionContext` / `KernelRegistry`
- Minimal `ExecutionProvider`
- `CpuExecutionProvider`
- Minimal `AccelerateExecutionProvider` on macOS
- Provider assignment, allocator injection, and runtime provider tracing
- CPU-side basic kernels
- Real image input and YOLO detection output
- Graph optimization entry and first optimization passes
- Memory observation, initializer materialization on demand, and buffer reuse demo

The memory optimization write-up is in [docs/blog_memory_optimization.md](./docs/blog_memory_optimization.md).
The phase5 EP design note is in [docs/execution_provider.md](./docs/execution_provider.md).
The phase5 summary post is in [docs/phase5_execution_provider_summary.md](./docs/phase5_execution_provider_summary.md).

## Quick Start

```bash
cmake -S . -B build_phase3 -DMINIORT_BUILD_OPTIMIZER_TOOLS=OFF
cmake --build build_phase3 -j4
./build_phase3/miniort_run models/yolov8n.onnx --image pic/bus.jpg

cmake -S . -B build_phase4 -DMINIORT_BUILD_OPTIMIZER_TOOLS=ON
cmake --build build_phase4 -j4
./build_phase4/miniort_optimize_model models/yolov8n.onnx --image pic/bus.jpg
```

## Phases

- `phase1`: inspect graph structure only
- `phase2`: see the minimal execution pipeline
- `phase3`: run CPU inference end to end
- `phase4`: graph optimization and memory optimization
- `phase5`: move the runtime backend path into a minimal `ExecutionProvider`
  - `CpuExecutionProvider` and `CpuTensorAllocator` are in place
  - On macOS the runtime prefers the minimal `AccelerateExecutionProvider`
  - The current `AccelerateExecutionProvider` covers part of the elementwise / matmul / conv path
  - Multi-EP partitioning is not implemented yet

## Main Tools

- `miniort_inspect`: inspect graph structure only
- `miniort_session_trace`: trace execution and value flow
- `miniort_run`: run end-to-end inference
- `miniort_memory_trace`: trace memory usage and tensor lifetime
- `miniort_detect_yolov8n`: export detection results
- `miniort_optimize_model`: optimize the graph before running YOLO
- `miniort_compare_providers`: compare the default provider path against CPU-only
