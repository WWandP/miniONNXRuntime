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
  - CMake looks it up through `find_package(Protobuf CONFIG REQUIRED)`

The repository already includes `third_party/onnx`, so no extra ONNX source download is needed.

## What It Shows

- Parse ONNX graph
- Optimize graph structure
- Run CPU kernels
- Trace tensor memory and buffer reuse

## Current Status

- ONNX model parsing and internal graph construction
- `Session` / `ExecutionContext` / `KernelRegistry`
- CPU-side basic kernels
- Real image input and YOLO detection output
- Graph optimization entry and first optimization passes
- Memory observation, initializer materialization on demand, and buffer reuse demo

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
  - There is no separate `phase5`; memory optimization is part of `phase4`

## Main Tools

- `miniort_inspect`: inspect graph structure only
- `miniort_session_trace`: trace execution and value flow
- `miniort_run`: run end-to-end inference
- `miniort_memory_trace`: trace memory usage and tensor lifetime
- `miniort_detect_yolov8n`: export detection results
- `miniort_optimize_model`: optimize the graph before running YOLO
