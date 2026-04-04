# miniONNXRuntime

一个围绕 `yolov8n.onnx` 的 C++ mini runtime，用来说明 ONNX 模型如何被加载、优化和执行。

![miniONNXRuntime showcase](./assets/readme_showcase.png)

展示页源文件在 [assets/readme_showcase.html](./assets/readme_showcase.html)。

```mermaid
flowchart LR
  A[ONNX Model] --> B[LoadOnnxGraph]
  B --> C[Graph]
  C --> D[Session]
  D --> E[ExecutionContext]
  E --> F[CPU Kernels]
  F --> G[YOLO Postprocess]
  G --> H[json / png]
```

## Current Status

当前已经完成：

- ONNX 模型解析与内部图构建
- `Session` / `ExecutionContext` / `KernelRegistry`
- CPU 侧基础 kernels
- 真实图片输入和 YOLO 检测输出
- 执行 profiling
- phase4 图优化入口

当前还会继续做：

- `ShapeSimplification`
- 更完整的图优化 pass
- `ExecutionProvider` 抽象
- buffer reuse / 内存优化
- 更通用的模型支持

图优化的整理版说明在 [docs/blog_graph_optimization.md](./docs/blog_graph_optimization.md)，实验记录保留在 [GRAPH_OPTIMIZATION.md](./GRAPH_OPTIMIZATION.md)。  
内存优化的基线和后续规划记录在 [docs/memory_optimization.md](./docs/memory_optimization.md)。

## Quick Start

```bash
cmake -S . -B build_phase3 -DMINIORT_BUILD_OPTIMIZER_TOOLS=OFF
cmake --build build_phase3 -j4
./build_phase3/miniort_run models/yolov8n.onnx --image pic/bus.jpg

cmake -S . -B build_phase4 -DMINIORT_BUILD_OPTIMIZER_TOOLS=ON
cmake --build build_phase4 -j4
./build_phase4/miniort_optimize_model models/yolov8n.onnx --image pic/bus.jpg
```

## Phase Overview

- `phase1`
  - 纯解析
  - 主要是模型解析、图结构和属性检查
- `phase2`
  - 加入 `Session`
  - 形成最小执行主线
- `phase3`
  - 加入各个算子的 CPU 实现
  - 跑通 `yolov8n.onnx` 推理和检测输出
- `phase4`
  - 在 phase3 基础上增加图优化入口和内存演示
  - 当前已经接入第一版 `ConstantFolding` / `DeadNodeCleanup`

## Repository Layout

```text
include/miniort/
  loader/          ONNX loader 对外接口
  model/           Graph / Node / Value / TensorInfo
  runtime/         Session / Tensor / ExecutionContext / KernelRegistry
  optimizer/       图优化入口
  tools/           图像与 YOLO 后处理工具接口

src/
  loader/          ONNX protobuf -> 内部 Graph
  runtime/         执行器与 builtin kernels
  optimizer/       图优化实现
  tools/           输入预处理与 YOLO 后处理

tools/
  miniort_inspect        静态查看图结构
  miniort_session_trace  查看执行主线和 value 流转
  miniort_memory_trace   查看执行过程中的内存占用与张量生命周期
  miniort_run            使用真实输入跑整图
  miniort_detect_yolov8n 导出检测结果和可视化
  miniort_optimize_model  优化图后再跑 YOLO
```

## Tool Arguments

### `miniort_inspect`

```bash
./build_phase1/miniort_inspect models/yolov8n.onnx
```

适合 `phase1`：纯解析、看图结构。

### `miniort_session_trace`

```bash
./build_phase3/miniort_session_trace models/yolov8n.onnx
```

适合 `phase2`：看执行主线和中间值流转。

### `miniort_run`

```bash
./build_phase3/miniort_run models/yolov8n.onnx --image pic/bus.jpg
```

适合 `phase3`：跑完整推理主线。

### `miniort_memory_trace`

```bash
./build_phase4/miniort_memory_trace models/yolov8n.onnx --image pic/bus.jpg
```

适合 `phase4`：看 tensor 生命周期、峰值和 buffer reuse。

### `miniort_detect_yolov8n`

```bash
./build_phase3/miniort_detect_yolov8n models/yolov8n.onnx --image pic/bus.jpg
```

适合 `phase3`：看最终检测结果和性能基线。

### `miniort_optimize_model`

```bash
./build_phase4/miniort_optimize_model models/yolov8n.onnx --image pic/bus.jpg
```

适合 `phase4`：先优化图，再跑同一套 YOLO 后处理。
