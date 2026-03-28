# miniONNXRuntime

一个面向 `yolov8n.onnx` 的教学型、实验型推理 runtime。

这个项目的目标不是复刻 ONNX Runtime，也不是追求通用 ONNX 兼容或高性能，而是用一个足够小、但能跑真实模型的 C++ 实现，把推理 runtime 的几条核心主线做清楚：

- ONNX 模型加载与内部 IR 建模
- 拓扑顺序执行
- `name -> Tensor` 运行时值表
- kernel 注册与调度
- shape / view / nn 类算子的最小语义实现
- 后续面向 buffer reuse、Execution Provider 和 CUDA 的演进基础

## Current Status

当前版本已经完成一个可运行的小闭环：

- 可读取 `yolov8n.onnx`
- 可构建最小内部图结构 `Graph / Node / Value / TensorInfo`
- 可建立节点索引、算子直方图和拓扑顺序
- 已实现最小执行器：
  - `Session`
  - `ExecutionContext`
  - `KernelRegistry`
- 已用朴素 CPU kernels 跑通 `yolov8n.onnx`
- 已支持真实图片输入、最小检测后处理、结果 `json` 导出和可视化图片导出

当前项目处于“主线已跑通，后续继续补工程能力和扩展能力”的阶段。

## Scope

当前版本明确支持：

- 单模型：`models/yolov8n.onnx`
- 单 batch：`N=1`
- 数据类型：`float32`
- 单设备：CPU
- 执行方式：拓扑顺序执行
- shape 模式：以静态 shape 为主，允许少量运行时 shape 计算

当前版本明确不做：

- 通用 ONNX 兼容
- 多模型泛化
- 多线程 / SIMD / GEMM 优化
- 内存规划 / buffer reuse
- 多 `ExecutionProvider`
- CUDA 路径

## Repository Layout

```text
include/miniort/
  loader/          ONNX loader 对外接口
  model/           Graph / Node / Value / TensorInfo
  runtime/         Session / Tensor / ExecutionContext / KernelRegistry
  tools/           图像加载等工具接口

src/
  loader/          ONNX protobuf -> 内部 Graph
  model/           图结构辅助逻辑
  runtime/         执行器与 builtin kernels
  tools/           输入预处理

tools/
  inspect_model.cc         静态看图
  session_trace.cc         看执行主线和 value 流转
  run_model.cc             带真实输入跑整图
  detect_yolov8n.cc        导出检测结果和可视化

docs/
  design_summary.md
  architecture.md
  dev_log.md
```

## Execution Path

当前执行主线可以概括成：

1. `LoadOnnxGraph(...)` 读取 ONNX 模型并构建内部 `Graph`
2. `Session` 持有 `Graph` 和 `KernelRegistry`
3. `ExecutionContext` 在一次 `Run()` 中保存 `name -> Tensor`
4. 先装载 initializer，再绑定 feeds
5. 按拓扑顺序逐节点查找并执行 kernel
6. 把输出 tensor 回写到 `ExecutionContext`
7. 对检测结果做最小后处理与导出

当前没有做显式 `ExecutionPlan`、生命周期分析或 arena。现阶段优先把值流转和算子语义做对。

## Build

依赖：

- CMake >= 3.20
- C++20 编译器
- 本机 `protobuf` 和 `protoc`

仓库内已 vendoring 当前需要的 ONNX proto 文件，构建时会本地生成 `onnx-ml.pb.cc/.h`。

```bash
cmake -S . -B build
cmake --build build -j4
```

如果本机没有 protobuf：

```bash
brew install protobuf
```

## Run

```bash
./build/miniort_inspect models/yolov8n.onnx --show-topology 8 --show-initializers 5
./build/miniort_session_trace models/yolov8n.onnx --max-nodes 16
./build/miniort_run models/yolov8n.onnx --image pic/bus.jpg
./build/miniort_detect_yolov8n models/yolov8n.onnx --image pic/bus.jpg
```

默认配置会编译全部工具。也可以按阶段裁剪：

```bash
cmake -S . -B build_phase1 \
  -DMINIORT_BUILD_INSPECT=ON \
  -DMINIORT_BUILD_RUNTIME_TOOLS=OFF \
  -DMINIORT_BUILD_DETECT_TOOL=OFF

cmake --build build_phase1 --target miniort_inspect
```

## Tooling

- `miniort_inspect`
  - 静态观察图结构、initializer、attribute 和拓扑顺序
- `miniort_session_trace`
  - 观察 `Session::Run()` 主线、kernel lookup、fallback 和 context 流转
- `miniort_run`
  - 加载真实图片，生成 `[1, 3, 640, 640]` 输入 tensor 并跑整图
- `miniort_detect_yolov8n`
  - 导出检测结果、`json` 和可视化图片

## Current Tradeoffs

这个项目当前刻意接受几个取舍：

- 先围绕单一真实模型裁剪，不先抽象通用框架
- 先做可解释、可调试的朴素实现，不先做性能优化
- 先用 `name -> Tensor` 的上下文把值流看清楚，不提前做复杂内存规划
- 当前 fallback / placeholder 机制主要服务调试，不代表最终正确性验证方案

## Roadmap

后续更值得推进的方向：

- 补单元测试和端到端回归测试
- 引入 value 生命周期分析
- 做 buffer reuse / 简单内存规划
- 抽象显式 `ExecutionProvider`
- 补更通用的模型子图支持，优先考虑 GPT-2
- 探索 CUDA 路径
- 再考虑 kernel 性能优化

## Docs

- [设计摘要](./docs/design_summary.md)
- [架构设计](./docs/architecture.md)
- [开发日志](./docs/dev_log.md)
