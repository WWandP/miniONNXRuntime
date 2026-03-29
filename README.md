# miniONNXRuntime

一个面向 `yolov8n.onnx` 的 C++ mini runtime，用来说明 ONNX 模型如何被解析、调度并执行。

当前项目在 macOS 上验证更充分，依赖安装和运行路径也更贴近这一环境；其他平台后续会继续补齐验证。

这个项目聚焦推理 runtime 的主线：

- ONNX 模型加载与内部图建模
- 拓扑顺序执行
- `name -> Tensor` 运行时值表
- kernel 注册与调度
- shape / view / nn 类算子的最小语义实现
- 为后续 buffer reuse、ExecutionProvider 和 CUDA 演进保留接口

## Overview

当前版本已经形成一个可运行的小闭环：

- 读取 `models/yolov8n.onnx`
- 构建内部 `Graph / Node / Value / TensorInfo`
- 生成节点索引、算子直方图和拓扑顺序
- 通过 `Session`、`ExecutionContext`、`KernelRegistry` 组织一次推理
- 使用 CPU kernels 跑通 `yolov8n.onnx`
- 支持真实图片输入、最小检测后处理、结果 `json` 导出和可视化图片导出

## Scope

当前版本支持：

- 单模型：`models/yolov8n.onnx`
- 单 batch：`N=1`
- 数据类型：`float32`
- 单设备：CPU
- 执行方式：拓扑顺序执行
- shape 模式：以静态 shape 为主，允许少量运行时 shape 计算

当前版本暂不覆盖：

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
  inspect_model.cc         静态查看图结构
  session_trace.cc         查看执行主线和 value 流转
  run_model.cc             使用真实输入跑整图
  detect_yolov8n.cc        导出检测结果和可视化
```

## Execution Flow

当前执行主线如下：

1. `LoadOnnxGraph(...)` 读取 ONNX 模型并构建内部 `Graph`
2. `Session` 持有 `Graph` 和 `KernelRegistry`
3. `ExecutionContext` 在一次 `Run()` 中保存 `name -> Tensor`
4. 先装载 initializer，再绑定 feeds
5. 按拓扑顺序逐节点查找并执行 kernel
6. 将输出 tensor 回写到 `ExecutionContext`
7. 对检测结果做最小后处理与导出

当前没有单独引入 `ExecutionPlan`、生命周期分析或 arena。现阶段优先把值流转和算子语义做对。

## Build

依赖：

- CMake >= 3.20
- C++20 编译器
- `protobuf` 和 `protoc`

当前构建与运行流程主要在 macOS 上完成验证。

仓库内已经 vendoring 了当前需要的 ONNX proto 文件，构建时会本地生成 `onnx-ml.pb.cc/.h`。

```bash
cmake -S . -B build
cmake --build build -j4
```

如果本机没有 protobuf：

```bash
brew install protobuf
```

## Run

### `miniort_inspect`

查看静态图结构、initializer、属性和拓扑顺序。

```bash
./build/miniort_inspect models/yolov8n.onnx --show-topology 8 --show-initializers 5
```

参数说明：

- `--show-topology N`
  - 显示前 `N` 个拓扑序节点。
- `--show-initializers N`
  - 显示前 `N` 个 initializer。
- `--filter-op OpType`
  - 仅显示指定 `op_type` 的节点。

适合用于：

- 快速确认模型是否被正确解析
- 查看节点分布和算子类型统计
- 排查 initializer、attribute、拓扑顺序问题

### `miniort_session_trace`

查看 `Session::Run()` 的执行主线、kernel lookup、fallback 和 context 流转。

```bash
./build/miniort_session_trace models/yolov8n.onnx --max-nodes 16
```

参数说明：

- `--quiet`
  - 关闭逐节点 trace 输出。
- `--strict-kernel`
  - 遇到未注册算子时直接报错。
- `--context-dump-limit N`
  - 限制最终 `ExecutionContext` 的打印数量。
- `--max-nodes N`
  - 最多执行前 `N` 个节点，便于局部跟踪。

适合用于：

- 观察执行顺序和中间值流转
- 调试 kernel 注册和 fallback 路径
- 缩小范围查看某一段子图

### `miniort_run`

使用真实图片跑整图，并打印最终上下文摘要。

```bash
./build/miniort_run models/yolov8n.onnx --image pic/bus.jpg
```

参数说明：

- `<model.onnx>`
  - 目标模型路径。
- `--image path`
  - 输入图片路径。
- `--verbose`
  - 输出运行时 trace。
- `--strict-kernel`
  - 遇到未注册算子时直接报错。
- `--context-dump-limit N`
  - 控制最终上下文打印条目数。
- `--max-nodes N`
  - 只执行前 `N` 个节点，便于局部验证。

适合用于：

- 跑通完整前向流程
- 观察图片输入如何变成 tensor
- 检查输出上下文和执行摘要

### `miniort_detect_yolov8n`

完成 `yolov8n` 推理、后处理、`json` 导出和可视化输出。

```bash
./build/miniort_detect_yolov8n models/yolov8n.onnx --image pic/bus.jpg
```

参数说明：

- `--image path`
  - 输入图片路径，必填。
- `--save-vis out.png`
  - 保存可视化结果。未指定时默认写入 `outputs/<image_stem>_yolov8n.png`。
- `--dump-json out.json`
  - 保存检测结果 JSON。未指定时默认写入 `outputs/<image_stem>_yolov8n.json`。
- `--score-threshold 0.25`
  - 过滤低置信度检测框。
- `--iou-threshold 0.45`
  - 设置 NMS 的 IoU 阈值。
- `--verbose`
  - 输出运行时 trace。

适合用于：

- 查看完整检测结果
- 导出可视化图片和结构化结果
- 做模型输出的快速验证

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
  - 面向静态图分析
- `miniort_session_trace`
  - 面向执行链路追踪
- `miniort_run`
  - 面向真实输入的整图运行
- `miniort_detect_yolov8n`
  - 面向检测结果导出与可视化

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
