# Architecture

## 1. 项目目标

`miniONNXRuntime` 当前不是一个追求通用 ONNX 兼容的框架，而是一个围绕 `yolov8n.onnx` 裁剪出来的教学型 runtime。

这一阶段的核心目标是：

- 用真实模型倒逼 runtime 的最小模块分层
- 把图加载、值流转、kernel 调度和执行主线做清楚
- 为后续 buffer reuse、Execution Provider、GPT-2 和 CUDA 留出演进空间

当前版本首先要做到的是：

- 一个真实可跑的 CPU mini runtime

而不是：

- 一个“接口看起来很完整”但没有跑通模型的抽象骨架

## 2. 第一版边界

当前版本明确支持：

- 单模型：`yolov8n.onnx`
- 单 batch：`N=1`
- 数据类型：`float32`
- 单设备：CPU
- 执行方式：拓扑顺序执行
- shape 模式：静态 shape 为主，少量运行时 shape 计算
- kernel 风格：朴素 C++ 实现，优先保证语义正确和可读性

当前版本明确不做：

- 通用 ONNX 兼容
- 多 `ExecutionProvider`
- 训练和控制流
- 动态 batch 泛化
- 通用 arena / memory planner
- 高性能 kernel 优化
- 通用图优化框架

## 3. 设计原则

这一版遵循几个明确的取舍：

1. 先跑通真实模型，再抽象公共层
2. 先把图执行和张量语义做对，再考虑性能
3. 先围绕单后端、单路径收敛，再抽象多后端
4. 先保留可观察性和可解释性，再做内存与性能优化

## 4. 模块分层

```text
miniONNXRuntime
|
+-- loader
|   +-- onnx_loader
|
+-- model
|   +-- TensorInfo
|   +-- Value
|   +-- Node
|   +-- Graph
|
+-- runtime
|   +-- Tensor
|   +-- ExecutionContext
|   +-- TensorAllocator
|   +-- KernelRegistry
|   +-- ExecutionProvider
|   +-- AccelerateExecutionProvider
|   +-- CpuExecutionProvider
|   +-- CpuTensorAllocator
|   +-- Session
|
+-- kernels
|   +-- basic_kernels
|   +-- elementwise_kernels
|   +-- shape_kernels
|   +-- nn_kernels
|
+-- tools
    +-- inspect_model
    +-- session_trace
    +-- run_model
    +-- detect_yolov8n
```

### loader

职责：

- 读取 `onnx::ModelProto`
- 提取 graph input / output / initializer / node / attribute
- 转成项目自己的最小内部 `Graph`
- 建立拓扑顺序和基础索引

这一层不负责执行，也不负责做复杂优化。

### model

职责：

- 表达最小图结构
- 保存 shape / dtype / 常量数据
- 支持节点索引、算子直方图和拓扑序

核心对象：

- `TensorInfo`
- `TensorData`
- `AttributeValue`
- `Value`
- `Node`
- `Graph`

### runtime

职责：

- 组织一次模型运行
- 管理运行时值表
- 通过 provider 装配 kernel 和 allocator
- 在当前阶段按拓扑顺序顺序执行

核心对象：

- `Tensor`
  - 运行时张量表示
- `ExecutionContext`
  - 当前 `Run()` 的 `name -> Tensor` 值表
- `TensorAllocator`
  - 运行期 tensor buffer 申请与回收接口
- `KernelRegistry`
  - `op_type -> kernel` 查找表
- `ExecutionProvider`
  - backend capability 的最小装配入口
- `AccelerateExecutionProvider`
  - Apple 平台上的最小加速 provider
- `CpuExecutionProvider`
  - 默认 CPU backend / fallback backend
- `Session`
  - 持有 `Graph`、`KernelRegistry`、providers 和运行选项

当前还没有单独引入：

- `ExecutionPlan`
- 多 EP partition
- provider-aware graph assignment
- 通用 arena / memory planner

这是刻意的。当前虽然已经引入了最小 `ExecutionProvider`，但仍然优先保持单一 CPU 路径闭环，不提前扩成空抽象。

### kernels

职责：

- 以最小实现表达 ONNX op 的执行语义

当前按语义拆分：

- `basic_kernels`
  - `Constant`
- `elementwise_kernels`
  - `Sigmoid` `Add` `Mul` `Div` `Sub` `Cast`
- `shape_kernels`
  - `Shape` `Gather` `Unsqueeze` `Concat` `Reshape`
  - `Range` `Split` `Expand` `Transpose` `Slice`
  - `ConstantOfShape` `ReduceMax` `ArgMax` `Softmax`
- `nn_kernels`
  - `Conv` `MaxPool` `Resize`

这些实现当前都偏朴素，目标是先跑通、先讲清楚。

## 5. 执行主线

当前主线如下：

1. `LoadOnnxGraph(...)` 读取 ONNX 模型
2. ONNX graph 被转换成内部 `Graph`
3. `Session` 按平台挂载默认 providers
   - Apple 平台：`Accelerate -> CPU`
   - 其他平台：`CPU`
4. providers 按顺序向 `KernelRegistry` 注册 kernels，已匹配 `op_type` 不再被后续 provider 覆盖
5. `Run()` 创建或接收一个 `ExecutionContext`
6. 若 context 尚未携带 allocator，则由 provider 提供默认 allocator
7. initializer 先加载进上下文
8. 运行时 feeds 绑定进上下文
9. 未提供的输入可按选项自动绑定 placeholder tensor
10. 按 `topological_order` 顺序执行每个 node
11. kernel 通过 `ExecutionContext` 申请输出 buffer，并把输出回写到 context
12. tensor 被覆盖或释放时，其底层 storage 可回收到 allocator
13. 工具层再读取最终输出做展示或后处理

当前执行模型的关键特点：

- 值流是显式的 `name -> Tensor`
- 所有中间结果当前仍主要常驻上下文
- 已有一层轻量 buffer 复用，但还不是 planner 驱动的 arena
- 节点已有最小 provider assignment
- 当前 fallback / placeholder 机制主要用于 trace 和教学调试

### 当前 EP 边界

当前 `ExecutionProvider` 只承载最小职责：

- 暴露 provider 名称
- 注册 kernels
- 提供默认 allocator
- 参与最小 provider assignment

当前还没有承载：

- graph partition
- 节点归属
- stream / async execution
- 编译后子图执行接口

这样做是为了先把单后端 CPU 路径的职责搬对，而不是过早做一层很空的“多后端接口”。

## 6. 当前已完成能力

当前版本已经完成：

- `yolov8n.onnx` 的最小 loader
- 图结构与元信息建模
- 拓扑顺序执行器
- 最小 `ExecutionProvider`
- `CpuExecutionProvider`
- Apple 平台上的最小 `AccelerateExecutionProvider`
- 围绕 `yolov8n` 所需算子的 CPU kernel 子集
- 图片预处理：
  - 读取图片
  - resize
  - `HWC -> NCHW`
  - `uint8 -> float32`
- 最小检测后处理
- `json` 与可视化图片导出

可以把当前项目理解成：

- 已经跑通的专用 mini runtime v0

而不是：

- 已经完成泛化能力和工程收尾的最终版本

## 7. 当前最重要的工程取舍

### 为什么当前只做最小 allocator，而不直接上 arena / planner

因为当前阶段更重要的是先把：

- producer / consumer 关系
- 张量 shape 语义
- kernel 输入输出约定
- 值在图中的流动方式

看清楚。没有这些前置，内存优化只会把系统复杂度提前拉高。

因此当前只先做到：

- 把 allocator 的归属放到 `ExecutionProvider`
- 把简单 buffer pool 从 `ExecutionContext` 内部实现提升为 backend capability

后面再决定是否把静态生命周期分析结果显式喂给 runtime。

### 为什么先做 `CpuExecutionProvider + AccelerateExecutionProvider`

因为当前最需要的不是一下子做复杂多后端执行，而是先验证：

- provider 装配是否合理
- provider assignment 是否真的能影响节点归属
- provider 优先级和 fallback 是否好用

更合理的顺序是：

1. 先把单后端 runtime 主线稳定下来
2. 再引入最小 CPU EP 抽象
3. 再挂上一条成本较低的第二 provider 路径验证自动选择
4. 最后再扩展多 EP、CUDA、Metal 或 CoreML

### 为什么允许面向 `yolov8n` 的裁剪

因为这个项目当前追求的是：

- 用真实模型逼出 runtime 核心模块

不是：

- 一开始就实现一个范围失控的通用框架

只要裁剪仍然保留图执行主线和张量语义，就仍然有很高的学习价值。

## 8. 后续演进方向

后续版本更值得投入的方向：

- 为 loader / kernel / runtime 补测试
- 做 value 生命周期分析
- 让 memory profile 与 runtime allocator 更好对齐
- 为图或节点补最小 provider 归属信息
- 在 CPU 路径稳定后探索 CUDA
- 从 `yolov8n` 扩到 GPT-2 一类新模型子图
- 再考虑更系统的 correctness/performance 对比

推荐的演进顺序：

1. 测试与文档收尾
2. CPU `ExecutionProvider` 与 allocator 收口
3. 生命周期分析与更明确的 memory planning
4. GPT-2 子图支持
5. CUDA 雏形
6. 性能优化

## 9. 当前对外描述建议

如果要对外介绍这个项目，更准确的说法是：

> 一个基于 C++ 实现、当前面向 `yolov8n` 裁剪的教学型 ONNX 推理 runtime。当前已完成模型加载、内部 IR 建模、拓扑顺序执行、kernel 调度、图片预处理和最小检测展示；后续计划继续推进 buffer reuse、Execution Provider、GPT-2 和 CUDA 路径。

这个表述比“我写了一个 mini onnxruntime”更准确，也更容易让别人理解项目当前阶段和后续方向。
