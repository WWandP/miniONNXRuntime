# Resume Notes: MiniORT And ONNX Runtime Design Influence

这份笔记用于后续整理简历和面试讲述。重点不是说 MiniORT 复刻了 ONNX Runtime，而是说明：

```text
研读 ONNX Runtime 的核心执行链路后，将其中的关键 runtime 思想抽象成 MiniORT 中可教学、可验证的简化实现。
```

## 项目定位

MiniORT 是一个教学型 ONNX 推理运行时，覆盖：

- ONNX 图解析
- 内部 Graph / Node / Tensor 表达
- 拓扑序执行
- Kernel Registry
- CPU / CUDA / Accelerate Execution Provider
- 图优化与内存观察
- YOLOv8n / GPT-2 / Qwen 推理示例

简历里不要写成完整替代 ONNX Runtime。更准确的说法是：

```text
自研教学型 ONNX Runtime，聚焦推理 runtime 核心链路和 CUDA mixed-provider 优化。
```

## 从 ONNX Runtime 吸收的设计点

### 1. Execution Provider 抽象

ONNX Runtime 的思路：

- 不同后端通过 Execution Provider 接入 runtime。
- EP 负责声明自己能执行哪些节点或子图。
- Runtime 根据 provider 优先级做图分配。

MiniORT 中的对应实现：

- `ExecutionProvider` 抽象。
- `CpuExecutionProvider` / `CudaExecutionProvider` / `AccelerateExecutionProvider`。
- 每个 provider 注册自己的 kernel。
- `Session` 根据 provider 支持的 op type 给节点分配执行后端。

当前源码中的处理逻辑分两步：

1. `Session` 构造时，让每个 provider 注册一次自己的 kernel。
2. 同时把该 provider 支持的 op type 收集成 `unordered_set`，保存到 `provider_supported_ops_`。
3. 后续给节点分配 provider 时，只需要按 provider 顺序做 `contains(node.op_type)`。

也就是说，当前实现不是每遇到一个节点就重新构造 provider kernel 表，而是先预计算每个 provider 的能力集合：

```text
provider -> RegisterKernels(provider_registry)
provider_registry -> supported_ops unordered_set
node assignment -> provider_supported_ops_[i].contains(node.op_type)
```

这对应一次明显的策略变化：

- 早期实现：`ResolveExecutionProviderForNode()` 内部对每个节点、每个 provider 都重新创建 `KernelRegistry` 并调用 `RegisterKernels()`，再判断 `Has(node.op_type)`。
- 后续优化：在 `Session` 构造期缓存 `provider_supported_ops_`，节点分配阶段只查集合。

这个变化可以讲成：

```text
最初 provider assignment 是按节点反复构造 provider registry 做能力判断，随着模型变大和 provider 增多，这种方式会产生重复开销；后续改成 Session 构造期预计算每个 provider 的 supported-op set，把分配阶段简化为 O(provider_count) 的 hash lookup。
```

可讲述为：

```text
参考 ONNX Runtime 的 EP 架构，在 MiniORT 中实现了轻量 Execution Provider 抽象和 kernel registry，使 CPU/CUDA/Accelerate 后端可以通过统一接口接入执行链路。
```

### 2. Graph Partitioning / Provider Assignment

ONNX Runtime 的思路：

- `GraphPartitioner` 调用 EP 的 `GetCapability`。
- EP 返回 `ComputeCapability`，可以是单节点，也可以是子图。
- Runtime 按 provider 优先级把图划分给不同 EP。

ORT 原生流程更准确地说是：

1. `GraphPartitioner` 按用户注册的 provider 顺序遍历 EP。这个顺序代表用户偏好，通常 CUDA / TensorRT 等在前，CPU 在最后。
2. 对当前 EP 创建 `GraphViewer` 和 `KernelLookup`。
3. 调用当前 EP 的 `GetCapability(graph_viewer, kernel_lookup, ...)`。
4. EP 返回一组 `ComputeCapability`，每个 capability 表示它可以执行的节点或子图。
5. `GraphPartitioner` 检查这些节点是否还没被更高优先级 EP 占用。
6. 如果可分配，就把节点标记为当前 EP；如果 capability 是子图，还可能做 fusion / compile。

ORT 默认 `IExecutionProvider::GetCapability()` 的逻辑接近：

```text
for node in graph:
  if kernel_lookup.LookUpKernel(node):
    return single-node ComputeCapability
```

但具体 EP 可以重写这个方法。CUDA EP 就会做更多判断：

- 是否真的有 CUDA kernel。
- 某些 op 是否因为属性/shape 不支持而 fallback CPU。
- shape-related 子图是否更适合留在 CPU。
- resource accountant 是否允许继续分配到 CUDA。
- 已经分配过的节点不再被重复抢占。

所以 ORT 原生不是简单的“op_type 大表查询”，而是：

```text
EP 报告 capability -> partitioner 按优先级接受 capability -> 节点或子图被赋给对应 EP
```

MiniORT 当前更像 ORT 默认 EP capability 逻辑的最小版本：

```text
provider supported-op set contains node.op_type -> assign node to provider
```

它没有实现完整的 `ComputeCapability`、子图 fusion、compile 和 CPU-preferred shape subgraph 规则，但保留了核心思想：

```text
provider 按优先级声明能力，runtime 根据能力做执行后端分配。
```

MiniORT 中的简化：

- 没有实现完整 `ComputeCapability` 和子图编译。
- 先按 `op_type` 是否有 kernel 做节点级 provider assignment。
- 增加 `miniort_provider_segments` 工具，把连续相同 provider 的节点合并成 segment 做分析。

这个设计直接指导了 CUDA 优化：

- 原始 YOLOv8n provider segments: `60`
- CUDA 边界算子优化后 provider segments: `9`
- 最大 CUDA segment: 从 `17` 个节点增长到 `218` 个节点

可讲述为：

```text
借鉴 ORT 图分区思想，实现 provider assignment 和 provider segment 分析工具，用于识别 YOLOv8n 中 CPU/CUDA 边界，指导后续 CUDA boundary op 优化。
```

### 3. OrtDevice / Tensor Device Residency

ONNX Runtime 的思路：

- Tensor 带有 device location。
- Runtime 知道一个 tensor 在 CPU 还是 GPU。
- CUDA EP 的中间输出可以留在 GPU，不需要每个 op 都拷回 CPU。

MiniORT 早期问题：

```text
CUDA op input:  CPU vector -> H2D
CUDA op output: D2H -> CPU vector
next CUDA op:   CPU vector -> H2D again
```

MiniORT 中的简化实现：

- 在 `Tensor` 中增加可选 CUDA storage：

```text
cuda_data
cuda_bytes
```

- CUDA op 可以直接消费/产出 CUDA-resident tensor。
- CPU 节点执行前才 materialize CUDA input 到 host。
- graph output 在执行结束时 materialize。

优化效果：

- Tensor residency 前 mixed CUDA latency: `77.5 ms`
- Tensor residency 后 mixed CUDA latency: `45.1 ms`
- 相对原始 CUDA baseline 降低约 `71.5%`

可讲述为：

```text
参考 ORT 的 tensor device location 思想，为 MiniORT Tensor 增加 CUDA-resident storage，使 CUDA 中间结果跨 op 保留在 GPU，显著减少重复 H2D/D2H。
```

### 4. DataTransferManager / Cross-device Copy

ONNX Runtime 的思路：

- 跨设备 copy 不散落在每个 kernel 内。
- 通过 `DataTransferManager` 和 provider 注册的 `GPUDataTransfer` 统一处理。
- ORT 中还有显式的 `MemcpyFromHost` / `MemcpyToHost` runtime op。

MiniORT 中的简化实现：

- 没有完整实现 DataTransferManager。
- 先做了 `MaterializeCudaTensor` / `MaterializeCudaInputsForNode`。
- 把 copy 从“每个 CUDA kernel 无脑 D2H”逐步改成“只有 CPU 边界或最终输出需要时才 materialize”。

可讲述为：

```text
借鉴 ORT centralized data transfer 的方向，在 MiniORT 中实现最小化 CUDA materialization 机制，避免 CUDA op 之间重复 host/device round trip。
```

### 5. ExecutionFrame / Allocation Planner / Memory Reuse

ONNX Runtime 的思路：

- `ExecutionFrame` 管理运行时 value。
- `SequentialExecutionPlan` 和 `AllocationPlanner` 记录每个 value 的分配位置、复用关系和释放时机。
- allocator / arena / reuse 逻辑和 kernel 执行解耦。

MiniORT 中的简化实现：

- 没有完整 allocation planner。
- 先实现几个收益明确的小机制：
  - CUDA buffer pool
  - cuBLAS handle reuse
  - cuDNN plan cache
  - CUDA initializer cache
  - dead tensor eviction
  - CUDA Conv bias on GPU

优化效果：

- 原始 mixed CUDA baseline: `157.8 ms`
- buffer pool + cuBLAS handle reuse: `127.6 ms`
- CUDA im2col: `76.9 ms`
- CUDA tensor residency: `45.1 ms`
- CUDA Concat/Split: `33.8 ms`
- CUDA Reshape/Resize: `26.8 ms`
- CUDA initializer cache + Conv bias on GPU: `13.6 ms`

可讲述为：

```text
参考 ORT allocation planning 的思想，将内存管理从单个 kernel 中抽离，先以 buffer pool、initializer cache 和 CUDA-resident output 的方式实现轻量内存复用。
```

## 当前可写进简历的结果

推荐简历版本：

```text
自研教学型 mini ONNX Runtime，支持 ONNX 图解析、拓扑执行、Kernel Registry、CPU/CUDA Execution Provider 以及 YOLOv8n/GPT/Qwen 推理示例；参考 ONNX Runtime 的 EP 分区、设备内存驻留和跨设备数据搬运思想，优化 CUDA mixed-provider 执行路径，在 RTX 4090 上将 YOLOv8n 推理延迟从 157.8ms 降至 13.6ms，超过本机 ONNX Runtime CPU 16-thread baseline（约 24.2ms p50）。
```

更短版本：

```text
研读 ONNX Runtime 核心执行链路，将 EP 分区、device residency、data transfer 和 memory reuse 思想抽象到自研 MiniORT 中；优化 YOLOv8n CUDA mixed-provider 路径，在 RTX 4090 上达到 13.6ms latency，超过本机 ORT CPU baseline。
```

## 面试讲述主线

可以按这个顺序讲：

1. 我先做了一个能跑通 ONNX 图的最小 runtime：loader、Graph、Session、KernelRegistry、CPU kernels。
2. 后来参考 ONNX Runtime，把后端执行抽象成 Execution Provider。
3. 接入 CUDA 后，最开始虽然比 MiniORT CPU 快，但 Nsight 显示主要瓶颈不是 GPU kernel，而是 per-op malloc/free 和 H2D/D2H。
4. 我继续读 ORT 的 ExecutionFrame、DataTransferManager、AllocationPlanner，意识到关键是 tensor location 和数据搬运策略。
5. 于是 MiniORT 里做了 CUDA tensor residency、materialization、buffer pool、initializer cache 和 boundary ops CUDA 化。
6. 最终 YOLOv8n 从 `157.8 ms` 优化到 `13.6 ms`。

## 不建议这样说

不要说：

```text
实现了一个完整 ONNX Runtime。
```

更好：

```text
实现了教学型 mini ONNX Runtime，覆盖推理 runtime 的核心机制。
```

不要说：

```text
性能超过 ONNX Runtime。
```

更好：

```text
在同机 YOLOv8n benchmark 下，MiniORT mixed CUDA 路径超过 ONNX Runtime CPU baseline。
```

不要只说：

```text
实现 CUDA 加速。
```

更好：

```text
基于 profiling 定位数据搬运和 provider 边界瓶颈，并通过 CUDA tensor residency 和 boundary op CUDA 化降低 latency。
```
