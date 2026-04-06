# Execution Provider 设计说明

## 1. 为什么现在引入 EP

在 `phase3/phase4` 之前，`miniONNXRuntime` 的执行路径基本可以概括成：

- `Session` 直接注册 builtin CPU kernels
- `ExecutionContext` 自己维护运行期 tensor 值表
- `ExecutionContext` 内部顺带维护一份简单的 buffer pool

这条路径足够把 `yolov8n.onnx` 跑通，也适合教学早期讲清楚执行主线。

但继续往后推进时，会出现两个越来越明显的问题：

1. `Session` 同时承担了 orchestration 和 backend 装配职责
2. 运行时内存复用逻辑开始需要一个更明确的归属

如果继续把 CPU path 的 kernel 注册、allocator、后续 memory planner 都直接堆在 `Session` 或
`ExecutionContext` 里，后面再补多后端、partition 或 CUDA 会越来越别扭。

因此 `phase5` 先做一件更值钱的事：

- 不急着扩模型
- 不急着做多 EP 分图
- 先把 CPU 路径以及一条最小 Apple 加速路径正式收敛到 `ExecutionProvider`

## 2. 当前最小 EP 目标

当前这版 EP 不是为了追求“接口完整”，而是为了先把边界放对。

这一版只覆盖两个能力：

- provider 注册 kernels
- provider 提供默认 allocator

也就是说，当前 `ExecutionProvider` 更像：

- backend capability 的最小装配入口

而不是：

- 已经完整承载 partition / stream / fence / graph compile 的成熟后端接口

## 3. 当前对象关系

当前代码里的关系可以概括成：

```text
Session
  -> 持有 Graph
  -> 持有 KernelRegistry
  -> 持有 providers
  -> Run() 时把 provider 提供的 allocator 注入 ExecutionContext

ExecutionProvider
  -> 暴露 Name()
  -> RegisterKernels(KernelRegistry&)
  -> CreateTensorAllocator()

CpuExecutionProvider
  -> 注册现有 builtin CPU kernels
  -> 创建 CpuTensorAllocator

AccelerateExecutionProvider
  -> 在 Apple 平台上注册一小部分 Accelerate kernels
  -> 当前先覆盖一组高频 elementwise / matmul 算子
  -> allocator 先复用 CpuTensorAllocator

ExecutionContext
  -> 管理 name -> Tensor 的运行时值表
  -> 对 kernel 暴露 Acquire*Buffer()
  -> 实际 buffer 获取/回收委托给 allocator

TensorAllocator
  -> 负责运行期 tensor storage 的申请与回收

CpuTensorAllocator
  -> 维护 float/int64 buffer pool
  -> 复用已释放 tensor 的底层 storage
```

## 4. 关键调用链

### Session 初始化阶段

当前 `Session` 默认会挂一组按优先级排序的 providers。

目前默认顺序是：

- Apple 平台：`Accelerate -> CPU`
- 其他平台：`CPU`

初始化时主要做两件事：

1. 清洗 provider 列表
2. 依次收集每个 provider 的 kernels，并按 first-match 规则合并进运行时 registry

这意味着：

- `KernelRegistry` 仍然是运行时查表入口
- 但 kernel 的来源已经不再由 `Session` 硬编码决定
- 前面 provider 已经声明支持的 `op_type`，后面的 provider 不再覆盖它

### Run 阶段

当前 `Run()` 主线中，和 provider 相关的关键步骤是：

1. `Session` 检查 `ExecutionContext` 是否已有 allocator
2. 如果没有，则向 provider 请求一个默认 allocator
3. allocator 被注入到 `ExecutionContext`
4. kernel 仍然通过 `ExecutionContext::Acquire*Buffer()` 获取输出 buffer
5. `ExecutionContext` 在覆盖或删除 tensor 时，把 storage 回收给 allocator

这样做以后，kernel 侧几乎不用感知 EP 的存在，但内存策略已经不再硬编码在 context 内部。

## 5. 当前 provider assignment 规则

当前 `Session` 初始化时，会先给每个节点计算 `execution_provider`。

现在支持的最小策略是：

- `ProviderAssignmentPolicy::kFirstMatch`

语义是：

- 按 provider 顺序检查谁声明支持当前 `op_type`
- 第一个匹配的 provider 获得该节点归属
- 如果没有 provider 支持，则节点被标成 `<unassigned>`

当前这层 assignment 会同时产出：

- 节点上的 `execution_provider`
- `SessionAssignmentSummary`
  - `assigned_nodes`
  - `unassigned_nodes`
  - `provider_node_counts`
  - `unassigned_op_types`

如果 `allow_unassigned_nodes=false`，`Session` 会在初始化阶段直接报错。

## 6. 为什么 allocator 先挂在 EP 下面

当前这样做有几个现实原因。

### 5.1 allocator 和 backend 强相关

即使现在只有 CPU，这个方向也很明确：

- CPU allocator 更可能是 host memory pool
- CUDA allocator 更可能绑定 device memory、stream 或 pinned memory

因此 allocator 更自然地属于 provider capability，而不是一个完全脱离 backend 的全局对象。

### 5.2 先保留 kernel 接口稳定

当前 kernel 代码大量依赖：

- `ExecutionContext::AcquireFloatBuffer(...)`
- `ExecutionContext::AcquireInt64Buffer(...)`

如果此时直接把 allocator 概念暴露到所有 kernel，会让改动面无谓扩大。

现在先让：

- kernel -> `ExecutionContext`
- `ExecutionContext` -> `TensorAllocator`

这是一个刻意的过渡层。好处是先把职责搬对，但不破坏现有 kernel 编码方式。

### 5.3 为后续 arena / planner 留挂点

后面如果做：

- 更系统的 lifetime analysis
- 简单 memory planner
- arena allocation

可以继续沿两种路径演进：

1. planner 产出更明确的复用策略，再驱动 allocator
2. allocator 从“buffer pool”升级到“arena + offset”一类模型

无论哪种路径，都比继续把逻辑塞进 `Session` 更干净。

## 7. 当前 Apple 加速路径为什么先选 Accelerate

在 Apple 平台上，当前没有直接做：

- `MetalExecutionProvider`
- `CoreMLExecutionProvider`

而是先做了一个最小 `AccelerateExecutionProvider`。

原因是：

- `Accelerate` 更适合做小步验证第二个 provider
- 它能直接复用当前 runtime 的单 op kernel 模型
- 不需要马上处理 GPU buffer / command queue / stream / 子图编译

当前 `Accelerate EP` 的定位不是“完整 Apple 后端”，而是：

- 验证多 provider 自动选择
- 验证 assignment 是否真的能落到第二个 provider
- 验证 first-match registry 合并是否合理

当前目前覆盖：

- `Sigmoid`
- `Add`
- `Mul`
- `Sub`
- `Div`
- `MatMul`
- `Gemm`

其中：

- `Sigmoid`
  - 走 vForce/vDSP 组合实现
- `Add` `Mul` `Sub` `Div`
  - 在 `float32 + 同 shape + 无 broadcast` 时走 vDSP 快路径
  - 其他情况回退到通用实现
- `MatMul` `Gemm`
  - 当前先支持 2D `float32`
  - `Gemm` 支持最小常见属性：`transA` `transB` `alpha` `beta`

对 elementwise 算子，只有在下面这种情况下走 Accelerate 快路径：

- `float32`
- 两个输入 shape 相同
- 不需要 broadcast

其他情况会在同一个 kernel 内回退到通用实现。

## 8. 当前没有做什么

这一版虽然已经引入 `ExecutionProvider`，但刻意没有继续做下面这些能力：

- 多 EP graph partition
- provider-specific kernel dispatch
- allocator stats / memory instrumentation 接口
- stream / async execution
- compiled graph / fused subgraph execute 接口
- 大范围 Apple 算子覆盖
- Metal / CoreML 路径

原因很简单：

- 当前项目虽然已经有 CPU + Accelerate 两条 provider 路径，但它们都还是单图顺序执行下的最小能力
- 这些能力现在加进去，大概率会先变成空抽象

当前最重要的是让“单后端路径”的边界先合理，再继续演进。

## 9. 和当前内存优化的关系

当前项目已经有两层和内存相关的东西：

1. `memory_profile`
   - 用静态图信息估算 tensor 生命周期和峰值内存
2. runtime allocator
   - 在执行期复用已释放 tensor 的底层 buffer

两者现在还是分开的：

- `memory_profile` 更像分析工具
- `CpuTensorAllocator` 更像运行期复用机制

这其实是合理的，因为当前阶段还在验证：

- 生命周期信息是否可信
- 哪些 tensor 真正适合复用
- 分析结果将来应该以什么形式喂给 runtime

后面更自然的方向是：

- 先让 `memory_profile` 和 runtime 行为对齐得更好
- 再考虑让 planner 显式影响 allocator 策略

## 10. 当前设计的优点和局限

### 优点

- `Session` 不再直接硬编码 CPU builtin kernels
- provider 顺序和节点 assignment 已经有了最小闭环
- Apple 平台上已经能自动优先选择 `Accelerate`
- allocator 的归属从 context 内部实现转成 provider capability
- 对现有 kernel 改动很小
- 后续扩展 CUDA / 多 EP / planner 时有更清晰的落点

### 局限

- 现在还没有真正的多 provider partition / execute plan
- `KernelRegistry` 仍然是合并后的全局 registry，而不是 provider-scoped dispatch table
- allocator 目前仍然只覆盖 `float32` / `int64`
- 运行期还没有显式的 memory plan 输入
- `Accelerate EP` 目前仍然只覆盖一组较小的 elementwise / gemm 算子子集

这些局限是当前阶段有意接受的，因为目标是：

- 先把架构方向做对
- 再逐步补机制

## 11. 推荐的后续演进顺序

围绕 EP，当前更合理的后续顺序是：

1. 扩大 `Accelerate EP` 覆盖的算子范围
2. 让内存 profile 和 runtime allocator 的行为更可对照
3. 视需要引入更明确的 allocator stats / tracing
4. 再考虑多 EP partition 骨架
5. 最后再结合新模型需求决定是否推进 GPT-2

如果后面直接跳到 `GPT-2`，项目很容易再次回到“补算子”的主线上。

而先把 EP 和 allocator 边界做好，后面新增模型带来的复杂度会更容易安放。
