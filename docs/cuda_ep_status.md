# CUDA EP 当前实现状态

这份文档记录的是 `miniONNXRuntime` 里 CUDA Execution Provider 目前已经实现到什么程度，重点放在“现在能做什么”和“还缺什么”。

## 结论先说

- **核心思路已经实现**
- 当前是一个 **教学型混合执行 CUDA EP**
- 不是完整的 device-aware GPU runtime
- 对 YOLOv8n 来说，**可以吃到一部分 CUDA 加速，但不会整图都跑在 GPU 上**

## 已实现内容

### 1. Provider 抽象已经打通

项目已经有统一的 `ExecutionProvider` 接口：

- `Name()`
- `RegisterKernels(KernelRegistry&)`
- `CreateTensorAllocator()`

这意味着 runtime 不再只依赖 CPU，后端切换是通过 provider 完成的。

### 2. CUDA Execution Provider 已接入

`CudaExecutionProvider` 已经实现并注册了一批 kernel：

- `Conv`
- `MatMul`
- `Gemm`
- `Add`
- `Sub`
- `Mul`
- `Div`
- `MaxPool`
- `Sigmoid`
- `SiLU`
- `Tanh`

其中一部分 kernel 使用 CUDA / cuBLAS 计算，出错时会回退到 CPU 实现。

### 3. Session 会自动分配 provider

当前默认 provider 顺序是：

1. `CUDA`
2. `Accelerate`（仅 macOS）
3. `CPU`

`Session` 使用 `kFirstMatch` 策略，节点会优先分配给第一个能处理该 `op_type` 的 provider。

### 4. 构建系统支持可选启用 CUDA EP

`CMakeLists.txt` 里已经有：

- `MINIORT_BUILD_CUDA_EP`
- `CUDA::cudart`
- `CUDA::cublas`

也就是说，CUDA EP 不是写在计划里的概念，而是已经进入构建链路了。

## YOLOv8n 的实际覆盖情况

根据当前仓库里的 `yolov8n.onnx` 图结构，模型里主要有这些节点类型：

- `Conv`
- `Mul`
- `Sigmoid`
- `Concat`
- `Reshape`
- `Unsqueeze`
- `Shape`
- `Add`
- `Gather`
- `Split`
- `Cast`
- `Expand`
- `Range`
- `Slice`
- `Transpose`
- `ConstantOfShape`
- `MaxPool`
- `Resize`
- `ArgMax`
- `Div`
- `ReduceMax`
- `Softmax`
- `Sub`

### 能走 CUDA 的部分

当前 CUDA EP 能覆盖到 YOLO 里的这几类：

- `Conv`
- `Mul`
- `Sigmoid`
- `Add`
- `MaxPool`
- `Div`
- `Sub`

按当前图统计，大约是 **204 / 459 个节点**，也就是大约 **44%**。

### 一定还会留在 CPU 的部分

这些常见节点目前 CUDA EP 还没有注册：

- `Constant`
- `Concat`
- `Reshape`
- `Unsqueeze`
- `Shape`
- `Gather`
- `Split`
- `Cast`
- `Expand`
- `Range`
- `Slice`
- `Transpose`
- `ConstantOfShape`
- `Resize`
- `ArgMax`
- `ReduceMax`
- `Softmax`

所以 YOLOv8n 现在的形态更准确地说是：

- **主干卷积能上 CUDA**
- **图结构和后处理仍大量依赖 CPU**
- **整体是混合执行，不是纯 CUDA**

## 还没实现的关键点

### 1. 真正的 device-aware tensor 存储

当前 `Tensor` 仍然是 host-memory-first 的设计，主要数据还是放在 `std::vector<float>` / `std::vector<int64_t>` 中。

也就是说：

- 还没有独立的 GPU tensor 存储抽象
- 还没有真正的 host/device residency 管理
- 还没有跨 provider 的显式数据搬运边界

### 2. 更完整的算子覆盖

YOLO 里常见的图变换和后处理节点还没进 CUDA EP，例如：

- `Concat`
- `Reshape`
- `Gather`
- `Transpose`
- `Resize`
- `Softmax`

这也是为什么图不能直接变成“几乎全 CUDA”。

### 3. 性能还不是最终形态

现在很多 CUDA kernel 仍然是：

- 先从 CPU 取输入
- 在 kernel 内部做临时 device buffer
- 计算后再拷回 host

这条路适合教学和验证 provider 思路，但还不是高性能 runtime 的最终版本。

## 适合怎么理解这版 CUDA EP

可以把它理解成三句话：

- **provider 抽象是通了**
- **部分关键算子已经能在 CUDA 上执行**
- **但 runtime 还没升级成完整 GPU 常驻内存模型**

所以它现在的价值主要是：

- 验证 provider 分配机制
- 验证 CUDA kernel 接入方式
- 验证 YOLO 这类模型的混合执行路径

## 相关文件

- [`CMakeLists.txt`](/home/weiwei.pan/code/miniONNXRuntime/CMakeLists.txt)
- [`src/runtime/session.cc`](/home/weiwei.pan/code/miniONNXRuntime/src/runtime/session.cc)
- [`src/runtime/cuda_execution_provider.cc`](/home/weiwei.pan/code/miniONNXRuntime/src/runtime/cuda_execution_provider.cc)
- [`docs/cuda_ep_adaptation_plan.md`](/home/weiwei.pan/code/miniONNXRuntime/docs/cuda_ep_adaptation_plan.md)

## 下一步建议

如果要继续推进 CUDA EP，下一批最值得补的是：

1. `Concat`
2. `Reshape`
3. `Transpose`
4. `Gather`
5. `Softmax`

这几类补上以后，YOLO 和 GPT 这两条线的覆盖面都会明显提升。
