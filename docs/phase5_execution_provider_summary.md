> 这篇文章想单独把 `phase5` 讲清楚。前面几个阶段，我主要是在把 `miniONNXRuntime` 跑通：先能读 ONNX，后来能跑 YOLO，再后来开始做图优化和内存复用。到了这一阶段，重点开始变了，不再只是“能不能跑”，而是“这个 runtime 的后端边界是不是清楚”。`Execution Provider` 就是在这个时候引进来的。当前这个项目里已经有两条 EP 路线：一条是默认兜底的 `CpuExecutionProvider`，另一条是在 macOS 上优先启用的 `AccelerateExecutionProvider`。

## 1. 先说清楚：EP 到底是什么

`Execution Provider` 这个词听起来有点大，其实在这个项目里可以先简单理解成：

- 一层用来描述“谁来执行这些算子”的抽象

如果再说得具体一点，它主要回答两个问题：

1. 哪些算子由哪个后端来执行
2. 这个后端用什么方式提供运行期内存

在完整的 ONNX Runtime 里，EP 往往还会继续负责更复杂的事情，比如图分区、子图下沉、异步执行、设备同步这些。但在 `miniONNXRuntime` 这个阶段，我没有一上来把这些都做进去，而是只保留了最小的一层：

- provider 注册自己的 kernels
- provider 提供自己的 allocator
- `Session` 根据 provider 顺序给节点分配执行归属

所以这篇文章里说的 `EP`，不是一个“大而全的后端系统”，而是一层刚好够用、但方向已经对了的 runtime 抽象。

## 2. 为什么这个阶段要先做 EP

如果回头看 `phase3` 和 `phase4`，那时候的执行路径其实很直接：

- `Session` 直接注册 CPU kernels
- `ExecutionContext` 里顺带维护 tensor 和简单的 buffer pool
- 默认所有东西都算“CPU runtime 自己的逻辑”

这条路在早期是很有效的，因为它短、简单，也容易把主线讲清楚。但项目继续往后走，就会开始别扭。

最明显的问题有两个。

第一个问题是，`Session` 管得太多了。  
它既要负责调度整张图，又要负责把具体后端装进去。短期看没问题，长期看就会越来越难扩。

第二个问题是，allocator 放哪不清楚。  
如果以后要继续做 Apple 路线，或者哪天真的加别的后端，那 allocator 其实应该和 backend 更靠近，而不是一直挂在一个很泛的 context 里。

所以 `phase5` 里我没有先跳去做新模型，也没有先去做多 EP 分图，而是先做了一件更基础的事：

- 把 CPU 路径和一条最小 Apple 路径正式收进 `ExecutionProvider`

这一步表面上看像是“重构”，但它其实决定了后面项目能不能继续长下去。

## 3. 这一阶段具体做了什么

这一阶段我做的事情，核心可以分成三块。

### 3.1 先把最小 EP 框架立起来

先抽出了几个最小对象：

- `ExecutionProvider`
- `CpuExecutionProvider`
- `TensorAllocator`
- `CpuTensorAllocator`

现在的关系可以简单理解成：

```text
Session
  -> 持有 providers
  -> 收集 provider 提供的 kernels
  -> Run() 时把 allocator 注入 ExecutionContext

ExecutionProvider
  -> Name()
  -> RegisterKernels(...)
  -> CreateTensorAllocator()

ExecutionContext
  -> 管运行时 tensor
  -> 对 kernel 暴露 Acquire*Buffer()
  -> 具体 buffer 获取和回收交给 allocator
```

这一步最直接的变化是：

- `Session` 不再自己硬编码 CPU kernel 注册
- allocator 不再继续藏在 `ExecutionContext` 里
- “后端能力”终于有了一个正式入口

这件事做完以后，项目的主线也变了。

以前更像这样：

```text
loader -> graph -> session -> builtin cpu kernels
```

现在变成了这样：

```text
loader -> graph -> session -> execution providers -> kernels
```

虽然只是多了一层，但这层的意义很大。它让 runtime 从“只有 CPU 的教学实现”，开始有了一点真正推理框架的味道。

### 3.2 再把节点归属也补齐

只有 provider 接口还不够，因为光有接口，并不能说明图里的节点到底归谁执行。

所以这一阶段还补了 provider assignment 这层信息，包括：

- `Node.execution_provider`
- `SessionAssignmentSummary`
- `ProviderAssignmentPolicy::kFirstMatch`
- 未分配节点的校验逻辑

当前规则很简单：

- 按 provider 顺序检查谁先支持这个 `op_type`
- 第一个匹配上的 provider 获得这个节点
- 如果没人支持，就标成 `<unassigned>`

我故意先把策略做得很简单，因为这个阶段的重点不是策略花哨，而是先把“节点属于谁”这件事明确下来。

这样做之后，很多工具都变得更有用了：

- `miniort_inspect` 可以看静态 assignment
- `miniort_session_trace` 可以看运行期命中
- `RunSummary` 可以按 provider 统计实际执行情况

也就是说，现在不只是“有 provider”，而是能看见它真的在参与调度。

### 3.3 最后把结果验证住

这轮改动动到了 `Session`、allocator、provider assignment 和一批 kernel，所以我也顺手把最小验证补齐了。

结果上可以简单说成：

- 原有 CPU 路径没有被改坏
- Apple 路线的 assignment 和执行都能正常命中
- `Conv`、`ConvSiLU` 这些后来补进去的路径也能稳定跑通

## 4. 这一阶段做了哪两个 provider

这一轮真正落地了两个 provider。

### 4.1 `CpuExecutionProvider`

这是基线，也是所有平台的兜底。

它负责：

- 承接原来已有的 CPU kernels
- 提供 `CpuTensorAllocator`

从结果上说，原来的 CPU 路径并没有被推翻，只是从“写死在 Session 里”变成了“正式挂到 EP 下面”。

### 4.2 `AccelerateExecutionProvider`

在 macOS 上，我没有先做 `MetalExecutionProvider`，也没有先做 `CoreMLExecutionProvider`，而是先做了一个最小的 `AccelerateExecutionProvider`。

这个选择其实很务实。

因为当前项目最需要验证的不是 GPU 资源管理，也不是子图编译，而是：

- 第二个 provider 能不能真的接进这套框架
- assignment 会不会真的把节点分过去
- 分过去以后，运行时能不能真走这条路径

`Accelerate` 很适合做这件事。  
它不像 `Metal` 那样一上来就要处理 command queue，也不像 `CoreML` 那样更适合子图整体下沉。它正好可以延续当前项目这种“单个 op kernel”风格。

当前默认 provider 顺序是：

- macOS: `Accelerate -> CPU`
- 其他平台: `CPU`

这个顺序的意思很简单：

- 在 Apple 平台上，先给 `Accelerate` 一个机会
- 如果它不支持，再回退到 CPU

## 5. 这一阶段在 macOS 上具体做了什么

当前 `AccelerateExecutionProvider` 已经支持一批常见算子，包括：

- `Sigmoid`
- `SiLU`
- `Add`
- `Mul`
- `Sub`
- `Div`
- `MatMul`
- `Gemm`
- `Conv`
- `ConvSiLU`

如果只是把这些名字列一遍，其实不太容易看出 Apple 路线和 CPU 路线到底差在哪。这里拿一个最有代表性的例子来说，也就是 `ConvSiLU`。

### 5.1 用 `ConvSiLU` 看 CPU 和 macOS 的区别

`ConvSiLU` 很适合拿来举例，因为它正好处在 YOLO 的主干上，而且它不是一个单纯的逐元素算子。

在 CPU 路线里，这个算子的思路比较直接：

- 先按普通卷积的方式把输出算出来
- 再逐个元素去做 `SiLU`

也就是说，CPU 版更像是一条朴素、清楚、容易验证正确性的实现路径。

对应到代码里，大致就是下面这样：

```cpp
auto output = RunConv2D(node, input, weight, bias, context);
for (auto& value : output.float_data) {
  value = value * (1.0f / (1.0f + std::exp(-value)));
}
```

这里的感觉很明显：

- 先把卷积结果算出来
- 再在一层很直接的循环里做激活

这类实现的好处是好懂，也容易先把语义做对。

在 macOS 的 `Accelerate` 路线里，我做的事情不太一样。

一开始，`ConvSiLU` 只是先复用了 CPU 的卷积主计算，然后把最后的 `SiLU` 换成了 Accelerate 的向量化实现。这样做的好处是能先把 provider 路线接通，先确认 assignment 和执行链路都是成立的。

但后来又继续往前推了一步，把卷积主计算也改成了更像 Accelerate 风格的写法：

- 先把卷积展开成 `im2col`
- 再用 `cblas_sgemm` 去做主计算
- 最后再接 Accelerate 版的 `SiLU`

对应代码里最关键的部分，大概长这样：

```cpp
FillIm2ColBuffer(batch_input, params, columns);
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            static_cast<int>(params.c_out),
            static_cast<int>(output_hw),
            static_cast<int>(kernel_dim),
            1.0f, weight_data.data(), static_cast<int>(kernel_dim),
            columns.data(), static_cast<int>(output_hw),
            0.0f, batch_output, static_cast<int>(output_hw));

ApplySiLUInPlaceAccelerate(output);
```

如果只看这几行，也能看出两边思路已经不一样了：

- CPU 路线是在原始张量布局上直接算
- `Accelerate` 路线会先把数据整理成更适合 BLAS 吃的矩阵形式
- 然后把主计算交给加速库去做
- 最后再用向量化方式补上激活

这样两边的差别就很清楚了：

- CPU 路线更偏向直接、朴素、可读
- macOS 路线更偏向把计算改写成适合加速库吃下去的形状

这其实就是 EP 这层抽象的意义之一。  
同样一个 `op_type`，在 runtime 这一层看起来还是 `ConvSiLU`，但到了不同 provider 下面，可以走完全不同的实现方式。

### 5.2 其他算子是什么思路

除了 `ConvSiLU` 之外，这一轮还顺手把一批更容易吃到 Accelerate 的算子也接进来了。

比如：

- `Add`、`Mul`、`Sub`、`Div`
- `Sigmoid`
- `MatMul`、`Gemm`

这里的思路也比较统一：

- 能直接映射到 `vDSP` / `vForce` 的，就尽量直接映射
- 能直接写成矩阵乘的，就尽量交给 `sgemm`
- 如果条件不合适，就回退到原来的通用实现

所以这条 Apple 路线并不是“把所有实现推倒重来”，而是：

- 在不破坏原有 CPU 路线的前提下
- 挑一批最值得、最容易接入加速库的算子先做起来

## 6. 这一阶段最后拿到了什么结果

### 6.1 先是结构上的结果

这轮做完以后，项目已经不再只是“CPU runtime + 一些 kernels”。

它现在已经有了这些比较清楚的边界：

- `Session` 负责 orchestration
- `ExecutionProvider` 负责提供 backend 能力
- allocator 也开始归到 provider 下面
- 节点归属信息能在初始化阶段明确算出来

这意味着后面不管是继续做 Apple 路线，还是将来想做多 EP、做 GPT-2，至少结构上已经不需要从头改一次。

### 6.2 然后是 macOS 上真实的混合执行

当前 `yolov8n.onnx` 在 macOS 上不是整张图都跑在 Apple 后端上，而是：

- 一部分节点走 `Accelerate`
- 剩下的节点继续走 `CPU`

但这已经足够说明这套 EP 框架不是摆设。

它不只是静态 assignment 看起来存在，而是运行时真的在混合执行。

### 6.3 原始 YOLO 图上的性能结果

对原始 YOLO 图，我用这条命令做过一轮对比：

```bash
./build_phase3/miniort_compare_providers models/yolov8n.onnx --image pic/bus.jpg --repeat 1
```

当前单次结果是：

- `mixed_ms = 2164.04`
- `cpu_only_ms = 11035.7`
- `speedup_pct = 80.3906`

这不是严格 benchmark，因为这里只是单次对比，还没有去做多次平均和方差统计。但它已经很能说明问题了：

- 这条 Apple 路线不是只有“结构上成立”
- 它对整图时间已经有明显影响

### 6.4 优化后图上的结果更直观

对优化后的 YOLO 图，我还用这条命令看过：

```bash
./build_phase4/miniort_optimize_model models/yolov8n.onnx --image pic/bus.jpg
```

优化前，图里主要有：

- `Conv: 64`
- `Sigmoid: 58`
- `Mul: 63`

优化后，变成了：

- `ConvSiLU: 57`
- `Conv: 7`
- `Sigmoid: 1`
- `Mul: 5`

这说明 `ConvRewrite` 的效果是很实的，大部分 `Conv + Sigmoid + Mul` 都被融合成了 `ConvSiLU`。

而当前优化后图的 provider 分配结果是：

- `Accelerate: 86`
- `CPU: 158`

实际执行 summary 也是：

- `Accelerate: visited=86 executed=86`
- `CPU: visited=158 executed=158`

这里最有意思的一点是，`Accelerate` 节点数从之前的 79 提升到了 86。这个差值正好是剩下那 7 个普通 `Conv`，说明卷积快路径接进去以后，Apple 路线已经把它们也接住了。

## 7. 这一阶段没有做什么

虽然 `phase5` 已经把 EP 框架立住了，但我也有意没有继续把范围扩太大。

当前还没有做的内容包括：

- 多 EP graph partition
- provider-specific graph compile
- stream / async execution
- Metal / CoreML execution provider
- 更完整的 memory planner / arena

所以更准确地说：

- `EP framework v1` 已经完成
- `CPU EP v1` 已经完成
- `Accelerate EP v1` 也已经成立
- 但更完整的多后端 runtime 还没开始

这其实是我故意保留的边界。因为如果在这一阶段就把 partition、async、Metal、CoreML 全都一起拉进来，项目很容易一下子变得太重，反而把最值钱的主线冲散了。

## 8. 最后怎么概括这一阶段

如果只用一句话来总结 `phase5`，我会这样说：

- 这一阶段把 `miniONNXRuntime` 从“能跑 CPU 的教学型 runtime”，推进成了“有明确后端边界的迷你推理框架”

这轮最重要的价值，不只是多写了一个 Apple provider，而是把下面这三件事都证明了：

1. `ExecutionProvider` 这层抽象在这个项目里是成立的
2. 第二个 provider 真的能影响节点 assignment 和实际执行
3. Apple 路线已经不只是 demo，而是对 YOLO 有真实收益

对我来说，这比单纯再补几个算子更重要。因为从这个阶段开始，这个项目的重点已经不只是“做出结果”，而是开始能讲清楚一个 runtime 为什么要这样组织。  
