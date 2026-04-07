> 这篇文章先从 ONNX Runtime 里的图优化作用讲起，再逐步收束到 `miniONNXRuntime` 的具体实现。文章会按“图优化的通用作用 -> 当前项目的优化管线 -> 典型优化案例 -> 代码实现”这个顺序展开，最后落到这套代码是如何跑起来的。

项目代码在 GitHub 上也可以直接看：
- `https://github.com/WWandP/miniONNXRuntime`

## 1. 图优化在 ONNX Runtime 里做什么

在 ONNX Runtime 这类推理引擎里，图优化的角色很明确：它是把静态模型变成可执行图的重要前置步骤。

ONNX 模型描述的是一张静态计算图，但 runtime 真正执行时，面对的是一串需要调度的节点、张量和 kernel 调用。图优化要做的，就是尽量在执行前把那些不需要真正跑的部分处理掉，把图整理成更适合执行的形态。

如果不做图优化，runtime 往往会遇到几类典型问题：

- 图里有大量只服务于 shape 推导的辅助节点
- 常量子图每次运行都被重复执行
- 一些显然可以合并的模式会被拆成多个 kernel 调用
- identity 类节点会制造额外的中间张量和调度开销
- 中间值越多，执行和调试都更重

所以，图优化在 ONNX Runtime 里的作用可以概括成一句话：

**在不改变模型语义的前提下，减少 runtime 需要实际执行和维护的图工作量。**

对于完整的 ONNX Runtime 来说，这一步通常会和执行计划、算子融合、设备分配、内存规划等机制一起工作。

实际的 ONNX Runtime 图优化远不止这些内容。这里为了适配 `miniONNXRuntime` 当前阶段，只先挑了几类最明显、最容易看见收益的优化来做：

- 常量折叠
- 死节点清理
- 常见模式融合
- 简单的 shape/view 简化

在 `miniONNXRuntime` 里，这件事尤其重要，因为当前项目还处于“教学型 mini runtime”阶段。我们先把图本身收拾干净，让 runtime 的主线更短、更清楚。

## 2. 回到 miniONNXRuntime

前一篇文章已经把 ONNX 模型如何被解析成内部图结构讲清楚了。完成解析之后，我又补上了 YOLO 相关的 CPU kernel、前处理和后处理，先把推理结果跑通。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f12afd3353b84c1b9e82330140ba2ca6.png#pic_center)

到这一步，`miniONNXRuntime` 已经可以完整跑出结果，接下来就进入性能上的优化阶段，也就是图优化。把视角收回来，`miniONNXRuntime` 现在做的图优化，就是围绕 `yolov8n.onnx` 这条主线，把能提前算的算掉、能删的删掉、能融合的融合掉、能简化的简化掉。

当前图优化相关代码大致分布在下面这些位置：

```text
miniONNXRuntime
├── include/miniort/optimizer/
│   └── graph_optimizer.h
├── src/optimizer/
│   └── graph_optimizer.cc
├── tools/
│   └── optimize_model.cc
└── src/runtime/
    ├── basic_kernels.cc
    ├── elementwise_kernels.cc
    ├── shape_kernels.cc
    └── nn_kernels.cc
```

这几个文件可以这样理解：

- `src/optimizer/graph_optimizer.cc`
  - 图优化的核心实现
  - 负责常量折叠、死节点清理、Conv 融合和 shape 简化
- `tools/optimize_model.cc`
  - 图优化的命令行入口
  - 负责把加载、优化、执行、输出串起来
- `src/runtime/nn_kernels.cc`
  - 主要放卷积和 YOLO 相关的网络层 kernel
  - 这里有 `Conv`、`ConvSiLU`、`MaxPool`、`Resize`、`Softmax`
- `src/runtime/shape_kernels.cc`
  - 主要放 shape 相关 kernel
  - 这里有 `Shape`、`Gather`、`Reshape`、`Range`、`Slice`、`Transpose`、`Expand`、`ArgMax`、`ReduceMax`
- `src/runtime/elementwise_kernels.cc`
  - 主要放逐元素算子
  - 这里有 `Add`、`Mul`、`Sub`、`Div`、`Sigmoid`、`Cast`
- `src/runtime/basic_kernels.cc`
  - 放基础常量类 kernel
  - 这里有 `Constant`

这里面最关键的几个函数是：

- `LoadOnnxGraph(...)`
  - 把 ONNX 文件读成内部 `Graph`
- `OptimizeGraph(...)`
  - 执行图优化管线，返回优化后的 `Graph`
- `RunConstantFolding(...)`
  - 折叠静态子图
- `RunDeadNodeCleanup(...)`
  - 清理不可达节点
- `RunConvRewrite(...)`
  - 识别并融合 `Conv -> Sigmoid -> Mul`
- `RunShapeSimplification(...)`
  - 简化 shape/view 类节点和恒等算子

工具入口是 `tools/optimize_model.cc`，它负责把“加载模型 -> 优化图 -> 跑推理 -> 输出结果”串起来。

1. 先用 `LoadOnnxGraph(...)` 读入 ONNX 模型
2. 调 `OptimizeGraph(...)` 做图优化
3. 再把优化后的图交给 `Session` 执行
4. 最后跑 YOLO 后处理，输出检测结果

这个优化模块已经接到实际执行链路里。

### 图优化的整体位置

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e18bb52ed66240b58b964034cec1045d.png#pic_center)


### 一个典型的优化对象

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/911b07fed78b417abdc376e3b6b1baa2.png#pic_center)

## 3. 当前做了哪些图优化

目前的图优化管线是固定顺序的四个 pass：

1. `ConstantFolding`
2. `DeadNodeCleanup`
3. `ConvRewrite`
4. `ShapeSimplification`

对应的配置项在 `include/miniort/optimizer/graph_optimizer.h` 的 `GraphOptimizationOptions` 里，默认会开启常量折叠、死节点清理、Conv 融合和 shape 简化。

### 一张总表

| Pass | 主要做什么 | 当前代码里的效果 |
| --- | --- | --- |
| `ConstantFolding` | 把只依赖常量的子图提前算出来 | 将静态子图折叠成 initializer |
| `DeadNodeCleanup` | 从最终输出反向清理不可达节点 | 删除死节点、无用 initializer 和无用 `value_info` |
| `ConvRewrite` | 识别 `Conv -> Sigmoid -> Mul` | 融合成 `ConvSiLU` |
| `ShapeSimplification` | 去掉 identity 节点和简单代数噪声 | 清理 shape/view 节点，并继续折叠可静态求值的子图 |

## 4. 图优化到底是怎么跑的

入口函数是 `OptimizeGraph(...)`，代码位置在 `src/optimizer/graph_optimizer.cc`。

它的结构很直接：按固定顺序执行 pass，记录 summary 和 timing，最后返回优化后的 `Graph`。

```cpp
Graph OptimizeGraph(Graph graph, const GraphOptimizationOptions& options,
                    std::ostream* trace, GraphOptimizationSummary* summary);
```

从实现上看，它采用的是一个很清晰的线性流水线：

```text
ConstantFolding
  -> DeadNodeCleanup
  -> ConvRewrite
  -> ShapeSimplification
```

这种顺序对应着明确的处理逻辑：

- 先折叠常量，先把静态垃圾清掉
- 再做一次可达性清理
- 然后尝试做结构性融合
- 最后做 shape/view 类的小优化和补刀式折叠

这样做的好处是实现简单，调试路径也短。

## 5. ConstantFolding：把静态子图提前算掉

### 它解决什么问题

先看一个最直接的理解：

- 有些节点的输入本来就是常量
- 这类节点每次推理算出来的结果都一样
- 既然结果不会变，就可以在初始化阶段先算好

`ConstantFolding` 做的事就是把这类节点提前算完。

这样到了真正推理的时候，这些结果已经在图里准备好了，runtime 直接复用，不用每次重复计算。

### 代码怎么做

这个 pass 的核心在 `RunConstantFolding(...)`。

它的工作方式可以理解成：

1. 先看图里的每个节点
2. 判断这个节点的输入是不是常量
3. 如果是，就在优化阶段直接算出来
4. 把结果放回图里，后面执行时直接用

真正“怎么算”的部分放在 `FoldConstantNode(...)` 里。

如果节点可以折叠，就：

- 把结果写入 `graph.initializers`
- 同时补上对应的 `graph.value_infos`
- 把原节点从图里移除

最后再调用 `RebuildGraphDerivedState(...)` 重建拓扑序、节点索引和 op 统计。

你可以把 `FoldConstantNode(...)` 理解成一组“节点计算器”。

比如：

- `Constant`
  - 直接把常量值拿出来
- `Shape`
  - 如果输入 shape 已知，就直接算出 shape
  - 公式上可以写成 `y = [d0, d1, ..., dn]`
- `Gather`
  - 如果输入张量和索引都已知，就直接按索引取值
  - 公式上可以写成 `y = x[index]`
- `Reshape`
  - 如果 reshape 的目标 shape 已知，就直接算出新 shape
  - 公式上要求元素总数不变：`prod(shape_in) = prod(shape_out)`
- `Range`
  - 如果起点、终点、步长都是常量，就直接生成结果序列
  - 公式上可以写成 `y = [start, start + step, ...]`
- `Cast`
  - 如果输入是常量，就直接做类型转换
  - 公式上可以写成 `y = cast(x, target_dtype)`

这一类节点的共同特点是：输入固定，输出也固定，所以很适合提前算完。

### 支持哪些算子

`FoldConstantNode(...)` 当前支持的范围包括：

- `Constant`
- `Shape`
- `Gather`
- `Unsqueeze`
- `Concat`
- `Reshape`
- `Range`
- `Cast`
- `ConstantOfShape`
- `Expand`
- `Transpose`
- `Slice`
- `ReduceMax`
- `ArgMax`
- `Sigmoid`
- 二元算子 `Add` / `Mul` / `Sub` / `Div`

这里有两个细节：

- 只有当输入已经是常量时，`ConstantFolding` 才会折叠节点
- `ShapeSimplification` 也会复用这套能力，并且会借助静态 shape 信息继续折叠一部分节点

更直观一点，可以把它们理解成这些计算：

- `Constant`
  - `y = const`
- `Unsqueeze`
  - 在指定轴上插入大小为 1 的维度
- `Concat`
  - `y = concat(x0, x1, ..., axis)`
- `ConstantOfShape`
  - `y = fill(value, shape)`
- `Expand`
  - `y = broadcast(x, shape)`
- `Transpose`
  - `y = permute(x, perm)`
- `Slice`
  - `y = x[start:end:step]`
- `ReduceMax`
  - `y = max(x, axis)`
- `ArgMax`
  - `y = argmax(x, axis)`
- `Sigmoid`
  - `y = 1 / (1 + e^{-x})`

### 为什么要把结果 materialize 成 initializer

这里的 `materialize` 可以直接理解成“落成固定值”或者“变成可直接复用的数据”。

因为这些结果已经在初始化阶段算好了，后面推理时只需要直接取用。

把结果写成 initializer 的好处是：

- 后续 pass 可以把它当常量使用
- runtime 加载时可以直接当固定数据装载
- 推理时少算一遍，整体更快

## 6. DeadNodeCleanup：删掉从输出看不到的东西

### 它解决什么问题

图优化做完以后，图里常常会留下“没人再用”的节点。

举个简单例子：

- 原来有一条路径 A -> B -> C
- 优化后，C 直接改成从 A 走了
- 那么 B 就变成了多余节点

这些节点继续留着没有意义，只会让图更乱。

### 代码怎么做

`RunDeadNodeCleanup(...)` 做的是“从输出往回找谁还活着”。

它的步骤很简单：

1. 从最终输出开始
2. 找到是谁生成了这个输出
3. 再看这个生成节点用了哪些输入
4. 继续往回追，直到不能再追
5. 没被追到的节点，就删掉

这个过程也可以理解成一次“图上的反向可达性分析”。

它内部先构建 `output -> producer` 的反查表，再从最终输出回溯到所有依赖节点。

### 为什么它很重要

很多优化都会留下临时痕迹。

`DeadNodeCleanup` 的作用就是把这些残留清掉，保证后续 pass 看到的是一张干净的图。

## 7. ConvRewrite：当前最有效果的优化

### 它解决什么问题

`yolov8n.onnx` 里有大量重复模式：

- `Conv -> Sigmoid -> Mul`

这个模式对应 YOLO 里常见的激活块。它在图里出现很多次，所以只要能融合，收益就很明显。

### 代码怎么匹配

这部分由 `RunConvRewrite(...)` 和 `MatchConvSiLUFusion(...)` 完成。

匹配条件比较严格：

- 当前节点必须是 `Conv`
- `Conv` 必须只有一个输出
- 这个输出必须被 `Sigmoid` 和 `Mul` 两个节点消费
- `Sigmoid` 的输出只能被这个 `Mul` 使用
- `Mul` 的两个输入必须正好是 `Conv` 输出和 `Sigmoid` 输出

满足条件后，就把这三节点融合成一个新节点：

- `op_type = "ConvSiLU"`
- 输入沿用 `Conv` 的输入
- 输出沿用 `Mul` 的输出
- 属性沿用 `Conv` 的属性

这三个节点合在一起，可以直接写成一个更直观的公式：

- `z = Conv(x, w, b)`
- `s = sigmoid(z)`
- `y = z * s`

融合后就变成：

- `y = ConvSiLU(x, w, b)`

如果把 `SiLU` 直接写成公式，就是：

- `SiLU(x) = x * sigmoid(x)`
- `sigmoid(x) = 1 / (1 + e^{-x})`

### 为什么这样融合可行

因为这个阶段我们已经在项目里实现了对应的 `ConvSiLU` CPU kernel。

也就是说，图优化和 kernel 实现是配套的：

- 图里把三个节点收成一个
- 执行时再由一个 kernel 直接完成这个复合语义

对应的 kernel 注册在 `src/runtime/nn_kernels.cc` 的 `RegisterNnKernels(...)` 里。

当前项目里已经实现的 YOLO 相关 CPU kernel 主要包括：

- `Conv`
- `ConvSiLU`
- `MaxPool`
- `Resize`
- `Softmax`

基础算子和 shape 算子则分别放在：

- `src/runtime/basic_kernels.cc`
- `src/runtime/elementwise_kernels.cc`
- `src/runtime/shape_kernels.cc`

如果你想看完整实现，也可以直接看这个仓库的代码：`https://github.com/WWandP/miniONNXRuntime`

`MatchConvSiLUFusion(...)` 负责做模式匹配，`RunConvRewrite(...)` 负责真正重写图。前者判断“能不能融”，后者决定“怎么替换”。

### 当前效果

按本地日志统计，在 `yolov8n.onnx` 上：

- 融合了 `57` 组 `Conv + Sigmoid + Mul`
- 节点数从 `459` 降到 `244`
- 运行时间有明显下降

这里更关键的是这个模式非常重复，所以融合收益非常集中。

## 8. ShapeSimplification：把 shape/view 噪声和恒等算子清掉

### 它解决什么问题

很多 ONNX 图里会有一堆“看起来在做事，实际上没改变语义”的节点，例如：

- identity `Transpose`
- identity `Reshape`
- full-range `Slice`
- 单输入 `Concat`
- `x + 0`
- `x * 1`
- `x / 1`

这些节点通常来自导出器、shape 处理逻辑或图规范化过程。

### 代码怎么做

`RunShapeSimplification(...)` 会按节点类型逐个尝试简化。

它做的事情主要有两类：

1. 直接消掉节点，并把输出重写回输入
2. 如果节点结果可以静态算出，就直接把它变成固定值写回图里

这一段会复用 `FoldConstantNode(...)` 的能力，所以它既能做结构上的删减，也能做一部分静态求值。

这一节里最容易直接理解的就是这些“恒等条件”：

- `Transpose`
  - 如果 `perm = [0, 1, 2, ...]`，就表示没变
- `Reshape`
  - 如果 `shape_in == shape_out`，就表示没变
- `Slice`
  - 如果 `start = 0`、`end = dim`、`step = 1`，就表示没切掉任何东西
- `Concat`
  - 如果只有一个输入，就等于原样返回
- `Add`
  - `x + 0 = x`
- `Sub`
  - `x - 0 = x`
- `Mul`
  - `x * 1 = x`
- `Div`
  - `x / 1 = x`

### 典型规则

- `Transpose`
  - 如果 permutation 是 identity，就删掉
- `Reshape`
  - 如果 reshape 前后 shape 等价，就删掉
- `Slice`
  - 如果 slice 覆盖完整范围，就删掉
- `Concat`
  - 如果只有一个输入，就直接改写引用
- `Add`
  - `x + 0 -> x`
- `Sub`
  - `x - 0 -> x`
- `Mul`
  - `x * 1 -> x`
  - 在 shape 已知时，`x * 0` 可以折成全零张量
- `Div`
  - `x / 1 -> x`

### 为什么这个 pass 还有价值

单个节点看起来没什么，但它有两个作用：

- 让图更干净
- 给后续结构性优化腾空间

特别是在像 YOLO 这种图里，shape/view 节点很多。把这些节点整理掉以后，图结构更容易读，后续做融合也更容易定位模式。

## 9. 代码是怎么串起来的

图优化的主入口是 `tools/optimize_model.cc`。

它的执行链路很清楚：

```text
LoadOnnxGraph
  -> PrintGraphSnapshot(before)
  -> OptimizeGraph
  -> PrintGraphSnapshot(after)
  -> Session::Run
  -> YOLO 后处理
```

也就是说，优化位于实际推理之前。

### 入口参数

`GraphOptimizationOptions` 目前暴露了几个开关：

- `enable_constant_folding`
- `enable_dead_node_cleanup`
- `enable_conv_silu_fusion`
- `enable_shape_simplification`
- `verbose`

这让后续调试很方便，可以单独关掉某个 pass 看图怎么变化。

### 统计信息

`GraphOptimizationSummary` 会记录：

- 优化前节点数
- 优化后节点数
- 跑了多少个 pass
- 实际应用了哪些 pass

配合 trace 输出，可以比较直观地观察每一轮优化到底做了什么。

`OptimizeGraph(...)` 里会按顺序调用这些 pass，并把每一步的执行结果累计到 `GraphOptimizationSummary` 里。这个 summary 里最直观的两个字段是 `nodes_before` 和 `nodes_after`。

这里也可以简单理解成：

- 先把能直接算出来的节点提前算掉
- 再把没用了的节点清掉
- 然后把很常见的模式合并掉
- 最后把图里明显多余的 shape/view 节点整理掉

## 10. 图优化的效果

### 图数量变化

| 阶段 | 节点数 | 变化 |
| --- | ---: | ---: |
| 图优化前 | 459 | - |
| 第一版图优化后 | 359 | -100 |

### 推理耗时变化

| 阶段 | `Session::Run` | 变化 |
| --- | ---: | ---: |
| 图优化前 | 13044.674 ms | - |
| 第一版图优化后 | 12479.051 ms | -565.623 ms |
| 相对提升 | - | 约 4.3% |

### 怎么理解这组结果

这组数据说明两件事：

- `ConstantFolding` 先把一批静态子图折掉了，所以节点数直接下降
- 节点少了以后，`Session::Run(...)` 也跟着变快了

对后续迭代来说，这个结果也很重要，因为它说明图优化已经能稳定转化成实际收益。

## 11. 这版图优化的边界

当前实现已经能解决一批很典型的问题，也仍然保留着阶段性版本的边界。

现在还没有做的事情包括：

- 通用 pass manager
- 更完整的 shape inference
- 更系统的 alias / 生命周期分析
- buffer reuse 和内存规划
- 更丰富的 pattern fusion
- 更广泛的模型兼容性
## 12. 总结

可以把当前图优化理解成三层：

- 第一层，`ConstantFolding` 把能静态算的先算掉
- 第二层，`ShapeSimplification` 把没必要执行的 shape/view 节点去掉
- 第三层，`ConvRewrite` 把最频繁的复合模式融合成一个 kernel

最后再用 `DeadNodeCleanup` 把遗留垃圾收尾。

可以概括成：

**把一张静态图逐步变成一张更适合执行的图。**

这也是图优化在推理引擎里最实用的价值。对 `miniONNXRuntime` 来说，当前的目标集中在与yolov8n.onnx相关的图优化，后续工作会围绕更多 pattern fusion、内存规划和更通用的执行框架展开。当前这一版只覆盖了关于yolo的最明显、最容易落地的图优化，完整的onnxruntime的优化内容还是非常多的，这里仅对几类做了介绍。
