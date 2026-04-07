# 图优化记录

这份文档记录 `phase4` 里围绕 `yolov8n.onnx` 做过的图优化工作。

## 优化总表

| 优化项 | 主要算子 / 模式 | 当前效果 | 典型收益 |
| --- | --- | --- | --- |
| `ConstantFolding` | `Constant`、`Shape`、`Gather`、`Unsqueeze`、`Concat`、`Reshape`、`Range`、`Cast`、`ConstantOfShape`、`Add`、`Mul`、`Sub`、`Div` | 折叠静态常量子图 | 节点数减少 `100` 个 |
| `DeadNodeCleanup` | 死节点、无用 initializer、无用 `value_info` | 作为折叠和重写后的清理步骤 | 当前模型上无额外节点删除 |
| `ShapeSimplification` | `Transpose`、`Reshape`、`Slice`、`Concat`、`Add`、`Sub`、`Mul`、`Div` | 去掉 shape/view 噪声和恒等算子 | 额外减少 `1` 个节点 |
| `ConvRewrite` | `Conv -> Sigmoid -> Mul` | 融合成 `ConvSiLU` | 融合 `57` 组，节点数降到 `244`，运行时间约降到 `8.37 s` |

## 目标

先保持 phase3 的推理结果不变，再尽量减少模型图对 runtime 提出的工作量。

当前图优化管线如下：

- `ConstantFolding`
- `DeadNodeCleanup`
- `ConvRewrite`
- `ShapeSimplification`

## 各个 Pass

### 1. ConstantFolding

这个 pass 会把只依赖常量的子图折叠成新的 initializer。

当前已经支持的折叠对象包括：

- `Constant`
- `Shape`
- `Gather`
- `Unsqueeze`
- `Concat`
- `Reshape`
- `Range`
- `Cast`
- `ConstantOfShape`
- 以及所有输入都为常量时的二元数值算子：
  - `Add`
  - `Mul`
  - `Sub`
  - `Div`

另外，这一版还会利用静态 shape 元信息，所以有些 `Shape` 节点即使不是从 runtime initializer 直接读出来，也能被静态折叠掉。

在 `yolov8n.onnx` 上的效果：

- 折叠掉 `100` 个节点
- 把这些结果 materialize 成新的 initializer
- 清掉了 detection head 周围一大批静态 shape 节点

### 2. DeadNodeCleanup

这个 pass 会从最终输出反向回溯，把不再可达的节点、initializer 和 `value_info` 清掉。

它通常在常量折叠和图重写之后更有意义。

在当前 `yolov8n` 跑法里，这个 pass 暂时没有再额外删掉节点，但它仍然是后续图重写的基础清理步骤。

### 3. ShapeSimplification

这个 pass 主要负责去掉 shape/view 类的无效节点，以及一部分简单代数恒等式。

当前规则包括：

- `Transpose`
  - 删除 identity permutation
- `Reshape`
  - 删除 identity reshape
- `Slice`
  - 删除 full-range slice
- `Concat`
  - 删除单输入 `Concat`
- `Add`
  - `x + 0 -> x`
- `Sub`
  - `x - 0 -> x`
- `Mul`
  - `x * 1 -> x`
  - 在输出 shape 可静态解析时，`x * 0 -> 0`
- `Div`
  - `x / 1 -> x`

在 `yolov8n.onnx` 上的效果：

- `Transpose / Reshape / Slice / Concat` 这批规则先用来清理 shape/view 噪声
- `Add / Mul / Sub / Div` 这批规则再进一步去掉一些恒等算子
- 实际命中数量不大，但它们能让图更干净，也能给后续重写留出空间

### 4. ConvRewrite

这是目前收益最大的一步。

这个 pass 专门识别 YOLO 里常见的：

- `Conv -> Sigmoid -> Mul`

然后把这三节点融合成一个 `ConvSiLU` 节点。

运行时侧已经有对应的 `ConvSiLU` kernel，所以融合后的节点可以直接执行，不需要再把中间的 `Sigmoid` 和 `Mul` 单独 materialize 出来。

在 `yolov8n.onnx` 上的效果：

- 融合了 `57` 组 `Conv + Sigmoid + Mul`
- 节点数从 `459` 降到 `244`
- 运行时间从大约 `13.35 s` 降到 `8.37 s`

## 为什么这个模型最适合做 Conv 融合

`yolov8n.onnx` 里有大量重复的激活块，典型形式就是：

- `Conv + Sigmoid + Mul`

这个模式在当前模型里出现得非常频繁，所以它带来的收益也是最大的。

## 性能样本

下面这些数据都是本地日志里的代表性样本，实际每次运行会有一点波动。

### 图优化前

- 节点数：`459`
- `Session::Run` 总耗时：大约 `11.88 s` 到 `13.35 s`
- 主要热点：
  - `Conv`：大约 `6.31 s` 到 `7.05 s`
  - `Mul`：大约 `4.12 s` 到 `4.67 s`
  - `Sigmoid`：大约 `0.17 s` 到 `0.24 s`
  - `Transpose`：大约 `0.42 s` 到 `0.50 s`
  - `Add`：大约 `0.34 s` 到 `0.40 s`
  - `Slice`：大约 `0.33 s` 到 `0.39 s`

### 第一轮图优化后

- `ConstantFolding` 折叠了 `100` 个节点
- `ShapeSimplification` 在某些运行里还能额外去掉 `1` 个节点
- 节点数从 `459` 降到 `358` 或 `359`
- `Session::Run` 大约到 `12.48 s` 左右，后续运行里也会有波动

### ConvRewrite 后

- `ConvRewrite` 融合了 `57` 组 `Conv + Sigmoid + Mul`
- 节点数从 `459` 降到 `244`
- `Session::Run` 大约降到 `8.37 s`
- 剩余热点里，`ConvSiLU` 仍然占大头，但整体已经明显短很多

### 激活块对比

融合前，激活块的总成本大致是：

- `Conv + Sigmoid + Mul`：大约 `11.89 s`

融合后，对应成本大致变成：

- `ConvSiLU` 加上剩余未融合的几个算子：大约 `6.98 s`

这就是当前图优化里最明显的收益来源。

### 输出稳定性

- 检测框数量保持 `5` 个
- JSON 输出和可视化输出的结构没有变化

## 小结

- `ConstantFolding` 负责把能静态求值的常量子图先折掉
- `ShapeSimplification` 负责清掉 shape/view 噪声和一部分恒等算子
- `ConvRewrite` 是当前最值钱的优化，直接缩短了 YOLO 主干里的激活块
- 现在这份 phase4 可以作为后续优化迭代的基线
