# GPT-2 ONNX Gap Analysis

基于简化后的模型：

- [models/gpt2/model.sim.onnx](/Volumes/ww/code/onnxruntime/minionnxruntime/models/gpt2/model.sim.onnx)

目标不是评估模型效果，而是盘点当前 `miniONNXRuntime` 距离跑通标准 GPT-2 前向还差哪些运行时能力。

## 模型概况

- 输入：`input_ids`，形状 `[batch, sequence]`，类型 `int64`
- 输出：`logits`，形状 `[batch, sequence, 50257]`，类型 `float32`
- 总节点数：`1358`
- initializer 数量：`171`

简化后模型的主要算子分布：

- `Unsqueeze`: 242
- `Gather`: 195
- `Concat`: 146
- `Reshape`: 146
- `Shape`: 133
- `Slice`: 72
- `Transpose`: 60
- `Squeeze`: 60
- `Add`: 49
- `Gemm`: 48
- `Mul`: 48
- `LayerNormalization`: 25
- `MatMul`: 25
- `Pow`: 24
- `Split`: 12
- `Cast`: 12
- `Div`: 12
- `Sub`: 12
- `Where`: 12
- `Softmax`: 12
- `Tanh`: 12
- `Range`: 1

这份统计说明两件事：

- `onnxsim` 已经把图清理得更适合阅读，但 GPT-2 仍然是一个 shape-heavy 图，尤其大量依赖 `Gather/Unsqueeze/Reshape/Concat`
- 真正决定能否跑通的主干不是节点总数，而是少数几个关键能力：embedding、LayerNorm、attention、GELU

## 当前 runtime 已具备的部分

从现有内核实现看，项目已经支持这些 GPT-2 图中会用到的算子：

- 已支持：`Add`、`Sub`、`Mul`、`Div`、`Cast`
- 已支持：`Shape`、`Gather`、`Unsqueeze`、`Concat`、`Reshape`、`Range`、`Split`、`Transpose`、`Slice`
- 已支持：`MatMul`、`Gemm`、`Softmax`

相关实现位置：

- [src/runtime/elementwise_kernels.cc](/Volumes/ww/code/onnxruntime/minionnxruntime/src/runtime/elementwise_kernels.cc)
- [src/runtime/shape_kernels.cc](/Volumes/ww/code/onnxruntime/minionnxruntime/src/runtime/shape_kernels.cc)
- [src/runtime/nn_kernels.cc](/Volumes/ww/code/onnxruntime/minionnxruntime/src/runtime/nn_kernels.cc)

这意味着当前项目已经有一部分“GPT-2 主干骨架”，并不是从零开始。

## 算子覆盖对比

下面按 `model.sim.onnx` 中实际出现的算子做一个更直接的覆盖表。

### 已支持，可直接复用

| Op | Count | 说明 |
|---|---:|---|
| `Add` | 49 | 逐元素广播实现已具备 |
| `Sub` | 12 | 逐元素广播实现已具备 |
| `Mul` | 48 | 逐元素广播实现已具备 |
| `Div` | 12 | 逐元素广播实现已具备 |
| `Cast` | 12 | 当前图里主要用于基础类型转换 |
| `Shape` | 133 | 形状读取已支持 |
| `Unsqueeze` | 242 | 当前实现可复用 |
| `Concat` | 146 | 当前实现可复用 |
| `Reshape` | 146 | 当前实现可复用 |
| `Range` | 1 | 当前实现可复用 |
| `Split` | 12 | 当前实现可复用 |
| `Transpose` | 60 | 多维 `float32/int64` 转置已支持 |
| `Slice` | 72 | 当前实现已支持正向切片主路径 |
| `Softmax` | 12 | attention score softmax 主体可复用 |

### 部分支持，需要扩展

| Op | Count | 当前状态 | 主要问题 |
|---|---:|---|---|
| `Gather` | 195 | 仅部分支持 | 只支持 `axis=0` 且仅限标量/1D，不足以做 embedding lookup |
| `MatMul` | 25 | 仅部分支持 | 只支持 2D `float32`，无法覆盖 attention 的 batch/head 维度 |
| `Gemm` | 48 | 基本可用但有限 | 仅 2D `float32`，可覆盖多数线性层，但仍偏教学实现 |

### 当前完全缺失

| Op | Count | 典型用途 |
|---|---:|---|
| `Squeeze` | 60 | shape 子图整理、attention 支路 |
| `LayerNormalization` | 25 | 每层 block 的 pre-norm 和最终 ln_f |
| `Pow` | 24 | `gelu_new` 中的 `x^3` |
| `Where` | 12 | causal mask 选择逻辑 |
| `Tanh` | 12 | `gelu_new` 的核心子表达式 |

从节点覆盖率看，当前 runtime 已经能“认得”大多数 GPT-2 图中的常见名字，但真正阻断主干的是少数几个高价值缺口。

## 完全缺失的关键能力

下面这些算子在 `model.sim.onnx` 里存在，但当前 runtime 没有对应 kernel：

- `LayerNormalization`
- `Pow`
- `Tanh`
- `Where`
- `Squeeze`

其中优先级最高的是：

- `LayerNormalization`
  - GPT-2 每个 block 会反复使用，图里有 `25` 个
  - 没有它，主干根本走不通
- `Tanh`
  - GPT-2 的 `gelu_new` 最终会展开成包含 `Tanh` 的表达式
- `Pow`
  - `gelu_new` 里需要 `x^3`
- `Where`
  - attention mask 路径依赖它
- `Squeeze`
  - 图里出现 `60` 次，缺失会频繁阻断 shape/attention 支路

## 名义支持但能力不够的部分

有些算子名字已经支持，但当前实现能力不足以跑 GPT-2。

### `Gather`

当前实现限制非常强：

- 只支持 `axis=0`
- 只支持标量或 1D tensor

实现里直接写了：

- [src/runtime/shape_kernels.cc:55](/Volumes/ww/code/onnxruntime/minionnxruntime/src/runtime/shape_kernels.cc#L55)

而 GPT-2 的 `Gather` 主要用于：

- token embedding 查表
- positional embedding 查表
- shape 子图中的更一般索引

要跑 GPT-2，`Gather` 至少要支持：

- 更高维输入
- 任意合法 `axis`
- embedding 场景下 `data=[vocab, hidden]`、`indices=[batch, seq]` 的输出拼接

### `MatMul`

当前 `MatMul` 只支持 `2D float32 tensors`：

- [src/runtime/nn_kernels.cc:20](/Volumes/ww/code/onnxruntime/minionnxruntime/src/runtime/nn_kernels.cc#L20)

但 GPT-2 attention 的 `Q @ K^T`、`attn @ V` 一般是 4D 语义：

- `[batch, heads, seq, head_dim]`
- `[batch, heads, seq, seq]`

所以即便图里名字叫 `MatMul`，现实现也接不住 attention 主干。

### `Gemm`

当前 `Gemm` 同样只支持 `2D float32 tensors`：

- [src/runtime/nn_kernels.cc:97](/Volumes/ww/code/onnxruntime/minionnxruntime/src/runtime/nn_kernels.cc#L97)

这对 GPT-2 的线性层通常够用，但前提是：

- 前面的 `Reshape/Transpose` 能把张量整理成 2D
- bias 广播和输出形状都符合现实现假设

`Gemm` 不是最先要补的点，但要注意它当前仍偏“教学实现”，不是通用版本。

### `Transpose`

当前 `Transpose` 对 `float32/int64` 多维转置是支持的：

- [src/runtime/shape_kernels.cc:352](/Volumes/ww/code/onnxruntime/minionnxruntime/src/runtime/shape_kernels.cc#L352)

这一点对 GPT-2 attention 是有价值的。它不是主要缺口。

### `Softmax`

当前 `Softmax` 是按任意 axis 做归一化：

- [src/runtime/nn_kernels.cc:485](/Volumes/ww/code/onnxruntime/minionnxruntime/src/runtime/nn_kernels.cc#L485)

这说明 attention score 的 softmax 主体大概率可复用。它也不是主要阻塞点。

## GPT-2 主干和缺口对应关系

从 GPT-2 前向主线看，当前缺口可以按模块理解：

### 1. 输入 embedding

需要：

- `Gather` 作为 embedding lookup
- `Add` 把 token embedding 和 positional embedding 相加

当前状态：

- `Add` 已有
- `Gather` 语义不够

结论：

- 输入阶段还不能跑通

### 2. LayerNorm

需要：

- `LayerNormalization`

当前状态：

- 完全缺失

结论：

- 每层 block 一开始就会卡住

### 3. Attention

需要：

- `Gemm` 生成 QKV
- `Split`
- `Reshape`
- `Transpose`
- 高维 `MatMul`
- `Where` 构造/应用 causal mask
- `Softmax`
- 再一次高维 `MatMul`

当前状态：

- `Split/Reshape/Transpose/Softmax` 基本有
- `Where` 缺失
- attention 所需的高维 `MatMul` 缺失

结论：

- attention 目前无法通过

### 4. MLP / GELU

需要：

- `Gemm`
- `Pow`
- `Mul`
- `Add`
- `Tanh`

当前状态：

- `Gemm/Mul/Add` 有
- `Pow/Tanh` 缺失

结论：

- MLP 激活部分无法通过

### 5. 最终 logits

需要：

- 输出前的最终 `LayerNormalization`
- 词表投影，通常是 `MatMul` 或 `Gemm`

当前状态：

- 最终 `LayerNormalization` 缺失

## 最小可行补齐顺序

如果目标是“尽快朝 GPT-2 靠近”，建议按下面顺序补：

1. `Squeeze`
2. `Tanh`
3. `Pow`
4. `LayerNormalization`
5. `Where`
6. 扩展 `Gather` 到 embedding 场景
7. 扩展 `MatMul` 到 3D/4D batch matmul

这样排序的原因：

- 前五项是纯 kernel 缺口，落地快
- `Gather` 和高维 `MatMul` 虽然工程量更大，但是真正决定 GPT-2 是否能过主干

## 结论

基于 [models/gpt2/model.sim.onnx](/Volumes/ww/code/onnxruntime/minionnxruntime/models/gpt2/model.sim.onnx)，当前项目离跑通标准 GPT-2 还差的不是“大量零散算子”，而是几个非常明确的能力缺口：

- 完全缺失：`LayerNormalization`、`Pow`、`Tanh`、`Where`、`Squeeze`
- 语义不足：`Gather`
- 维度能力不足：`MatMul`

反过来看，现有项目已经拥有不少可复用积木：

- `Add/Sub/Mul/Div/Cast`
- `Reshape/Transpose/Slice/Concat/Split/Range/Shape`
- `Gemm`
- `Softmax`

所以更准确的判断是：

- 这个项目还不能直接跑 GPT-2
- 但已经具备了一条清晰的、可迭代推进到 GPT-2 的实现路径
