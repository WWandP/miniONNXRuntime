# Phase5 GPT-2 CPU Bring-up

这份记录基于：

- 模型：`openai-community/gpt2`
- ONNX：`models/gpt2/model.sim.onnx`
- 入口：`miniort_run_gpt`

目标是把当前 phase5 的最小 `ExecutionProvider` 路线，从原本偏 YOLO/CNN 的图推进到标准 GPT-2 的 CPU 前向。

## 已完成内容

### 1. 模型准备

- 下载了标准 GPT-2 本地模型目录
- 导出并简化得到：
  - [models/gpt2/model.sim.onnx](/Volumes/ww/code/onnxruntime/minionnxruntime/models/gpt2/model.sim.onnx)

### 2. 缺口分析

整理了 GPT-2 ONNX 与当前 runtime 的差距：

- [docs/gpt2_onnx_gap_analysis.md](/Volumes/ww/code/onnxruntime/minionnxruntime/docs/gpt2_onnx_gap_analysis.md)

### 3. CPU kernel 补齐

为 GPT-2 主干补了第一批关键 CPU 能力：

- `Squeeze`
- `Tanh`
- `Pow`
- `Where`
- `LayerNormalization`
- 通用 `Gather`
- batched `MatMul`

同时补了最小 bool mask materialization，使 attention mask 路径可执行。

### 4. GPT 专用入口

新增了 phase5 风格的 GPT 调试入口：

- [tools/run_gpt.cc](/Volumes/ww/code/onnxruntime/minionnxruntime/tools/run_gpt.cc)

对应可执行文件：

- `./build_phase4/miniort_run_gpt`

支持：

- `--tokens`
- `--cpu-only`
- `--strict`
- `--start-node`
- `--max-nodes`
- `--top-k`
- `--verbose`
- `--quiet`

## 逐段推进结果

同一组输入 token：

- `464,3616,286,1204,318`

在 `--strict --cpu-only` 下推进：

- `max_nodes=64`：通过
- `max_nodes=128`：通过
- `max_nodes=256`：通过
- `max_nodes=512`：通过
- `max_nodes=1024`：通过
- 全图 `1358` 节点：通过

全图结果记录在：

- [outputs/gpt2/gpt2_full.txt](/Volumes/ww/code/onnxruntime/minionnxruntime/outputs/gpt2/gpt2_full.txt)

关键结论：

- `executed=1358`
- `skipped=0`
- `materialized_outputs=0`

这意味着当前 CPU runtime 已经可以完整执行这份 GPT-2 ONNX 图。

## 数值对齐结果

同一输入下，对比了：

- PyTorch
- ONNX Runtime
- miniONNXRuntime

PyTorch 与 ONNX Runtime 的最后一个位置 top-5 token 完全一致：

- `407`
- `284`
- `262`
- `326`
- `257`

PyTorch vs ONNX Runtime 的最大绝对误差：

- `0.0003662109375`

miniONNXRuntime 全图跑完后的 `last_token_topk` 也是：

- `407`
- `284`
- `262`
- `326`
- `257`

因此当前结论可以写成：

- miniONNXRuntime 已经在这份 GPT-2 简化 ONNX 上实现 CPU 全图跑通
- 最终 top-k 排序与 PyTorch / ONNX Runtime 对齐

## 第一版语义化输入输出

为了让输入和输出不再停留在 token id 层，新增了一个文本包装脚本：

- [tools/run_gpt_text.py](/Volumes/ww/code/onnxruntime/minionnxruntime/tools/run_gpt_text.py)

这个脚本做的事情：

1. 用本地 Hugging Face tokenizer 把 prompt 编码成 token ids
2. 一次调用 `miniort_run_gpt --generate N`
3. 在单进程内复用同一个 `Session` 做多轮贪心续写
4. 把生成后的 token ids 再解码回文本

它不是最终产品形态，但足以作为“第一版有语义输入输出”的最小闭环。

示例命令：

```bash
/Volumes/ww/miniconda3/envs/norm/bin/python tools/run_gpt_text.py \
  --model-dir models/gpt2 \
  --model models/gpt2/model.sim.onnx \
  --binary ./build_phase4/miniort_run_gpt \
  --prompt "The meaning of life is" \
  --max-new-tokens 2 \
  --cpu-only \
  --strict
```

示例结果：

```text
input_text:
The meaning of life is

input_token_ids:
[464, 3616, 286, 1204, 318]

full_token_ids:
[464, 3616, 286, 1204, 318, 407, 262]

output_text:
The meaning of life is not the
```

当前版本已经把最粗的开销去掉了：生成多个 token 时，不再为每个 token 单独启动一次二进制，也不再为每个 token 重新加载整份 ONNX 模型。现在的边界变成了“同一进程内仍然是每一步都重跑整段 prompt+history 的前向”，所以语义闭环已经更顺了，但性能仍然受限于没有 KV cache。

## 当前边界

目前这条 GPT-2 路线已经具备：

- ONNX 加载
- CPU 全图执行
- 最终 logits top-k 摘要
- 文本 prompt 到文本续写的最小包装链路

当前还没有系统化做的部分：

- 更严格的逐元素 logits 对齐导出
- 更完整的文本生成采样策略
- KV cache / 增量解码
- tokenizer 完整内嵌到 C++ runtime
- 单进程内进一步沉淀成面向文本生成的稳定 C++ 接口

## 建议的下一步

1. 把 `run_gpt_text.py` 的 encode/decode 能力逐步向 C++ 入口内收
2. 给 `miniort_run_gpt` 增加可选 logits dump，做更严格的数值对齐
3. 如果继续推进性能，再考虑 KV cache 和 provider 优化
