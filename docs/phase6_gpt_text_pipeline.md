# Phase6 GPT Text Pipeline

这份文档对应当前面向用户保留的两条 GPT 入口：

- `./scripts/run_phase.sh phase6`
- `./scripts/run_phase.sh phase6-kv`

目标不是把 GPT 路线讲成完整 LLM 产品，而是把这条教学主线讲清楚：

- baseline 文本生成怎么跑
- KV cache 文本生成怎么跑
- 输出里哪些字段最值得看
- 中间日志分别代表什么

## 适合看什么

`phase6` 更适合看：

- prompt 文本怎么变成 token ids
- `miniort_run_gpt` 怎么做多步 greedy 生成
- 最终输出文本和 token ids 是否合理
- 当前 macOS provider 路径有没有跑通

`phase6-kv` 更适合看：

- GPT-2 的 `prefill` / `decode` 双图怎么接起来
- KV cache 为什么能避免重复计算整段历史 token
- provider 分布和执行节点数如何变化

## 运行前提

需要这些本地文件：

- `models/gpt2/model.kv_prefill.onnx`
- `models/gpt2/model.kv_decode.onnx`
- `models/gpt2/vocab.json`
- `models/gpt2/merges.txt`
- `examples/gpt2_tiny/story_prompt.txt`

仓库当前的统一脚本会自动检查这些文件是否存在。

## 模型来源与导出环境

当前 GPT 资产默认来自：

- 模型：`openai-community/gpt2`
- 来源：Hugging Face
- 本地目录：`models/gpt2/`

仓库里已经有拉取和导出脚本：

- [scripts/fetch_gpt2.sh](/Volumes/ww/code/onnxruntime/minionnxruntime/scripts/fetch_gpt2.sh)

它会做两件事：

- 下载 Hugging Face 原始模型与 tokenizer 资产
- 在 Python 环境满足依赖时导出 ONNX 文件

### Python 环境要求

如果你需要自己重导模型，至少要准备：

- Python 3
- `torch`
- `transformers`
- `onnx`

如果还想自动生成简化后的 ONNX，则额外需要：

- `onnxsim`

脚本当前会自动寻找一个带这些依赖的 Python 解释器。也可以显式指定：

```bash
PYTHON_BIN=/path/to/python ./scripts/fetch_gpt2.sh
```

### 推荐做法

对这个项目，更推荐的方式是：

- 提交导出脚本和文档
- 不把大 ONNX 模型文件放进 git 历史
- 通过脚本拉取或本地导出模型

### 下载与导出命令

只下载 Hugging Face 资产并尝试导出基础 ONNX：

```bash
./scripts/fetch_gpt2.sh
```

同时导出 KV cache 图：

```bash
EXPORT_KV_CACHE=1 ./scripts/fetch_gpt2.sh
```

### 默认文件约定

脚本和 phase 入口当前约定这些文件名：

- `model.onnx`
- `model.kv_prefill.onnx`
- `model.kv_decode.onnx`
- `vocab.json`
- `merges.txt`

其中：

- `phase6` 默认使用 `model.kv_prefill.onnx`
- `phase6-kv` 使用 `model.kv_prefill.onnx + model.kv_decode.onnx`

## 推荐命令

### 1. baseline

```bash
./scripts/run_phase.sh phase6
```

默认行为：

- 读取 `examples/gpt2_tiny/story_prompt.txt`
- 使用 `models/gpt2/model.kv_prefill.onnx`
- 生成 `48` 个新 token
- 走默认 macOS provider 路径

### 2. KV cache

```bash
./scripts/run_phase.sh phase6-kv
```

默认行为：

- 读取 `examples/gpt2_tiny/story_prompt.txt`
- prefill 使用 `models/gpt2/model.kv_prefill.onnx`
- decode 使用 `models/gpt2/model.kv_decode.onnx`
- 生成 `48` 个新 token
- 走默认 macOS provider 路径

### 3. 常见自定义方式

只改 prompt 文件：

```bash
GPT_PROMPT_FILE=examples/gpt2_tiny/story_prompt.txt ./scripts/run_phase.sh phase6
```

只改生成长度：

```bash
GPT_GENERATE=32 ./scripts/run_phase.sh phase6
```

同时改 prompt 和生成长度：

```bash
GPT_PROMPT_FILE=examples/gpt2_tiny/story_prompt.txt GPT_GENERATE=32 ./scripts/run_phase.sh phase6-kv
```

## 运行链路

### phase6

可以把 `phase6` 理解成：

```text
prompt text
  -> GPT-2 tokenizer encode
  -> input token ids
  -> model.kv_prefill.onnx
  -> greedy generate N steps
  -> full token ids
  -> tokenizer decode
  -> output text
```

这里没有 KV cache，所以每一步续写都会基于更长的 token 序列再次跑整张 `prefill` 图。

### phase6-kv

可以把 `phase6-kv` 理解成：

```text
prompt text
  -> tokenizer encode
  -> prefill model
  -> logits + present KV
  -> pick next token
  -> decode model
  -> new logits + updated KV
  -> repeat
```

它的关键点是：

- 第一步用整段 prompt 建立 cache
- 后面每一步只喂新 token 和历史 cache
- 不再重复重算全部历史 token

## 输出怎么看

`phase6` / `phase6-kv` 的输出现在和其他 phase 一样，先有 banner 和 step，再打印运行结果。

一个典型输出会包含这些部分：

### 1. phase banner

```text
============================================================
[phase6] GPT Text Generation
goal: 看 GPT-2 在 macOS provider 路径下如何完成文本 prompt 到文本输出。
============================================================
```

这里说明你当前跑的是哪条 GPT 路线，以及这条 phase 的观察重点。

### 2. step 标题

例如：

```text
[1/4] Resolve Prompt And Tokenizer
[2/4] Load Model Graphs
[3/4] Run Generation
[4/4] Summarize Outputs
```

这几步分别表示：

- `Resolve Prompt And Tokenizer`
  - 读取 prompt
  - 用 GPT-2 tokenizer 编码成 token ids
- `Load Model Graphs`
  - 加载 ONNX 图
  - `phase6` 默认加载 `model.kv_prefill.onnx`
  - `phase6-kv` 会同时加载 prefill / decode 两张图
- `Run Generation`
  - 实际执行推理与 greedy 续写
- `Summarize Outputs`
  - 输出文本、token ids、执行摘要

### 3. `last_token_topk`

示例：

```text
last_token_topk
  - token_id=262 logit=...
  - token_id=286 logit=...
```

表示：

- 取最后一个位置的 logits
- 按分数从高到低列出 top-k 候选 token

它的主要用途是：

- 看模型最后一步更偏向哪些 token
- 做 greedy 生成调试
- 对比不同 provider / 不同路径下 top-k 是否一致

### 4. `full_token_ids`

示例：

```text
full_token_ids:
[7454, 2402, 257, ...]
```

表示：

- 初始 prompt token ids
- 加上后续生成出来的新 token ids
- 这是最终完整 token 序列，不只是新增部分

### 5. `input_text`

示例：

```text
input_text:
Once upon a time
```

表示原始输入 prompt 文本。

### 6. `input_token_ids`

示例：

```text
input_token_ids:
[7454, 2402, 257, 640]
```

表示 tokenizer 编码后的原始 prompt token 序列。

它只对应输入，不包含新增 token。

### 7. `output_text`

示例：

```text
output_text:
Once upon a time, the world was ...
```

表示最终解码后的完整文本。

这里通常包含：

- 原始 prompt
- 新生成的续写内容

如果看到重复或机械延续，不一定是 runtime 错误，很多时候只是当前 greedy decoding 的自然结果。

### 8. `summary`

示例：

```text
summary executed=139618 skipped=0 materialized_outputs=0
```

这三项分别表示：

- `executed`
  - 成功执行了多少个节点
- `skipped`
  - 有多少节点因为缺 kernel 或执行失败被跳过
- `materialized_outputs`
  - 有多少输出是 runtime 根据元数据补出来的 placeholder，而不是 kernel 真算出来的

对 GPT 主线来说，理想情况通常是：

- `skipped=0`
- `materialized_outputs=0`

这说明主干没有靠占位输出硬撑过去。

### 9. `session.run end`

示例：

```text
session.run end executed=139618 skipped=0 materialized_outputs=0 released_tensors=0
```

这是更完整的运行摘要。

额外多出来的：

- `released_tensors`
  - 运行过程中按生命周期释放了多少临时 tensor

当前 GPT 默认入口没有打开更激进的 eviction 策略时，这个值可能是 `0`。

### 10. `provider execution summary`

示例：

```text
provider execution summary
  - Accelerate: visited=26802 executed=26802 skipped=0 materialized_outputs=0
  - CPU: visited=112816 executed=112816 skipped=0 materialized_outputs=0
```

这一段最重要。

它表示每个 provider 实际承担了多少节点：

- `visited`
  - 跑到了多少节点
- `executed`
  - 真正执行成功了多少节点
- `skipped`
  - 该 provider 下被跳过的节点数
- `materialized_outputs`
  - 该 provider 路径下占位输出的次数

如果看到：

- `Accelerate` 有值
- `CPU` 也有值

说明当前是混合 provider 路径，不是全图都在 Apple Accelerate 上。

这在当前项目里是正常的。

### 11. `[result]`

示例：

```text
[result] phase6 complete
  你现在看到的是 baseline 文本生成视角。
```

这表示 phase 结束，提醒你当前看到的是哪种视角：

- baseline
- 或 KV cache

## 如何判断是否正常

一个“基本正常”的 GPT 输出通常满足：

- 有 `input_text`
- 有 `output_text`
- 有 `full_token_ids`
- `summary skipped=0`
- `materialized_outputs=0`
- `provider execution summary` 中至少有 `CPU`

在 macOS 上，如果 `Accelerate` 也出现，说明 Apple provider 已经参与执行。

## 当前边界

这条 GPT 路线当前仍然是教学型入口，不是完整对话产品。

它还没有这些能力：

- sampling
- repetition penalty
- 通用 chat template
- C++ 内嵌 tokenizer
- 更稳定的对话接口抽象
- 更严格的逐元素 logits dump 与对齐

所以更适合拿来做：

- 文本模型执行演示
- provider / KV cache 对比
- runtime 输出解释

而不是直接当通用聊天产品入口。

## 相关文档

- [docs/phase5_gpt2_mac_accelerate_baseline.md](/Volumes/ww/code/onnxruntime/minionnxruntime/docs/phase5_gpt2_mac_accelerate_baseline.md)
- [docs/phase5_gpt2_kv_cache_bringup.md](/Volumes/ww/code/onnxruntime/minionnxruntime/docs/phase5_gpt2_kv_cache_bringup.md)
- [docs/phase5_gpt2_development_journal.md](/Volumes/ww/code/onnxruntime/minionnxruntime/docs/phase5_gpt2_development_journal.md)
