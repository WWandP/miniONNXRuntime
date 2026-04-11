# Phase6 GPT Text Pipeline

这份记录描述的是在 `phase5` 的 CPU GPT-2 bring-up 之上，继续补齐“文本输入 -> 文本输出”最小闭环的阶段。

基线依赖：

- 模型：`models/gpt2/model.sim.onnx`
- C++ 入口：`miniort_run_gpt`
- Python 包装：`tools/run_gpt_text.py`

`phase5` 的重点是：

- 把 GPT-2 简化 ONNX 图完整跑通
- 补齐 CPU kernel 缺口
- 对齐最终 top-k

`phase6` 的重点则变成：

- 给文本模型一个专用入口
- 把 tokenizer 前后处理接上
- 从 token 级调试推进到语义化输入输出

## 当前入口

### 1. `miniort_run_gpt`

位置：

- [tools/run_gpt.cc](/Volumes/ww/code/onnxruntime/minionnxruntime/tools/run_gpt.cc)

用途：

- 直接喂 token ids
- 也支持直接喂文本 prompt
- 也支持从配置文件读取 prompt、生成长度和运行选项
- 支持 `--generate N` 做单进程内多步贪心生成
- 支持 `--cpu-only`、`--strict`、`--quiet`
- 适合做模型执行、top-k、节点范围调试

示例：

```bash
./build_phase4/miniort_run_gpt \
  models/gpt2/model.sim.onnx \
  --prompt "The meaning of life is" \
  --model-dir models/gpt2 \
  --generate 2 \
  --top-k 5 \
  --cpu-only \
  --strict
```

也可以继续直接传 token ids：

```bash
./build_phase4/miniort_run_gpt \
  models/gpt2/model.sim.onnx \
  --tokens 464,3616,286,1204,318 \
  --generate 1 \
  --top-k 5 \
  --cpu-only \
  --strict
```

也可以把参数写到配置文件里：

```bash
./build_phase4/miniort_run_gpt \
  --config examples/gpt2_tiny/story_generate.cfg
```

### 2. `run_gpt_text.py`

位置：

- [tools/run_gpt_text.py](/Volumes/ww/code/onnxruntime/minionnxruntime/tools/run_gpt_text.py)

用途：

- 用本地 Hugging Face tokenizer 做 encode/decode
- 把文本 prompt 转成 token ids
- 调用 `miniort_run_gpt`
- 再把生成结果解码回文本

这条脚本路径现在更多是对照和兼容入口。

示例：

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

## 当前结果

这条链路已经具备：

- 语义化文本输入
- 语义化文本输出
- 单进程内多步贪心生成
- 与现有 CPU runtime 的直接联通
- 单二进制 C++ 文本续写入口

一个最小示例输出是：

```text
The meaning of life is not the
```

## 与 YOLO 路线的关系

当前项目仍然以 `yolov8n.onnx` 为主教学样本，GPT 路线不是替代 YOLO，而是补出另一条更接近文本模型 runtime 的主线。

现在目录上已经对齐为：

- YOLO：`models/yolov8n.onnx`
- GPT：`models/gpt2/`
- GPT 调试输出：`outputs/gpt2/`

这样模型资产、输出结果和工具入口都不再混在临时目录里。

## 当前边界

`phase6` 这版仍然是“先跑通、再整理”的版本，还没有做到这些能力：

- sampling 策略
- C++ 内嵌 tokenizer
- 更稳定的文本生成接口抽象
- 更严格的逐元素 logits dump 与对齐
- cache 模型的自动导出和统一调度

所以它现在更像是：

- 一个文本模型最小闭环
- 一个调试与教学入口

而不是一个完整的 LLM inference runtime。

## 下一步建议

1. 把 `run_gpt_text.py` 中的 tokenizer 能力逐步向 C++ 内收
2. 给 `miniort_run_gpt` 增加 logits dump 之类的对齐辅助能力
3. 如果继续推进性能，再把 KV cache 和增量解码继续打磨成一条完整对比链路
