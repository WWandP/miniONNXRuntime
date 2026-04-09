# Model Assets

这个目录放的是本地模型资产，但不是所有文件都会进 git。

当前约定：

- YOLO 演示模型：
  - `models/yolov8n.onnx`
- GPT-2 文本模型目录：
  - `models/gpt2/`

`models/gpt2/` 里的大文件不会提交到仓库，默认由本地脚本下载和生成。

## GPT-2 目录内容

跑当前 GPT 文本链路时，通常需要这些文件：

- `config.json`
- `generation_config.json`
- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.json`
- `merges.txt`
- `model.onnx`
- `model.sim.onnx`

其中：

- 前 7 个来自 Hugging Face 的 `openai-community/gpt2`
- `model.onnx` 和 `model.sim.onnx` 是本地导出产物

## 获取方式

推荐直接使用：

```bash
./scripts/fetch_gpt2.sh
```

这个脚本会：

1. 下载 GPT-2 原始权重和 tokenizer 文件到 `models/gpt2/`
2. 如果当前 Python 环境已经安装 `torch`、`transformers`、`onnx`
   - 自动导出 `models/gpt2/model.onnx`
3. 如果额外安装了 `onnxsim`
   - 继续生成 `models/gpt2/model.sim.onnx`

如果只下载成功但没有导出成功，脚本会给出缺失依赖提示。

## 当前文本链路

准备好 `models/gpt2/` 后，可以直接跑：

```bash
/Volumes/ww/miniconda3/envs/norm/bin/python tools/run_gpt_text.py \
  --prompt "The meaning of life is" \
  --max-new-tokens 1 \
  --cpu-only \
  --strict
```
