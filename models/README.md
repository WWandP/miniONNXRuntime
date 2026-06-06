# Model Assets

This directory is the local model cache. Large model files are intentionally not
committed to git.

Use this command first when setting up a new checkout:

```bash
./scripts/download_models.sh status
```

It prints the exact files expected by the default phase scripts.

## Unified Download Entry

```bash
./scripts/download_models.sh yolo
./scripts/download_models.sh gpt2
./scripts/download_models.sh qwen
./scripts/download_models.sh all
```

`all` tries the three tracks in order. Text-model files are large, so it is also
reasonable to prepare only the model track you need.

## YOLOv8n

Required file:

```text
models/yolov8n.onnx
```

Default command:

```bash
./scripts/download_models.sh yolo
```

## GPT-2 KV Cache

Required files for `phase6-kv`:

```text
models/gpt2/model.kv_prefill.onnx
models/gpt2/model.kv_decode.onnx
models/gpt2/vocab.json
models/gpt2/merges.txt
```

Default command:

```bash
./scripts/download_models.sh gpt2
```

By default this uses the repository's shared Google Drive archive of exported KV
assets and requires `gdown`:

```bash
python -m pip install gdown
./scripts/download_models.sh gpt2
```

If you want to regenerate from Hugging Face locally instead:

```bash
GPT2_SOURCE=hf-export EXPORT_KV_CACHE=1 ./scripts/download_models.sh gpt2
```

The local export path requires a Python environment with:

```text
torch
transformers
onnx
```

`onnxsim` is optional and only produces `model.sim.onnx`.

## Qwen2.5-0.5B KV Cache

Required files for `phase7`:

```text
models/qwen2_5_0_5b_instruct/model.kv_prefill.onnx
models/qwen2_5_0_5b_instruct/model.kv_decode.onnx
models/qwen2_5_0_5b_instruct/vocab.json
models/qwen2_5_0_5b_instruct/merges.txt
```

Recommended optional tokenizer/config files:

```text
models/qwen2_5_0_5b_instruct/tokenizer.json
models/qwen2_5_0_5b_instruct/tokenizer_config.json
models/qwen2_5_0_5b_instruct/config.json
models/qwen2_5_0_5b_instruct/generation_config.json
```

Default command:

```bash
./scripts/download_models.sh qwen
```

The Qwen ONNX files are large. The script supports the current shared Google
Drive folder when `gdown` is installed:

```bash
python -m pip install gdown
./scripts/download_models.sh qwen
```

You can also download or export the files yourself and place them in
`models/qwen2_5_0_5b_instruct/`. To export KV ONNX files from a local Qwen
checkpoint:

```bash
python scripts/export_qwen_kv_onnx.py \
  --model-dir models/qwen2_5_0_5b_instruct
```

Then verify:

```bash
./scripts/download_models.sh status
```
