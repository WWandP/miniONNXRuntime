# Phase7

## 看什么

- Qwen 默认 KV cache 路径怎么跑（prefill/decode 双图）
- `summary`、`provider execution summary` 在 Qwen 场景下怎么看

## 运行命令

```bash
./scripts/run_phase.sh phase7
```

CUDA 构建运行：

```bash
MINIORT_BUILD_CUDA_EP=ON CMAKE_BUILD_TYPE=Release BUILD_DIR=build_cuda_release \
  ./scripts/run_phase.sh phase7
```

开启 CUDA EP 后会走当前 CUDA provider 路径；默认 `build_local` 则是普通 CPU/default 路径。

## 输出重点

- `last_token_topk`
- `full_token_ids`
- `input_text`
- `output_text`
- `summary`
- `provider execution summary`

## 适合谁看

想把 GPT 教学线扩展到更大文本模型（Qwen）时看这里。
