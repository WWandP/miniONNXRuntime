# Chat Web Demo

这个 demo 提供一个最小聊天网页，后端调用 `miniort_run_gpt`，默认系统提示词为：

`你是一个聊天助手。`

## 启动

```bash
python tools/chat_web_demo.py
```

默认地址：

- `http://127.0.0.1:8080`

## 默认模型选择逻辑

脚本优先自动选择：

1. KV INT8：
   - `model.kv_prefill.int8.onnx`
   - `model.kv_decode.int8.onnx`
2. KV FP：
   - `model.kv_prefill.onnx`
   - `model.kv_decode.onnx`
3. baseline：
   - `model.baseline.int8.onnx`
   - `model.baseline.onnx`

默认模型目录：

- `models/qwen2_5_0_5b_instruct`

## 常用参数

```bash
python tools/chat_web_demo.py \
  --host 0.0.0.0 \
  --port 8080 \
  --miniort-bin build_local/miniort_run_gpt \
  --model-dir models/qwen2_5_0_5b_instruct \
  --generate 48 \
  --strict
```

## 说明

- 前端每次发送消息时，后端会把历史拼接进 prompt，形成多轮对话上下文。
- 当前版本是演示应用，适合验证端到端链路，不是高并发服务实现。
