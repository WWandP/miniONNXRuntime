# Qwen 推理后续工作计划（2026-04-16）

## 当前状态
- 已可在 miniORT 上跑通 Qwen2.5-0.5B-Instruct（baseline + KV cache，含 INT8 导出版本）。
- 关键缺失算子已补齐（CPU + macOS Accelerate 路径）。
- `miniort_run_gpt --kv-cache --strict` 可执行完整流程。
- 已有基础 Web 聊天演示页 `tools/chat_web_demo.py`。

## 主要问题
- 单次命令模型加载耗时高（尤其是 `collect_initializers`），导致每轮对话首 token 体感慢。
- 目前 Web Demo 每次请求会启动一次 `miniort_run_gpt`，重复加载模型。
- 仍有若干热点算子在 decode 阶段耗时明显（`Expand`/`Mul`/`MatMul`/`Transpose`）。

## 下次优先级（P0 -> P2）

## P0: 先把“可用性”做出来
1. 常驻进程 REPL / 服务化推理
- 目标：模型只加载一次，多轮对话复用 session。
- 建议：新增 `miniort_chat_repl`（C++）或 Python 长驻桥接进程。
- 验收：同模型同参数下，第 2 轮起明显快于第 1 轮（不再重复 30s+ 加载）。

2. Web Demo 请求超时与状态可视化
- 目标：避免前端长时间停在“思考中”。
- 内容：客户端超时、后端错误分类、耗时分段展示（加载/推理/总耗时）。
- 验收：超时时前端有明确提示，页面不假死。

## P1: 性能优化
1. 算子级优化（优先 decode 热点）
- 优先：`Expand`、`Mul`、`MatMul`、`Transpose`。
- 方法：减少临时分配、广播 fast-path、减少重复 shape 计算。
- 验收：decode 单步时间较当前基线下降（目标 20%+）。

2. 常量处理策略复盘
- 目标：仅保留稳定收益路径，避免引入回归。
- 说明：此前尝试过的 Constant 激进复用/提升有回归风险，需要在单测覆盖下逐步推进。
- 验收：`miniort_runtime_tests` 全绿，Qwen strict 路径无新报错。

## P2: 工程化与文档
1. 增加可复现 benchmark 脚本
- 固定 prompt、generate、模型路径，自动输出分项耗时表。
- 验收：一条命令生成对比结果，便于回归检查。

2. 文档补全
- 更新 Qwen 端到端指南（下载、导出、运行、Web Demo、常见报错）。
- 补“性能调优记录”章节，包含已验证无效或有风险方案。

## 建议基线命令
```bash
./build_local/miniort_run_gpt \
  --kv-cache \
  --kv-cache-prefill-model models/qwen2_5_0_5b_instruct/model.kv_prefill.int8.onnx \
  --kv-cache-decode-model models/qwen2_5_0_5b_instruct/model.kv_decode.int8.onnx \
  --model-dir models/qwen2_5_0_5b_instruct \
  --prompt "你好，介绍一下你自己" \
  --generate 8 \
  --quiet
```

## 收尾更新（2026-04-18）

### 本轮已完成（低风险、教学导向）
- `Session` provider 分配优化：
  - 在 Session 初始化时缓存各 provider 支持的 `op_type` 集合；
  - 节点分配阶段不再对每个节点重复 `RegisterKernels()`。
- `Expand` 优化（CPU 路径）：
  - 增加同 shape、标量扩展、仅前缀维扩展 fast-path；
  - 通用广播路径改为无临时索引分配的线性推进。
- elementwise 广播优化（CPU 路径）：
  - `Add/Mul/Sub/Div` 增加同 shape / 标量广播 fast-path，减少通用广播索引开销。
- `MatMul` 优化（macOS Accelerate 路径）：
  - 增加 `batch_count == 1` 直接 `cblas_sgemm`；
  - 增加线性 batch 映射 fast-path，减少 `UnravelIndex/ComputeBroadcastOffset` 调度开销。
- `Where` 优化（CPU + Accelerate）：
  - 增加“condition/x/y 与输出同 shape” fast-path。

### 影响文件
- `include/miniort/runtime/session.h`
- `src/runtime/session.cc`
- `src/runtime/shape_kernels.cc`
- `src/runtime/elementwise_kernels.cc`
- `src/runtime/accelerate_execution_provider.cc`

### 回归与冒烟结果
- `cmake --build build_local -j4`：通过
- `ctest --test-dir build_local --output-on-failure`：通过（`1/1`）
- Qwen strict 冒烟：
  - baseline：`executed=7149 skipped=0`
  - KV cache（`--generate 1`）：`executed=14498 skipped=0`

### 当前结论
- 功能闭环稳定，Qwen baseline + KV cache strict 路径可重复通过。
- 本轮优化已覆盖“低成本、可讲解”的热点路径，符合教学项目定位。
- 若准备收尾，可转入文档整理/提交归档，不必继续做激进性能改造。
