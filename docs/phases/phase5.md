# Phase5

## 看什么

- 默认 provider 和 CPU-only 的差异
- 哪些节点落到默认 provider 路径（例如 CUDA / Accelerate / CPU fallback）
- provider 切换后大致速度差

## 运行命令

```bash
./scripts/run_phase.sh phase5
```

## 输出重点

- `provider_compare`
- `warmup`
- `repeat`
- `allow_missing`
- `mixed_ms`
- `cpu_only_ms`
- `delta_ms`
- `speedup_pct`
- `mixed_latency_ms`
- `cpu_only_latency_ms`

## 指标含义

- `warmup`：正式计时前先跑几次，不计入统计，用来避开首次运行的初始化抖动。
- `repeat`：正式计时次数。
- `allow_missing`：是否允许缺失 kernel 或未分配节点。默认是 `false`，用于避免把跳过节点的结果当成真实性能。
- `mixed_ms`：默认 provider 路径的平均延迟。启用 CUDA 时，这条路径会优先尝试 CUDA；不支持的算子会按 provider 顺序回退。
- `cpu_only_ms`：同一模型、同一输入在纯 CPU provider 下的平均延迟。
- `delta_ms`：`cpu_only_ms - mixed_ms`。正数表示默认 provider 路径更快，负数表示更慢。
- `speedup_pct`：`delta_ms / cpu_only_ms * 100`。正数越大，说明相对 CPU-only 的提升越明显。
- `mixed_latency_ms` / `cpu_only_latency_ms`：正式计时样本的分布统计，包含 `mean`、`min`、`p50`、`p95`、`max`。
- `p50`：中位数延迟，更接近日常稳定表现。
- `p95`：95 分位延迟，用来看尾部抖动。

## 适合谁看

想理解 Execution Provider 抽象和 macOS provider 价值时看这里。
