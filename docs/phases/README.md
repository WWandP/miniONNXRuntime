# Phase 指南

这组文档只面向用户，重点回答三件事：

- 每个阶段怎么跑
- 跑完重点看什么
- 下一步适合看哪一阶段

## 阶段顺序

1. [phase1](./phase1.md): 静态图结构
2. [phase2](./phase2.md): 最小执行主线
3. [phase3](./phase3.md): 完整推理
4. [phase4](./phase4.md): 图优化与内存
5. [phase5](./phase5.md): Execution Provider
6. [phase6](./phase6.md): GPT 文本生成与 KV cache

## 统一入口

所有阶段都通过同一个脚本运行：

```bash
./scripts/run_phase.sh <phase>
```

常用命令：

```bash
./scripts/run_phase.sh phase1
./scripts/run_phase.sh phase3
./scripts/run_phase.sh phase4-opt
./scripts/run_phase.sh phase5
./scripts/run_phase.sh phase6
./scripts/run_phase.sh phase6-kv
```
