# Phase4

## 看什么

- 图优化前后差异
- 哪些节点被折叠或清理
- tensor 生命周期和内存峰值

## 运行命令

优化视角：

```bash
./scripts/run_phase.sh phase4-opt
```

内存视角：

```bash
./scripts/run_phase.sh phase4-memory
```

## 输出重点

- optimization summary
- 优化前后节点数量
- live tensor / peak bytes
- tensor 释放时机

## 适合谁看

想理解“图为什么会变小”和“内存为什么会下降”时看这里。
