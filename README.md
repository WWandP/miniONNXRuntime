# miniONNXRuntime

一个用于教学和自我训练的最小推理运行时项目。

目标不是复刻 ONNX Runtime，而是围绕固定模型子集，逐步实现：

- 图表示
- 图优化
- 执行器
- 内存/Buffer 管理
- 部分算子核函数与 tile 思路

当前项目重点：

- 参考 ONNX Runtime 的架构思路
- 用 Python 实现一个可讲解、可实验、可面试展示的教学型 runtime
- 以 YOLO 或 GPT-2 的固定子图/固定算子集为例推进

建议阅读顺序：

1. [设计摘要](./docs/design_summary.md)
2. [开发日志](./docs/dev_log.md)
3. [架构草图](./docs/architecture.md)

