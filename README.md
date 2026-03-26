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
- 用 C++ 实现一个可讲解、可实验、可面试展示的教学型 runtime
- 第一阶段以 `yolov8n.onnx` 为唯一目标模型推进

当前第一阶段能力：

- 读取 `yolov8n.onnx`
- 构建内部图结构
- 打印输入输出、initializer、算子分布和拓扑顺序摘要

当前 `LoadOnnxGraph(...)` 这一步不只是“把 ONNX 字段复制进来”，还会完成一层最小内部建模和预处理，包括：

- 把 `ModelProto/GraphProto` 转成项目自己的 `Graph/Node/Value/TensorInfo`
- 提取输入、输出、initializer、节点和模型元信息
- 解析 tensor 的 dtype 和 shape 维度信息
- 区分真正的 graph inputs 和同时出现在 input 里的 initializer
- 建立节点名索引、算子直方图和拓扑顺序

当前也有意做了裁剪，暂时还没有完整保留：

- `NodeProto` 的 attributes
- `TensorProto` 的原始常量数据
- 更完整的 ONNX type/shape 体系
- 执行器、kernel 注册和真正推理逻辑

构建与运行：

```bash
cd minionnxruntime
cmake -S . -B build
cmake --build build -j4
./build/miniort_inspect models/yolov8n.onnx --show-topology 8 --show-initializers 5
```

依赖说明：

- 仓库内已 vendoring 当前用到的 ONNX proto 文件，位置在 `third_party/onnx/`
- 构建时会使用本机安装的 `protobuf` 和 `protoc` 生成 `onnx-ml.pb.cc/.h`
- 不再依赖外部 `onnxruntime/build/...`

如果本机没有 protobuf，可以先安装：

```bash
brew install protobuf
```

建议阅读顺序：

1. [设计摘要](./docs/design_summary.md)
2. [开发日志](./docs/dev_log.md)
3. [架构草图](./docs/architecture.md)
