# Dev Log

## 2026-03-24

- 初始化 `miniONNXRuntime` 项目骨架
- 明确项目定位为教学型 runtime，而不是 ONNX Runtime 复刻
- 确定当前学习重点：
  - 图优化
  - 执行框架
  - CPU 内存管理
  - kernel / tile 逻辑
- 确定第一阶段优先围绕 YOLO 类模型推进

## 2026-03-26

- 把项目目标改成以 `yolov8n.onnx` 为唯一目标模型倒逼框架设计
- 重写架构文档，明确第一版边界：
  - `batch=1`
  - `float32`
  - CPU
  - 顺序执行
  - 朴素 C++ kernel
- 把技术路线从 Python 切换到 C++
- 完成第一阶段 C++ 骨架：
  - `model` 内部图结构
  - `loader` 的 ONNX protobuf 读取和内部图构建
  - `tools/inspect_model` 的模型摘要打印工具
- 当前结果：
  - 已能加载 `minionnxruntime/models/yolov8n.onnx`
  - 已能输出输入输出、initializer、算子分布和拓扑顺序摘要
  - 已通过本地 CMake 编译和命令行运行验证
- 下一步计划：
  - 补 `runtime` 的 `Session/ExecutionContext/KernelRegistry`
  - 再开始逐步接入 kernel

### 2026-03-26 继续

- 清理 Python 遗留目录，项目主线收敛为 C++
- 补 VS Code 工作区设置、构建任务和调试配置，方便直接断点研究 `miniort_inspect`
- 把 ONNX 依赖改为仓库内 vendoring `onnx-ml.proto`，构建时使用本机 `protobuf/protoc` 生成代码
- 扩展 `Graph/Node/Value`：
  - `Value` 可携带最小常量数据
  - `Node` 可携带基础 attributes
- 扩展 loader：
  - 解析 initializer 的基础常量数据
  - 解析 `float/int/string/list/tensor` 这类常见 attribute
- 扩展 `inspect_model`：
  - 输出节点 attributes
  - 默认只展示固定数量的节点和 initializer 预览，便于快速看图
- 当前判断：
  - 对 `yolov8n` 这条路线，现有解析层已经足够进入下一阶段
  - 暂时不优先补子图/控制流相关解析
- 下一阶段更值得研究的是 `value` 的 producer/consumer 关系、最后使用位置和生命周期分析，这些会直接服务后续内存优化

## 2026-03-28

- 完成第二步：最小执行器骨架
  - `Session`
  - `ExecutionContext`
  - `KernelRegistry`
  - `Run()` 主循环
  - placeholder / fallback trace 机制
- 新增分阶段工具入口：
  - `miniort_session_trace`
  - `miniort_run`
- 把图片预处理接进 runtime：
  - 读取图片
  - resize 到模型输入大小
  - `HWC -> NCHW`
  - `uint8 -> float32`

### 2026-03-28 继续

- 完成第三步：围绕 `yolov8n` 持续补算子
- 先补 shape / elementwise / view 类算子，再补 `Conv / MaxPool / Resize`
- 把 kernel 从单文件拆成：
  - `basic_kernels`
  - `elementwise_kernels`
  - `shape_kernels`
  - `nn_kernels`
  - `kernel_utils`
- 当前已能整图跑通 `yolov8n.onnx`
  - `summary executed=459 skipped=0 materialized_outputs=0`

### 2026-03-28 展示链路

- 新增 `miniort_detect_yolov8n`
- 当前可以：
  - 跑完整图
  - 读取最终 `output`
  - 做最小检测后处理
  - 导出 `outputs/*.json`
  - 导出 `outputs/*.png`
- 当前 `Conv` 仍是朴素 CPU 实现，但已经做过一轮常数项优化

### 2026-03-28 当前判断

- 现阶段已经达到“最小条件跑通”
- 后续优先级更适合先补文档、术语、测试和展示链路收尾
- `ExecutionProvider / CUDA / buffer reuse` 继续放到后续版本

## 2026-03-30 图优化前性能基线

- 记录一版图优化前的完整性能数据，作为后续对比基线
- 测试命令：
  - `./build_phase4/miniort_detect_yolov8n models/yolov8n.onnx --image pic/bus.jpg`
- 运行结果摘要：
  - loader 总耗时约 `116.2 ms`
    - `loader.parse_model_proto`: `21.354 ms`
    - `loader.collect_initializers`: `92.485 ms`
  - 图像预处理总耗时约 `107.5 ms`
    - `image_loader.resize`: `69.199 ms`
    - `image_loader.load`: `16.995 ms`
    - `image_loader.normalize_and_pack`: `4.253 ms`
  - `Session::Run` 总耗时约 `11.875 s`
    - `kernel.Conv`: `6.309 s`
    - `kernel.Mul`: `4.124 s`
    - `kernel.Transpose`: `420.468 ms`
    - `kernel.Add`: `340.180 ms`
    - `kernel.Slice`: `333.282 ms`
  - 后处理：
    - `detect.decode_and_nms`: `0.088 ms`
    - `detect.save_visualization`: `286.635 ms`
- 当前结论：
  - 推理主耗时仍然集中在 `Conv` 和 `Mul`
  - `Transpose` / `Slice` / `Add` 这类 shape/view 算子也有明显开销
  - 后续图优化优先观察这些中间图是否能被折叠或消除

## 2026-03-30 第一版图优化

- 给 phase4 补上第一版可执行图优化入口，目标是把“跑通”升级成“可优化、可对比”
- 本轮优化拆成两部分：
  - `ConstantFolding`
  - `DeadNodeCleanup`
- 其中 `ShapeSimplification` 先保留为入口，后续继续扩展
- 实现方式：
  - 为 `phase4` 单独保留优化入口 `miniort_optimize_model`
  - 在优化器里先遍历整图，识别可静态求值的常量子图
  - 可折叠的节点会被替换成新的 initializer，并同步写回 `value_infos`
  - 折叠后再做一次从输出端反推的活跃节点清理，删除不再通向最终输出的节点
  - 优化后的图继续走同一套 YOLO 推理 / 解码 / NMS / 可视化流程，保证和 phase3 结果可直接对比
- 这版支持静态折叠的算子主要包括：
  - `Constant`
  - `Shape`
  - `Gather`
  - `Unsqueeze`
  - `Concat`
  - `Reshape`
  - `Range`
  - `Cast`
  - `ConstantOfShape`
  - `Expand`
  - `Transpose`
  - `Slice`
  - `ReduceMax`
  - `ArgMax`
  - `Sigmoid`
  - `Add`
  - `Mul`
  - `Sub`
  - `Div`
- 优化命令：
  - `./build_phase4/miniort_optimize_model models/yolov8n.onnx --image pic/bus.jpg`
- 实际结果：
  - 节点数从 `459` 降到 `359`
  - `ConstantFolding` 折叠了 `100` 个节点，并把对应结果 materialize 成新的常量
  - `DeadNodeCleanup` 在这一版没有再额外删掉节点
  - `Session::Run` 仍然能完整跑通，检测结果保持为 `5` 个框
  - `outputs/bus_yolov8n.json` 和 `outputs/bus_yolov8n.png` 正常导出
- 性能对比：
  - `Session::Run` 从 `13044.674 ms` 降到 `12479.051 ms`
  - 约减少 `565.623 ms`，提升约 `4.3%`
  - `Conv`、`Mul`、`Add` 等主热点都有小幅下降
  - `Transpose` / `Slice` 仍然有明显耗时，说明下一步更值得继续做 shape/view 子图简化
- 当前判断：
- 第一版图优化已经证明“图变短”可以转化为实际收益
- 但这只是第一层收益，真正更大的空间还在 `ShapeSimplification` 和 `Conv` 周边模式上
- phase4 现在可以作为后续图优化迭代的长期基线

## 2026-04-06 phase5 起步：最小 CPU Execution Provider

- 开始进入 `phase5`，目标不是直接扩模型，而是先把 runtime 后端边界抽清楚
- 新增最小 `ExecutionProvider` 抽象：
  - 先只定义 `Name()`
  - 先只定义 `RegisterKernels(KernelRegistry&)`
- 新增 `CpuExecutionProvider`
  - 把现有 builtin CPU kernels 通过 provider 注册进 `KernelRegistry`
- 调整 `Session`
  - 默认不再直接硬编码注册 builtin kernels
  - 改为默认挂载一个 `CpuExecutionProvider`
  - 保留默认构造路径，现有工具入口基本不需要改
- 当前结果：
  - `build_phase3` / `build_phase4` 编译通过
  - `miniort_session_trace` 已能继续加载并执行 `yolov8n.onnx`
- 当前判断：
  - 这一步先把“CPU 路径”从 `Session` 里拆出来，给后续 allocator / partition / 多 EP 留出干净挂点
  - 暂时还不引入多 EP 分图，也不急着抽 allocator 接口，避免过度设计
- 下一步计划：
  - 给 provider 增加更明确的职责边界说明
  - 评估 allocator / memory planner 以后应该挂在 EP 还是单独 runtime 层
  - 再决定是否开始做 `MatMul/Gemm` 这类更适合 GPT-2 的 CPU kernel 能力准备

### 2026-04-06 继续：把 allocator 挂到 CPU EP

- 继续收敛 `phase5` 的运行时边界，不引入多 EP，只先处理 CPU 路径的内存分配归属
- 新增 `TensorAllocator` 抽象
  - 先只覆盖运行期最常见的两类 buffer：`float32` / `int64`
  - 先保留现有向量复用语义，不改 kernel 接口
- 新增 `CpuTensorAllocator`
  - 把原来 `ExecutionContext` 里维护的 buffer pool 迁过去
- 调整 `ExecutionContext`
  - 改为持有可注入的 allocator
  - `Acquire*Buffer()` / storage recycle 改为委托给 allocator
- 调整 `Session`
  - 默认从 provider 请求 allocator 并注入到 context
- 当前判断：
  - 这样做以后，kernel 仍然只依赖 `ExecutionContext`
  - 但内存复用策略的归属已经从 context 内部实现，转成了“由 CPU EP 提供”
  - 后面如果做 arena / planner，可以继续沿 allocator 这条线演进，而不必再把逻辑塞回 `Session`

### 2026-04-06 继续：补 provider assignment 策略和校验

- 继续完善 `phase5` 的 CPU EP 收口，不急着做真正的多 EP partition
- 给 `SessionOptions` 增加 provider assignment 相关配置：
  - `provider_assignment_policy`
  - `allow_unassigned_nodes`
- 当前先显式支持一条最小策略：
  - `kFirstMatch`
  - 按 provider 顺序选择第一个声明支持该 `op_type` 的 provider
- `Session` 初始化阶段现在会：
  - 生成 assignment summary
  - 在需要时对未分配节点做显式校验并抛错
- 当前判断：
  - 这一步把原本隐含在实现细节里的“谁来执行节点”逻辑提升成了可配置的 runtime 行为
  - 虽然还是单一 CPU provider，但后面往多 EP 走时，不需要再从零重构 assignment 入口

### 2026-04-06 继续：补最小 Accelerate EP

- 在 macOS 上补一条最小 Apple 加速路径，但仍然归在 `phase5` 内，不单独开新 phase
- 先做运行级 smoke test：
  - `Accelerate` 可实际链接并执行简单向量加法
  - `Metal` 可拿到默认 device
- 最终实现选择先落 `AccelerateExecutionProvider`
  - 原因是它更适合验证第二个 provider，而不必一下子处理 Metal/CoreML 的复杂资源管理或子图编译
- 当前实现：
  - 默认 provider 顺序在 Apple 平台变成 `Accelerate -> CPU`
  - `KernelRegistry` 合并改成 first-match 保留，避免后面的 CPU kernel 覆盖前面的 Accelerate kernel
  - `Accelerate EP` 当前先支持：
    - `Add`
    - `Mul`
  - 对 `float32 + 同 shape + 无 broadcast` 走 vDSP 快路径
  - 其他情况在同一个 kernel 内回退到通用实现
- 当前结果：
  - `miniort_session_trace` 可看到 `providers=Accelerate,CPU`
  - `miniort_inspect` 的 assignment summary 可看到 `Accelerate: 77`
  - 测试已覆盖 Apple 平台下 `Add` 默认优先分配给 `Accelerate`
- 当前判断：
  - 这一步已经证明当前 EP 框架不只是“能挂第二个 provider”，而是 provider 优先级、assignment 和实际 kernel 调度能对齐
  - 但这还只是最小 Accelerate 路径，不等于完整 Apple backend

### 2026-04-06 继续：补 provider trace / execution summary

- 继续完善 `phase5` 的可观察性，目标是把“节点分给谁”和“谁真的执行了”区分开
- 扩展 `RunSummary`
  - 新增 provider 维度的 `visited / executed / skipped / materialized_outputs` 统计
- `session.run` 结束时新增 `provider execution summary`
  - 这样 trace 里可以同时看到：
    - `provider assignment summary`
    - `provider execution summary`
- 给 `miniort_session_trace` 增加参数：
  - `--start-node N`
  - `--max-nodes N`
  - `--image path`
- 这样可以：
  - 用真实图片输入跑 trace
  - 只观察某一段 topo 区间
  - 直接验证后段 `Add/Mul` 是否命中 `Accelerate`
- 实际验证命令：
  - `./build_phase3/miniort_session_trace models/yolov8n.onnx --image pic/bus.jpg --max-nodes 260`
- 实际结果：
  - trace 中可见多处：
    - `provider=Accelerate`
    - `kernel Mul produced ... via Accelerate`
    - `kernel Add produced ... via Accelerate`
  - 当前窗口内 `provider execution summary` 显示：
    - `Accelerate: visited=45 executed=45 skipped=0 materialized_outputs=0`
- 当前判断：
  - 现在已经不只是“assignment 把节点分给 Accelerate”，而是能证明真实执行也命中了 Accelerate kernel
  - 这让 `phase5` 的 Apple 自动选择路径具备了可调试、可展示的证据链

### 2026-04-06 继续：扩 Accelerate 算子池到 Sigmoid / MatMul / Gemm

- 继续扩大 `AccelerateExecutionProvider` 的有效覆盖范围，让它不只停留在最小 `Add/Mul`
- 本轮新增支持：
  - `Sigmoid`
  - `Sub`
  - `Div`
  - `MatMul`
  - `Gemm`
- 实现方式：
  - `Sigmoid` 使用 Accelerate 的向量指数/倒数组合路径
  - `Sub` / `Div` 使用 vDSP 快路径，并保留通用 fallback
  - `MatMul` / `Gemm` 先补一版通用 CPU kernel，再给 Accelerate EP 挂 `cblas_sgemm` 快路径
  - `Gemm` 当前支持最小常见属性：
    - `transA`
    - `transB`
    - `alpha`
    - `beta`
    - 以及常见 bias 形状
- 测试补充：
  - 补 `MatMul` 数值正确性测试
  - 补 `Gemm` 数值正确性测试
  - Apple 平台下测试 `Sigmoid/Add/Mul/Sub/Div/MatMul/Gemm` 默认都优先 assignment 给 `Accelerate`
- 当前结果：
  - 本地 assignment 检查结果为：
    - `Sigmoid:Accelerate`
    - `Add:Accelerate`
    - `Mul:Accelerate`
    - `Sub:Accelerate`
    - `Div:Accelerate`
    - `MatMul:Accelerate`
    - `Gemm:Accelerate`
  - `miniort_runtime_tests` / `ctest` 继续通过
- 工程收口：
  - Apple 平台构建为 `miniort_runtime` 增加 `ACCELERATE_NEW_LAPACK=1`
  - 这样可以避免 `cblas_sgemm` 的旧接口弃用警告持续刷屏
- 当前判断：
  - `Accelerate EP` 已经从“最小 elementwise provider”扩展成了一个可承接后续 GPT-2 前置探索的 Apple provider 雏形

### 2026-04-06 继续：记录 YOLO 上的 Accelerate 命中与性能对比

- 为了确认 Apple 路径不只是“能分配节点”，还实际带来整图收益，补了一轮真实图片上的命中和性能对比
- 命中统计方式：
  - 使用 `miniort_session_trace models/yolov8n.onnx --image pic/bus.jpg --max-nodes 0`
  - 从 trace 中提取 `via Accelerate` 的实际执行记录
- 当前整图实际命中的 Accelerate 算子分布：
  - `Mul`: `63`
  - `Sigmoid`: `58`
  - `Add`: `14`
  - `Div`: `1`
  - `Sub`: `1`
- 当前静态 assignment summary 显示：
  - `Accelerate: 137`
  - 与实际命中计数相符
- 新增对比工具：
  - `miniort_compare_providers`
  - 同机比较默认 `Accelerate + CPU` 与纯 `CPU`
- 实测命令：
  - `./build_phase3/miniort_compare_providers models/yolov8n.onnx --image pic/bus.jpg --repeat 1`
- 当前单次结果：
  - `mixed_ms = 7190.52`
  - `cpu_only_ms = 11093.6`
  - `delta_ms = 3903.05`
  - `speedup_pct = 35.183`
- 当前判断：
  - 这一步已经证明当前 `AccelerateExecutionProvider` 不只是局部 trace 里可见，而是对整图推理时间已经带来明显收益
  - 目前还只是单次对比，不是严格 benchmark；后面如果要做更严谨展示，建议补 `repeat=3/5` 的均值结果

### 2026-04-06 继续：优化 Conv / ConvSiLU 的 Accelerate 路线

- 为了让 Apple 路线不只停留在 elementwise / matmul，而是真正接住 YOLO 的主干卷积，本轮把 `AccelerateExecutionProvider` 里的卷积实现推进到了 `im2col + sgemm`
- 当前改动：
  - 将 `Accelerate` 侧 `Conv2D` 从参考级三重循环实现替换为：
    - `im2col`
    - `cblas_sgemm`
    - bias 行加
  - `ConvSiLU` 继续复用这条卷积路径，卷积完成后再用 Accelerate 向量化做 `SiLU`
  - 同时把普通 `Conv` 也注册到 `AccelerateExecutionProvider`
- 测试补充：
  - 增加普通 `Conv` 数值正确性测试
  - 保留 `ConvSiLU` 数值正确性测试
  - Apple 平台下默认 provider 归属测试现在也覆盖 `Conv`
  - `miniort_runtime_tests` / `ctest` 均通过
- 优化后图验证：
  - 使用 `./build_phase4/miniort_optimize_model models/yolov8n.onnx --image pic/bus.jpg`
  - 优化后图仍然是：
    - `ConvSiLU: 57`
    - `Conv: 7`
  - provider assignment summary 变为：
    - `Accelerate: 86`
    - `CPU: 158`
  - 对比前一版 `Accelerate: 79`，说明剩余 `7` 个 `Conv` 也已经切到 `Accelerate`
  - provider execution summary 同样显示：
    - `Accelerate: visited=86 executed=86`
- 原始图性能对比更新：
  - 使用 `./build_phase3/miniort_compare_providers models/yolov8n.onnx --image pic/bus.jpg --repeat 1`
  - 本轮单次结果：
    - `mixed_ms = 2164.04`
    - `cpu_only_ms = 11035.7`
    - `delta_ms = 8871.69`
    - `speedup_pct = 80.3906`
- 当前判断：
  - 这一步意味着 Apple 路线已经不只是“加速激活/逐元素算子”，而是开始吃到 YOLO 主干中最关键的一批卷积节点
  - 当前实现仍然是 `CPU memory + Accelerate BLAS` 路线，不是 GPU / Metal；但作为 phase5 范围内的 Apple EP，已经足够说明 EP 框架和后端优化路线是成立的

## 模板

后续记录建议按这个格式补：

### YYYY-MM-DD

- 今日目标
- 阅读了哪些源码
- 做了哪些实现
- 卡住的问题
- 下一步计划
