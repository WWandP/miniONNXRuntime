# 内存优化记录

这份文档记录 `miniONNXRuntime` 在进入 buffer reuse 之前的内存基线，以及后续内存优化的演进点。

## 当前基线

在 `miniort_memory_trace` 入口上，对 `models/yolov8n.onnx` 做一次最小运行观察，当前得到的基线数据如下：

- graph 节点数：`459`
- graph value 数量：`595`
- 估算峰值内存：`12.02 MiB`
- 观察到的峰值内存：`12.02 MiB`
- 峰值出现的节点位置：`node_index=0`
- 观察到的峰值 live tensor 数量：`127`

这组数据的含义是：

- 当前 runtime 仍然把大量中间值常驻在 `ExecutionContext` 中
- 初始化权重会先全部加载进上下文
- 在尚未做 buffer reuse 之前，峰值内存基本由 initializer + 初始活跃值共同决定

## 当前可观察性

现在已经有一个专门的观察入口：

- `miniort_memory_trace`

它会打印：

- 静态的 tensor 生命周期计划
- 每个节点执行后的 live tensor 数量
- 当前 live tensor 名称
- 估算峰值和观察峰值

## 第一版优化

当前已经开始做的第一步优化是“按最后一次使用释放中间 tensor”：

- 在 `Session` 里记录每个 tensor 的最后使用点
- 当某个中间 tensor 已经没有后续消费者时，立即从 `ExecutionContext` 中移除
- 这一步先降低常驻 tensor 数量，再继续往 buffer 复用推进

这一步目前只在 `miniort_memory_trace` 入口上显式打开，用来做前后对比和基线观察。

这使得后续 buffer reuse 的修改可以直接和当前基线对比。

## 当前复用实验

第一版 buffer pool 已经接到运行时里，当前观察到：

- 在 `miniort_memory_trace` 上跑 `models/yolov8n.onnx`
- 截断到前 `120` 个节点时
- `released_tensors=20`

这说明：

- 运行时已经可以识别并回收一部分中间 tensor
- 被回收的 storage 能进入 pool，为后续输出复用做准备
- 目前这一步还属于“开始复用”，不是完整的 memory planner

## Initializer 优化

目前已经把 initializer 改成按需 materialize，而不是在 `Session::Run()` 一开始统一加载到 `ExecutionContext`。

这次在 `miniort_memory_trace` 上观察到：

- `initializer_count=127`
- `initializer_bytes=12.02 MiB`
- `runtime_initializer_count=2`
- `runtime_initializer_bytes=1.75 KiB`
- `final_context` 从原来的 `148` 个 tensor 降到 `23`
- 这意味着大部分 initializer 不再在 runtime context 里重复持有
- 只有真正被当前运行路径访问到的 initializer 才会进入 context cache

对应的观察值也发生了变化：

- 观察到的峰值内存：`1.75 KiB`
- 峰值出现的节点位置：`node_index=0`
- 观察到的峰值 live tensor 数量：`2`
- 估算的总峰值内存：`12.02 MiB + 1.75 KiB`

这组数据说明：

- runtime context 的峰值已经被明显压低
- 但这时看到的数值主要是 context 层峰值，不再等同于“graph + context”双份权重的总和
- 后面如果要继续对比总内存，还需要把 graph 侧的 initializer 存储单独统计出来
- 现在可以直接看 `approx_total_bytes_at_peak` 来理解整体占用趋势

## 第一阶段优化目标

buffer 优化的第一步不是做复杂 arena，而是先做生命周期驱动的回收：

- 识别 tensor 的最后一次使用点
- 在最后一次使用后尽快释放对应存储
- 保持执行结果不变

这一步的目标是：

- 降低 `ExecutionContext` 中同时常驻的张量数量
- 降低峰值内存
- 为后续真正的 buffer reuse / memory planner 铺路

## 后续要补的点

- 从“释放张量”过渡到“复用 buffer”
- 统计每个 buffer 的分配与复用次数
- 记录优化前后的峰值差异
- 再进一步把 planner 独立成更清晰的模块
- 给 initializer 侧加独立的字节统计，和 context 峰值分开看

## 优化前后对比

以下是 `models/yolov8n.onnx` 在 `miniort_memory_trace` 上的对比记录：

| 指标 | 优化前 | 现在 |
| --- | --- | --- |
| initializer_count | `127` | `127` |
| initializer_bytes | `12.02 MiB` | `12.02 MiB` |
| runtime_initializer_count | `127` | `2` |
| runtime_initializer_bytes | `12.02 MiB` | `1.75 KiB` |
| observed_peak_bytes | `12.02 MiB` | `1.75 KiB` |
| observed_peak_live_tensors | `127` | `2` |
| approx_total_bytes_at_peak | `24.04 MiB` | `12.03 MiB` |

这张表的结论很直接：

- graph 侧 initializer 没有变少
- runtime 侧不再把整批 initializer 常驻到 context
- 总峰值内存从“双份权重”的量级，降到了“单份权重 + 少量 runtime 临时值”的量级
