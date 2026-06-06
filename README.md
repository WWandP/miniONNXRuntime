# miniONNXRuntime

一个面向教学的 ONNX Runtime 迷你实现。
它围绕三条教学主线展开：

- `yolov8n.onnx`：看视觉模型如何被解析、优化、执行，以及如何做基础内存优化
- `gpt2` ONNX 图族：看文本模型如何做 prompt 编码、greedy 生成、provider 执行和 `KV cache`
- `qwen2.5-0.5b` ONNX 图族：看更大文本模型如何被解析、优化、执行，以及如何做基础内存优化

![miniONNXRuntime banner](./assets/readme_banner.png)

English version: [README.en.md](./README.en.md)

## 环境要求

构建前需要：

- CMake 3.20+
- 支持 C++20 的编译器
- Protobuf
  - 需要 `protoc`
  - CMake 会优先尝试 `find_package(Protobuf CONFIG QUIET)`，失败时回退到系统自带的 `FindProtobuf`

项目自带了用于解析 ONNX 的 `third_party/onnx`，不需要额外单独下载 ONNX 代码。

## 安装依赖

### Linux

如果你想用仓库自带脚本自动补齐依赖，可以直接执行：

```bash
# 自动检查并安装 cmake / protobuf / protoc
./scripts/setup_linux_env.sh
```

脚本会优先尝试：

- `conda-forge`
- 如果没有 `conda`，则回退到 `apt-get`

如果你想手动安装，Ubuntu / Debian 可以执行：

```bash
sudo apt update
sudo apt install -y build-essential cmake git libprotobuf-dev protobuf-compiler
```

### macOS

如果你在 macOS 上，先装 Homebrew 依赖：

```bash
brew install cmake protobuf git
```

## 下载模型

由于模型文件较大，无法直接上传到 GitHub。请先运行：

```bash
# 查看本地还缺哪些模型文件
./scripts/download_models.sh status

# 按需下载，也可以用 all 一次准备默认资产
./scripts/download_models.sh yolo
./scripts/download_models.sh gpt2
./scripts/download_models.sh qwen
```

统一入口支持：

- YOLOv8n：下载 `models/yolov8n.onnx`
- GPT-2 KV：准备 `models/gpt2/` 下的 prefill/decode ONNX 和 tokenizer 文件
- Qwen2.5-0.5B KV：准备 `models/qwen2_5_0_5b_instruct/` 下的 prefill/decode ONNX 和 tokenizer 文件

Qwen ONNX 文件较大，脚本会优先使用 `gdown` 拉取当前共享文件夹；也可以手动下载或用
`scripts/export_qwen_kv_onnx.py` 从本地 checkpoint 导出。详细文件约定见 [models/README.md](./models/README.md)。
如果使用默认的 Google Drive 归档下载 GPT-2/Qwen，请先安装 `gdown`：`python -m pip install gdown`。

## 快速开始

依赖装好之后，Linux / macOS 的构建和运行方式是一致的：

```bash
# 打开 optimizer tools，方便 phase4 直接可用
cmake -S . -B build_local -DMINIORT_BUILD_OPTIMIZER_TOOLS=ON

# 编译全部工具
cmake --build build_local -j4

# phase1: 先看静态图结构
./scripts/run_phase.sh phase1

# phase3: 跑一次完整 CPU 推理
./scripts/run_phase.sh phase3

# phase4: 看图优化前后差异
./scripts/run_phase.sh phase4-opt

# phase5: 看 provider 路径对比
./scripts/run_phase.sh phase5

# phase6: 跑 GPT-2 baseline
./scripts/run_phase.sh phase6

# phase6-kv: 跑 GPT-2 KV cache
./scripts/run_phase.sh phase6-kv

# phase7: 跑 Qwen KV cache（默认）
./scripts/run_phase.sh phase7
```

只想先构建和测试的话：

```bash
./scripts/run_phase.sh build
./scripts/run_phase.sh test
```

想按顺序完整跑一遍的话：

```bash
./scripts/run_phase.sh all
```

说明：`all` 当前覆盖默认主线 `phase1 -> phase5`。文本模型阶段（`phase6` / `phase6-kv` / `phase7`）需要先下载模型。

如果要在 CUDA 设备上运行 phase，会使用当前 CUDA ExecutionProvider 路径，包括 2026-06-06 这组 CUDA 相关优化。示例：

```bash
MINIORT_BUILD_CUDA_EP=ON CMAKE_BUILD_TYPE=Release BUILD_DIR=build_cuda_release \
  ./scripts/run_phase.sh phase7
```

默认不设置 `MINIORT_BUILD_CUDA_EP=ON` 时，`run_phase.sh` 使用普通 CPU/default 构建。

## 2026-06-06 CUDA 相关优化

这里整理了 2026-06-06 前后完成的一组 CUDA 路径优化，覆盖 YOLOv8n、GPT-2 KV-cache 和 Qwen2.5-0.5B KV-cache。下面数字来自同机本地实验，测试设备为 NVIDIA GeForce RTX 4090；ORT CPU 参考使用 ONNX Runtime 1.24.4 `CPUExecutionProvider`，ORT CUDA 参考使用 ONNX Runtime 1.23.2 `CUDAExecutionProvider`。

| 模型 / 路径 | 优化重点 | MiniORT CUDA | ORT CUDA 参考 | ORT CPU 参考 |
| --- | --- | --- | --- | --- |
| YOLOv8n mixed CUDA | device residency、CUDA buffer pool、CUDA im2col、Concat/Split/Resize 覆盖、dead tensor eviction、planned memory reuse | `repeat=50` mean `4.96 ms`，p50 `4.99 ms` | mean `1.86 ms`，p50 `1.86 ms` | mean `12.87 ms`，p50 `12.84 ms` |
| GPT-2 KV CUDA | KV-cache 双图、CUDA hot-path 覆盖、greedy argmax、warmed benchmark、`--graph-opt` | `generate=96` 约 `227.40 tokens/s`（generation mean `422.16 ms`，prefill mean `11.13 ms`） | 约 `438.65 tokens/s`（generation mean `218.85 ms`，prefill mean `1.80 ms`） | 约 `83.46 tokens/s`（generation mean `1150.24 ms`，prefill mean `16.54 ms`） |
| Qwen2.5-0.5B KV CUDA | eager CUDA initializer prepare、CUDA/cuBLAS warmup、RMSNorm primitive coverage、graph-scoped Constant reuse、tail-dimension CUDA broadcast | `generate=8` 约 `61.65 tokens/s`（generation mean `129.77 ms`，prefill mean `15.16 ms`） | 约 `171.36 tokens/s`（generation mean `46.68 ms`，prefill mean `4.75 ms`） | 约 `15.45 tokens/s`（generation mean `517.85 ms`，prefill mean `52.98 ms`） |

优化记录：

- [2026-06-06 CUDA 相关优化](./docs/optimization_summary.md)

## 学习路径

| Phase | 看什么 | 对应命令 | 说明文档 |
| --- | --- | --- | --- |
| `phase1` | 静态图结构 | `./scripts/run_phase.sh phase1` | [phase1](./docs/phases/phase1.md) / [EN](./docs/phases/phase1.en.md) |
| `phase2` | 最小执行主线 | `./scripts/run_phase.sh phase2` | [phase2](./docs/phases/phase2.md) / [EN](./docs/phases/phase2.en.md) |
| `phase3` | 完整 CPU 推理 | `./scripts/run_phase.sh phase3` | [phase3](./docs/phases/phase3.md) / [EN](./docs/phases/phase3.en.md) |
| `phase4` | 图优化与内存观察 | `./scripts/run_phase.sh phase4-opt` / `phase4-memory` | [phase4](./docs/phases/phase4.md) / [EN](./docs/phases/phase4.en.md) |
| `phase5` | `ExecutionProvider` 抽象与 provider 对比 | `./scripts/run_phase.sh phase5` | [phase5](./docs/phases/phase5.md) / [EN](./docs/phases/phase5.en.md) |
| `phase6` | GPT-2 baseline 文本生成 | `./scripts/run_phase.sh phase6` | [phase6](./docs/phases/phase6.md) / [EN](./docs/phases/phase6.en.md) |
| `phase6-kv` | GPT-2 KV cache 推理 | `./scripts/run_phase.sh phase6-kv` | [phase6](./docs/phases/phase6.md) / [EN](./docs/phases/phase6.en.md) |
| `phase7` | Qwen KV cache 推理 | `./scripts/run_phase.sh phase7` | [phase7](./docs/phases/phase7.md) / [EN](./docs/phases/phase7.en.md) |

## 主要入口

| 工具 | 更适合看什么 | 典型场景 |
| --- | --- | --- |
| `miniort_inspect` | 图结构、输入输出、op histogram | 第一次看模型 |
| `miniort_session_trace` | 前几个节点如何执行、value 怎么流转 | 学最小执行主线 |
| `miniort_run` | 一次完整推理的 timing 和 summary | 验证整图执行 |
| `miniort_memory_trace` | live tensor、peak bytes、释放时机 | 看内存与生命周期 |
| `miniort_optimize_model` | 优化前后图差异、优化后再运行 | 看 phase4 |
| `miniort_compare_providers` | 默认 provider 和 CPU-only 的差异 | 看 phase5 |
| `miniort_detect_yolov8n` | 最终检测结果和输出文件 | 看 demo 效果 |
| `miniort_run_gpt`（GPT-2） | GPT-2 文本生成与 KV cache 推理 | 看 GPT-2 文本模型执行 |
| `miniort_run_qwen`（Qwen） | Qwen KV cache 推理 | 看 Qwen 文本模型执行 |
| `tools/chat_web_demo.py`（Qwen） | 简单聊天网页 + 后端调用 `miniort_run_qwen` | 快速演示 Qwen 对话效果 |

## 文本模型入口（GPT / Qwen）

运行前先用 `./scripts/download_models.sh status` 检查本地模型文件；缺哪条线就下载哪条线。

- GPT-2：`./scripts/run_phase.sh phase6` / `./scripts/run_phase.sh phase6-kv`
- Qwen（默认 KV cache）：`./scripts/run_phase.sh phase7`

Qwen 进一步说明可看：

- `docs/phases/phase7.md`
- `docs/optimization_summary.md`
- `examples/qwen2_5_0_5b/kv_generate.cfg`

运行示例：

```bash
# phase7（默认 KV cache）
./scripts/run_phase.sh phase7

# 直接调用二进制（KV cache 配置）
./build_local/miniort_run_qwen --config examples/qwen2_5_0_5b/kv_generate.cfg
```

## 仓库结构

```text
miniONNXRuntime
├── include/ / src/   # runtime、loader、optimizer、tool 的核心实现
├── tools/            # 命令行入口
├── models/ / pic/    # 本地演示模型和图片
├── docs/             # 面向用户的正式文档
├── notes/            # 草稿、实验记录、内部笔记
└── scripts/          # 环境安装与统一构建/运行入口
```
