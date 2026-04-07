# miniONNXRuntime

一个面向教学的 ONNX Runtime 迷你实现。
它围绕 `yolov8n.onnx` 演示模型如何被解析、优化、执行，以及如何做基础内存优化。

![miniONNXRuntime banner](./assets/readme_banner.png)

英文版说明见 [README.en.md](./README.en.md)。
教学主线见 [docs/phases/README.md](./docs/phases/README.md)。

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

## 这个项目展示什么

- 解析 ONNX 图
- 优化图结构
- 执行 CPU / Apple provider kernels
- 跟踪 tensor 内存和 buffer reuse

## 文档入口

- 中文 phase 主线：[docs/phases/README.md](./docs/phases/README.md)
- 英文 phase 主线：[docs/phases/README.en.md](./docs/phases/README.en.md)

## 快速开始

依赖装好之后，Linux / macOS 的构建和运行方式是一致的：

```bash
# 打开 optimizer tools，方便 phase4 直接可用
cmake -S . -B build_local -DMINIORT_BUILD_OPTIMIZER_TOOLS=ON

# 编译全部工具
cmake --build build_local -j4

# phase1: 先看静态图结构
./scripts/run_phase.sh phase1

# phase3: 再跑一次完整 CPU 推理
./scripts/run_phase.sh phase3

# phase4: 看图优化前后差异
./scripts/run_phase.sh phase4-opt

# phase5: 看 provider 路径对比
./scripts/run_phase.sh phase5
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

## 学习路径

| Phase | 看什么 | 对应命令 | 说明文档 |
| --- | --- | --- | --- |
| `phase1` | 静态图结构 | `./scripts/run_phase.sh phase1` | [phase1](/home/weiwei.pan/code/miniONNXRuntime/docs/phases/phase1.md) / [EN](/home/weiwei.pan/code/miniONNXRuntime/docs/phases/phase1.en.md) |
| `phase2` | 最小执行主线 | `./scripts/run_phase.sh phase2` | [phase2](/home/weiwei.pan/code/miniONNXRuntime/docs/phases/phase2.md) / [EN](/home/weiwei.pan/code/miniONNXRuntime/docs/phases/phase2.en.md) |
| `phase3` | 完整 CPU 推理 | `./scripts/run_phase.sh phase3` | [phase3](/home/weiwei.pan/code/miniONNXRuntime/docs/phases/phase3.md) / [EN](/home/weiwei.pan/code/miniONNXRuntime/docs/phases/phase3.en.md) |
| `phase4` | 图优化与内存观察 | `./scripts/run_phase.sh phase4-opt` / `phase4-memory` | [phase4](/home/weiwei.pan/code/miniONNXRuntime/docs/phases/phase4.md) / [EN](/home/weiwei.pan/code/miniONNXRuntime/docs/phases/phase4.en.md) |
| `phase5` | `ExecutionProvider` 抽象与 provider 对比 | `./scripts/run_phase.sh phase5` | [phase5](/home/weiwei.pan/code/miniONNXRuntime/docs/phases/phase5.md) / [EN](/home/weiwei.pan/code/miniONNXRuntime/docs/phases/phase5.en.md) |

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
