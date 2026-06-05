# CUDA Baseline Notes

这份记录用于跟踪 YOLOv8n 在 CUDA Execution Provider 上的性能基线。

## 2026-06-04 Baseline

### Environment

- Time: `2026-06-04 20:44:46 CST`
- Commit: `afa48b5`
- Working tree: contains benchmark/script/doc changes not committed yet
- GPU: `NVIDIA GeForce RTX 4090`
- Driver: `590.48.01`
- NVIDIA-SMI CUDA version: `13.1`
- nvcc: `13.2.51`
- GPU memory: `24564 MiB`

### Build

```bash
cmake -S . -B build_cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DMINIORT_BUILD_CUDA_EP=ON \
  -DMINIORT_BUILD_OPTIMIZER_TOOLS=ON

cmake --build build_cuda --target miniort_compare_providers miniort_inspect miniort_run -j4
```

Relevant CMake cache:

- `CMAKE_BUILD_TYPE=Release`
- `MINIORT_BUILD_CUDA_EP=ON`
- `CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`
- `CMAKE_CUDA_ARCHITECTURES=native`
- `CUDAToolkit=13.2.51`
- `CMAKE_CXX_COMPILER=/usr/bin/c++`

### Assets

- Model: `models/yolov8n.onnx`
- Model sha256: `c03e8dc50385b49f150a8de1b3fa6f3f8a0ba0ce2a458a353179c084f8012f35`
- Image: `pic/bus.jpg`
- Image sha256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`

Model summary:

- Graph: `main_graph`
- Producer: `pytorch 2.2.0`
- Opset: `ai.onnx=17`
- Input: `images`, `float32`, `[1, 3, 640, 640]`
- Output: `output0`, `float32`, `[1, 84, 8400]`
- Nodes: `261`
- Initializers: `127`

Provider assignment:

- CUDA: `198`
- CPU: `63`
- Unassigned: `0`

### Primary Benchmark

Command:

```bash
./build_cuda/miniort_compare_providers \
  models/yolov8n.onnx \
  --image pic/bus.jpg \
  --warmup 3 \
  --repeat 20
```

Important: `miniort_compare_providers` now defaults to strict behavior:

- `allow_missing_kernels=false`
- `allow_unassigned_nodes=false`

Result:

```text
provider_compare
  warmup=3
  repeat=20
  allow_missing=false
  mixed_ms=157.819
  cpu_only_ms=652.405
  delta_ms=494.585
  speedup_pct=75.810
  mixed_latency_ms mean=157.819 min=155.632 p50=157.552 p95=159.928 max=160.341
  cpu_only_latency_ms mean=652.405 min=647.074 p50=651.486 p95=659.378 max=663.174
```

Baseline interpretation:

- Primary CUDA/default latency: `mixed_ms=157.819 ms`
- CPU-only reference latency: `cpu_only_ms=652.405 ms`
- Relative speedup vs CPU-only: `75.810%`
- Stable CUDA/default median: `p50=157.552 ms`
- CUDA/default tail in this run: `p95=159.928 ms`

### Strict Run Sanity Check

Command:

```bash
./build_cuda/miniort_run models/yolov8n.onnx --image pic/bus.jpg --strict
```

Execution summary:

```text
provider_counts:
  - CUDA: 198
  - CPU: 63

session.run end executed=261 skipped=0 materialized_outputs=0 released_tensors=0
```

Timing summary from the traced run:

```text
session.run.total: 315.035 ms
kernel.Conv: 254.507 ms
kernel.Mul: 23.556 ms
kernel.Sigmoid: 17.401 ms
kernel.Concat: 5.704 ms
kernel.Resize: 3.154 ms
kernel.Split: 2.992 ms
kernel.Add: 1.927 ms
kernel.Softmax: 1.837 ms
```

Note: the traced `miniort_run --strict` timing is materially higher than the no-trace wall-clock benchmark from
`miniort_compare_providers`. For optimization comparisons, use `miniort_compare_providers` as the primary benchmark
and use `miniort_run --strict` to verify correctness and identify coarse hotspots.

### Optimization Leads

- `Conv` dominates the traced run.
- Current CUDA implementation still performs many per-op `cudaMalloc` and host-device copies.
- The YOLO graph has many `Conv + Sigmoid + Mul` patterns; fusion and device-resident tensors are likely high-value next steps.
- Keep comparing against this baseline with the same model, image, build type, warmup, and repeat settings.

## 2026-06-04 First Bottleneck Analysis

### Tools Used

This round used three levels of tools:

| Tool | Command / Entry | Purpose | Result |
| --- | --- | --- | --- |
| `miniort_compare_providers` | `./build_cuda/miniort_compare_providers models/yolov8n.onnx --image pic/bus.jpg --warmup 3 --repeat 20` | Primary no-trace wall-clock benchmark for CUDA/default vs CPU-only | CUDA/default mean `157.819 ms`, CPU-only mean `652.405 ms`, speedup `75.810%` |
| `miniort_inspect` | `./build_cuda/miniort_inspect models/yolov8n.onnx` | Static graph and provider assignment check | `198` nodes assigned to CUDA, `63` nodes assigned to CPU, `0` unassigned |
| `miniort_run --strict` | `./build_cuda/miniort_run models/yolov8n.onnx --image pic/bus.jpg --strict` | Correctness/sanity check plus coarse runtime hotspot view | `261` nodes executed, `0` skipped, `0` materialized outputs; traced run showed `Conv` as the largest coarse hotspot |
| `nsys` / Nsight Systems | see command below | CUDA API and GPU activity profiling | Showed the current path is dominated by allocation/free and host-device transfers, not GPU math |
| Source inspection | `src/runtime/cuda_execution_provider.cc` | Confirm runtime behavior behind the profile | CUDA kernels allocate temporary device buffers, copy CPU tensors to GPU, execute, copy outputs back, and free buffers per op |

### Analysis Result Summary

The current CUDA path is functionally correct and already much faster than CPU-only, but it is not yet shaped like a
real GPU-resident inference runtime.

Observed facts:

- `miniort_compare_providers` gives the primary baseline:
  - CUDA/default: `157.819 ms`
  - CPU-only: `652.405 ms`
  - speedup: `75.810%`
- `miniort_inspect` shows CUDA covers most compute-like nodes:
  - CUDA: `198`
  - CPU: `63`
  - unassigned: `0`
- `miniort_run --strict` verifies no fake speedup from skipped nodes:
  - executed: `261`
  - skipped: `0`
  - materialized outputs: `0`
- Nsight Systems shows actual GPU kernels are tiny compared with runtime overhead:
  - GPU kernels: about `0.84 ms/run`
  - GPU H2D memcpy: about `21.28 ms/run`
  - GPU D2H memcpy: about `8.01 ms/run`
  - CUDA API `cudaMemcpy`: about `41.3 ms/run`
  - CUDA API `cudaMalloc`: about `34.4 ms/run`
  - CUDA API `cudaFree`: about `26.6 ms/run`

Conclusion:

The next useful work is runtime/data-movement optimization, not low-level kernel math optimization. The first target
should be reducing per-op allocation/free and repeated host-device round-trips.

### Nsight Systems Command

```bash
/usr/local/cuda/bin/nsys profile \
  --stats=true \
  --force-overwrite=true \
  --output=/tmp/miniort_yolo_cuda_nsys \
  ./build_cuda/miniort_compare_providers \
    models/yolov8n.onnx \
    --image pic/bus.jpg \
    --warmup 1 \
    --repeat 2
```

This profiles three CUDA/default runs total: one warmup run plus two measured runs.

Benchmark output under profiling:

```text
mixed_ms=164.433
cpu_only_ms=661.813
speedup_pct=75.154
mixed_latency_ms mean=164.433 min=163.572 p50=164.433 p95=165.208 max=165.294
```

### CUDA API Summary

Across the three CUDA/default runs:

```text
cudaMemcpy: 1551 calls, 123.916 ms
cudaMalloc: 2127 calls, 103.212 ms
cudaFree:   2319 calls,  79.680 ms
cudaDeviceSynchronize: 1155 calls, 6.443 ms
cudaLaunchKernel: 579 calls, 2.549 ms
```

Approximate per CUDA/default run:

```text
cudaMemcpy: about 517 calls/run, 41.3 ms/run
cudaMalloc: about 709 calls/run, 34.4 ms/run
cudaFree:   about 773 calls/run, 26.6 ms/run
```

### GPU Activity Summary

Across the three CUDA/default runs:

```text
GPU kernels total: about 2.522 ms
GPU H2D memcpy time: 63.852 ms
GPU D2H memcpy time: 24.037 ms
```

Approximate per CUDA/default run:

```text
GPU kernels: about 0.84 ms/run
H2D memcpy: about 21.28 ms/run
D2H memcpy: about 8.01 ms/run
```

Memcpy volume across the three CUDA/default runs:

```text
H2D: 1443.857 MB, 972 copies
D2H:  538.704 MB, 579 copies
```

Approximate per CUDA/default run:

```text
H2D: about 481.3 MB/run, 324 copies/run
D2H: about 179.6 MB/run, 193 copies/run
```

### Interpretation

The current CUDA path is not primarily limited by raw GPU compute. In this baseline, actual GPU kernel time is under
`1 ms/run`, while host-device copies and CUDA allocation/free overhead are far larger. The implementation currently
treats CUDA as a per-op accelerator:

1. read input from CPU-side `Tensor`
2. `cudaMalloc` temporary buffers
3. copy host to device
4. launch CUDA kernel or cuBLAS
5. copy device output back to host
6. `cudaFree` temporary buffers

This explains why CUDA is faster than CPU-only but still far slower than what the GPU should be capable of for YOLOv8n.

### Current Bottleneck Ranking

1. Per-op device allocation/free overhead.
2. Per-op host-device copies, especially repeated intermediate tensor round-trips.
3. CPU-side `im2col` in `RunCudaConv2D`, followed by copying the column buffer to GPU.
4. CPU fallback nodes such as `Concat`, `Split`, `Resize`, `Slice`, `Reshape`, `Transpose`, and `Softmax`.
5. Actual GPU math kernels, which are not the dominant cost in the current profile.

### Suggested Analysis Order

1. Add more structured CUDA timing counters around CUDA EP internals, grouped by op and by phase:
   `malloc`, `h2d`, `kernel/cublas`, `d2h`, `free`, and CPU preprocessing such as `im2col`.
2. Reuse CUDA allocations or introduce a simple device buffer pool.
3. Keep intermediate tensors device-resident across consecutive CUDA nodes.
4. Move Conv `im2col` to CUDA or switch Conv to a GPU-native implementation.
5. Add CUDA support for fused `ConvSiLU`, then revisit graph optimization on CUDA.

## 2026-06-04 Provider Segment Analysis

### Tool Added

Added:

```text
tools/provider_segments.cc
```

Built as:

```text
miniort_provider_segments
```

Purpose:

- inspect the graph after provider assignment
- group consecutive topological nodes with the same `execution_provider`
- count segment nodes, op types, boundary inputs, and boundary outputs
- estimate whether provider-level subgraph execution can reduce host-device round trips

### Command

```bash
cmake -S . -B build_cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DMINIORT_BUILD_CUDA_EP=ON \
  -DMINIORT_BUILD_OPTIMIZER_TOOLS=ON

cmake --build build_cuda --target miniort_provider_segments -j4

./build_cuda/miniort_provider_segments models/yolov8n.onnx
```

Full output was also saved locally during analysis:

```text
/tmp/miniort_yolo_provider_segments.txt
```

### Segment Summary

```text
provider_segment_summary
  total_segments=60
  - CPU: segments=30 nodes=63 max_segment_nodes=22
  - CUDA: segments=30 nodes=198 max_segment_nodes=17
```

CUDA segment length histogram:

```text
nodes=1:  6 segments
nodes=3:  4 segments
nodes=4:  1 segment
nodes=5:  3 segments
nodes=6:  2 segments
nodes=7:  2 segments
nodes=8:  1 segment
nodes=9:  4 segments
nodes=11: 2 segments
nodes=12: 2 segments
nodes=14: 2 segments
nodes=17: 1 segment
```

CPU segment length histogram:

```text
nodes=1:  21 segments
nodes=2:   6 segments
nodes=3:   1 segment
nodes=5:   1 segment
nodes=22:  1 segment
```

The initial large CPU segment is the constant block:

```text
segment[1] provider=CPU topo=[1,22] nodes=22 ops={Constant:22}
```

Most later CPU segments are small graph-structure or post-processing operators:

```text
Concat
Split
Resize
Reshape
Softmax
Transpose
Gather
Shape
Slice
```

### Interpretation

Provider segmentation is useful, but the current YOLO graph does not form one large CUDA island. It forms many medium
and small CUDA islands separated by CPU nodes:

```text
CUDA segment -> CPU Split/Concat/Resize/... -> CUDA segment -> CPU ...
```

This means a segment executor can reduce transfers inside each CUDA segment, but there will still be many CPU/CUDA
boundaries unless more ops move to CUDA.

The first CUDA island after constants is typical:

```text
segment[2] provider=CUDA topo=[23,30] nodes=8 ops={Conv:2, Sigmoid:3, Mul:3}
segment[3] provider=CPU  topo=[31,31] nodes=1 ops={Split:1}
segment[4] provider=CUDA topo=[32,38] nodes=7 ops={Conv:2, Mul:2, Sigmoid:2, Add:1}
segment[5] provider=CPU  topo=[39,39] nodes=1 ops={Concat:1}
```

So there are two related optimization tracks:

1. **Segment/device-residency optimization**: keep tensors on CUDA within each CUDA segment.
2. **CUDA op coverage optimization**: add CUDA support for frequent boundary ops like `Concat`, `Split`, `Resize`, and
   `Reshape` to merge adjacent CUDA segments.

### Next Design Implication

A minimal device-residency design should not assume one giant CUDA subgraph. It should handle repeated boundary
crossings cheaply:

- keep CUDA segment inputs on device when already available
- materialize CPU tensors only when a CPU segment actually needs them
- keep final outputs on CPU only at graph output or explicit inspection points
- prefer adding CUDA support for boundary ops that split otherwise long CUDA chains

The likely staged implementation order is:

1. Add segment analysis tooling. Done.
2. Add `ExecutionContext` support for CUDA-resident tensor storage metadata.
3. Teach a small set of CUDA kernels to consume/produce device tensors without round-tripping.
4. Start with elementwise chains inside CUDA segments.
5. Add CUDA `Concat`/`Split`/`Resize`/`Reshape` selectively to merge segments.
6. Revisit fused `ConvSiLU` after basic device residency is working.

## 2026-06-04 ONNX Runtime Reference Notes

Reference repository:

```text
/home/weiwei.pan/code/onnxruntime
commit: 2070b28ce1
```

### Reference Files Read

| Area | ONNX Runtime files | What to learn |
| --- | --- | --- |
| Provider partitioning | `onnxruntime/core/framework/graph_partitioner.h`, `onnxruntime/core/framework/graph_partitioner.cc` | EPs report `ComputeCapability`; partitioner assigns nodes/subgraphs by provider priority |
| CUDA capability | `onnxruntime/core/providers/cuda/cuda_execution_provider.cc` | CUDA EP checks kernel availability and may deliberately leave some shape/control nodes on CPU |
| Runtime values | `onnxruntime/core/framework/execution_frame.h`, `onnxruntime/core/framework/execution_frame.cc` | Runtime stores values by index in an `ExecutionFrame`, not by repeated string map lookup |
| Allocation plan | `onnxruntime/core/framework/sequential_execution_plan.h`, `onnxruntime/core/framework/allocation_planner.cc` | Each value has planned device location, allocation kind, reuse info, and release actions |
| Data transfer | `onnxruntime/core/framework/data_transfer.h`, `onnxruntime/core/framework/data_transfer_manager.h`, `onnxruntime/core/framework/data_transfer_manager.cc` | Cross-device copies are centralized through a data transfer registry |
| CUDA copy implementation | `onnxruntime/core/providers/cuda/gpu_data_transfer.h`, `onnxruntime/core/providers/cuda/gpu_data_transfer.cc` | CUDA copy logic handles CPU/GPU, GPU/GPU, sync/async copies outside normal kernels |
| Explicit copy kernels | `onnxruntime/core/providers/cuda/cuda_execution_provider.cc` (`MemcpyFromHost`, `MemcpyToHost`) | ORT represents required boundary copies as explicit runtime work, not ad hoc copies inside every op |

### Key ORT Concepts

1. **Execution provider capability is richer than a simple op registry**

   ORT asks each EP for `ComputeCapability` over a `GraphViewer`. A capability can be a single node or a subgraph.
   Higher-priority EPs claim nodes first. Compiling EPs can return subgraphs with metadata and ORT may fuse them into
   provider-owned nodes.

   MiniORT currently does a simpler version: `ResolveExecutionProviderForNode()` checks whether a provider registered
   a kernel for `node.op_type`.

2. **Tensor values carry device location**

   ORT `Tensor` has a `Location()` with an `OrtDevice`. The execution planner uses that to decide where a value should
   be allocated and which provider/allocator owns the buffer.

   MiniORT currently has only CPU-side `Tensor` storage:

   ```text
   float_data
   int64_data
   ```

   That is the main reason CUDA kernels keep copying to and from host memory.

3. **Cross-device copies are centralized**

   ORT uses `DataTransferManager` to find an `IDataTransfer` implementation for source and destination devices.
   CUDA registers `GPUDataTransfer`, which handles:

   ```text
   CPU -> GPU
   GPU -> CPU
   GPU -> GPU
   CPU -> CPU fallback
   ```

   MiniORT currently performs `cudaMemcpy` directly inside each CUDA kernel wrapper.

4. **ExecutionFrame allocates outputs based on planned location**

   ORT's `ExecutionFrame::GetOrCreateNodeOutputMLValue()` creates output values using an allocation plan. The plan
   includes the location where the tensor should live. This prevents the runtime from defaulting every output back to
   CPU.

   MiniORT currently creates outputs with CPU vectors, even when the node's provider is CUDA.

5. **Memory reuse is planned separately from kernel execution**

   ORT has allocation/reuse/release planning in `SequentialExecutionPlan` and `AllocationPlanner`. This is much larger
   than MiniORT needs right now, but the core idea matters: kernel execution should not own every allocation decision.

### What MiniORT Should Borrow First

Do not copy ORT's full architecture yet. The smallest useful subset is:

1. Add a minimal device identity:

   ```cpp
   enum class DeviceType { kCPU, kCUDA };

   struct DeviceLocation {
     DeviceType type;
     int device_id;
   };
   ```

2. Add optional device storage to `Tensor` or to a side table in `ExecutionContext`:

   ```cpp
   struct DeviceTensorStorage {
     DeviceLocation location;
     void* data;
     std::size_t bytes;
     bool owns_data;
   };
   ```

3. Add a tiny `DataTransfer` layer:

   ```text
   CopyTensor CPU -> CUDA
   CopyTensor CUDA -> CPU
   CopyTensor CUDA -> CUDA
   ```

4. Teach `ExecutionContext` to answer:

   ```text
   FindTensorOnCPU(name)
   FindTensorOnCUDA(name)
   EnsureTensorOnCPU(name)
   EnsureTensorOnCUDA(name)
   ```

5. Teach CUDA kernels to consume and produce CUDA-resident buffers when possible.

6. Defer full ORT-like allocation planning. Start with a simple CUDA buffer pool or per-run cache, then improve reuse.

### Design Implication For The Current CUDA Work

Provider segments alone are not enough. ORT's model suggests the real unit of improvement is:

```text
value location + data transfer manager + provider-aware output allocation
```

Once MiniORT has those three, segment execution becomes natural:

```text
segment boundary input: EnsureTensorOnCUDA
CUDA node output: allocate CUDA tensor
next CUDA node input: read CUDA tensor directly
CPU boundary: EnsureTensorOnCPU only when needed
```

This is the right next implementation direction before spending effort on low-level CUDA kernel math.

## Optimization Attempt 1: CUDA Runtime Object Reuse

Date: 2026-06-04

Goal: take the easiest ONNX Runtime-inspired allocation idea first. ORT has allocator/planner infrastructure; MiniORT
does not need the full system yet, so this attempt keeps the API surface unchanged and only reuses temporary CUDA
runtime objects inside the CUDA EP.

### Code Change

Files:

```text
src/runtime/cuda_execution_provider.cc
```

Changes:

1. Added a small process-local `CudaBufferPool`.
   - `DeviceBuffer` now acquires buffers from the pool.
   - `DeviceBuffer` returns buffers to the pool on destruction instead of immediately calling `cudaFree`.
   - The pool uses `std::multimap<size, ptr>` and reuses the first buffer whose size is at least the requested size.
   - If returning a buffer to the pool fails, it falls back to `cudaFree`.

2. Reused one process-local cuBLAS handle.
   - Previous code created a `CublasHandle` inside every CUDA MatMul/Gemm/Conv call.
   - New code uses `GetCublasHandle()`.

This is intentionally simple and benchmark-oriented. Memory can stay cached until process exit, which is acceptable for
the current YOLO single-session measurement.

### Benchmark Command

```bash
./build_cuda/miniort_compare_providers models/yolov8n.onnx --image pic/bus.jpg --warmup 3 --repeat 20
```

Baseline before this change:

```text
mixed_ms=157.819
cpu_only_ms=652.405
speedup_pct=75.810
mixed_latency_ms mean=157.819 min=155.632 p50=157.552 p95=159.928 max=160.341
```

After buffer pool only:

```text
mixed_ms=132.806
cpu_only_ms=653.592
speedup_pct=79.681
mixed_latency_ms mean=132.806 min=131.592 p50=132.312 p95=135.432 max=135.456
```

After buffer pool + cuBLAS handle reuse:

```text
mixed_ms=127.564
cpu_only_ms=670.419
speedup_pct=80.973
mixed_latency_ms mean=127.564 min=123.253 p50=124.728 p95=139.529 max=145.916
```

Measured improvement versus baseline:

```text
latency_delta = 157.819 ms - 127.564 ms = 30.255 ms
relative_latency_reduction = 19.171%
```

### Nsight Systems Command

```bash
/usr/local/cuda/bin/nsys profile --stats=true --force-overwrite=true \
  --output=/tmp/miniort_yolo_cuda_nsys_pool_cublas \
  ./build_cuda/miniort_compare_providers models/yolov8n.onnx --image pic/bus.jpg --warmup 1 --repeat 2
```

Nsight profiles 3 mixed-provider runs here: 1 warmup run plus 2 measured runs.

Before this change:

```text
cudaMemcpy:            1551 calls, 123.916 ms
cudaMalloc:            2127 calls, 103.212 ms
cudaFree:              2319 calls,  79.680 ms
cudaDeviceSynchronize: 1155 calls,   6.443 ms
cudaEventCreate:       3456 calls,   0.785 ms
cudaEventDestroy:      3456 calls,   0.622 ms
```

After buffer pool only:

```text
cudaMemcpy:            1551 calls, 125.380 ms
cudaMalloc:             582 calls,  87.858 ms
cudaFree:               774 calls,   7.135 ms
cudaDeviceSynchronize: 1155 calls,  26.214 ms
cudaEventCreate:       3456 calls,   0.785 ms
cudaEventDestroy:      3456 calls,   0.622 ms
```

After buffer pool + cuBLAS handle reuse:

```text
cudaMemcpy:            1551 calls, 153.100 ms
cudaMalloc:               9 calls,  79.371 ms
cudaFree:                10 calls,   1.157 ms
cudaDeviceSynchronize:  391 calls,  21.553 ms
cudaEventCreate:        18 calls,   0.007 ms
cudaEventDestroy:       18 calls,   0.006 ms
```

### Interpretation

This optimization worked, but it also confirms the next bottleneck:

```text
GPU kernels are still tiny.
Host/device copies are still unchanged: 1551 cudaMemcpy calls over the profiled mixed-provider runs.
```

So the next meaningful resume-friendly result should target value residency:

```text
CPU Tensor only -> Tensor can also hold CUDA storage
per-op cudaMemcpy -> centralized EnsureTensorOnCPU/EnsureTensorOnCUDA
CUDA op output copied to CPU immediately -> keep CUDA output on device until a CPU op or final output needs it
```

That is the small MiniORT version of ORT's `OrtDevice` + `DataTransferManager` + `ExecutionFrame` allocation planning.

### Validation

```bash
cmake --build build_cuda --target miniort_compare_providers miniort_run -j4
./build_cuda/miniort_run models/yolov8n.onnx --image pic/bus.jpg --strict
ctest --test-dir build_local --output-on-failure
```

Results:

```text
strict run: executed=261 skipped=0 materialized_outputs=0
ctest: 100% tests passed, 0 tests failed out of 1
```

## Optimization Attempt 2: Move Conv Im2Col To CUDA

Date: 2026-06-04

Question: should we start optimizing individual CUDA kernels with shared memory and stream splitting?

Conclusion: for the current YOLO run, the first useful Conv optimization is not shared memory. The old Conv path did:

```text
CPU FillIm2ColBuffer -> cudaMemcpy expanded columns H2D -> cuBLAS SGEMM -> cudaMemcpy output D2H
```

That makes Conv expensive because the expanded im2col matrix is much larger than the original input tensor. So this
attempt moved im2col into a CUDA kernel and changed Conv to:

```text
cudaMemcpy input H2D -> CUDA Im2Col2DKernel -> cuBLAS SGEMM -> cudaMemcpy output D2H
```

This still uses im2col + GEMM, but avoids CPU-side im2col work and avoids copying the expanded columns from host to
device.

### Code Change

Files:

```text
src/runtime/cuda_elementwise_kernels.h
src/runtime/cuda_elementwise_kernels.cu
src/runtime/cuda_execution_provider.cc
```

Changes:

1. Added `LaunchCudaIm2Col2D`.
2. Added `Im2Col2DKernel`.
3. Changed CUDA Conv to upload the original input tensor and run im2col on GPU.
4. Removed the unused CPU `FillIm2ColBuffer` helper from CUDA EP.

### Benchmark Command

```bash
./build_cuda/miniort_compare_providers models/yolov8n.onnx --image pic/bus.jpg --warmup 3 --repeat 20
```

Before this attempt, after buffer pool + cuBLAS handle reuse:

```text
mixed_ms=127.564
cpu_only_ms=670.419
speedup_pct=80.973
mixed_latency_ms mean=127.564 min=123.253 p50=124.728 p95=139.529 max=145.916
```

After CUDA im2col:

```text
mixed_ms=76.869
cpu_only_ms=690.759
speedup_pct=88.872
mixed_latency_ms mean=76.869 min=74.846 p50=76.385 p95=78.727 max=86.974
```

Measured improvement versus previous optimized baseline:

```text
latency_delta = 127.564 ms - 76.869 ms = 50.695 ms
relative_latency_reduction = 39.741%
```

Measured improvement versus original strict baseline:

```text
latency_delta = 157.819 ms - 76.869 ms = 80.950 ms
relative_latency_reduction = 51.293%
```

### Nsight Systems Command

```bash
/usr/local/cuda/bin/nsys profile --stats=true --force-overwrite=true \
  --output=/tmp/miniort_yolo_cuda_nsys_cuda_im2col \
  ./build_cuda/miniort_compare_providers models/yolov8n.onnx --image pic/bus.jpg --warmup 1 --repeat 2
```

Nsight profiles 3 mixed-provider runs here: 1 warmup run plus 2 measured runs.

Before CUDA im2col:

```text
H2D memcpy total: 1443.857 MB, 972 copies
D2H memcpy total:  538.704 MB, 579 copies
cudaMemcpy API:    153.100 ms, 1551 calls
```

After CUDA im2col:

```text
H2D memcpy total:  821.469 MB, 972 copies
D2H memcpy total:  538.704 MB, 579 copies
cudaMemcpy API:    102.300 ms, 1551 calls
Im2Col2DKernel:      1.855 ms, 192 launches
```

Interpretation:

```text
H2D volume dropped by 622.388 MB over 3 profiled mixed-provider runs.
That is about 207.463 MB less H2D transfer per YOLO run.
The new im2col CUDA kernels add only about 0.618 ms per YOLO run.
```

### cuDNN Note

NVIDIA's optimized Conv implementation is cuDNN. That is the better long-term backend than MiniORT's hand-written
im2col path.

Current machine state:

```text
Found cuDNN runtime libraries:
  /usr/local/lib/ollama/mlx_cuda_v13/libcudnn*.so.9

Did not find development headers:
  cudnn.h
  cudnn_version.h
```

So cuDNN cannot be linked cleanly from this project yet. Once headers are installed, the next implementation should add
an optional CMake path:

```text
MINIORT_BUILD_CUDNN=ON
find cudnn.h + libcudnn
CUDA Conv path: cuDNN first, fallback to CUDA im2col + cuBLAS
```

Resume wording for the current result should avoid claiming cuDNN. The honest claim is:

```text
Moved Conv im2col from CPU to a custom CUDA kernel, reducing YOLOv8n H2D transfer by ~207 MB/run and cutting mixed
provider latency from 127.6 ms to 76.9 ms on RTX 4090.
```

### Validation

```bash
cmake --build build_cuda --target miniort_compare_providers miniort_run miniort_runtime_tests -j4
ctest --test-dir build_cuda --output-on-failure
./build_cuda/miniort_run models/yolov8n.onnx --image pic/bus.jpg --strict
ctest --test-dir build_local --output-on-failure
git diff --check
```

Results:

```text
strict run: executed=261 skipped=0 materialized_outputs=0
build_cuda ctest: 100% tests passed, 0 tests failed out of 1
build_local ctest: 100% tests passed, 0 tests failed out of 1
```

## Optimization Attempt 5: CUDA Concat/Split Boundary Ops

Date: 2026-06-04

Goal: reduce CPU/CUDA boundaries after adding CUDA tensor residency.

Provider segment analysis after tensor residency still showed that YOLO had many boundaries caused by shape/data movement
ops. `Concat` and `Split` were good first targets because they are common YOLO boundary ops and mostly perform contiguous
memory movement.

### Code Change

Files:

```text
src/runtime/cuda_execution_provider.cc
```

Changes:

1. Added CUDA `Concat` for float32 tensors using CUDA device-to-device copies.
2. Added CUDA `Split` for float32 tensors using CUDA device-to-device copies.
3. Registered both ops in CUDA EP.
4. Added fallback materialization when CUDA binary ops fall back to CPU broadcast logic.

### Provider Segment Change

Before adding CUDA `Concat`/`Split`, original segment analysis had:

```text
total_segments=60
CUDA nodes=198
CPU nodes=63
```

After adding CUDA tensor residency plus CUDA `Concat`/`Split`:

```text
total_segments=21
CUDA nodes=226
CPU nodes=35
max CUDA segment nodes=98
```

This is the clearest evidence that the graph is becoming more CUDA-resident instead of bouncing between providers.

### Benchmark Command

```bash
./build_cuda/miniort_compare_providers models/yolov8n.onnx --image pic/bus.jpg --warmup 3 --repeat 20
```

Before CUDA `Concat`/`Split`, after tensor residency:

```text
mixed_ms=45.058
cpu_only_ms=700.158
speedup_pct=93.565
mixed_latency_ms mean=45.058 min=44.493 p50=44.804 p95=46.299 max=46.679
```

After CUDA `Concat`/`Split`:

```text
mixed_ms=33.835
cpu_only_ms=695.722
speedup_pct=95.137
mixed_latency_ms mean=33.835 min=32.692 p50=33.506 p95=34.888 max=40.277
```

Measured improvement versus tensor residency baseline:

```text
latency_delta = 45.058 ms - 33.835 ms = 11.223 ms
relative_latency_reduction = 24.908%
```

Measured improvement versus original strict baseline:

```text
latency_delta = 157.819 ms - 33.835 ms = 123.984 ms
relative_latency_reduction = 78.562%
```

Compared with Python ONNX Runtime CPU reference:

```text
MiniORT mixed CUDA:         p50 about 33.5 ms
Python ORT CPU, 16 threads: p50 about 24.2 ms
```

This is still slower than ORT, but now close enough to be a credible educational-runtime optimization result.

### Nsight Systems Command

```bash
/usr/local/cuda/bin/nsys profile --stats=true --force-overwrite=true \
  --output=/tmp/miniort_yolo_cuda_nsys_concat_split \
  ./build_cuda/miniort_compare_providers models/yolov8n.onnx --image pic/bus.jpg --warmup 1 --repeat 2
```

After CUDA `Concat`/`Split`, over 3 profiled mixed-provider runs:

```text
H2D memcpy total: 270.593 MB, 417 copies
D2H memcpy total: 215.098 MB, 216 copies
D2D memcpy total: 159.936 MB, 1494 copies
```

Compared with tensor residency before CUDA `Concat`/`Split`:

```text
H2D: 327.252 MB -> 270.593 MB
D2H: 253.325 MB -> 215.098 MB
```

Interpretation:

```text
Concat/Split moved work from CPU boundary copies into GPU-local D2D copies.
This reduced host/device traffic and made CUDA segments longer.
```

### Current Best Result

Best strict MiniORT YOLO result so far:

```text
mixed_ms=33.835
p50=33.506
speedup_pct=95.137 versus MiniORT CPU-only
```

End-to-end optimization chain:

```text
157.819 ms baseline
127.564 ms buffer pool + cuBLAS handle reuse
 76.869 ms CUDA im2col
 77.537 ms cuDNN + plan cache
 45.058 ms CUDA tensor residency
 33.835 ms CUDA Concat/Split boundary ops
```

### Validation

```bash
cmake --build build_cuda --target miniort_compare_providers miniort_run miniort_runtime_tests miniort_provider_segments -j4
ctest --test-dir build_cuda --output-on-failure
ctest --test-dir build_local --output-on-failure
./build_cuda/miniort_run models/yolov8n.onnx --image pic/bus.jpg --strict
git diff --check
```

Results:

```text
strict run: executed=261 skipped=0 materialized_outputs=0
build_cuda ctest: 100% tests passed, 0 tests failed out of 1
build_local ctest: 100% tests passed, 0 tests failed out of 1
```

## Optimization Attempt 6: CUDA Reshape/Resize Boundary Ops

Date: 2026-06-04

Goal: reduce the remaining CPU/CUDA boundaries after CUDA tensor residency and CUDA `Concat`/`Split`.

### Code Change

Files:

```text
src/runtime/cuda_execution_provider.cc
src/runtime/cuda_elementwise_kernels.h
src/runtime/cuda_elementwise_kernels.cu
```

Changes:

1. Added CUDA `Reshape` as a metadata-only op.
   - For CUDA-resident `float32` tensors, it shares the existing device buffer and only changes shape metadata.
   - For host-only `float32` or `int64` tensors, it preserves host data behavior.
2. Added CUDA `Resize` for the YOLO-supported path:
   - 4D NCHW `float32`
   - `nearest`
   - `asymmetric`
   - `floor`
   - scales input
3. Registered both ops in CUDA EP.

### Provider Segment Change

Before this attempt, after CUDA tensor residency plus CUDA `Concat`/`Split`:

```text
total_segments=21
CUDA nodes=226
CPU nodes=35
max CUDA segment nodes=98
```

After CUDA `Reshape`:

```text
total_segments=13
CUDA nodes=231
CPU nodes=30
max CUDA segment nodes=103
```

After CUDA `Reshape` + CUDA `Resize`:

```text
total_segments=9
CUDA nodes=233
CPU nodes=28
max CUDA segment nodes=218
```

This is a useful structural improvement: the early YOLO backbone/head path now forms one large CUDA segment from
topological node `23` through `240`.

### Benchmark Command

```bash
./build_cuda/miniort_compare_providers models/yolov8n.onnx --image pic/bus.jpg --warmup 3 --repeat 20
```

Before this attempt:

```text
mixed_ms=33.835
p50=33.506
speedup_pct=95.137
```

After CUDA `Reshape` + CUDA `Resize`:

```text
mixed_ms=26.807
cpu_only_ms=707.491
speedup_pct=96.211
mixed_latency_ms mean=26.807 min=26.494 p50=26.742 p95=27.404 max=27.659
```

Measured improvement versus previous best:

```text
latency_delta = 33.835 ms - 26.807 ms = 7.028 ms
relative_latency_reduction = 20.771%
```

Measured improvement versus original strict baseline:

```text
latency_delta = 157.819 ms - 26.807 ms = 131.012 ms
relative_latency_reduction = 83.014%
```

Updated end-to-end optimization chain:

```text
157.819 ms baseline
127.564 ms buffer pool + cuBLAS handle reuse
 76.869 ms CUDA im2col
 77.537 ms cuDNN + plan cache
 45.058 ms CUDA tensor residency
 33.835 ms CUDA Concat/Split boundary ops
 26.807 ms CUDA Reshape/Resize boundary ops
```

### Validation

```bash
cmake --build build_cuda --target miniort_compare_providers miniort_provider_segments miniort_run miniort_runtime_tests -j4
./build_cuda/miniort_provider_segments models/yolov8n.onnx
./build_cuda/miniort_compare_providers models/yolov8n.onnx --image pic/bus.jpg --warmup 3 --repeat 20
./build_cuda/miniort_run models/yolov8n.onnx --image pic/bus.jpg --strict
ctest --test-dir build_cuda --output-on-failure
ctest --test-dir build_local --output-on-failure
git diff --check
```

Results:

```text
strict run: executed=261 skipped=0 materialized_outputs=0
build_cuda ctest: 100% tests passed, 0 tests failed out of 1
build_local ctest: 100% tests passed, 0 tests failed out of 1
```

## Optimization Attempt 7: CUDA Initializer Cache And Conv Bias

Date: 2026-06-04

Goal: keep model weights and Conv outputs on GPU. After CUDA `Reshape`/`Resize`, Nsight still showed substantial
H2D/D2H traffic. Source inspection found two causes:

1. CUDA initializer uploads were tied to each per-run `ExecutionContext`.
2. Conv bias handling copied Conv output back to CPU, added bias on CPU, then later CUDA ops uploaded it again.

### Code Change

Files:

```text
src/runtime/cuda_execution_provider.cc
src/runtime/cuda_elementwise_kernels.h
src/runtime/cuda_elementwise_kernels.cu
```

Changes:

1. Added a process-local CUDA initializer cache for `float32` initializer tensors.
   - Cache key includes initializer name, dtype, byte size, and shape.
   - Repeated runs can bind the cached device buffer instead of uploading weights again.
2. Added `AddChannelBias2DKernel`.
   - Applies NCHW Conv bias directly on CUDA output buffers.
3. Changed cuDNN Conv and im2col Conv paths to keep biased Conv outputs CUDA-resident.

### Benchmark Command

```bash
./build_cuda/miniort_compare_providers models/yolov8n.onnx --image pic/bus.jpg --warmup 3 --repeat 20
```

Before this attempt:

```text
mixed_ms=26.807
p50=26.742
speedup_pct=96.211
```

After initializer cache only:

```text
mixed_ms=26.104
p50=26.053
speedup_pct=96.217
```

After initializer cache + CUDA Conv bias:

```text
mixed_ms=13.576
cpu_only_ms=707.122
speedup_pct=98.080
mixed_latency_ms mean=13.576 min=12.823 p50=13.339 p95=14.188 max=17.502
```

Measured improvement versus previous best:

```text
latency_delta = 26.807 ms - 13.576 ms = 13.231 ms
relative_latency_reduction = 49.355%
```

Measured improvement versus original strict baseline:

```text
latency_delta = 157.819 ms - 13.576 ms = 144.243 ms
relative_latency_reduction = 91.398%
```

Updated end-to-end optimization chain:

```text
157.819 ms baseline
127.564 ms buffer pool + cuBLAS handle reuse
 76.869 ms CUDA im2col
 77.537 ms cuDNN + plan cache
 45.058 ms CUDA tensor residency
 33.835 ms CUDA Concat/Split boundary ops
 26.807 ms CUDA Reshape/Resize boundary ops
 13.576 ms CUDA initializer cache + Conv bias on GPU
```

### Nsight Systems

Command:

```bash
/usr/local/cuda/bin/nsys profile --stats=true --force-overwrite=true \
  --output=/tmp/miniort_yolo_cuda_nsys_cuda_bias \
  ./build_cuda/miniort_compare_providers models/yolov8n.onnx --image pic/bus.jpg --warmup 1 --repeat 2
```

Before this attempt, after CUDA `Reshape`/`Resize`, over 3 profiled mixed-provider runs:

```text
H2D: 241.332 MB, 402 copies
D2H: 196.896 MB, 201 copies
D2D: 159.936 MB, 1494 copies
```

After initializer cache only:

```text
H2D: 216.162 MB, 274 copies
D2H: 196.896 MB, 201 copies
D2D: 159.936 MB, 1494 copies
```

After initializer cache + CUDA Conv bias:

```text
H2D:  35.014 MB, 148 copies
D2H:  15.725 MB,  12 copies
D2D: 159.936 MB, 1494 copies
```

Interpretation:

```text
The big win was not the weight cache alone. The big win was keeping biased Conv outputs on GPU.
This removed repeated Conv-output D2H/H2D round trips between CUDA Conv and following CUDA activation/elementwise ops.
```

### Validation

```bash
cmake --build build_cuda --target miniort_compare_providers miniort_run miniort_runtime_tests -j4
./build_cuda/miniort_run models/yolov8n.onnx --image pic/bus.jpg --strict
./build_cuda/miniort_compare_providers models/yolov8n.onnx --image pic/bus.jpg --warmup 3 --repeat 20
ctest --test-dir build_cuda --output-on-failure
ctest --test-dir build_local --output-on-failure
git diff --check
```

Results:

```text
strict run: executed=261 skipped=0 materialized_outputs=0
build_cuda ctest: 100% tests passed, 0 tests failed out of 1
build_local ctest: 100% tests passed, 0 tests failed out of 1
```

## Reference: Python ONNX Runtime YOLOv8n

Date: 2026-06-04

Goal: compare MiniORT against a mature ONNX Runtime implementation using the same `models/yolov8n.onnx` and
`pic/bus.jpg` input.

### Environment

Conda env:

```text
local-voice-demo
python=/home/weiwei.pan/miniconda3/envs/local-voice-demo/bin/python
onnxruntime=1.23.2
available_providers=['AzureExecutionProvider', 'CPUExecutionProvider']
```

Note: this env has `onnxruntime_gpu-1.23.2.dist-info`, but CUDA EP is not actually available:

```text
ort.get_available_providers() = ['AzureExecutionProvider', 'CPUExecutionProvider']
ort.get_device() = CPU
```

Requesting CUDA EP falls back to CPU:

```text
Specified provider 'CUDAExecutionProvider' is not in available provider names.
```

### Tool

Added:

```text
tools/benchmark_onnxruntime_yolo.py
```

It uses the same rough preprocessing as MiniORT:

```text
RGB -> resize to 640x640 -> float32 / 255 -> NCHW [1,3,640,640]
```

The measurement times only `session.run`, not image load/resize.

### Commands And Results

Default ORT CPU threading:

```bash
conda run -n local-voice-demo python tools/benchmark_onnxruntime_yolo.py \
  models/yolov8n.onnx --image pic/bus.jpg --warmup 10 --repeat 100
```

```text
latency_ms mean=24.330 min=12.545 p50=12.769 p95=46.485 max=53.922
```

Fixed thread scan:

```bash
conda run -n local-voice-demo python tools/benchmark_onnxruntime_yolo.py \
  models/yolov8n.onnx --image pic/bus.jpg --warmup 20 --repeat 100 --threads N
```

```text
threads=1:  mean=63.053 p50=61.608 p95=69.253
threads=4:  mean=32.687 p50=31.433 p95=44.343
threads=8:  mean=26.533 p50=26.598 p95=28.796
threads=16: mean=25.878 p50=24.231 p95=37.123
threads=24: mean=72.302 p50=69.962 p95=151.559
threads=32: mean=67.094 p50=66.570 p95=95.310
```

Graph optimization disabled, 16 threads:

```bash
conda run -n local-voice-demo python tools/benchmark_onnxruntime_yolo.py \
  models/yolov8n.onnx --image pic/bus.jpg --warmup 20 --repeat 100 --threads 16 --disable-graph-opt
```

```text
latency_ms mean=29.826 min=19.660 p50=26.299 p95=46.801 max=48.550
```

### Interpretation

Python ONNX Runtime CPU is much faster than MiniORT CPU and also faster than MiniORT's current mixed CUDA path:

```text
ORT CPU, 16 threads:       p50 about 24 ms
MiniORT CUDA im2col best:  mean about 76 ms
MiniORT cuDNN cached path: mean about 77-100 ms in repeated runs
MiniORT CPU-only:          mean about 670 ms
```

This confirms that there is still a large runtime gap. The gap is not just Conv kernel math:

```text
1. ORT CPU kernels are highly optimized and multithreaded.
2. ORT has mature memory planning and avoids MiniORT's per-op allocation/copy style.
3. MiniORT still copies CUDA op outputs back to CPU and uploads the next CUDA op input again.
4. MiniORT's provider partitioning still has many CPU/CUDA boundaries on YOLOv8n.
```

The next meaningful MiniORT optimization should not be another single CUDA kernel tweak. It should target:

```text
Tensor residency on CUDA + centralized CPU/CUDA copy planning + CUDA segment execution
```

That is the likely path to close the gap with ORT more than further im2col/shared-memory tuning.

## Optimization Attempt 4: CUDA Tensor Residency

Date: 2026-06-04

Goal: close the gap with ONNX Runtime by reducing per-op CPU/GPU round trips.

The previous CUDA path still did this for many CUDA nodes:

```text
CUDA op input:  CPU vector -> H2D
CUDA op output: D2H -> CPU vector
next CUDA op:   CPU vector -> H2D again
```

This attempt adds a minimal CUDA-resident tensor path:

```text
Tensor.cuda_data + Tensor.cuda_bytes
CUDA op output can stay on GPU
next CUDA op reads Tensor.cuda_data directly
CPU node boundary materializes CUDA inputs back to host
graph outputs are materialized after execution
```

This is the smallest useful version of ONNX Runtime's `OrtDevice` / device tensor / data transfer planning idea.

### Code Change

Files:

```text
include/miniort/runtime/tensor.h
include/miniort/runtime/cuda_execution_provider.h
src/runtime/tensor.cc
src/runtime/cuda_execution_provider.cc
src/runtime/session.cc
```

Changes:

1. Added optional CUDA storage to `Tensor`:

```text
std::shared_ptr<void> cuda_data
std::size_t cuda_bytes
```

2. Added CUDA materialization helpers:

```text
MaterializeCudaTensor(name, context)
MaterializeCudaInputsForNode(node, context)
```

3. `Session::Run` now materializes CUDA inputs before CPU nodes and materializes graph outputs at the end.
4. CUDA Conv, MaxPool, unary ops, and binary float ops can produce device-only outputs.
5. CUDA Conv/input/weights use cached tensor CUDA storage instead of always uploading host vectors.

### Benchmark Command

```bash
./build_cuda/miniort_compare_providers models/yolov8n.onnx --image pic/bus.jpg --warmup 3 --repeat 20
```

Before tensor residency, cuDNN cached path:

```text
mixed_ms=77.537
cpu_only_ms=671.272
speedup_pct=88.449
mixed_latency_ms mean=77.537 min=76.181 p50=77.354 p95=79.096 max=81.300
```

After tensor residency, cuDNN enabled:

```text
mixed_ms=45.058
cpu_only_ms=700.158
speedup_pct=93.565
mixed_latency_ms mean=45.058 min=44.493 p50=44.804 p95=46.299 max=46.679
```

After tensor residency, cuDNN disabled:

```text
mixed_ms=45.039
cpu_only_ms=697.053
speedup_pct=93.539
mixed_latency_ms mean=45.039 min=44.299 p50=45.052 p95=46.394 max=46.443
```

Interpretation:

```text
Tensor residency reduced mixed-provider latency by about 32.5 ms versus the previous cuDNN cached path.
cuDNN and CUDA im2col are now roughly tied because the dominant improvement is avoiding transfers, not changing Conv math.
```

Measured improvement versus original strict baseline:

```text
latency_delta = 157.819 ms - 45.058 ms = 112.761 ms
relative_latency_reduction = 71.450%
```

Compared with Python ONNX Runtime CPU reference:

```text
MiniORT mixed CUDA:        p50 about 44.8 ms
Python ORT CPU, 16 threads: p50 about 24.2 ms
```

This is still slower than ORT, but the gap is now much more defensible for a small educational runtime.

### Nsight Systems Command

```bash
/usr/local/cuda/bin/nsys profile --stats=true --force-overwrite=true \
  --output=/tmp/miniort_yolo_cuda_nsys_residency \
  ./build_cuda/miniort_compare_providers models/yolov8n.onnx --image pic/bus.jpg --warmup 1 --repeat 2
```

Nsight profiles 3 mixed-provider runs here: 1 warmup run plus 2 measured runs.

Before tensor residency:

```text
H2D memcpy total: 821.469 MB, 972 copies
D2H memcpy total: 538.704 MB, 579 copies
```

After tensor residency:

```text
H2D memcpy total: 327.252 MB, 447 copies
D2H memcpy total: 253.325 MB, 285 copies
```

Per YOLO run reduction:

```text
H2D: about 164.739 MB less per run
D2H: about  95.126 MB less per run
```

### Remaining Bottleneck

The next gap is provider boundaries:

```text
Concat / Split / Resize / Reshape / Transpose still run on CPU.
CPU boundaries force CUDA tensors to materialize back to host.
```

Next practical optimization:

```text
Add CUDA kernels for common YOLO boundary ops, starting with Concat(axis=1) and Split(axis=1).
```

### Validation

```bash
cmake --build build_cuda --target miniort_compare_providers miniort_run miniort_runtime_tests -j4
ctest --test-dir build_cuda --output-on-failure
ctest --test-dir build_local --output-on-failure
./build_cuda/miniort_run models/yolov8n.onnx --image pic/bus.jpg --strict
git diff --check
```

Results:

```text
strict run: executed=261 skipped=0 materialized_outputs=0
build_cuda ctest: 100% tests passed, 0 tests failed out of 1
build_local ctest: 100% tests passed, 0 tests failed out of 1
```

## Optimization Attempt 3: Optional cuDNN Conv Backend

Date: 2026-06-04

cuDNN development packages were installed after the previous attempt:

```text
libcudnn9-cuda-13         9.22.0.52-1
libcudnn9-dev-cuda-13     9.22.0.52-1
libcudnn9-headers-cuda-13 9.22.0.52-1
```

Environment validation:

```bash
find /usr /usr/local -name 'cudnn.h' -o -name 'cudnn_version.h' 2>/dev/null
ldconfig -p | grep libcudnn
g++ /tmp/check_cudnn.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudnn -o /tmp/check_cudnn
/tmp/check_cudnn
```

Result:

```text
cudnnGetVersion() = 92200
```

### Code Change

Files:

```text
CMakeLists.txt
src/runtime/cuda_execution_provider.cc
```

Changes:

1. Added `MINIORT_BUILD_CUDNN`.
2. CMake now finds `cudnn.h` and `libcudnn.so`.
3. CUDA Conv now tries cuDNN first when available.
4. If cuDNN fails or the Conv padding is asymmetric, it falls back to the CUDA im2col + cuBLAS path.
5. Added a small cuDNN Conv plan cache keyed by Conv shape/attributes:

```text
Conv params -> cudnnConvolutionFwdAlgo_t + workspace_size
```

This mirrors the ONNX Runtime idea that provider execution plans should be prepared/cached instead of recomputed for
every node execution.

### Configure Command

```bash
cmake -S . -B build_cuda \
  -DMINIORT_BUILD_CUDA_EP=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native
```

CMake result:

```text
cuDNN enabled: include=/usr/include/x86_64-linux-gnu library=/usr/lib/x86_64-linux-gnu/libcudnn.so
```

### Benchmark Command

```bash
./build_cuda/miniort_compare_providers models/yolov8n.onnx --image pic/bus.jpg --warmup 3 --repeat 20
```

CUDA im2col baseline, cuDNN disabled:

```text
mixed_ms=76.187
cpu_only_ms=670.461
speedup_pct=88.637
mixed_latency_ms mean=76.187 min=75.421 p50=75.900 p95=77.503 max=78.236
```

cuDNN first attempt, without plan cache:

```text
mixed_ms=87.678
cpu_only_ms=672.852
speedup_pct=86.969
mixed_latency_ms mean=87.678 min=86.476 p50=87.320 p95=89.411 max=89.752
```

cuDNN with plan cache:

```text
mixed_ms=77.537
cpu_only_ms=671.272
speedup_pct=88.449
mixed_latency_ms mean=77.537 min=76.181 p50=77.354 p95=79.096 max=81.300
```

Interpretation:

```text
cuDNN integration is working.
The plan cache reduced cuDNN latency by 10.141 ms versus the uncached cuDNN path.
For this MiniORT YOLO setup, cuDNN is roughly tied with the custom CUDA im2col path, but not clearly faster yet.
```

The reason is that the current runtime still copies every CUDA op output back to CPU and uploads every next CUDA op
input again. cuDNN optimizes Conv math, but it cannot fix MiniORT's tensor residency problem by itself.

### Nsight Systems Command

```bash
/usr/local/cuda/bin/nsys profile --stats=true --force-overwrite=true \
  --output=/tmp/miniort_yolo_cuda_nsys_cudnn_cached \
  ./build_cuda/miniort_compare_providers models/yolov8n.onnx --image pic/bus.jpg --warmup 1 --repeat 2
```

Nsight confirms cuDNN kernels are used:

```text
_5x_cudnn_ampere_scudnn_winograd...
cutlass__5x_cudnn::Kernel...
cudnn::engines_precompiled::nchwToNhwcKernel...
cudnn::engines_precompiled::nhwcToNchwKernel...
```

Memory transfer stayed the same as CUDA im2col:

```text
H2D memcpy total: 821.469 MB, 972 copies
D2H memcpy total: 538.704 MB, 579 copies
```

### Resume-Friendly Summary

Good claim:

```text
Integrated optional cuDNN Conv backend with cached algorithm/workspace planning; reduced uncached cuDNN path from
87.7 ms to 77.5 ms on YOLOv8n and verified cuDNN kernel execution with Nsight Systems.
```

Important nuance:

```text
cuDNN did not materially beat the custom CUDA im2col path in this MiniORT version because end-to-end latency is now
dominated by CPU/GPU tensor movement rather than Conv kernel math alone.
```

### Validation

```bash
cmake --build build_cuda --target miniort_compare_providers miniort_run miniort_runtime_tests -j4
ctest --test-dir build_cuda --output-on-failure
./build_cuda/miniort_run models/yolov8n.onnx --image pic/bus.jpg --strict
ctest --test-dir build_local --output-on-failure
git diff --check
```

Results:

```text
strict run: executed=261 skipped=0 materialized_outputs=0
build_cuda ctest: 100% tests passed, 0 tests failed out of 1
build_local ctest: 100% tests passed, 0 tests failed out of 1
```
