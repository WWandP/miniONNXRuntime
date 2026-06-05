#include "miniort/runtime/cuda_execution_provider.h"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#ifdef MINIORT_BUILD_CUDNN
#include <cudnn.h>
#endif

#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <functional>
#include <unordered_map>
#include <vector>

#include "cuda_elementwise_kernels.h"
#include "kernel_utils.h"
#include "miniort/runtime/cpu_tensor_allocator.h"

namespace miniort {

namespace {

class CudaError : public std::runtime_error {
 public:
  explicit CudaError(const std::string& message) : std::runtime_error(message) {}
};

void CheckCuda(cudaError_t status, const std::string& context) {
  if (status != cudaSuccess) {
    throw CudaError(context + ": " + cudaGetErrorString(status));
  }
}

void CheckCublas(cublasStatus_t status, const std::string& context) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw CudaError(context + ": cuBLAS status " + std::to_string(static_cast<int>(status)));
  }
}

#ifdef MINIORT_BUILD_CUDNN
void CheckCudnn(cudnnStatus_t status, const std::string& context) {
  if (status != CUDNN_STATUS_SUCCESS) {
    throw CudaError(context + ": " + cudnnGetErrorString(status));
  }
}
#endif

class CudaBufferPool {
 public:
  struct Allocation {
    void* data{nullptr};
    std::size_t size{0};
  };

  ~CudaBufferPool() {
    for (const auto& [size, data] : free_buffers_) {
      (void)size;
      (void)cudaFree(data);
    }
  }

  CudaBufferPool(const CudaBufferPool&) = delete;
  CudaBufferPool& operator=(const CudaBufferPool&) = delete;

  Allocation Acquire(std::size_t bytes) {
    if (bytes == 0) {
      return {};
    }

    const auto it = free_buffers_.lower_bound(bytes);
    if (it != free_buffers_.end()) {
      Allocation allocation{it->second, it->first};
      free_buffers_.erase(it);
      return allocation;
    }

    void* data = nullptr;
    CheckCuda(cudaMalloc(&data, bytes), "cudaMalloc");
    return {data, bytes};
  }

  void Release(void* data, std::size_t bytes) noexcept {
    if (data == nullptr) {
      return;
    }

    try {
      free_buffers_.emplace(bytes, data);
    } catch (...) {
      (void)cudaFree(data);
    }
  }

 private:
  CudaBufferPool() = default;

  friend CudaBufferPool& GetCudaBufferPool();

  std::multimap<std::size_t, void*> free_buffers_;
};

CudaBufferPool& GetCudaBufferPool() {
  static CudaBufferPool pool;
  return pool;
}

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t bytes) {
    const auto allocation = GetCudaBufferPool().Acquire(bytes);
    data_ = allocation.data;
    size_ = allocation.size;
  }

  ~DeviceBuffer() {
    Reset();
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this == &other) {
      return *this;
    }
    Reset();
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
    return *this;
  }

  void* data() { return data_; }
  const void* data() const { return data_; }
  std::size_t size() const { return size_; }

  std::shared_ptr<void> ReleaseShared() {
    void* data = data_;
    const auto size = size_;
    data_ = nullptr;
    size_ = 0;
    return std::shared_ptr<void>(data, [size](void* ptr) {
      if (ptr != nullptr) {
        GetCudaBufferPool().Release(ptr, size);
      }
    });
  }

 private:
  void Reset() noexcept {
    if (data_ == nullptr) {
      return;
    }
    GetCudaBufferPool().Release(data_, size_);
    data_ = nullptr;
    size_ = 0;
  }

  void* data_{nullptr};
  std::size_t size_{0};
};

struct CachedCudaInitializer {
  std::shared_ptr<void> data;
  std::size_t bytes{0};
};

std::string MakeCudaInitializerCacheKey(const Tensor& tensor, std::size_t bytes) {
  std::string key = tensor.name;
  key += "|";
  key += tensor.dtype;
  key += "|";
  key += std::to_string(bytes);
  key += "|";
  for (const auto dim : tensor.shape) {
    key += std::to_string(dim);
    key += ",";
  }
  return key;
}

std::unordered_map<std::string, CachedCudaInitializer>& GetCudaInitializerCache() {
  static std::unordered_map<std::string, CachedCudaInitializer> cache;
  return cache;
}

std::mutex& GetCudaInitializerCacheMutex() {
  static std::mutex mutex;
  return mutex;
}

bool TryBindCachedCudaInitializer(Tensor& tensor, std::size_t bytes) {
  if (!tensor.is_initializer) {
    return false;
  }

  const auto key = MakeCudaInitializerCacheKey(tensor, bytes);
  std::lock_guard<std::mutex> lock(GetCudaInitializerCacheMutex());
  const auto it = GetCudaInitializerCache().find(key);
  if (it == GetCudaInitializerCache().end()) {
    return false;
  }
  if (it->second.data == nullptr || it->second.bytes < bytes) {
    return false;
  }
  tensor.cuda_data = it->second.data;
  tensor.cuda_bytes = it->second.bytes;
  return true;
}

void CacheCudaInitializer(const Tensor& tensor, std::size_t bytes) {
  if (!tensor.is_initializer || tensor.cuda_data == nullptr) {
    return;
  }

  const auto key = MakeCudaInitializerCacheKey(tensor, bytes);
  std::lock_guard<std::mutex> lock(GetCudaInitializerCacheMutex());
  GetCudaInitializerCache()[key] = CachedCudaInitializer{tensor.cuda_data, tensor.cuda_bytes};
}

Tensor MakeCudaFloatOutput(const std::string& name, const std::vector<std::int64_t>& shape) {
  Tensor tensor;
  tensor.name = name;
  tensor.dtype = "float32";
  tensor.shape = shape;
  tensor.is_placeholder = false;
  return tensor;
}

float* MutableCudaFloatData(Tensor& tensor, const std::string& op_type) {
  if (tensor.dtype != "float32") {
    throw std::runtime_error(op_type + " requires float32 tensor data: " + tensor.name);
  }
  const auto bytes = GetElementCount(tensor.shape) * sizeof(float);
  if (bytes == 0) {
    return nullptr;
  }
  if (tensor.cuda_data == nullptr) {
    if (TryBindCachedCudaInitializer(tensor, bytes)) {
      return static_cast<float*>(tensor.cuda_data.get());
    }
    const auto& host_data = RequireFloatData(tensor, op_type);
    DeviceBuffer device(bytes);
    CheckCuda(cudaMemcpy(device.data(), host_data.data(), bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy H2D " + op_type + " input");
    tensor.cuda_bytes = device.size();
    tensor.cuda_data = device.ReleaseShared();
    CacheCudaInitializer(tensor, bytes);
  }
  if (tensor.cuda_bytes < bytes) {
    throw std::runtime_error(op_type + " CUDA tensor storage is too small: " + tensor.name);
  }
  return static_cast<float*>(tensor.cuda_data.get());
}

const float* CudaFloatData(Tensor& tensor, const std::string& op_type) {
  return MutableCudaFloatData(tensor, op_type);
}

void BindCudaFloatOutput(Tensor& tensor, DeviceBuffer&& buffer) {
  tensor.cuda_bytes = buffer.size();
  tensor.cuda_data = buffer.ReleaseShared();
}

class CublasHandle {
 public:
  CublasHandle() {
    CheckCublas(cublasCreate(&handle_), "cublasCreate");
  }

  ~CublasHandle() {
    if (handle_ != nullptr) {
      (void)cublasDestroy(handle_);
    }
  }

  CublasHandle(const CublasHandle&) = delete;
  CublasHandle& operator=(const CublasHandle&) = delete;

  cublasHandle_t get() const { return handle_; }

 private:
  cublasHandle_t handle_{nullptr};
};

cublasHandle_t GetCublasHandle() {
  static CublasHandle handle;
  return handle.get();
}

#ifdef MINIORT_BUILD_CUDNN
class CudnnHandle {
 public:
  CudnnHandle() {
    CheckCudnn(cudnnCreate(&handle_), "cudnnCreate");
  }

  ~CudnnHandle() {
    if (handle_ != nullptr) {
      (void)cudnnDestroy(handle_);
    }
  }

  CudnnHandle(const CudnnHandle&) = delete;
  CudnnHandle& operator=(const CudnnHandle&) = delete;

  cudnnHandle_t get() const { return handle_; }

 private:
  cudnnHandle_t handle_{nullptr};
};

cudnnHandle_t GetCudnnHandle() {
  static CudnnHandle handle;
  return handle.get();
}

class CudnnTensorDescriptor {
 public:
  CudnnTensorDescriptor() {
    CheckCudnn(cudnnCreateTensorDescriptor(&descriptor_), "cudnnCreateTensorDescriptor");
  }

  ~CudnnTensorDescriptor() {
    if (descriptor_ != nullptr) {
      (void)cudnnDestroyTensorDescriptor(descriptor_);
    }
  }

  CudnnTensorDescriptor(const CudnnTensorDescriptor&) = delete;
  CudnnTensorDescriptor& operator=(const CudnnTensorDescriptor&) = delete;

  cudnnTensorDescriptor_t get() const { return descriptor_; }

 private:
  cudnnTensorDescriptor_t descriptor_{nullptr};
};

class CudnnFilterDescriptor {
 public:
  CudnnFilterDescriptor() {
    CheckCudnn(cudnnCreateFilterDescriptor(&descriptor_), "cudnnCreateFilterDescriptor");
  }

  ~CudnnFilterDescriptor() {
    if (descriptor_ != nullptr) {
      (void)cudnnDestroyFilterDescriptor(descriptor_);
    }
  }

  CudnnFilterDescriptor(const CudnnFilterDescriptor&) = delete;
  CudnnFilterDescriptor& operator=(const CudnnFilterDescriptor&) = delete;

  cudnnFilterDescriptor_t get() const { return descriptor_; }

 private:
  cudnnFilterDescriptor_t descriptor_{nullptr};
};

class CudnnConvolutionDescriptor {
 public:
  CudnnConvolutionDescriptor() {
    CheckCudnn(cudnnCreateConvolutionDescriptor(&descriptor_), "cudnnCreateConvolutionDescriptor");
  }

  ~CudnnConvolutionDescriptor() {
    if (descriptor_ != nullptr) {
      (void)cudnnDestroyConvolutionDescriptor(descriptor_);
    }
  }

  CudnnConvolutionDescriptor(const CudnnConvolutionDescriptor&) = delete;
  CudnnConvolutionDescriptor& operator=(const CudnnConvolutionDescriptor&) = delete;

  cudnnConvolutionDescriptor_t get() const { return descriptor_; }

 private:
  cudnnConvolutionDescriptor_t descriptor_{nullptr};
};
#endif

struct Conv2DParams {
  std::size_t n{0};
  std::size_t c_in{0};
  std::size_t h_in{0};
  std::size_t w_in{0};
  std::size_t c_out{0};
  std::size_t k_h{0};
  std::size_t k_w{0};
  std::int64_t pad_top{0};
  std::int64_t pad_left{0};
  std::int64_t pad_bottom{0};
  std::int64_t pad_right{0};
  std::int64_t dilation_h{1};
  std::int64_t dilation_w{1};
  std::int64_t stride_h{1};
  std::int64_t stride_w{1};
  std::int64_t h_out{0};
  std::int64_t w_out{0};
};

#ifdef MINIORT_BUILD_CUDNN
struct CudnnConvPlan {
  cudnnConvolutionFwdAlgo_t algorithm{CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM};
  std::size_t workspace_size{0};
};

std::string MakeCudnnConvPlanKey(const Conv2DParams& params) {
  return std::to_string(params.n) + ":" + std::to_string(params.c_in) + ":" + std::to_string(params.h_in) + ":" +
         std::to_string(params.w_in) + ":" + std::to_string(params.c_out) + ":" + std::to_string(params.k_h) + ":" +
         std::to_string(params.k_w) + ":" + std::to_string(params.pad_top) + ":" +
         std::to_string(params.pad_left) + ":" + std::to_string(params.stride_h) + ":" +
         std::to_string(params.stride_w) + ":" + std::to_string(params.dilation_h) + ":" +
         std::to_string(params.dilation_w) + ":" + std::to_string(params.h_out) + ":" +
         std::to_string(params.w_out);
}

CudnnConvPlan GetOrCreateCudnnConvPlan(cudnnHandle_t handle, const CudnnTensorDescriptor& input_desc,
                                       const CudnnFilterDescriptor& weight_desc,
                                       const CudnnConvolutionDescriptor& conv_desc,
                                       const CudnnTensorDescriptor& output_desc, const Conv2DParams& params) {
  static std::map<std::string, CudnnConvPlan> plan_cache;
  const auto key = MakeCudnnConvPlanKey(params);
  const auto it = plan_cache.find(key);
  if (it != plan_cache.end()) {
    return it->second;
  }

  int returned_algo_count = 0;
  cudnnConvolutionFwdAlgoPerf_t perf_results[CUDNN_CONVOLUTION_FWD_ALGO_COUNT]{};
  CheckCudnn(cudnnGetConvolutionForwardAlgorithm_v7(handle, input_desc.get(), weight_desc.get(), conv_desc.get(),
                                                    output_desc.get(), CUDNN_CONVOLUTION_FWD_ALGO_COUNT,
                                                    &returned_algo_count, perf_results),
             "cudnnGetConvolutionForwardAlgorithm_v7");
  if (returned_algo_count <= 0) {
    throw CudaError("cuDNN Conv did not return a forward algorithm");
  }

  CudnnConvPlan plan;
  plan.algorithm = perf_results[0].algo;
  CheckCudnn(cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc.get(), weight_desc.get(), conv_desc.get(),
                                                     output_desc.get(), plan.algorithm, &plan.workspace_size),
             "cudnnGetConvolutionForwardWorkspaceSize");
  plan_cache.emplace(key, plan);
  return plan;
}
#endif

float ReadFloatAttribute(const Node& node, const std::string& name, float default_value) {
  const auto it = node.attributes.find(name);
  return it == node.attributes.end() ? default_value : it->second.float_value;
}

Conv2DParams ResolveConv2DParams(const Node& node, const Tensor& input, const Tensor& weight, const Tensor* bias) {
  if (input.shape.size() != 4 || weight.shape.size() != 4) {
    throw std::runtime_error("Conv currently only supports 2D NCHW tensors");
  }

  const auto group = ReadIntAttribute(node, "group", 1);
  if (group != 1) {
    throw std::runtime_error("Conv currently only supports group=1");
  }

  const auto dilations = ReadIntsAttribute(node, "dilations", {1, 1});
  const auto strides = ReadIntsAttribute(node, "strides", {1, 1});
  const auto pads = ReadIntsAttribute(node, "pads", {0, 0, 0, 0});
  if (dilations.size() != 2 || strides.size() != 2 || pads.size() != 4) {
    throw std::runtime_error("Conv attribute rank is not supported");
  }

  Conv2DParams params;
  params.n = static_cast<std::size_t>(input.shape[0]);
  params.c_in = static_cast<std::size_t>(input.shape[1]);
  params.h_in = static_cast<std::size_t>(input.shape[2]);
  params.w_in = static_cast<std::size_t>(input.shape[3]);
  params.c_out = static_cast<std::size_t>(weight.shape[0]);
  const auto weight_c_in = static_cast<std::size_t>(weight.shape[1]);
  params.k_h = static_cast<std::size_t>(weight.shape[2]);
  params.k_w = static_cast<std::size_t>(weight.shape[3]);

  if (params.c_in != weight_c_in) {
    throw std::runtime_error("Conv input channel count does not match weight");
  }
  if (bias != nullptr && RequireFloatData(*bias, "Conv").size() != params.c_out) {
    throw std::runtime_error("Conv bias size does not match output channels");
  }

  params.pad_top = pads[0];
  params.pad_left = pads[1];
  params.pad_bottom = pads[2];
  params.pad_right = pads[3];
  params.dilation_h = dilations[0];
  params.dilation_w = dilations[1];
  params.stride_h = strides[0];
  params.stride_w = strides[1];

  const auto effective_kh = static_cast<std::int64_t>((params.k_h - 1) * params.dilation_h + 1);
  const auto effective_kw = static_cast<std::int64_t>((params.k_w - 1) * params.dilation_w + 1);
  params.h_out = (static_cast<std::int64_t>(params.h_in) + params.pad_top + params.pad_bottom - effective_kh) /
                     params.stride_h +
                 1;
  params.w_out = (static_cast<std::int64_t>(params.w_in) + params.pad_left + params.pad_right - effective_kw) /
                     params.stride_w +
                 1;
  if (params.h_out <= 0 || params.w_out <= 0) {
    throw std::runtime_error("Conv output shape is invalid");
  }

  return params;
}

template <typename FloatOp, typename IntOp>
Tensor RunBinaryNumericFallback(const Node& node, ExecutionContext& context, const std::string& op_type,
                                FloatOp eval_float, IntOp eval_int) {
  const auto& lhs = RequireTensor(context, node.inputs.at(0));
  const auto& rhs = RequireTensor(context, node.inputs.at(1));
  const auto output_shape = ComputeBroadcastShape(lhs.shape, rhs.shape, op_type);
  const auto output_strides = ComputeStrides(output_shape);
  const auto lhs_strides = ComputeStrides(lhs.shape);
  const auto rhs_strides = ComputeStrides(rhs.shape);
  const auto element_count = GetElementCount(output_shape);

  if (lhs.dtype == "int64" && rhs.dtype == "int64") {
    const auto& lhs_data = RequireInt64Data(lhs, op_type);
    const auto& rhs_data = RequireInt64Data(rhs, op_type);
    auto output = MakeInt64Output(node.outputs.at(0), output_shape, context);
    for (std::size_t i = 0; i < element_count; ++i) {
      const auto output_index = UnravelIndex(i, output_shape, output_strides);
      const auto lhs_offset = ComputeBroadcastOffset(output_index, lhs.shape, lhs_strides);
      const auto rhs_offset = ComputeBroadcastOffset(output_index, rhs.shape, rhs_strides);
      output.int64_data[i] = eval_int(lhs_data[lhs_offset], rhs_data[rhs_offset]);
    }
    return output;
  }

  const auto* lhs_float_data = lhs.dtype == "float32" ? &RequireFloatData(lhs, op_type) : nullptr;
  const auto* lhs_int_data = lhs.dtype == "int64" ? &RequireInt64Data(lhs, op_type) : nullptr;
  const auto* rhs_float_data = rhs.dtype == "float32" ? &RequireFloatData(rhs, op_type) : nullptr;
  const auto* rhs_int_data = rhs.dtype == "int64" ? &RequireInt64Data(rhs, op_type) : nullptr;

  auto output = MakeFloatOutput(node.outputs.at(0), output_shape, context);
  for (std::size_t i = 0; i < element_count; ++i) {
    const auto output_index = UnravelIndex(i, output_shape, output_strides);
    const auto lhs_offset = ComputeBroadcastOffset(output_index, lhs.shape, lhs_strides);
    const auto rhs_offset = ComputeBroadcastOffset(output_index, rhs.shape, rhs_strides);
    const auto lhs_value =
        lhs_float_data != nullptr ? (*lhs_float_data)[lhs_offset] : static_cast<float>((*lhs_int_data)[lhs_offset]);
    const auto rhs_value =
        rhs_float_data != nullptr ? (*rhs_float_data)[rhs_offset] : static_cast<float>((*rhs_int_data)[rhs_offset]);
    output.float_data[i] = eval_float(lhs_value, rhs_value);
  }
  return output;
}

void ApplyGemmBias(Tensor& output, const Tensor* bias) {
  if (bias == nullptr) {
    return;
  }
  const auto& bias_data = RequireFloatData(*bias, "CUDA Gemm");
  if (output.shape.size() != 2) {
    throw std::runtime_error("CUDA Gemm output must be 2D");
  }
  const auto m = static_cast<std::size_t>(output.shape[0]);
  const auto n = static_cast<std::size_t>(output.shape[1]);

  if (bias->shape.empty() && bias_data.size() == 1) {
    for (auto& value : output.float_data) {
      value += bias_data[0];
    }
    return;
  }
  if (bias->shape.size() == 1 && bias_data.size() == n) {
    for (std::size_t i = 0; i < m; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        output.float_data[i * n + j] += bias_data[j];
      }
    }
    return;
  }
  if (bias->shape.size() == 1 && bias_data.size() == m) {
    for (std::size_t i = 0; i < m; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        output.float_data[i * n + j] += bias_data[i];
      }
    }
    return;
  }
  if (bias->shape.size() == 2 &&
      static_cast<std::size_t>(bias->shape[0]) == m &&
      static_cast<std::size_t>(bias->shape[1]) == n &&
      bias_data.size() == m * n) {
    for (std::size_t i = 0; i < m * n; ++i) {
      output.float_data[i] += bias_data[i];
    }
    return;
  }

  throw std::runtime_error("CUDA Gemm bias shape is not supported");
}

Tensor RunCudaMatMul(const Node& node, const Tensor& lhs, const Tensor& rhs, ExecutionContext& context) {
  const auto& lhs_data = RequireFloatData(lhs, "CUDA MatMul");
  const auto& rhs_data = RequireFloatData(rhs, "CUDA MatMul");
  if (lhs.shape.size() < 2 || rhs.shape.size() < 2) {
    throw std::runtime_error("CUDA MatMul currently requires rank >= 2 float32 tensors");
  }

  const auto m = static_cast<std::size_t>(lhs.shape[lhs.shape.size() - 2]);
  const auto k = static_cast<std::size_t>(lhs.shape[lhs.shape.size() - 1]);
  const auto rhs_k = static_cast<std::size_t>(rhs.shape[rhs.shape.size() - 2]);
  const auto n = static_cast<std::size_t>(rhs.shape[rhs.shape.size() - 1]);
  if (k != rhs_k) {
    throw std::runtime_error("CUDA MatMul inner dimensions do not match");
  }

  const std::vector<std::int64_t> lhs_batch_shape(lhs.shape.begin(), lhs.shape.end() - 2);
  const std::vector<std::int64_t> rhs_batch_shape(rhs.shape.begin(), rhs.shape.end() - 2);
  const auto output_batch_shape = ComputeBroadcastShape(lhs_batch_shape, rhs_batch_shape, "MatMul");

  std::vector<std::int64_t> output_shape = output_batch_shape;
  output_shape.push_back(static_cast<std::int64_t>(m));
  output_shape.push_back(static_cast<std::int64_t>(n));

  auto output = MakeFloatOutput(node.outputs.at(0), output_shape, context);
  const auto output_batch_strides = ComputeStrides(output_batch_shape);
  const auto lhs_full_strides = ComputeStrides(lhs.shape);
  const auto rhs_full_strides = ComputeStrides(rhs.shape);
  const auto batch_count = GetElementCount(output_batch_shape);

  const std::size_t lhs_matrix_elements = m * k;
  const std::size_t rhs_matrix_elements = k * n;
  const std::size_t out_matrix_elements = m * n;

  DeviceBuffer lhs_device(lhs_matrix_elements * sizeof(float));
  DeviceBuffer rhs_device(rhs_matrix_elements * sizeof(float));
  DeviceBuffer out_device(out_matrix_elements * sizeof(float));
  const auto handle = GetCublasHandle();

  const float alpha = 1.0f;
  const float beta = 0.0f;

  for (std::size_t batch = 0; batch < batch_count; ++batch) {
    const auto batch_index = UnravelIndex(batch, output_batch_shape, output_batch_strides);
    const auto lhs_batch_offset =
        lhs_batch_shape.empty() ? 0 : ComputeBroadcastOffset(batch_index, lhs_batch_shape, lhs_full_strides);
    const auto rhs_batch_offset =
        rhs_batch_shape.empty() ? 0 : ComputeBroadcastOffset(batch_index, rhs_batch_shape, rhs_full_strides);
    const auto lhs_base = lhs_batch_shape.empty() ? 0 : lhs_batch_offset;
    const auto rhs_base = rhs_batch_shape.empty() ? 0 : rhs_batch_offset;
    const auto output_base = batch * out_matrix_elements;

    CheckCuda(cudaMemcpy(lhs_device.data(), lhs_data.data() + lhs_base, lhs_matrix_elements * sizeof(float),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy H2D lhs");
    CheckCuda(cudaMemcpy(rhs_device.data(), rhs_data.data() + rhs_base, rhs_matrix_elements * sizeof(float),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy H2D rhs");

    // cuBLAS assumes column-major storage. Using swapped operands maps our
    // row-major MatMul into an equivalent column-major GEMM.
    CheckCublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(n), static_cast<int>(m),
                            static_cast<int>(k), &alpha, static_cast<const float*>(rhs_device.data()),
                            static_cast<int>(n), static_cast<const float*>(lhs_device.data()), static_cast<int>(k),
                            &beta, static_cast<float*>(out_device.data()), static_cast<int>(n)),
                "cublasSgemm");

    CheckCuda(cudaMemcpy(output.float_data.data() + output_base, out_device.data(), out_matrix_elements * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy D2H output");
  }

  return output;
}

Tensor RunCudaGemm(const Node& node, const Tensor& a, const Tensor& b, const Tensor* c, ExecutionContext& context) {
  const auto& a_data = RequireFloatData(a, "CUDA Gemm");
  const auto& b_data = RequireFloatData(b, "CUDA Gemm");
  if (a.shape.size() != 2 || b.shape.size() != 2) {
    throw std::runtime_error("CUDA Gemm currently only supports 2D float32 tensors");
  }

  const auto trans_a = ReadIntAttribute(node, "transA", 0) != 0;
  const auto trans_b = ReadIntAttribute(node, "transB", 0) != 0;
  const float alpha = ReadFloatAttribute(node, "alpha", 1.0f);
  const float beta = 0.0f;
  const float bias_scale = ReadFloatAttribute(node, "beta", 1.0f);

  const auto a_rows = static_cast<std::size_t>(a.shape[0]);
  const auto a_cols = static_cast<std::size_t>(a.shape[1]);
  const auto b_rows = static_cast<std::size_t>(b.shape[0]);
  const auto b_cols = static_cast<std::size_t>(b.shape[1]);

  const auto m = trans_a ? a_cols : a_rows;
  const auto k_a = trans_a ? a_rows : a_cols;
  const auto k_b = trans_b ? b_cols : b_rows;
  const auto n = trans_b ? b_rows : b_cols;
  if (k_a != k_b) {
    throw std::runtime_error("CUDA Gemm inner dimensions do not match");
  }

  auto output = MakeFloatOutput(node.outputs.at(0),
                                {static_cast<std::int64_t>(m), static_cast<std::int64_t>(n)},
                                context);

  const std::size_t a_elements = a_rows * a_cols;
  const std::size_t b_elements = b_rows * b_cols;
  const std::size_t out_elements = m * n;

  DeviceBuffer a_device(a_elements * sizeof(float));
  DeviceBuffer b_device(b_elements * sizeof(float));
  DeviceBuffer out_device(out_elements * sizeof(float));
  const auto handle = GetCublasHandle();

  CheckCuda(cudaMemcpy(a_device.data(), a_data.data(), a_elements * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy H2D Gemm A");
  CheckCuda(cudaMemcpy(b_device.data(), b_data.data(), b_elements * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy H2D Gemm B");

  const auto op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  const auto op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

  // Map row-major Gemm to column-major cuBLAS by swapping A/B and output axes.
  CheckCublas(cublasSgemm(handle, op_b, op_a, static_cast<int>(n), static_cast<int>(m),
                          static_cast<int>(k_a), &alpha, static_cast<const float*>(b_device.data()),
                          static_cast<int>(b_cols), static_cast<const float*>(a_device.data()),
                          static_cast<int>(a_cols), &beta, static_cast<float*>(out_device.data()),
                          static_cast<int>(n)),
              "cublasSgemm Gemm");

  CheckCuda(cudaMemcpy(output.float_data.data(), out_device.data(), out_elements * sizeof(float),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H Gemm output");

  if (c != nullptr) {
    if (bias_scale != 1.0f) {
      Tensor scaled_bias = *c;
      scaled_bias.float_data = c->float_data;
      for (auto& value : scaled_bias.float_data) {
        value *= bias_scale;
      }
      ApplyGemmBias(output, &scaled_bias);
    } else {
      ApplyGemmBias(output, c);
    }
  }

  return output;
}

#ifdef MINIORT_BUILD_CUDNN
Tensor RunCudnnConv2D(const Node& node, Tensor& input, Tensor& weight, Tensor* bias,
                      const Conv2DParams& params, ExecutionContext& context) {
  if (params.pad_top != params.pad_bottom || params.pad_left != params.pad_right) {
    throw CudaError("cuDNN Conv only handles symmetric padding in this path");
  }

  auto output = MakeFloatOutput(node.outputs.at(0),
                                {static_cast<std::int64_t>(params.n), static_cast<std::int64_t>(params.c_out),
                                 params.h_out, params.w_out},
                                context);
  const auto output_hw = static_cast<std::size_t>(params.h_out) * static_cast<std::size_t>(params.w_out);

  DeviceBuffer output_device(output.float_data.size() * sizeof(float));

  CudnnTensorDescriptor input_desc;
  CudnnFilterDescriptor weight_desc;
  CudnnTensorDescriptor output_desc;
  CudnnConvolutionDescriptor conv_desc;
  const auto handle = GetCudnnHandle();

  CheckCudnn(cudnnSetTensor4dDescriptor(input_desc.get(), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                        static_cast<int>(params.n), static_cast<int>(params.c_in),
                                        static_cast<int>(params.h_in), static_cast<int>(params.w_in)),
             "cudnnSetTensor4dDescriptor input");
  CheckCudnn(cudnnSetFilter4dDescriptor(weight_desc.get(), CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                        static_cast<int>(params.c_out), static_cast<int>(params.c_in),
                                        static_cast<int>(params.k_h), static_cast<int>(params.k_w)),
             "cudnnSetFilter4dDescriptor weight");
  CheckCudnn(cudnnSetTensor4dDescriptor(output_desc.get(), CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                        static_cast<int>(params.n), static_cast<int>(params.c_out),
                                        static_cast<int>(params.h_out), static_cast<int>(params.w_out)),
             "cudnnSetTensor4dDescriptor output");
  CheckCudnn(cudnnSetConvolution2dDescriptor(conv_desc.get(), static_cast<int>(params.pad_top),
                                             static_cast<int>(params.pad_left), static_cast<int>(params.stride_h),
                                             static_cast<int>(params.stride_w), static_cast<int>(params.dilation_h),
                                             static_cast<int>(params.dilation_w), CUDNN_CROSS_CORRELATION,
                                             CUDNN_DATA_FLOAT),
             "cudnnSetConvolution2dDescriptor");
  CheckCudnn(cudnnSetConvolutionMathType(conv_desc.get(), CUDNN_DEFAULT_MATH),
             "cudnnSetConvolutionMathType");

  const auto plan =
      GetOrCreateCudnnConvPlan(handle, input_desc, weight_desc, conv_desc, output_desc, params);
  DeviceBuffer workspace(plan.workspace_size);

  const float alpha = 1.0f;
  const float beta = 0.0f;
  CheckCudnn(cudnnConvolutionForward(handle, &alpha, input_desc.get(), CudaFloatData(input, "cuDNN Conv"),
                                     weight_desc.get(), CudaFloatData(weight, "cuDNN Conv"), conv_desc.get(),
                                     plan.algorithm, workspace.data(),
                                     plan.workspace_size, &beta, output_desc.get(), output_device.data()),
             "cudnnConvolutionForward");

  if (bias != nullptr) {
    CheckCuda(LaunchCudaAddChannelBias2D(static_cast<float*>(output_device.data()), CudaFloatData(*bias, "cuDNN Conv"),
                                         params.n, params.c_out, static_cast<std::size_t>(params.h_out),
                                         static_cast<std::size_t>(params.w_out)),
              "Conv bias kernel launch");
  }
  output.float_data.clear();
  BindCudaFloatOutput(output, std::move(output_device));

  return output;
}
#endif

Tensor RunCudaConv2DIm2Col(const Node& node, Tensor& input, Tensor& weight, Tensor* bias,
                           const Conv2DParams& params, ExecutionContext& context) {
  auto output = MakeFloatOutput(node.outputs.at(0),
                                {static_cast<std::int64_t>(params.n), static_cast<std::int64_t>(params.c_out),
                                 params.h_out, params.w_out},
                                context);

  const auto input_hw = params.h_in * params.w_in;
  const auto output_hw = static_cast<std::size_t>(params.h_out) * static_cast<std::size_t>(params.w_out);
  const auto kernel_dim = params.c_in * params.k_h * params.k_w;

  DeviceBuffer columns_device(kernel_dim * output_hw * sizeof(float));
  DeviceBuffer output_device(params.c_out * output_hw * sizeof(float));
  const auto handle = GetCublasHandle();

  const float alpha = 1.0f;
  const float beta = 0.0f;

  for (std::size_t batch = 0; batch < params.n; ++batch) {
    const auto* batch_input =
        CudaFloatData(input, "CUDA Conv") + batch * params.c_in * input_hw;
    auto* batch_output = output.float_data.data() + batch * params.c_out * output_hw;

    CheckCuda(LaunchCudaIm2Col2D(batch_input, static_cast<float*>(columns_device.data()), params.c_in, params.h_in,
                                 params.w_in, static_cast<std::size_t>(params.h_out),
                                 static_cast<std::size_t>(params.w_out), params.k_h, params.k_w, params.stride_h,
                                 params.stride_w, params.dilation_h, params.dilation_w, params.pad_top,
                                 params.pad_left),
              "Im2Col kernel launch");

    CheckCublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(output_hw),
                            static_cast<int>(params.c_out), static_cast<int>(kernel_dim), &alpha,
                            static_cast<const float*>(columns_device.data()), static_cast<int>(output_hw),
                            CudaFloatData(weight, "CUDA Conv"), static_cast<int>(kernel_dim), &beta,
                            static_cast<float*>(output_device.data()), static_cast<int>(output_hw)),
                "cublasSgemm Conv");

    if (bias != nullptr && params.n == 1) {
      CheckCuda(LaunchCudaAddChannelBias2D(static_cast<float*>(output_device.data()), CudaFloatData(*bias, "CUDA Conv"),
                                           1, params.c_out, static_cast<std::size_t>(params.h_out),
                                           static_cast<std::size_t>(params.w_out)),
                "Conv bias kernel launch");
    }

    if (params.n != 1) {
      CheckCuda(cudaMemcpy(batch_output, output_device.data(), params.c_out * output_hw * sizeof(float),
                           cudaMemcpyDeviceToHost),
                "cudaMemcpy D2H Conv output");
    }

    if (bias != nullptr && params.n != 1) {
      const auto& bias_data = RequireFloatData(*bias, "CUDA Conv");
      for (std::size_t oc = 0; oc < params.c_out; ++oc) {
        const float bias_value = bias_data[oc];
        auto* output_plane = batch_output + oc * output_hw;
        for (std::size_t i = 0; i < output_hw; ++i) {
          output_plane[i] += bias_value;
        }
      }
    }
  }

  if (params.n == 1) {
    output.float_data.clear();
    BindCudaFloatOutput(output, std::move(output_device));
  }

  return output;
}

Tensor RunCudaConv2D(const Node& node, const Tensor& input, const Tensor& weight, Tensor* bias,
                     ExecutionContext& context) {
  auto* input_tensor = context.FindTensor(node.inputs.at(0));
  auto* weight_tensor = context.FindTensor(node.inputs.at(1));
  if (input_tensor == nullptr || weight_tensor == nullptr) {
    throw std::runtime_error("missing CUDA Conv input");
  }
  const auto params = ResolveConv2DParams(node, *input_tensor, *weight_tensor, bias);
#ifdef MINIORT_BUILD_CUDNN
  try {
    return RunCudnnConv2D(node, *input_tensor, *weight_tensor, bias, params, context);
  } catch (const CudaError&) {
  }
#endif
  return RunCudaConv2DIm2Col(node, *input_tensor, *weight_tensor, bias, params, context);
}

Tensor RunMaxPoolFallback(const Node& node, const Tensor& input, ExecutionContext& context) {
  const auto& input_data = RequireFloatData(input, "MaxPool");
  if (input.shape.size() != 4) {
    throw std::runtime_error("MaxPool currently only supports 2D NCHW tensors");
  }

  const auto kernel_shape = ReadIntsAttribute(node, "kernel_shape", {});
  const auto strides = ReadIntsAttribute(node, "strides", {1, 1});
  const auto pads = ReadIntsAttribute(node, "pads", {0, 0, 0, 0});
  const auto dilations = ReadIntsAttribute(node, "dilations", {1, 1});
  const auto ceil_mode = ReadIntAttribute(node, "ceil_mode", 0);
  if (kernel_shape.size() != 2 || strides.size() != 2 || pads.size() != 4 || dilations.size() != 2) {
    throw std::runtime_error("MaxPool attribute rank is not supported");
  }
  if (ceil_mode != 0) {
    throw std::runtime_error("MaxPool currently only supports ceil_mode=0");
  }

  const auto n = static_cast<std::size_t>(input.shape[0]);
  const auto c = static_cast<std::size_t>(input.shape[1]);
  const auto h_in = static_cast<std::size_t>(input.shape[2]);
  const auto w_in = static_cast<std::size_t>(input.shape[3]);
  const auto k_h = static_cast<std::size_t>(kernel_shape[0]);
  const auto k_w = static_cast<std::size_t>(kernel_shape[1]);
  const auto stride_h = strides[0];
  const auto stride_w = strides[1];
  const auto dilation_h = dilations[0];
  const auto dilation_w = dilations[1];
  const auto pad_top = pads[0];
  const auto pad_left = pads[1];
  const auto pad_bottom = pads[2];
  const auto pad_right = pads[3];

  const auto effective_kh = static_cast<std::int64_t>((k_h - 1) * dilation_h + 1);
  const auto effective_kw = static_cast<std::int64_t>((k_w - 1) * dilation_w + 1);
  const auto h_out = (static_cast<std::int64_t>(h_in) + pad_top + pad_bottom - effective_kh) / stride_h + 1;
  const auto w_out = (static_cast<std::int64_t>(w_in) + pad_left + pad_right - effective_kw) / stride_w + 1;
  if (h_out <= 0 || w_out <= 0) {
    throw std::runtime_error("MaxPool output shape is invalid");
  }

  auto output = MakeCudaFloatOutput(node.outputs.at(0),
                                    {static_cast<std::int64_t>(n), static_cast<std::int64_t>(c), h_out, w_out});

  const auto input_hw = h_in * w_in;
  const auto output_hw = static_cast<std::size_t>(h_out) * static_cast<std::size_t>(w_out);
  for (std::size_t batch = 0; batch < n; ++batch) {
    for (std::size_t channel = 0; channel < c; ++channel) {
      for (std::int64_t oh = 0; oh < h_out; ++oh) {
        for (std::int64_t ow = 0; ow < w_out; ++ow) {
          float best = -std::numeric_limits<float>::infinity();
          for (std::size_t kh = 0; kh < k_h; ++kh) {
            for (std::size_t kw = 0; kw < k_w; ++kw) {
              const auto ih = oh * stride_h - pad_top + static_cast<std::int64_t>(kh) * dilation_h;
              const auto iw = ow * stride_w - pad_left + static_cast<std::int64_t>(kw) * dilation_w;
              if (ih < 0 || iw < 0 || ih >= static_cast<std::int64_t>(h_in) ||
                  iw >= static_cast<std::int64_t>(w_in)) {
                continue;
              }
              const auto input_index = ((batch * c + channel) * input_hw) +
                                       static_cast<std::size_t>(ih) * w_in + static_cast<std::size_t>(iw);
              best = std::max(best, input_data[input_index]);
            }
          }
          const auto output_index = ((batch * c + channel) * output_hw) +
                                    static_cast<std::size_t>(oh) * static_cast<std::size_t>(w_out) +
                                    static_cast<std::size_t>(ow);
          output.float_data[output_index] = best;
        }
      }
    }
  }
  return output;
}

Tensor RunCudaMaxPool2D(const Node& node, const Tensor& input, ExecutionContext& context) {
  auto* input_tensor = context.FindTensor(node.inputs.at(0));
  if (input_tensor == nullptr) {
    throw std::runtime_error("missing CUDA MaxPool input");
  }
  if (input.shape.size() != 4) {
    throw std::runtime_error("MaxPool currently only supports 2D NCHW tensors");
  }

  const auto kernel_shape = ReadIntsAttribute(node, "kernel_shape", {});
  const auto strides = ReadIntsAttribute(node, "strides", {1, 1});
  const auto pads = ReadIntsAttribute(node, "pads", {0, 0, 0, 0});
  const auto dilations = ReadIntsAttribute(node, "dilations", {1, 1});
  const auto ceil_mode = ReadIntAttribute(node, "ceil_mode", 0);
  if (kernel_shape.size() != 2 || strides.size() != 2 || pads.size() != 4 || dilations.size() != 2) {
    throw std::runtime_error("MaxPool attribute rank is not supported");
  }
  if (ceil_mode != 0) {
    throw std::runtime_error("MaxPool currently only supports ceil_mode=0");
  }

  const auto n = static_cast<std::size_t>(input.shape[0]);
  const auto c = static_cast<std::size_t>(input.shape[1]);
  const auto h_in = static_cast<std::size_t>(input.shape[2]);
  const auto w_in = static_cast<std::size_t>(input.shape[3]);
  const auto k_h = static_cast<std::size_t>(kernel_shape[0]);
  const auto k_w = static_cast<std::size_t>(kernel_shape[1]);
  const auto stride_h = strides[0];
  const auto stride_w = strides[1];
  const auto dilation_h = dilations[0];
  const auto dilation_w = dilations[1];
  const auto pad_top = pads[0];
  const auto pad_left = pads[1];
  const auto pad_bottom = pads[2];
  const auto pad_right = pads[3];

  const auto effective_kh = static_cast<std::int64_t>((k_h - 1) * dilation_h + 1);
  const auto effective_kw = static_cast<std::int64_t>((k_w - 1) * dilation_w + 1);
  const auto h_out = (static_cast<std::int64_t>(h_in) + pad_top + pad_bottom - effective_kh) / stride_h + 1;
  const auto w_out = (static_cast<std::int64_t>(w_in) + pad_left + pad_right - effective_kw) / stride_w + 1;
  if (h_out <= 0 || w_out <= 0) {
    throw std::runtime_error("MaxPool output shape is invalid");
  }

  auto output = MakeFloatOutput(node.outputs.at(0),
                                {static_cast<std::int64_t>(n), static_cast<std::int64_t>(c), h_out, w_out},
                                context);
  const auto output_count = n * c * static_cast<std::size_t>(h_out) * static_cast<std::size_t>(w_out);

  DeviceBuffer output_device(output_count * sizeof(float));
  CheckCuda(LaunchCudaMaxPool2D(CudaFloatData(*input_tensor, "CUDA MaxPool"), static_cast<float*>(output_device.data()),
                                n, c, h_in, w_in, static_cast<std::size_t>(h_out), static_cast<std::size_t>(w_out),
                                k_h, k_w, stride_h, stride_w, dilation_h, dilation_w, pad_top, pad_left),
            "MaxPool kernel launch");
  output.float_data.clear();
  BindCudaFloatOutput(output, std::move(output_device));
  return output;
}

Tensor RunCudaUnaryFloatOp(const std::string& op_name, const std::string& output_name, Tensor& input,
                           ExecutionContext& context,
                           const std::function<cudaError_t(const float*, float*, std::size_t)>& launcher) {
  (void)context;
  const auto element_count = GetElementCount(input.shape);
  auto output = MakeCudaFloatOutput(output_name, input.shape);

  DeviceBuffer output_device(element_count * sizeof(float));

  CheckCuda(launcher(CudaFloatData(input, op_name), static_cast<float*>(output_device.data()), element_count),
            op_name + " kernel launch");
  BindCudaFloatOutput(output, std::move(output_device));

  return output;
}

Tensor RunUnaryFloatFallback(const std::string& output_name, const Tensor& input, ExecutionContext& context,
                             const std::function<float(float)>& eval) {
  const auto& input_data = RequireFloatData(input, "CUDA unary fallback");
  auto output = MakeOutputLikeWithReusedStorage(output_name, input, context);
  for (std::size_t i = 0; i < input_data.size(); ++i) {
    output.float_data[i] = eval(input_data[i]);
  }
  return output;
}

Tensor RunCudaBinaryFloatOp(const Node& node, ExecutionContext& context, const std::string& op_name,
                            CudaBinaryFloatOp op_kind) {
  const auto& lhs = RequireTensor(context, node.inputs.at(0));
  const auto& rhs = RequireTensor(context, node.inputs.at(1));

  const auto eval_float = [&](float lhs_value, float rhs_value) -> float {
    switch (op_kind) {
      case CudaBinaryFloatOp::kAdd:
        return lhs_value + rhs_value;
      case CudaBinaryFloatOp::kSub:
        return lhs_value - rhs_value;
      case CudaBinaryFloatOp::kMul:
        return lhs_value * rhs_value;
      case CudaBinaryFloatOp::kDiv:
        if (rhs_value == 0.0f) {
          throw std::runtime_error("Div divisor must not be zero");
        }
        return lhs_value / rhs_value;
    }
    throw std::runtime_error("unsupported CUDA binary float op");
  };

  const auto eval_int = [&](std::int64_t lhs_value, std::int64_t rhs_value) -> std::int64_t {
    switch (op_kind) {
      case CudaBinaryFloatOp::kAdd:
        return lhs_value + rhs_value;
      case CudaBinaryFloatOp::kSub:
        return lhs_value - rhs_value;
      case CudaBinaryFloatOp::kMul:
        return lhs_value * rhs_value;
      case CudaBinaryFloatOp::kDiv:
        if (rhs_value == 0) {
          throw std::runtime_error("Div divisor must not be zero");
        }
        return lhs_value / rhs_value;
    }
    throw std::runtime_error("unsupported CUDA binary int op");
  };

  if (lhs.dtype != "float32" || rhs.dtype != "float32") {
    return RunBinaryNumericFallback(node, context, op_name, eval_float, eval_int);
  }

  auto* lhs_tensor = context.FindTensor(node.inputs.at(0));
  auto* rhs_tensor = context.FindTensor(node.inputs.at(1));
  if (lhs_tensor == nullptr || rhs_tensor == nullptr) {
    throw std::runtime_error("missing CUDA binary input");
  }
  const auto lhs_count = GetElementCount(lhs.shape);
  const auto rhs_count = GetElementCount(rhs.shape);

  if (lhs.shape != rhs.shape && lhs_count != 1 && rhs_count != 1) {
    MaterializeCudaInputsForNode(node, context);
    return RunBinaryNumericFallback(node, context, op_name, eval_float, eval_int);
  }

  const auto output_shape = ComputeBroadcastShape(lhs.shape, rhs.shape, op_name);
  const auto output_count = GetElementCount(output_shape);
  auto output = MakeCudaFloatOutput(node.outputs.at(0), output_shape);

  if (op_kind == CudaBinaryFloatOp::kDiv) {
    const auto& rhs_data = RequireFloatData(rhs, op_name);
    if ((rhs_count == 1 && rhs_data.front() == 0.0f) ||
        (rhs_count > 1 && std::any_of(rhs_data.begin(), rhs_data.end(), [](float value) { return value == 0.0f; }))) {
      throw std::runtime_error("Div divisor must not be zero");
    }
  }

  if (output_count == 0) {
    return output;
  }

  DeviceBuffer output_device(output_count * sizeof(float));

  if (lhs_count == 1 && rhs_count == 1) {
    MaterializeCudaInputsForNode(node, context);
    const auto& lhs_data = RequireFloatData(lhs, op_name);
    const auto& rhs_data = RequireFloatData(rhs, op_name);
    DeviceBuffer rhs_device(sizeof(float));
    CheckCuda(cudaMemcpy(rhs_device.data(), rhs_data.data(), sizeof(float), cudaMemcpyHostToDevice),
              "cudaMemcpy H2D rhs scalar");
    CheckCuda(LaunchCudaBinaryFloatScalarLeft(op_kind, lhs_data.front(), static_cast<const float*>(rhs_device.data()),
                                             static_cast<float*>(output_device.data()), 1),
              op_name + " scalar-scalar kernel launch");
  } else if (lhs_count == 1) {
    MaterializeCudaTensor(node.inputs.at(0), context);
    const auto& lhs_data = RequireFloatData(lhs, op_name);
    CheckCuda(LaunchCudaBinaryFloatScalarLeft(op_kind, lhs_data.front(), CudaFloatData(*rhs_tensor, op_name),
                                             static_cast<float*>(output_device.data()), output_count),
              op_name + " scalar-left kernel launch");
  } else if (rhs_count == 1) {
    MaterializeCudaTensor(node.inputs.at(1), context);
    const auto& rhs_data = RequireFloatData(rhs, op_name);
    CheckCuda(LaunchCudaBinaryFloatScalarRight(op_kind, CudaFloatData(*lhs_tensor, op_name), rhs_data.front(),
                                               static_cast<float*>(output_device.data()), output_count),
              op_name + " scalar-right kernel launch");
  } else {
    CheckCuda(LaunchCudaBinaryFloat(op_kind, CudaFloatData(*lhs_tensor, op_name),
                                    CudaFloatData(*rhs_tensor, op_name),
                                    static_cast<float*>(output_device.data()), output_count),
              op_name + " kernel launch");
  }

  BindCudaFloatOutput(output, std::move(output_device));
  return output;
}

Tensor RunCudaConcat(const Node& node, ExecutionContext& context) {
  if (node.inputs.empty()) {
    throw std::runtime_error("Concat requires at least one input");
  }
  std::vector<Tensor*> inputs;
  inputs.reserve(node.inputs.size());
  for (const auto& input_name : node.inputs) {
    if (input_name.empty()) {
      continue;
    }
    auto* input = context.FindTensor(input_name);
    if (input == nullptr) {
      throw std::runtime_error("missing Concat input: " + input_name);
    }
    inputs.push_back(input);
  }
  if (inputs.empty()) {
    throw std::runtime_error("Concat requires at least one input");
  }

  const auto rank = inputs.front()->shape.size();
  auto axis = NormalizeAxis(ReadIntAttribute(node, "axis", 0), rank, "Concat");
  const auto axis_index = static_cast<std::size_t>(axis);
  std::vector<std::int64_t> output_shape = inputs.front()->shape;
  output_shape[axis_index] = 0;
  for (const auto* input : inputs) {
    if (input->dtype != "float32") {
      throw std::runtime_error("CUDA Concat currently supports float32 only");
    }
    if (input->shape.size() != rank) {
      throw std::runtime_error("CUDA Concat rank mismatch");
    }
    for (std::size_t i = 0; i < rank; ++i) {
      if (i == axis_index) {
        continue;
      }
      if (input->shape[i] != output_shape[i]) {
        throw std::runtime_error("CUDA Concat non-axis dimensions must match");
      }
    }
    output_shape[axis_index] += input->shape[axis_index];
  }

  std::size_t outer = 1;
  for (std::size_t i = 0; i < axis_index; ++i) {
    outer *= static_cast<std::size_t>(output_shape[i]);
  }
  std::size_t inner = 1;
  for (std::size_t i = axis_index + 1; i < output_shape.size(); ++i) {
    inner *= static_cast<std::size_t>(output_shape[i]);
  }
  const auto output_axis = static_cast<std::size_t>(output_shape[axis_index]);

  auto output = MakeCudaFloatOutput(node.outputs.at(0), output_shape);
  DeviceBuffer output_device(GetElementCount(output_shape) * sizeof(float));
  auto* output_ptr = static_cast<float*>(output_device.data());

  for (std::size_t outer_index = 0; outer_index < outer; ++outer_index) {
    std::size_t axis_offset = 0;
    for (auto* input : inputs) {
      const auto input_axis = static_cast<std::size_t>(input->shape[axis_index]);
      const auto copy_elements = input_axis * inner;
      const auto input_offset = outer_index * input_axis * inner;
      const auto output_offset = (outer_index * output_axis + axis_offset) * inner;
      CheckCuda(cudaMemcpy(output_ptr + output_offset, CudaFloatData(*input, "CUDA Concat") + input_offset,
                           copy_elements * sizeof(float), cudaMemcpyDeviceToDevice),
                "cudaMemcpy D2D Concat");
      axis_offset += input_axis;
    }
  }

  BindCudaFloatOutput(output, std::move(output_device));
  return output;
}

std::vector<Tensor> RunCudaSplit(const Node& node, ExecutionContext& context) {
  auto* data = context.FindTensor(node.inputs.at(0));
  if (data == nullptr) {
    throw std::runtime_error("missing Split input");
  }
  if (data->dtype != "float32") {
    throw std::runtime_error("CUDA Split currently supports float32 only");
  }
  const auto rank = data->shape.size();
  auto axis = NormalizeAxis(ReadIntAttribute(node, "axis", 0), rank, "Split");
  const auto axis_index = static_cast<std::size_t>(axis);

  std::vector<std::int64_t> splits;
  if (node.inputs.size() > 1 && !node.inputs.at(1).empty()) {
    splits = RequireInt64Data(RequireTensor(context, node.inputs.at(1)), "Split");
  } else {
    if (data->shape[axis_index] % static_cast<std::int64_t>(node.outputs.size()) != 0) {
      throw std::runtime_error("CUDA Split cannot infer equal split sizes");
    }
    splits.assign(node.outputs.size(), data->shape[axis_index] / static_cast<std::int64_t>(node.outputs.size()));
  }
  if (splits.size() != node.outputs.size()) {
    throw std::runtime_error("CUDA Split sizes/output count mismatch");
  }

  std::size_t outer = 1;
  for (std::size_t i = 0; i < axis_index; ++i) {
    outer *= static_cast<std::size_t>(data->shape[i]);
  }
  std::size_t inner = 1;
  for (std::size_t i = axis_index + 1; i < data->shape.size(); ++i) {
    inner *= static_cast<std::size_t>(data->shape[i]);
  }
  const auto input_axis = static_cast<std::size_t>(data->shape[axis_index]);
  const auto* input_ptr = CudaFloatData(*data, "CUDA Split");

  std::vector<Tensor> outputs;
  outputs.reserve(node.outputs.size());
  std::size_t axis_offset = 0;
  for (std::size_t output_index = 0; output_index < node.outputs.size(); ++output_index) {
    std::vector<std::int64_t> output_shape = data->shape;
    output_shape[axis_index] = splits[output_index];
    auto output = MakeCudaFloatOutput(node.outputs[output_index], output_shape);
    DeviceBuffer output_device(GetElementCount(output_shape) * sizeof(float));
    auto* output_ptr = static_cast<float*>(output_device.data());
    const auto output_axis = static_cast<std::size_t>(splits[output_index]);
    for (std::size_t outer_index = 0; outer_index < outer; ++outer_index) {
      const auto input_offset = (outer_index * input_axis + axis_offset) * inner;
      const auto output_offset = outer_index * output_axis * inner;
      CheckCuda(cudaMemcpy(output_ptr + output_offset, input_ptr + input_offset, output_axis * inner * sizeof(float),
                           cudaMemcpyDeviceToDevice),
                "cudaMemcpy D2D Split");
    }
    BindCudaFloatOutput(output, std::move(output_device));
    outputs.push_back(std::move(output));
    axis_offset += output_axis;
  }

  return outputs;
}

Tensor RunCudaReshape(const Node& node, ExecutionContext& context) {
  const auto& data = RequireTensor(context, node.inputs.at(0));
  const auto& shape_tensor = RequireTensor(context, node.inputs.at(1));
  const auto output_shape = ResolveReshapeDims(data, shape_tensor);

  Tensor output;
  output.name = node.outputs.at(0);
  output.dtype = data.dtype;
  output.shape = output_shape;
  output.is_placeholder = false;
  output.cuda_data = data.cuda_data;
  output.cuda_bytes = data.cuda_bytes;

  if (data.dtype == "float32") {
    if (data.cuda_data == nullptr) {
      output.float_data = data.float_data;
    }
  } else if (data.dtype == "int64") {
    output.int64_data = data.int64_data;
  } else {
    throw std::runtime_error("CUDA Reshape currently supports float32/int64 only");
  }

  return output;
}

Tensor RunCudaResize(const Node& node, ExecutionContext& context) {
  auto* input = context.FindTensor(node.inputs.at(0));
  if (input == nullptr) {
    throw std::runtime_error("missing Resize input");
  }
  if (input->dtype != "float32") {
    throw std::runtime_error("CUDA Resize currently supports float32 only");
  }
  if (input->shape.size() != 4) {
    throw std::runtime_error("CUDA Resize currently only supports 4D NCHW tensors");
  }

  const auto mode_it = node.attributes.find("mode");
  const auto coord_it = node.attributes.find("coordinate_transformation_mode");
  const auto nearest_it = node.attributes.find("nearest_mode");
  const auto mode = mode_it == node.attributes.end() ? std::string("nearest") : mode_it->second.string_value;
  const auto coord_mode =
      coord_it == node.attributes.end() ? std::string("asymmetric") : coord_it->second.string_value;
  const auto nearest_mode =
      nearest_it == node.attributes.end() ? std::string("floor") : nearest_it->second.string_value;
  if (mode != "nearest" || coord_mode != "asymmetric" || nearest_mode != "floor") {
    throw std::runtime_error("CUDA Resize currently only supports nearest+asymmetric+floor");
  }

  if (node.inputs.size() < 3 || node.inputs.at(2).empty()) {
    throw std::runtime_error("CUDA Resize currently expects scales input");
  }
  const auto& scales = RequireFloatData(RequireTensor(context, node.inputs.at(2)), "CUDA Resize");
  if (scales.size() != 4) {
    throw std::runtime_error("CUDA Resize currently expects 4D scales");
  }

  const auto n_out = static_cast<std::int64_t>(std::floor(static_cast<double>(input->shape[0]) * scales[0]));
  const auto c_out = static_cast<std::int64_t>(std::floor(static_cast<double>(input->shape[1]) * scales[1]));
  const auto h_out = static_cast<std::int64_t>(std::floor(static_cast<double>(input->shape[2]) * scales[2]));
  const auto w_out = static_cast<std::int64_t>(std::floor(static_cast<double>(input->shape[3]) * scales[3]));
  if (n_out != input->shape[0] || c_out != input->shape[1]) {
    throw std::runtime_error("CUDA Resize currently requires batch/channel scales to keep dimensions unchanged");
  }
  if (h_out <= 0 || w_out <= 0) {
    throw std::runtime_error("CUDA Resize output shape is invalid");
  }

  const auto n = static_cast<std::size_t>(input->shape[0]);
  const auto c = static_cast<std::size_t>(input->shape[1]);
  const auto h_in = static_cast<std::size_t>(input->shape[2]);
  const auto w_in = static_cast<std::size_t>(input->shape[3]);
  const auto h_out_size = static_cast<std::size_t>(h_out);
  const auto w_out_size = static_cast<std::size_t>(w_out);

  auto output = MakeCudaFloatOutput(node.outputs.at(0), {input->shape[0], input->shape[1], h_out, w_out});
  DeviceBuffer output_device(GetElementCount(output.shape) * sizeof(float));
  CheckCuda(LaunchCudaResizeNearest2D(CudaFloatData(*input, "CUDA Resize"), static_cast<float*>(output_device.data()),
                                      n, c, h_in, w_in, h_out_size, w_out_size, scales[2], scales[3]),
            "Resize nearest kernel launch");
  BindCudaFloatOutput(output, std::move(output_device));
  return output;
}

}  // namespace

void MaterializeCudaTensor(const std::string& name, ExecutionContext& context) {
  auto* tensor = context.FindTensor(name);
  if (tensor == nullptr || tensor->dtype != "float32" || tensor->cuda_data == nullptr) {
    return;
  }
  const auto element_count = GetElementCount(tensor->shape);
  const auto bytes = element_count * sizeof(float);
  if (bytes == 0 || !tensor->float_data.empty()) {
    return;
  }
  tensor->float_data = context.AcquireFloatBuffer(element_count);
  tensor->float_data.resize(element_count);
  CheckCuda(cudaMemcpy(tensor->float_data.data(), tensor->cuda_data.get(), bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H materialize " + name);
}

void MaterializeCudaInputsForNode(const Node& node, ExecutionContext& context) {
  for (const auto& input : node.inputs) {
    if (!input.empty()) {
      MaterializeCudaTensor(input, context);
    }
  }
}

std::string_view CudaExecutionProvider::Name() const {
  return "CUDA";
}

void CudaExecutionProvider::RegisterKernels(KernelRegistry& registry) const {
  registry.Register("Sigmoid", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    auto* input = context.FindTensor(node.inputs.at(0));
    if (input == nullptr) {
      throw std::runtime_error("missing input tensor: " + node.inputs.at(0));
    }
    Tensor output;
    try {
      output = RunCudaUnaryFloatOp("Sigmoid", node.outputs.at(0), *input, context,
                                   [](const float* input_ptr, float* output_ptr, std::size_t count) {
                                     return LaunchCudaSigmoid(input_ptr, output_ptr, count);
                                   });
    } catch (const CudaError& ex) {
      MaterializeCudaTensor(node.inputs.at(0), context);
      output = RunUnaryFloatFallback(node.outputs.at(0), *input, context,
                                     [](float value) { return 1.0f / (1.0f + std::exp(-value)); });
      if (trace != nullptr) {
        *trace << "    kernel Sigmoid fell back to CPU reason=" << ex.what() << "\n";
      }
    }
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Sigmoid produced " << node.outputs.at(0) << " via CUDA\n";
    }
  });

  registry.Register("SiLU", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    auto* input = context.FindTensor(node.inputs.at(0));
    if (input == nullptr) {
      throw std::runtime_error("missing input tensor: " + node.inputs.at(0));
    }
    Tensor output;
    try {
      output = RunCudaUnaryFloatOp("SiLU", node.outputs.at(0), *input, context,
                                   [](const float* input_ptr, float* output_ptr, std::size_t count) {
                                     return LaunchCudaSiLU(input_ptr, output_ptr, count);
                                   });
    } catch (const CudaError& ex) {
      MaterializeCudaTensor(node.inputs.at(0), context);
      output = RunUnaryFloatFallback(node.outputs.at(0), *input, context,
                                     [](float value) { return value * (1.0f / (1.0f + std::exp(-value))); });
      if (trace != nullptr) {
        *trace << "    kernel SiLU fell back to CPU reason=" << ex.what() << "\n";
      }
    }
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel SiLU produced " << node.outputs.at(0) << " via CUDA\n";
    }
  });

  registry.Register("Tanh", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    auto* input = context.FindTensor(node.inputs.at(0));
    if (input == nullptr) {
      throw std::runtime_error("missing input tensor: " + node.inputs.at(0));
    }
    Tensor output;
    try {
      output = RunCudaUnaryFloatOp("Tanh", node.outputs.at(0), *input, context,
                                   [](const float* input_ptr, float* output_ptr, std::size_t count) {
                                     return LaunchCudaTanh(input_ptr, output_ptr, count);
                                   });
    } catch (const CudaError& ex) {
      MaterializeCudaTensor(node.inputs.at(0), context);
      output = RunUnaryFloatFallback(node.outputs.at(0), *input, context,
                                     [](float value) { return std::tanh(value); });
      if (trace != nullptr) {
        *trace << "    kernel Tanh fell back to CPU reason=" << ex.what() << "\n";
      }
    }
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Tanh produced " << node.outputs.at(0) << " via CUDA\n";
    }
  });

  registry.Register("Add", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    Tensor output;
    try {
      output = RunCudaBinaryFloatOp(node, context, "Add", CudaBinaryFloatOp::kAdd);
    } catch (const CudaError& ex) {
      MaterializeCudaInputsForNode(node, context);
      output = RunBinaryNumericFallback(node, context, "Add",
                                        [](float lhs, float rhs) { return lhs + rhs; },
                                        [](std::int64_t lhs, std::int64_t rhs) { return lhs + rhs; });
      if (trace != nullptr) {
        *trace << "    kernel Add fell back to CPU reason=" << ex.what() << "\n";
      }
    }
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Add produced " << node.outputs.at(0) << " via CUDA\n";
    }
  });

  registry.Register("Sub", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    Tensor output;
    try {
      output = RunCudaBinaryFloatOp(node, context, "Sub", CudaBinaryFloatOp::kSub);
    } catch (const CudaError& ex) {
      MaterializeCudaInputsForNode(node, context);
      output = RunBinaryNumericFallback(node, context, "Sub",
                                        [](float lhs, float rhs) { return lhs - rhs; },
                                        [](std::int64_t lhs, std::int64_t rhs) { return lhs - rhs; });
      if (trace != nullptr) {
        *trace << "    kernel Sub fell back to CPU reason=" << ex.what() << "\n";
      }
    }
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Sub produced " << node.outputs.at(0) << " via CUDA\n";
    }
  });

  registry.Register("Mul", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    Tensor output;
    try {
      output = RunCudaBinaryFloatOp(node, context, "Mul", CudaBinaryFloatOp::kMul);
    } catch (const CudaError& ex) {
      MaterializeCudaInputsForNode(node, context);
      output = RunBinaryNumericFallback(node, context, "Mul",
                                        [](float lhs, float rhs) { return lhs * rhs; },
                                        [](std::int64_t lhs, std::int64_t rhs) { return lhs * rhs; });
      if (trace != nullptr) {
        *trace << "    kernel Mul fell back to CPU reason=" << ex.what() << "\n";
      }
    }
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Mul produced " << node.outputs.at(0) << " via CUDA\n";
    }
  });

  registry.Register("Div", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    Tensor output;
    try {
      output = RunCudaBinaryFloatOp(node, context, "Div", CudaBinaryFloatOp::kDiv);
    } catch (const CudaError& ex) {
      MaterializeCudaInputsForNode(node, context);
      output = RunBinaryNumericFallback(
          node, context, "Div",
          [](float lhs, float rhs) {
            if (rhs == 0.0f) {
              throw std::runtime_error("Div divisor must not be zero");
            }
            return lhs / rhs;
          },
          [](std::int64_t lhs, std::int64_t rhs) {
            if (rhs == 0) {
              throw std::runtime_error("Div divisor must not be zero");
            }
            return lhs / rhs;
          });
      if (trace != nullptr) {
        *trace << "    kernel Div fell back to CPU reason=" << ex.what() << "\n";
      }
    }
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Div produced " << node.outputs.at(0) << " via CUDA\n";
    }
  });

  registry.Register("MatMul", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& lhs = RequireTensor(context, node.inputs.at(0));
    const auto& rhs = RequireTensor(context, node.inputs.at(1));
    auto output = RunCudaMatMul(node, lhs, rhs, context);
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel MatMul produced " << node.outputs.at(0) << " via CUDA\n";
    }
  });

  registry.Register("Conv", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& input = RequireTensor(context, node.inputs.at(0));
    const auto& weight = RequireTensor(context, node.inputs.at(1));
    Tensor* bias = nullptr;
    if (node.inputs.size() > 2 && !node.inputs.at(2).empty()) {
      bias = context.FindTensor(node.inputs.at(2));
      if (bias == nullptr) {
        throw std::runtime_error("missing Conv bias input");
      }
    }

    auto output = RunCudaConv2D(node, input, weight, bias, context);
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Conv produced " << node.outputs.at(0) << " via CUDA\n";
    }
  });

  registry.Register("MaxPool", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& input = RequireTensor(context, node.inputs.at(0));
    Tensor output;
    try {
      output = RunCudaMaxPool2D(node, input, context);
    } catch (const CudaError& ex) {
      output = RunMaxPoolFallback(node, input, context);
      if (trace != nullptr) {
        *trace << "    kernel MaxPool fell back to CPU reason=" << ex.what() << "\n";
      }
    }
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel MaxPool produced " << node.outputs.at(0) << " via CUDA\n";
    }
  });

  registry.Register("Gemm", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& a = RequireTensor(context, node.inputs.at(0));
    const auto& b = RequireTensor(context, node.inputs.at(1));
    const Tensor* c = nullptr;
    if (node.inputs.size() > 2 && !node.inputs.at(2).empty()) {
      c = &RequireTensor(context, node.inputs.at(2));
    }
    auto output = RunCudaGemm(node, a, b, c, context);
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Gemm produced " << node.outputs.at(0) << " via CUDA\n";
    }
  });

  registry.Register("Concat", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    auto output = RunCudaConcat(node, context);
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Concat produced " << node.outputs.at(0) << " via CUDA\n";
    }
  });

  registry.Register("Split", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    auto outputs = RunCudaSplit(node, context);
    for (auto& output : outputs) {
      if (trace != nullptr) {
        *trace << "    kernel Split produced " << output.name << " via CUDA\n";
      }
      context.BindTensor(std::move(output));
    }
  });

  registry.Register("Reshape", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    auto output = RunCudaReshape(node, context);
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Reshape produced " << node.outputs.at(0) << " via CUDA\n";
    }
  });

  registry.Register("Resize", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    auto output = RunCudaResize(node, context);
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Resize produced " << node.outputs.at(0) << " via CUDA\n";
    }
  });
}

std::shared_ptr<TensorAllocator> CudaExecutionProvider::CreateTensorAllocator() const {
  return std::make_shared<CpuTensorAllocator>();
}

bool IsCudaExecutionProviderAvailable() {
  int device_count = 0;
  const auto status = cudaGetDeviceCount(&device_count);
  return status == cudaSuccess && device_count > 0;
}

}  // namespace miniort
