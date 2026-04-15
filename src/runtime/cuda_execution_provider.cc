#include "miniort/runtime/cuda_execution_provider.h"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <functional>
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

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t bytes) {
    if (bytes == 0) {
      return;
    }
    CheckCuda(cudaMalloc(&data_, bytes), "cudaMalloc");
    size_ = bytes;
  }

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
    }
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
    if (data_ != nullptr) {
      (void)cudaFree(data_);
    }
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
    return *this;
  }

  void* data() { return data_; }
  const void* data() const { return data_; }
  std::size_t size() const { return size_; }

 private:
  void* data_{nullptr};
  std::size_t size_{0};
};

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

void FillIm2ColBuffer(const float* batch_input, const Conv2DParams& params, std::vector<float>& columns) {
  const auto output_hw = static_cast<std::size_t>(params.h_out) * static_cast<std::size_t>(params.w_out);
  const auto kernel_dim = params.c_in * params.k_h * params.k_w;
  columns.assign(kernel_dim * output_hw, 0.0f);

  const auto input_hw = params.h_in * params.w_in;
  std::size_t kernel_index = 0;
  for (std::size_t ic = 0; ic < params.c_in; ++ic) {
    const auto* input_plane = batch_input + ic * input_hw;
    for (std::size_t kh = 0; kh < params.k_h; ++kh) {
      for (std::size_t kw = 0; kw < params.k_w; ++kw, ++kernel_index) {
        auto* col_row = columns.data() + kernel_index * output_hw;
        for (std::int64_t oh = 0; oh < params.h_out; ++oh) {
          const auto ih = oh * params.stride_h + static_cast<std::int64_t>(kh) * params.dilation_h - params.pad_top;
          for (std::int64_t ow = 0; ow < params.w_out; ++ow) {
            const auto iw =
                ow * params.stride_w + static_cast<std::int64_t>(kw) * params.dilation_w - params.pad_left;
            float value = 0.0f;
            if (ih >= 0 && ih < static_cast<std::int64_t>(params.h_in) &&
                iw >= 0 && iw < static_cast<std::int64_t>(params.w_in)) {
              value = input_plane[static_cast<std::size_t>(ih) * params.w_in + static_cast<std::size_t>(iw)];
            }
            col_row[static_cast<std::size_t>(oh) * static_cast<std::size_t>(params.w_out) +
                    static_cast<std::size_t>(ow)] = value;
          }
        }
      }
    }
  }
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
  CublasHandle handle;

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
    CheckCublas(cublasSgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(n), static_cast<int>(m),
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
  CublasHandle handle;

  CheckCuda(cudaMemcpy(a_device.data(), a_data.data(), a_elements * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy H2D Gemm A");
  CheckCuda(cudaMemcpy(b_device.data(), b_data.data(), b_elements * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy H2D Gemm B");

  const auto op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  const auto op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

  // Map row-major Gemm to column-major cuBLAS by swapping A/B and output axes.
  CheckCublas(cublasSgemm(handle.get(), op_b, op_a, static_cast<int>(n), static_cast<int>(m),
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

Tensor RunCudaConv2D(const Node& node, const Tensor& input, const Tensor& weight, const Tensor* bias,
                     ExecutionContext& context) {
  const auto& input_data = RequireFloatData(input, "CUDA Conv");
  const auto& weight_data = RequireFloatData(weight, "CUDA Conv");
  const std::vector<float>* bias_data = nullptr;
  if (bias != nullptr) {
    bias_data = &RequireFloatData(*bias, "CUDA Conv");
  }

  const auto params = ResolveConv2DParams(node, input, weight, bias);
  auto output = MakeFloatOutput(node.outputs.at(0),
                                {static_cast<std::int64_t>(params.n), static_cast<std::int64_t>(params.c_out),
                                 params.h_out, params.w_out},
                                context);

  const auto input_hw = params.h_in * params.w_in;
  const auto output_hw = static_cast<std::size_t>(params.h_out) * static_cast<std::size_t>(params.w_out);
  const auto kernel_dim = params.c_in * params.k_h * params.k_w;
  std::vector<float> columns;
  columns.reserve(kernel_dim * output_hw);

  DeviceBuffer weight_device(weight_data.size() * sizeof(float));
  DeviceBuffer columns_device(kernel_dim * output_hw * sizeof(float));
  DeviceBuffer output_device(params.c_out * output_hw * sizeof(float));
  CublasHandle handle;

  CheckCuda(cudaMemcpy(weight_device.data(), weight_data.data(), weight_data.size() * sizeof(float),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy H2D Conv weights");

  const float alpha = 1.0f;
  const float beta = 0.0f;

  for (std::size_t batch = 0; batch < params.n; ++batch) {
    const auto* batch_input = input_data.data() + batch * params.c_in * input_hw;
    auto* batch_output = output.float_data.data() + batch * params.c_out * output_hw;

    FillIm2ColBuffer(batch_input, params, columns);
    CheckCuda(cudaMemcpy(columns_device.data(), columns.data(), columns.size() * sizeof(float),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy H2D Conv columns");

    CheckCublas(cublasSgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(output_hw),
                            static_cast<int>(params.c_out), static_cast<int>(kernel_dim), &alpha,
                            static_cast<const float*>(columns_device.data()), static_cast<int>(output_hw),
                            static_cast<const float*>(weight_device.data()), static_cast<int>(kernel_dim), &beta,
                            static_cast<float*>(output_device.data()), static_cast<int>(output_hw)),
                "cublasSgemm Conv");

    CheckCuda(cudaMemcpy(batch_output, output_device.data(), params.c_out * output_hw * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy D2H Conv output");

    if (bias_data != nullptr) {
      for (std::size_t oc = 0; oc < params.c_out; ++oc) {
        const float bias_value = (*bias_data)[oc];
        auto* output_plane = batch_output + oc * output_hw;
        for (std::size_t i = 0; i < output_hw; ++i) {
          output_plane[i] += bias_value;
        }
      }
    }
  }

  return output;
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

  auto output = MakeFloatOutput(node.outputs.at(0),
                                {static_cast<std::int64_t>(n), static_cast<std::int64_t>(c), h_out, w_out},
                                context);

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
  const auto& input_data = RequireFloatData(input, "CUDA MaxPool");
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

  DeviceBuffer input_device(input_data.size() * sizeof(float));
  DeviceBuffer output_device(output_count * sizeof(float));
  CheckCuda(cudaMemcpy(input_device.data(), input_data.data(), input_data.size() * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy H2D MaxPool input");
  CheckCuda(LaunchCudaMaxPool2D(static_cast<const float*>(input_device.data()), static_cast<float*>(output_device.data()),
                                n, c, h_in, w_in, static_cast<std::size_t>(h_out), static_cast<std::size_t>(w_out),
                                k_h, k_w, stride_h, stride_w, dilation_h, dilation_w, pad_top, pad_left),
            "MaxPool kernel launch");
  CheckCuda(cudaDeviceSynchronize(), "MaxPool synchronize");
  CheckCuda(cudaMemcpy(output.float_data.data(), output_device.data(), output_count * sizeof(float),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H MaxPool output");
  return output;
}

Tensor RunCudaUnaryFloatOp(const std::string& op_name, const std::string& output_name, const Tensor& input,
                           ExecutionContext& context,
                           const std::function<cudaError_t(const float*, float*, std::size_t)>& launcher) {
  const auto& input_data = RequireFloatData(input, op_name);
  auto output = MakeOutputLikeWithReusedStorage(output_name, input, context);
  const auto element_count = input_data.size();

  DeviceBuffer input_device(element_count * sizeof(float));
  DeviceBuffer output_device(element_count * sizeof(float));

  CheckCuda(cudaMemcpy(input_device.data(), input_data.data(), element_count * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy H2D unary input");
  CheckCuda(launcher(static_cast<const float*>(input_device.data()), static_cast<float*>(output_device.data()),
                     element_count),
            op_name + " kernel launch");
  CheckCuda(cudaDeviceSynchronize(), op_name + " synchronize");
  CheckCuda(cudaMemcpy(output.float_data.data(), output_device.data(), element_count * sizeof(float),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H unary output");

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

  const auto& lhs_data = RequireFloatData(lhs, op_name);
  const auto& rhs_data = RequireFloatData(rhs, op_name);
  const auto lhs_count = lhs_data.size();
  const auto rhs_count = rhs_data.size();

  if (lhs.shape != rhs.shape && lhs_count != 1 && rhs_count != 1) {
    return RunBinaryNumericFallback(node, context, op_name, eval_float, eval_int);
  }

  const auto output_shape = ComputeBroadcastShape(lhs.shape, rhs.shape, op_name);
  const auto output_count = GetElementCount(output_shape);
  auto output = MakeFloatOutput(node.outputs.at(0), output_shape, context);

  if (op_kind == CudaBinaryFloatOp::kDiv) {
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
    DeviceBuffer rhs_device(sizeof(float));
    CheckCuda(cudaMemcpy(rhs_device.data(), rhs_data.data(), sizeof(float), cudaMemcpyHostToDevice),
              "cudaMemcpy H2D rhs scalar");
    CheckCuda(LaunchCudaBinaryFloatScalarLeft(op_kind, lhs_data.front(), static_cast<const float*>(rhs_device.data()),
                                             static_cast<float*>(output_device.data()), 1),
              op_name + " scalar-scalar kernel launch");
  } else if (lhs_count == 1) {
    DeviceBuffer rhs_device(output_count * sizeof(float));
    CheckCuda(cudaMemcpy(rhs_device.data(), rhs_data.data(), output_count * sizeof(float), cudaMemcpyHostToDevice),
              "cudaMemcpy H2D rhs");
    CheckCuda(LaunchCudaBinaryFloatScalarLeft(op_kind, lhs_data.front(), static_cast<const float*>(rhs_device.data()),
                                             static_cast<float*>(output_device.data()), output_count),
              op_name + " scalar-left kernel launch");
  } else if (rhs_count == 1) {
    DeviceBuffer lhs_device(output_count * sizeof(float));
    CheckCuda(cudaMemcpy(lhs_device.data(), lhs_data.data(), output_count * sizeof(float), cudaMemcpyHostToDevice),
              "cudaMemcpy H2D lhs");
    CheckCuda(LaunchCudaBinaryFloatScalarRight(op_kind, static_cast<const float*>(lhs_device.data()), rhs_data.front(),
                                               static_cast<float*>(output_device.data()), output_count),
              op_name + " scalar-right kernel launch");
  } else {
    DeviceBuffer lhs_device(output_count * sizeof(float));
    DeviceBuffer rhs_device(output_count * sizeof(float));
    CheckCuda(cudaMemcpy(lhs_device.data(), lhs_data.data(), output_count * sizeof(float), cudaMemcpyHostToDevice),
              "cudaMemcpy H2D lhs");
    CheckCuda(cudaMemcpy(rhs_device.data(), rhs_data.data(), output_count * sizeof(float), cudaMemcpyHostToDevice),
              "cudaMemcpy H2D rhs");
    CheckCuda(LaunchCudaBinaryFloat(op_kind, static_cast<const float*>(lhs_device.data()),
                                    static_cast<const float*>(rhs_device.data()),
                                    static_cast<float*>(output_device.data()), output_count),
              op_name + " kernel launch");
  }

  CheckCuda(cudaDeviceSynchronize(), op_name + " synchronize");
  CheckCuda(cudaMemcpy(output.float_data.data(), output_device.data(), output_count * sizeof(float),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H binary output");
  return output;
}

}  // namespace

std::string_view CudaExecutionProvider::Name() const {
  return "CUDA";
}

void CudaExecutionProvider::RegisterKernels(KernelRegistry& registry) const {
  registry.Register("Sigmoid", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& input = RequireTensor(context, node.inputs.at(0));
    Tensor output;
    try {
      output = RunCudaUnaryFloatOp("Sigmoid", node.outputs.at(0), input, context,
                                   [](const float* input_ptr, float* output_ptr, std::size_t count) {
                                     return LaunchCudaSigmoid(input_ptr, output_ptr, count);
                                   });
    } catch (const CudaError& ex) {
      output = RunUnaryFloatFallback(node.outputs.at(0), input, context,
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
    const auto& input = RequireTensor(context, node.inputs.at(0));
    Tensor output;
    try {
      output = RunCudaUnaryFloatOp("SiLU", node.outputs.at(0), input, context,
                                   [](const float* input_ptr, float* output_ptr, std::size_t count) {
                                     return LaunchCudaSiLU(input_ptr, output_ptr, count);
                                   });
    } catch (const CudaError& ex) {
      output = RunUnaryFloatFallback(node.outputs.at(0), input, context,
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
    const auto& input = RequireTensor(context, node.inputs.at(0));
    Tensor output;
    try {
      output = RunCudaUnaryFloatOp("Tanh", node.outputs.at(0), input, context,
                                   [](const float* input_ptr, float* output_ptr, std::size_t count) {
                                     return LaunchCudaTanh(input_ptr, output_ptr, count);
                                   });
    } catch (const CudaError& ex) {
      output = RunUnaryFloatFallback(node.outputs.at(0), input, context,
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
    const Tensor* bias = nullptr;
    if (node.inputs.size() > 2 && !node.inputs.at(2).empty()) {
      bias = &RequireTensor(context, node.inputs.at(2));
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
