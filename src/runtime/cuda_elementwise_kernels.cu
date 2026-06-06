#include "cuda_elementwise_kernels.h"

#include <cstdint>
#include <cmath>

namespace miniort {

namespace {

constexpr int kThreadsPerBlock = 256;

template <typename Fn>
__global__ void UnaryKernel(const float* input, float* output, std::size_t count, Fn fn) {
  const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  output[index] = fn(input[index]);
}

template <typename Fn>
__global__ void BinaryKernel(const float* lhs, const float* rhs, float* output, std::size_t count, Fn fn) {
  const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  output[index] = fn(lhs[index], rhs[index]);
}

template <typename Fn>
__global__ void BinaryScalarLeftKernel(float lhs_scalar, const float* rhs, float* output, std::size_t count, Fn fn) {
  const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  output[index] = fn(lhs_scalar, rhs[index]);
}

template <typename Fn>
__global__ void BinaryScalarRightKernel(const float* lhs, float rhs_scalar, float* output, std::size_t count, Fn fn) {
  const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  output[index] = fn(lhs[index], rhs_scalar);
}

template <typename Fn>
__global__ void BinaryVectorLeftKernel(const float* lhs_vector, const float* rhs, float* output, std::size_t count,
                                       std::size_t vector_size, Fn fn) {
  const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  output[index] = fn(lhs_vector[index % vector_size], rhs[index]);
}

template <typename Fn>
__global__ void BinaryVectorRightKernel(const float* lhs, const float* rhs_vector, float* output, std::size_t count,
                                        std::size_t vector_size, Fn fn) {
  const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  output[index] = fn(lhs[index], rhs_vector[index % vector_size]);
}

__global__ void MaxPool2DKernel(const float* input, float* output, std::size_t n, std::size_t c, std::size_t h_in,
                                std::size_t w_in, std::size_t h_out, std::size_t w_out, std::size_t k_h,
                                std::size_t k_w, std::int64_t stride_h, std::int64_t stride_w,
                                std::int64_t dilation_h, std::int64_t dilation_w, std::int64_t pad_top,
                                std::int64_t pad_left) {
  const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto total = n * c * h_out * w_out;
  if (index >= total) {
    return;
  }

  const auto ow = index % w_out;
  const auto oh = (index / w_out) % h_out;
  const auto channel = (index / (w_out * h_out)) % c;
  const auto batch = index / (w_out * h_out * c);

  float best = -INFINITY;
  const auto input_hw = h_in * w_in;
  for (std::size_t kh = 0; kh < k_h; ++kh) {
    for (std::size_t kw = 0; kw < k_w; ++kw) {
      const auto ih = static_cast<std::int64_t>(oh) * stride_h - pad_top + static_cast<std::int64_t>(kh) * dilation_h;
      const auto iw = static_cast<std::int64_t>(ow) * stride_w - pad_left + static_cast<std::int64_t>(kw) * dilation_w;
      if (ih < 0 || iw < 0 || ih >= static_cast<std::int64_t>(h_in) || iw >= static_cast<std::int64_t>(w_in)) {
        continue;
      }
      const auto input_index = ((batch * c + channel) * input_hw) +
                               static_cast<std::size_t>(ih) * w_in +
                               static_cast<std::size_t>(iw);
      best = fmaxf(best, input[input_index]);
    }
  }
  output[index] = best;
}

__global__ void Im2Col2DKernel(const float* __restrict__ input, float* __restrict__ columns, std::size_t c_in,
                               std::size_t h_in, std::size_t w_in, std::size_t h_out, std::size_t w_out,
                               std::size_t k_h, std::size_t k_w, std::int64_t stride_h, std::int64_t stride_w,
                               std::int64_t dilation_h, std::int64_t dilation_w, std::int64_t pad_top,
                               std::int64_t pad_left) {
  const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto output_hw = h_out * w_out;
  const auto kernel_dim = c_in * k_h * k_w;
  const auto total = kernel_dim * output_hw;
  if (index >= total) {
    return;
  }

  const auto output_offset = index % output_hw;
  const auto kernel_index = index / output_hw;
  const auto ow = output_offset % w_out;
  const auto oh = output_offset / w_out;
  const auto kw = kernel_index % k_w;
  const auto kh = (kernel_index / k_w) % k_h;
  const auto channel = kernel_index / (k_h * k_w);

  const auto ih = static_cast<std::int64_t>(oh) * stride_h + static_cast<std::int64_t>(kh) * dilation_h - pad_top;
  const auto iw = static_cast<std::int64_t>(ow) * stride_w + static_cast<std::int64_t>(kw) * dilation_w - pad_left;
  if (ih < 0 || iw < 0 || ih >= static_cast<std::int64_t>(h_in) || iw >= static_cast<std::int64_t>(w_in)) {
    columns[index] = 0.0f;
    return;
  }

  const auto input_index = (channel * h_in + static_cast<std::size_t>(ih)) * w_in + static_cast<std::size_t>(iw);
  columns[index] = input[input_index];
}

__global__ void ResizeNearest2DKernel(const float* input, float* output, std::size_t n, std::size_t c,
                                      std::size_t h_in, std::size_t w_in, std::size_t h_out, std::size_t w_out,
                                      float scale_h, float scale_w) {
  const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto total = n * c * h_out * w_out;
  if (index >= total) {
    return;
  }

  const auto ow = index % w_out;
  const auto oh = (index / w_out) % h_out;
  const auto channel = (index / (w_out * h_out)) % c;
  const auto batch = index / (w_out * h_out * c);
  auto ih = static_cast<std::size_t>(floorf(static_cast<float>(oh) / scale_h));
  auto iw = static_cast<std::size_t>(floorf(static_cast<float>(ow) / scale_w));
  ih = min(ih, h_in - 1);
  iw = min(iw, w_in - 1);

  const auto input_index = ((batch * c + channel) * h_in + ih) * w_in + iw;
  output[index] = input[input_index];
}

__global__ void AddChannelBias2DKernel(float* output, const float* bias, std::size_t n, std::size_t c,
                                       std::size_t h, std::size_t w) {
  const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto total = n * c * h * w;
  if (index >= total) {
    return;
  }
  const auto channel = (index / (h * w)) % c;
  output[index] += bias[channel];
}

__global__ void AddGemmBiasKernel(float* output, const float* bias, std::size_t m, std::size_t n,
                                  CudaGemmBiasKind kind, float scale) {
  const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto total = m * n;
  if (index >= total) {
    return;
  }

  float bias_value = 0.0f;
  switch (kind) {
    case CudaGemmBiasKind::kScalar:
      bias_value = bias[0];
      break;
    case CudaGemmBiasKind::kColumn:
      bias_value = bias[index % n];
      break;
    case CudaGemmBiasKind::kRow:
      bias_value = bias[index / n];
      break;
    case CudaGemmBiasKind::kFull:
      bias_value = bias[index];
      break;
  }
  output[index] += scale * bias_value;
}

__global__ void TransposeFloatKernel(const float* input, float* output, std::size_t count, std::size_t rank,
                                     const std::int64_t* input_strides, const std::int64_t* output_strides,
                                     const std::int64_t* perm) {
  const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }

  auto remaining = index;
  std::size_t input_offset = 0;
  for (std::size_t dim = 0; dim < rank; ++dim) {
    const auto stride = static_cast<std::size_t>(output_strides[dim]);
    const auto coord = stride == 0 ? 0 : remaining / stride;
    if (stride != 0) {
      remaining %= stride;
    }
    input_offset += coord * static_cast<std::size_t>(input_strides[perm[dim]]);
  }
  output[index] = input[input_offset];
}

__global__ void SoftmaxFloatKernel(const float* input, float* output, std::size_t rows, std::size_t axis_dim,
                                   std::size_t inner) {
  const auto row_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row_index >= rows) {
    return;
  }

  const auto outer_index = row_index / inner;
  const auto inner_index = row_index % inner;
  const auto base = (outer_index * axis_dim) * inner + inner_index;

  float max_value = -INFINITY;
  for (std::size_t axis_index = 0; axis_index < axis_dim; ++axis_index) {
    max_value = fmaxf(max_value, input[base + axis_index * inner]);
  }

  float sum = 0.0f;
  for (std::size_t axis_index = 0; axis_index < axis_dim; ++axis_index) {
    const auto offset = base + axis_index * inner;
    const auto value = expf(input[offset] - max_value);
    output[offset] = value;
    sum += value;
  }

  const auto inv_sum = 1.0f / sum;
  for (std::size_t axis_index = 0; axis_index < axis_dim; ++axis_index) {
    const auto offset = base + axis_index * inner;
    output[offset] *= inv_sum;
  }
}

__device__ std::size_t BroadcastOffset(std::size_t flat_index, std::size_t rank,
                                       const std::int64_t* output_strides, const std::int64_t* input_shape,
                                       const std::int64_t* input_strides) {
  std::size_t offset = 0;
  auto remaining = flat_index;
  for (std::size_t dim = 0; dim < rank; ++dim) {
    const auto output_stride = static_cast<std::size_t>(output_strides[dim]);
    const auto coord = output_stride == 0 ? 0 : remaining / output_stride;
    if (output_stride != 0) {
      remaining %= output_stride;
    }
    if (input_shape[dim] != 1) {
      offset += coord * static_cast<std::size_t>(input_strides[dim]);
    }
  }
  return offset;
}

__global__ void WhereFloatInt64CondKernel(const std::int64_t* condition, const float* x, const float* y,
                                          float* output, std::size_t count, std::size_t rank,
                                          const std::int64_t* output_strides,
                                          const std::int64_t* condition_shape,
                                          const std::int64_t* condition_strides,
                                          const std::int64_t* x_shape, const std::int64_t* x_strides,
                                          const std::int64_t* y_shape, const std::int64_t* y_strides) {
  const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }

  const auto condition_offset = BroadcastOffset(index, rank, output_strides, condition_shape, condition_strides);
  const auto x_offset = BroadcastOffset(index, rank, output_strides, x_shape, x_strides);
  const auto y_offset = BroadcastOffset(index, rank, output_strides, y_shape, y_strides);
  output[index] = condition[condition_offset] != 0 ? x[x_offset] : y[y_offset];
}

__global__ void LayerNormalizationKernel(const float* input, const float* scale, const float* bias, float* output,
                                         std::size_t normalized_size, float epsilon) {
  extern __shared__ float scratch[];
  const auto row = static_cast<std::size_t>(blockIdx.x);
  const auto tid = static_cast<std::size_t>(threadIdx.x);
  const auto row_base = row * normalized_size;

  float partial_sum = 0.0f;
  for (std::size_t i = tid; i < normalized_size; i += blockDim.x) {
    partial_sum += input[row_base + i];
  }
  scratch[tid] = partial_sum;
  __syncthreads();

  for (std::size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  const float mean = scratch[0] / static_cast<float>(normalized_size);

  float partial_variance = 0.0f;
  for (std::size_t i = tid; i < normalized_size; i += blockDim.x) {
    const auto diff = input[row_base + i] - mean;
    partial_variance += diff * diff;
  }
  scratch[tid] = partial_variance;
  __syncthreads();

  for (std::size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }
  const float inv_stddev = rsqrtf(scratch[0] / static_cast<float>(normalized_size) + epsilon);

  for (std::size_t i = tid; i < normalized_size; i += blockDim.x) {
    output[row_base + i] = ((input[row_base + i] - mean) * inv_stddev) * scale[i] + bias[i];
  }
}

__global__ void ReduceMeanLastDimKernel(const float* input, float* output, std::size_t cols) {
  extern __shared__ float scratch[];
  const auto row = static_cast<std::size_t>(blockIdx.x);
  const auto tid = static_cast<std::size_t>(threadIdx.x);
  const auto row_base = row * cols;

  float partial = 0.0f;
  for (std::size_t i = tid; i < cols; i += blockDim.x) {
    partial += input[row_base + i];
  }
  scratch[tid] = partial;
  __syncthreads();

  for (std::size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      scratch[tid] += scratch[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[row] = scratch[0] / static_cast<float>(cols);
  }
}

struct SigmoidFn {
  __device__ float operator()(float value) const {
    return 1.0f / (1.0f + expf(-value));
  }
};

struct SiLUFn {
  __device__ float operator()(float value) const {
    return value * (1.0f / (1.0f + expf(-value)));
  }
};

struct TanhFn {
  __device__ float operator()(float value) const {
    return tanhf(value);
  }
};

struct SqrtFn {
  __device__ float operator()(float value) const {
    return sqrtf(value);
  }
};

struct SquareFn {
  __device__ float operator()(float value) const {
    return value * value;
  }
};

struct AddFn {
  __device__ float operator()(float lhs, float rhs) const {
    return lhs + rhs;
  }
};

struct SubFn {
  __device__ float operator()(float lhs, float rhs) const {
    return lhs - rhs;
  }
};

struct MulFn {
  __device__ float operator()(float lhs, float rhs) const {
    return lhs * rhs;
  }
};

struct DivFn {
  __device__ float operator()(float lhs, float rhs) const {
    return lhs / rhs;
  }
};

int BlockCount(std::size_t count) {
  return static_cast<int>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
}

template <typename Fn>
cudaError_t LaunchUnary(const float* input, float* output, std::size_t count, Fn fn) {
  if (count == 0) {
    return cudaSuccess;
  }
  UnaryKernel<<<BlockCount(count), kThreadsPerBlock>>>(input, output, count, fn);
  return cudaGetLastError();
}

template <typename Fn>
cudaError_t LaunchBinary(const float* lhs, const float* rhs, float* output, std::size_t count, Fn fn) {
  if (count == 0) {
    return cudaSuccess;
  }
  BinaryKernel<<<BlockCount(count), kThreadsPerBlock>>>(lhs, rhs, output, count, fn);
  return cudaGetLastError();
}

template <typename Fn>
cudaError_t LaunchBinaryScalarLeft(float lhs_scalar, const float* rhs, float* output, std::size_t count, Fn fn) {
  if (count == 0) {
    return cudaSuccess;
  }
  BinaryScalarLeftKernel<<<BlockCount(count), kThreadsPerBlock>>>(lhs_scalar, rhs, output, count, fn);
  return cudaGetLastError();
}

template <typename Fn>
cudaError_t LaunchBinaryScalarRight(const float* lhs, float rhs_scalar, float* output, std::size_t count, Fn fn) {
  if (count == 0) {
    return cudaSuccess;
  }
  BinaryScalarRightKernel<<<BlockCount(count), kThreadsPerBlock>>>(lhs, rhs_scalar, output, count, fn);
  return cudaGetLastError();
}

template <typename Fn>
cudaError_t LaunchBinaryVectorLeft(const float* lhs_vector, const float* rhs, float* output, std::size_t count,
                                   std::size_t vector_size, Fn fn) {
  if (count == 0) {
    return cudaSuccess;
  }
  BinaryVectorLeftKernel<<<BlockCount(count), kThreadsPerBlock>>>(lhs_vector, rhs, output, count, vector_size, fn);
  return cudaGetLastError();
}

template <typename Fn>
cudaError_t LaunchBinaryVectorRight(const float* lhs, const float* rhs_vector, float* output, std::size_t count,
                                    std::size_t vector_size, Fn fn) {
  if (count == 0) {
    return cudaSuccess;
  }
  BinaryVectorRightKernel<<<BlockCount(count), kThreadsPerBlock>>>(lhs, rhs_vector, output, count, vector_size, fn);
  return cudaGetLastError();
}

}  // namespace

cudaError_t LaunchCudaSigmoid(const float* input, float* output, std::size_t count) {
  return LaunchUnary(input, output, count, SigmoidFn{});
}

cudaError_t LaunchCudaSiLU(const float* input, float* output, std::size_t count) {
  return LaunchUnary(input, output, count, SiLUFn{});
}

cudaError_t LaunchCudaTanh(const float* input, float* output, std::size_t count) {
  return LaunchUnary(input, output, count, TanhFn{});
}

cudaError_t LaunchCudaSqrt(const float* input, float* output, std::size_t count) {
  return LaunchUnary(input, output, count, SqrtFn{});
}

cudaError_t LaunchCudaSquare(const float* input, float* output, std::size_t count) {
  return LaunchUnary(input, output, count, SquareFn{});
}

cudaError_t LaunchCudaBinaryFloat(CudaBinaryFloatOp op, const float* lhs, const float* rhs, float* output,
                                  std::size_t count) {
  switch (op) {
    case CudaBinaryFloatOp::kAdd:
      return LaunchBinary(lhs, rhs, output, count, AddFn{});
    case CudaBinaryFloatOp::kSub:
      return LaunchBinary(lhs, rhs, output, count, SubFn{});
    case CudaBinaryFloatOp::kMul:
      return LaunchBinary(lhs, rhs, output, count, MulFn{});
    case CudaBinaryFloatOp::kDiv:
      return LaunchBinary(lhs, rhs, output, count, DivFn{});
  }
  return cudaErrorInvalidValue;
}

cudaError_t LaunchCudaBinaryFloatScalarLeft(CudaBinaryFloatOp op, float lhs_scalar, const float* rhs, float* output,
                                            std::size_t count) {
  switch (op) {
    case CudaBinaryFloatOp::kAdd:
      return LaunchBinaryScalarLeft(lhs_scalar, rhs, output, count, AddFn{});
    case CudaBinaryFloatOp::kSub:
      return LaunchBinaryScalarLeft(lhs_scalar, rhs, output, count, SubFn{});
    case CudaBinaryFloatOp::kMul:
      return LaunchBinaryScalarLeft(lhs_scalar, rhs, output, count, MulFn{});
    case CudaBinaryFloatOp::kDiv:
      return LaunchBinaryScalarLeft(lhs_scalar, rhs, output, count, DivFn{});
  }
  return cudaErrorInvalidValue;
}

cudaError_t LaunchCudaBinaryFloatScalarRight(CudaBinaryFloatOp op, const float* lhs, float rhs_scalar, float* output,
                                             std::size_t count) {
  switch (op) {
    case CudaBinaryFloatOp::kAdd:
      return LaunchBinaryScalarRight(lhs, rhs_scalar, output, count, AddFn{});
    case CudaBinaryFloatOp::kSub:
      return LaunchBinaryScalarRight(lhs, rhs_scalar, output, count, SubFn{});
    case CudaBinaryFloatOp::kMul:
      return LaunchBinaryScalarRight(lhs, rhs_scalar, output, count, MulFn{});
    case CudaBinaryFloatOp::kDiv:
      return LaunchBinaryScalarRight(lhs, rhs_scalar, output, count, DivFn{});
  }
  return cudaErrorInvalidValue;
}

cudaError_t LaunchCudaBinaryFloatVectorLeft(CudaBinaryFloatOp op, const float* lhs_vector, const float* rhs,
                                            float* output, std::size_t count, std::size_t vector_size) {
  switch (op) {
    case CudaBinaryFloatOp::kAdd:
      return LaunchBinaryVectorLeft(lhs_vector, rhs, output, count, vector_size, AddFn{});
    case CudaBinaryFloatOp::kSub:
      return LaunchBinaryVectorLeft(lhs_vector, rhs, output, count, vector_size, SubFn{});
    case CudaBinaryFloatOp::kMul:
      return LaunchBinaryVectorLeft(lhs_vector, rhs, output, count, vector_size, MulFn{});
    case CudaBinaryFloatOp::kDiv:
      return LaunchBinaryVectorLeft(lhs_vector, rhs, output, count, vector_size, DivFn{});
  }
  return cudaErrorInvalidValue;
}

cudaError_t LaunchCudaBinaryFloatVectorRight(CudaBinaryFloatOp op, const float* lhs, const float* rhs_vector,
                                             float* output, std::size_t count, std::size_t vector_size) {
  switch (op) {
    case CudaBinaryFloatOp::kAdd:
      return LaunchBinaryVectorRight(lhs, rhs_vector, output, count, vector_size, AddFn{});
    case CudaBinaryFloatOp::kSub:
      return LaunchBinaryVectorRight(lhs, rhs_vector, output, count, vector_size, SubFn{});
    case CudaBinaryFloatOp::kMul:
      return LaunchBinaryVectorRight(lhs, rhs_vector, output, count, vector_size, MulFn{});
    case CudaBinaryFloatOp::kDiv:
      return LaunchBinaryVectorRight(lhs, rhs_vector, output, count, vector_size, DivFn{});
  }
  return cudaErrorInvalidValue;
}

cudaError_t LaunchCudaMaxPool2D(const float* input, float* output, std::size_t n, std::size_t c, std::size_t h_in,
                                std::size_t w_in, std::size_t h_out, std::size_t w_out, std::size_t k_h,
                                std::size_t k_w, std::int64_t stride_h, std::int64_t stride_w,
                                std::int64_t dilation_h, std::int64_t dilation_w, std::int64_t pad_top,
                                std::int64_t pad_left) {
  const auto count = n * c * h_out * w_out;
  if (count == 0) {
    return cudaSuccess;
  }
  MaxPool2DKernel<<<BlockCount(count), kThreadsPerBlock>>>(
      input, output, n, c, h_in, w_in, h_out, w_out, k_h, k_w, stride_h, stride_w, dilation_h, dilation_w, pad_top,
      pad_left);
  return cudaGetLastError();
}

cudaError_t LaunchCudaIm2Col2D(const float* input, float* columns, std::size_t c_in, std::size_t h_in,
                               std::size_t w_in, std::size_t h_out, std::size_t w_out, std::size_t k_h,
                               std::size_t k_w, std::int64_t stride_h, std::int64_t stride_w,
                               std::int64_t dilation_h, std::int64_t dilation_w, std::int64_t pad_top,
                               std::int64_t pad_left) {
  const auto count = c_in * k_h * k_w * h_out * w_out;
  if (count == 0) {
    return cudaSuccess;
  }
  Im2Col2DKernel<<<BlockCount(count), kThreadsPerBlock>>>(input, columns, c_in, h_in, w_in, h_out, w_out, k_h, k_w,
                                                          stride_h, stride_w, dilation_h, dilation_w, pad_top,
                                                          pad_left);
  return cudaGetLastError();
}

cudaError_t LaunchCudaResizeNearest2D(const float* input, float* output, std::size_t n, std::size_t c,
                                      std::size_t h_in, std::size_t w_in, std::size_t h_out, std::size_t w_out,
                                      float scale_h, float scale_w) {
  const auto count = n * c * h_out * w_out;
  if (count == 0) {
    return cudaSuccess;
  }
  ResizeNearest2DKernel<<<BlockCount(count), kThreadsPerBlock>>>(input, output, n, c, h_in, w_in, h_out, w_out,
                                                                 scale_h, scale_w);
  return cudaGetLastError();
}

cudaError_t LaunchCudaAddChannelBias2D(float* output, const float* bias, std::size_t n, std::size_t c,
                                       std::size_t h, std::size_t w) {
  const auto count = n * c * h * w;
  if (count == 0) {
    return cudaSuccess;
  }
  AddChannelBias2DKernel<<<BlockCount(count), kThreadsPerBlock>>>(output, bias, n, c, h, w);
  return cudaGetLastError();
}

cudaError_t LaunchCudaAddGemmBias(float* output, const float* bias, std::size_t m, std::size_t n,
                                  CudaGemmBiasKind kind, float scale) {
  const auto count = m * n;
  if (count == 0) {
    return cudaSuccess;
  }
  AddGemmBiasKernel<<<BlockCount(count), kThreadsPerBlock>>>(output, bias, m, n, kind, scale);
  return cudaGetLastError();
}

cudaError_t LaunchCudaTransposeFloat(const float* input, float* output, std::size_t count, std::size_t rank,
                                     const std::int64_t* input_strides, const std::int64_t* output_strides,
                                     const std::int64_t* perm) {
  if (count == 0) {
    return cudaSuccess;
  }
  TransposeFloatKernel<<<BlockCount(count), kThreadsPerBlock>>>(input, output, count, rank, input_strides,
                                                                output_strides, perm);
  return cudaGetLastError();
}

cudaError_t LaunchCudaSoftmaxFloat(const float* input, float* output, std::size_t rows, std::size_t axis_dim,
                                   std::size_t inner) {
  if (rows == 0 || axis_dim == 0) {
    return cudaSuccess;
  }
  SoftmaxFloatKernel<<<BlockCount(rows), kThreadsPerBlock>>>(input, output, rows, axis_dim, inner);
  return cudaGetLastError();
}

cudaError_t LaunchCudaWhereFloatInt64Cond(const std::int64_t* condition, const float* x, const float* y,
                                          float* output, std::size_t count, std::size_t rank,
                                          const std::int64_t* output_strides,
                                          const std::int64_t* condition_shape,
                                          const std::int64_t* condition_strides,
                                          const std::int64_t* x_shape, const std::int64_t* x_strides,
                                          const std::int64_t* y_shape, const std::int64_t* y_strides) {
  if (count == 0) {
    return cudaSuccess;
  }
  WhereFloatInt64CondKernel<<<BlockCount(count), kThreadsPerBlock>>>(
      condition, x, y, output, count, rank, output_strides, condition_shape, condition_strides, x_shape, x_strides,
      y_shape, y_strides);
  return cudaGetLastError();
}

cudaError_t LaunchCudaLayerNormalization(const float* input, const float* scale, const float* bias, float* output,
                                         std::size_t rows, std::size_t normalized_size, float epsilon) {
  if (rows == 0 || normalized_size == 0) {
    return cudaSuccess;
  }
  constexpr int threads = 256;
  LayerNormalizationKernel<<<static_cast<unsigned int>(rows), threads, threads * sizeof(float)>>>(
      input, scale, bias, output, normalized_size, epsilon);
  return cudaGetLastError();
}

cudaError_t LaunchCudaReduceMeanLastDim(const float* input, float* output, std::size_t rows, std::size_t cols) {
  if (rows == 0 || cols == 0) {
    return cudaSuccess;
  }
  constexpr int threads = 256;
  ReduceMeanLastDimKernel<<<static_cast<unsigned int>(rows), threads, threads * sizeof(float)>>>(input, output, cols);
  return cudaGetLastError();
}

}  // namespace miniort
