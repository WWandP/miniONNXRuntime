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

}  // namespace miniort
