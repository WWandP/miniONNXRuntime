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

}  // namespace miniort
