#pragma once

#include <cstdint>
#include <cstddef>

#include <cuda_runtime_api.h>

namespace miniort {

enum class CudaBinaryFloatOp {
  kAdd,
  kSub,
  kMul,
  kDiv,
};

cudaError_t LaunchCudaSigmoid(const float* input, float* output, std::size_t count);
cudaError_t LaunchCudaSiLU(const float* input, float* output, std::size_t count);
cudaError_t LaunchCudaTanh(const float* input, float* output, std::size_t count);
cudaError_t LaunchCudaBinaryFloat(CudaBinaryFloatOp op, const float* lhs, const float* rhs, float* output,
                                  std::size_t count);
cudaError_t LaunchCudaBinaryFloatScalarLeft(CudaBinaryFloatOp op, float lhs_scalar, const float* rhs, float* output,
                                            std::size_t count);
cudaError_t LaunchCudaBinaryFloatScalarRight(CudaBinaryFloatOp op, const float* lhs, float rhs_scalar, float* output,
                                             std::size_t count);
cudaError_t LaunchCudaMaxPool2D(const float* input, float* output, std::size_t n, std::size_t c, std::size_t h_in,
                                std::size_t w_in, std::size_t h_out, std::size_t w_out, std::size_t k_h,
                                std::size_t k_w, std::int64_t stride_h, std::int64_t stride_w,
                                std::int64_t dilation_h, std::int64_t dilation_w, std::int64_t pad_top,
                                std::int64_t pad_left);
cudaError_t LaunchCudaIm2Col2D(const float* input, float* columns, std::size_t c_in, std::size_t h_in,
                               std::size_t w_in, std::size_t h_out, std::size_t w_out, std::size_t k_h,
                               std::size_t k_w, std::int64_t stride_h, std::int64_t stride_w,
                               std::int64_t dilation_h, std::int64_t dilation_w, std::int64_t pad_top,
                               std::int64_t pad_left);
cudaError_t LaunchCudaResizeNearest2D(const float* input, float* output, std::size_t n, std::size_t c,
                                      std::size_t h_in, std::size_t w_in, std::size_t h_out, std::size_t w_out,
                                      float scale_h, float scale_w);
cudaError_t LaunchCudaAddChannelBias2D(float* output, const float* bias, std::size_t n, std::size_t c,
                                       std::size_t h, std::size_t w);

}  // namespace miniort
