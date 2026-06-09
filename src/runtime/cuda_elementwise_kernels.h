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

enum class CudaGemmBiasKind {
  kScalar,
  kColumn,
  kRow,
  kFull,
};

cudaError_t LaunchCudaSigmoid(const float* input, float* output, std::size_t count);
cudaError_t LaunchCudaSiLU(const float* input, float* output, std::size_t count);
cudaError_t LaunchCudaTanh(const float* input, float* output, std::size_t count);
cudaError_t LaunchCudaSqrt(const float* input, float* output, std::size_t count);
cudaError_t LaunchCudaSquare(const float* input, float* output, std::size_t count);
cudaError_t LaunchCudaBinaryFloat(CudaBinaryFloatOp op, const float* lhs, const float* rhs, float* output,
                                  std::size_t count);
cudaError_t LaunchCudaBinaryFloatScalarLeft(CudaBinaryFloatOp op, float lhs_scalar, const float* rhs, float* output,
                                            std::size_t count);
cudaError_t LaunchCudaBinaryFloatScalarRight(CudaBinaryFloatOp op, const float* lhs, float rhs_scalar, float* output,
                                             std::size_t count);
cudaError_t LaunchCudaBinaryFloatVectorLeft(CudaBinaryFloatOp op, const float* lhs_vector, const float* rhs,
                                            float* output, std::size_t count, std::size_t vector_size);
cudaError_t LaunchCudaBinaryFloatVectorRight(CudaBinaryFloatOp op, const float* lhs, const float* rhs_vector,
                                             float* output, std::size_t count, std::size_t vector_size);
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
cudaError_t LaunchCudaAddChannelBiasSiLU2D(float* output, const float* bias, std::size_t n, std::size_t c,
                                           std::size_t h, std::size_t w);
cudaError_t LaunchCudaAddGemmBias(float* output, const float* bias, std::size_t m, std::size_t n,
                                  CudaGemmBiasKind kind, float scale);
cudaError_t LaunchCudaTransposeFloat(const float* input, float* output, std::size_t count, std::size_t rank,
                                     const std::int64_t* input_strides, const std::int64_t* output_strides,
                                     const std::int64_t* perm);
cudaError_t LaunchCudaSoftmaxFloat(const float* input, float* output, std::size_t rows, std::size_t axis_dim,
                                   std::size_t inner);
cudaError_t LaunchCudaWhereFloatInt64Cond(const std::int64_t* condition, const float* x, const float* y,
                                          float* output, std::size_t count, std::size_t rank,
                                          const std::int64_t* output_strides,
                                          const std::int64_t* condition_shape,
                                          const std::int64_t* condition_strides,
                                          const std::int64_t* x_shape, const std::int64_t* x_strides,
                                          const std::int64_t* y_shape, const std::int64_t* y_strides);
cudaError_t LaunchCudaLayerNormalization(const float* input, const float* scale, const float* bias, float* output,
                                         std::size_t rows, std::size_t normalized_size, float epsilon);
cudaError_t LaunchCudaReduceMeanLastDim(const float* input, float* output, std::size_t rows, std::size_t cols);

}  // namespace miniort
