#include "builtin_kernel_groups.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(__AVX__)
#include <immintrin.h>
#define MINIORT_ENABLE_AVX 1
#endif

#if defined(__SSE__) || defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#define MINIORT_ENABLE_SSE 1
#endif

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

#include "kernel_utils.h"

namespace miniort {

namespace {

Tensor RunMatMul(const std::string& output_name, const Tensor& lhs, const Tensor& rhs, ExecutionContext& context) {
  const auto& lhs_data = RequireFloatData(lhs, "MatMul");
  const auto& rhs_data = RequireFloatData(rhs, "MatMul");
  if (lhs.shape.size() < 2 || rhs.shape.size() < 2) {
    throw std::runtime_error("MatMul currently requires rank >= 2 float32 tensors");
  }

  const auto m = static_cast<std::size_t>(lhs.shape[lhs.shape.size() - 2]);
  const auto k = static_cast<std::size_t>(lhs.shape[lhs.shape.size() - 1]);
  const auto rhs_k = static_cast<std::size_t>(rhs.shape[rhs.shape.size() - 2]);
  const auto n = static_cast<std::size_t>(rhs.shape[rhs.shape.size() - 1]);
  if (k != rhs_k) {
    throw std::runtime_error("MatMul inner dimensions do not match");
  }

  const std::vector<std::int64_t> lhs_batch_shape(lhs.shape.begin(), lhs.shape.end() - 2);
  const std::vector<std::int64_t> rhs_batch_shape(rhs.shape.begin(), rhs.shape.end() - 2);
  const auto output_batch_shape = ComputeBroadcastShape(lhs_batch_shape, rhs_batch_shape, "MatMul");

  std::vector<std::int64_t> output_shape = output_batch_shape;
  output_shape.push_back(static_cast<std::int64_t>(m));
  output_shape.push_back(static_cast<std::int64_t>(n));

  auto output = MakeFloatOutput(output_name, output_shape, context);
  const auto output_batch_strides = ComputeStrides(output_batch_shape);
  const auto lhs_full_strides = ComputeStrides(lhs.shape);
  const auto rhs_full_strides = ComputeStrides(rhs.shape);

  const auto batch_count = GetElementCount(output_batch_shape);
  for (std::size_t batch = 0; batch < batch_count; ++batch) {
    const auto batch_index = UnravelIndex(batch, output_batch_shape, output_batch_strides);
    const auto lhs_batch_offset = lhs_batch_shape.empty() ? 0 : ComputeBroadcastOffset(batch_index, lhs_batch_shape, lhs_full_strides);
    const auto rhs_batch_offset = rhs_batch_shape.empty() ? 0 : ComputeBroadcastOffset(batch_index, rhs_batch_shape, rhs_full_strides);
    const auto lhs_base = lhs_batch_shape.empty() ? 0 : lhs_batch_offset;
    const auto rhs_base = rhs_batch_shape.empty() ? 0 : rhs_batch_offset;
    const auto output_base = batch * m * n;

#if defined(__APPLE__)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                1.0f,
                lhs_data.data() + lhs_base, static_cast<int>(k),
                rhs_data.data() + rhs_base, static_cast<int>(n),
                0.0f,
                output.float_data.data() + output_base, static_cast<int>(n));
#else
    std::fill(output.float_data.begin() + static_cast<std::ptrdiff_t>(output_base),
              output.float_data.begin() + static_cast<std::ptrdiff_t>(output_base + m * n), 0.0f);

    for (std::size_t i = 0; i < m; ++i) {
      const auto* lhs_row_ptr = lhs_data.data() + lhs_base + i * k;
      auto* out_row_ptr = output.float_data.data() + output_base + i * n;
      for (std::size_t kk = 0; kk < k; ++kk) {
        const float lhs_value = lhs_row_ptr[kk];
        const auto* rhs_row_ptr = rhs_data.data() + rhs_base + kk * n;
        for (std::size_t j = 0; j < n; ++j) {
          out_row_ptr[j] += lhs_value * rhs_row_ptr[j];
        }
      }
    }
#endif
  }
  return output;
}

void ApplyGemmBias(Tensor& output, const Tensor* bias) {
  if (bias == nullptr) {
    return;
  }
  const auto& bias_data = RequireFloatData(*bias, "Gemm");
  if (output.shape.size() != 2) {
    throw std::runtime_error("Gemm output must be 2D");
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

  throw std::runtime_error("Gemm bias shape is not supported");
}

Tensor RunGemm2D(const Node& node, const Tensor& a, const Tensor& b, const Tensor* c, ExecutionContext& context) {
  const auto& a_data = RequireFloatData(a, "Gemm");
  const auto& b_data = RequireFloatData(b, "Gemm");
  if (a.shape.size() != 2 || b.shape.size() != 2) {
    throw std::runtime_error("Gemm currently only supports 2D float32 tensors");
  }

  const auto trans_a = ReadIntAttribute(node, "transA", 0) != 0;
  const auto trans_b = ReadIntAttribute(node, "transB", 0) != 0;
  const auto alpha_attr = node.attributes.find("alpha");
  const auto beta_attr = node.attributes.find("beta");
  const float alpha = alpha_attr == node.attributes.end() ? 1.0f : alpha_attr->second.float_value;
  const float beta = beta_attr == node.attributes.end() ? 1.0f : beta_attr->second.float_value;

  const auto a_rows = static_cast<std::size_t>(a.shape[0]);
  const auto a_cols = static_cast<std::size_t>(a.shape[1]);
  const auto b_rows = static_cast<std::size_t>(b.shape[0]);
  const auto b_cols = static_cast<std::size_t>(b.shape[1]);

  const auto m = trans_a ? a_cols : a_rows;
  const auto k_a = trans_a ? a_rows : a_cols;
  const auto k_b = trans_b ? b_cols : b_rows;
  const auto n = trans_b ? b_rows : b_cols;
  if (k_a != k_b) {
    throw std::runtime_error("Gemm inner dimensions do not match");
  }

  auto output = MakeFloatOutput(node.outputs.at(0), {static_cast<std::int64_t>(m), static_cast<std::int64_t>(n)}, context);
  std::fill(output.float_data.begin(), output.float_data.end(), 0.0f);

#if defined(__APPLE__)
  cblas_sgemm(CblasRowMajor,
              trans_a ? CblasTrans : CblasNoTrans,
              trans_b ? CblasTrans : CblasNoTrans,
              static_cast<int>(m), static_cast<int>(n), static_cast<int>(k_a),
              alpha,
              a_data.data(), static_cast<int>(a_cols),
              b_data.data(), static_cast<int>(b_cols),
              0.0f,
              output.float_data.data(), static_cast<int>(n));
#else
  for (std::size_t i = 0; i < m; ++i) {
    auto* out_row_ptr = output.float_data.data() + i * n;
    const auto* a_row_ptr = trans_a ? nullptr : a_data.data() + i * a_cols;
    for (std::size_t kk = 0; kk < k_a; ++kk) {
      const auto a_value = trans_a ? a_data[kk * a_cols + i] : a_row_ptr[kk];
      const auto* b_row_ptr = trans_b ? nullptr : b_data.data() + kk * b_cols;
      for (std::size_t j = 0; j < n; ++j) {
        const auto b_value = trans_b ? b_data[j * b_cols + kk] : b_row_ptr[j];
        out_row_ptr[j] += alpha * a_value * b_value;
      }
    }
  }
#endif

  if (c != nullptr) {
    if (beta != 1.0f) {
      Tensor scaled_bias = *c;
      scaled_bias.float_data = c->float_data;
      for (auto& value : scaled_bias.float_data) {
        value *= beta;
      }
      ApplyGemmBias(output, &scaled_bias);
    } else {
      ApplyGemmBias(output, c);
    }
  }

  return output;
}

enum class ConvPostOp {
  kNone,
  kSiLU,
};

#if defined(MINIORT_ENABLE_AVX)
inline __m256 MultiplyAdd(__m256 accum, __m256 lhs, __m256 rhs) {
#if defined(__FMA__)
  return _mm256_fmadd_ps(lhs, rhs, accum);
#else
  return _mm256_add_ps(accum, _mm256_mul_ps(lhs, rhs));
#endif
}
#endif

#if defined(MINIORT_ENABLE_SSE)
inline __m128 MultiplyAdd(__m128 accum, __m128 lhs, __m128 rhs) {
#if defined(__FMA__)
  return _mm_fmadd_ps(lhs, rhs, accum);
#else
  return _mm_add_ps(accum, _mm_mul_ps(lhs, rhs));
#endif
}
#endif

Tensor RunConv2D(const Node& node, const Tensor& input, const Tensor& weight, const Tensor* bias,
                 ExecutionContext& context, ConvPostOp post_op = ConvPostOp::kNone) {
  const auto& input_data = RequireFloatData(input, "Conv");
  const auto& weight_data = RequireFloatData(weight, "Conv");
  const std::vector<float>* bias_data = nullptr;
  if (bias != nullptr) {
    bias_data = &RequireFloatData(*bias, "Conv");
  }

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

  const auto n = static_cast<std::size_t>(input.shape[0]);
  const auto c_in = static_cast<std::size_t>(input.shape[1]);
  const auto h_in = static_cast<std::size_t>(input.shape[2]);
  const auto w_in = static_cast<std::size_t>(input.shape[3]);
  const auto c_out = static_cast<std::size_t>(weight.shape[0]);
  const auto w_c_in = static_cast<std::size_t>(weight.shape[1]);
  const auto k_h = static_cast<std::size_t>(weight.shape[2]);
  const auto k_w = static_cast<std::size_t>(weight.shape[3]);

  if (c_in != w_c_in) {
    throw std::runtime_error("Conv input channel count does not match weight");
  }
  if (bias_data != nullptr && bias_data->size() != c_out) {
    throw std::runtime_error("Conv bias size does not match output channels");
  }

  const auto pad_top = pads[0];
  const auto pad_left = pads[1];
  const auto pad_bottom = pads[2];
  const auto pad_right = pads[3];
  const auto dilation_h = dilations[0];
  const auto dilation_w = dilations[1];
  const auto stride_h = strides[0];
  const auto stride_w = strides[1];

  const auto effective_kh = static_cast<std::int64_t>((k_h - 1) * dilation_h + 1);
  const auto effective_kw = static_cast<std::int64_t>((k_w - 1) * dilation_w + 1);
  const auto h_out = (static_cast<std::int64_t>(h_in) + pad_top + pad_bottom - effective_kh) / stride_h + 1;
  const auto w_out = (static_cast<std::int64_t>(w_in) + pad_left + pad_right - effective_kw) / stride_w + 1;
  if (h_out <= 0 || w_out <= 0) {
    throw std::runtime_error("Conv output shape is invalid");
  }

  auto output = MakeFloatOutput(node.outputs.at(0),
                                {static_cast<std::int64_t>(n), static_cast<std::int64_t>(c_out), h_out, w_out},
                                context);

  const auto input_hw = h_in * w_in;
  const auto output_hw = static_cast<std::size_t>(h_out) * static_cast<std::size_t>(w_out);
  const auto kernel_hw = k_h * k_w;
  const auto output_w = static_cast<std::size_t>(w_out);
  const bool is_pointwise_identity_spatial =
      k_h == 1 && k_w == 1 && stride_h == 1 && stride_w == 1 &&
      dilation_h == 1 && dilation_w == 1 &&
      pad_top == 0 && pad_left == 0 && pad_bottom == 0 && pad_right == 0 &&
      h_in == static_cast<std::size_t>(h_out) && w_in == static_cast<std::size_t>(w_out);

  if (is_pointwise_identity_spatial) {
    const auto run_output_planes = [&](std::size_t begin, std::size_t end) {
      for (std::size_t plane_index = begin; plane_index < end; ++plane_index) {
        const auto batch = plane_index / c_out;
        const auto oc = plane_index % c_out;
        const auto* batch_input = input_data.data() + batch * c_in * input_hw;
        auto* batch_output = output.float_data.data() + batch * c_out * output_hw;
        auto* output_plane = batch_output + oc * output_hw;
        const float bias_value = bias_data != nullptr ? (*bias_data)[oc] : 0.0f;
        std::fill_n(output_plane, output_hw, bias_value);

        const auto* weight_oc = weight_data.data() + oc * c_in;
        for (std::size_t ic = 0; ic < c_in; ++ic) {
          const auto* input_plane = batch_input + ic * input_hw;
          const float weight_value = weight_oc[ic];
#if defined(MINIORT_ENABLE_AVX)
          const auto weight_vec = _mm256_set1_ps(weight_value);
          std::size_t i = 0;
          for (; i + 8 <= output_hw; i += 8) {
            const auto input_vec = _mm256_loadu_ps(input_plane + i);
            auto output_vec = _mm256_loadu_ps(output_plane + i);
            output_vec = MultiplyAdd(output_vec, input_vec, weight_vec);
            _mm256_storeu_ps(output_plane + i, output_vec);
          }
          for (; i < output_hw; ++i) {
            output_plane[i] += input_plane[i] * weight_value;
          }
#elif defined(MINIORT_ENABLE_SSE)
          const auto weight_vec = _mm_set1_ps(weight_value);
          std::size_t i = 0;
          for (; i + 4 <= output_hw; i += 4) {
            const auto input_vec = _mm_loadu_ps(input_plane + i);
            auto output_vec = _mm_loadu_ps(output_plane + i);
            output_vec = MultiplyAdd(output_vec, input_vec, weight_vec);
            _mm_storeu_ps(output_plane + i, output_vec);
          }
          for (; i < output_hw; ++i) {
            output_plane[i] += input_plane[i] * weight_value;
          }
#else
          for (std::size_t i = 0; i < output_hw; ++i) {
            output_plane[i] += input_plane[i] * weight_value;
          }
#endif
        }

        if (post_op == ConvPostOp::kSiLU) {
          for (std::size_t i = 0; i < output_hw; ++i) {
            const auto value = output_plane[i];
            output_plane[i] = value * (1.0f / (1.0f + std::exp(-value)));
          }
        }
      }
    };

    const auto plane_work = c_in * output_hw;
    std::size_t min_output_planes_per_thread = 4;
    if (plane_work < 200000) {
      min_output_planes_per_thread = 16;
    } else if (plane_work < 800000) {
      min_output_planes_per_thread = 8;
    }
    ParallelFor(n * c_out, min_output_planes_per_thread, run_output_planes);
    return output;
  }

  const bool is_3x3_pad1_dilation1 =
      k_h == 3 && k_w == 3 &&
      dilation_h == 1 && dilation_w == 1 &&
      pad_top == 1 && pad_left == 1 && pad_bottom == 1 && pad_right == 1 &&
      (stride_h == 1 || stride_h == 2) && stride_h == stride_w;

  if (is_3x3_pad1_dilation1) {
    const auto run_output_planes = [&](std::size_t begin, std::size_t end) {
      for (std::size_t plane_index = begin; plane_index < end; ++plane_index) {
        const auto batch = plane_index / c_out;
        const auto oc = plane_index % c_out;
        const auto* batch_input = input_data.data() + batch * c_in * input_hw;
        auto* batch_output = output.float_data.data() + batch * c_out * output_hw;
        auto* output_plane = batch_output + oc * output_hw;
        const float bias_value = bias_data != nullptr ? (*bias_data)[oc] : 0.0f;
        std::fill_n(output_plane, output_hw, bias_value);

        const auto* weight_oc = weight_data.data() + oc * c_in * kernel_hw;
        for (std::size_t ic = 0; ic < c_in; ++ic) {
          const auto* input_plane = batch_input + ic * input_hw;
          const auto* weight_ic = weight_oc + ic * kernel_hw;
          const float w00 = weight_ic[0];
          const float w01 = weight_ic[1];
          const float w02 = weight_ic[2];
          const float w10 = weight_ic[3];
          const float w11 = weight_ic[4];
          const float w12 = weight_ic[5];
          const float w20 = weight_ic[6];
          const float w21 = weight_ic[7];
          const float w22 = weight_ic[8];

          const std::size_t interior_begin =
              std::min(output_w, static_cast<std::size_t>((1 + stride_w - 1) / stride_w));
          const std::size_t interior_end =
              w_in >= 2 ? std::min(output_w, static_cast<std::size_t>(
                                                 (static_cast<std::int64_t>(w_in) - 2) / stride_w + 1))
                        : 0;

          for (std::size_t oh = 0; oh < static_cast<std::size_t>(h_out); ++oh) {
            const auto ih_base = static_cast<std::int64_t>(oh) * stride_h - 1;
            const float* row0 = (ih_base >= 0 && ih_base < static_cast<std::int64_t>(h_in))
                                    ? input_plane + static_cast<std::size_t>(ih_base) * w_in
                                    : nullptr;
            const float* row1 = (ih_base + 1 >= 0 && ih_base + 1 < static_cast<std::int64_t>(h_in))
                                    ? input_plane + static_cast<std::size_t>(ih_base + 1) * w_in
                                    : nullptr;
            const float* row2 = (ih_base + 2 >= 0 && ih_base + 2 < static_cast<std::int64_t>(h_in))
                                    ? input_plane + static_cast<std::size_t>(ih_base + 2) * w_in
                                    : nullptr;
            auto* output_row = output_plane + oh * output_w;

            const auto add_edge = [&](std::size_t ow) {
              const auto iw_base = static_cast<std::int64_t>(ow) * stride_w - 1;
              const auto sample = [&](const float* row, std::int64_t iw) {
                if (row == nullptr || iw < 0 || iw >= static_cast<std::int64_t>(w_in)) {
                  return 0.0f;
                }
                return row[static_cast<std::size_t>(iw)];
              };
              output_row[ow] += sample(row0, iw_base) * w00 +
                                sample(row0, iw_base + 1) * w01 +
                                sample(row0, iw_base + 2) * w02 +
                                sample(row1, iw_base) * w10 +
                                sample(row1, iw_base + 1) * w11 +
                                sample(row1, iw_base + 2) * w12 +
                                sample(row2, iw_base) * w20 +
                                sample(row2, iw_base + 1) * w21 +
                                sample(row2, iw_base + 2) * w22;
            };

            for (std::size_t ow = 0; ow < interior_begin; ++ow) {
              add_edge(ow);
            }
            std::size_t ow = interior_begin;
#if defined(MINIORT_ENABLE_AVX)
            if (stride_w == 1) {
              const auto w00v = _mm256_set1_ps(w00);
              const auto w01v = _mm256_set1_ps(w01);
              const auto w02v = _mm256_set1_ps(w02);
              const auto w10v = _mm256_set1_ps(w10);
              const auto w11v = _mm256_set1_ps(w11);
              const auto w12v = _mm256_set1_ps(w12);
              const auto w20v = _mm256_set1_ps(w20);
              const auto w21v = _mm256_set1_ps(w21);
              const auto w22v = _mm256_set1_ps(w22);
              for (; ow + 8 <= interior_end; ow += 8) {
                const auto iw = ow - 1;
                auto value = _mm256_loadu_ps(output_row + ow);
                if (row0 != nullptr) {
                  value = MultiplyAdd(value, _mm256_loadu_ps(row0 + iw), w00v);
                  value = MultiplyAdd(value, _mm256_loadu_ps(row0 + iw + 1), w01v);
                  value = MultiplyAdd(value, _mm256_loadu_ps(row0 + iw + 2), w02v);
                }
                if (row1 != nullptr) {
                  value = MultiplyAdd(value, _mm256_loadu_ps(row1 + iw), w10v);
                  value = MultiplyAdd(value, _mm256_loadu_ps(row1 + iw + 1), w11v);
                  value = MultiplyAdd(value, _mm256_loadu_ps(row1 + iw + 2), w12v);
                }
                if (row2 != nullptr) {
                  value = MultiplyAdd(value, _mm256_loadu_ps(row2 + iw), w20v);
                  value = MultiplyAdd(value, _mm256_loadu_ps(row2 + iw + 1), w21v);
                  value = MultiplyAdd(value, _mm256_loadu_ps(row2 + iw + 2), w22v);
                }
                _mm256_storeu_ps(output_row + ow, value);
              }
            }
#elif defined(MINIORT_ENABLE_SSE)
            if (stride_w == 1) {
              const auto w00v = _mm_set1_ps(w00);
              const auto w01v = _mm_set1_ps(w01);
              const auto w02v = _mm_set1_ps(w02);
              const auto w10v = _mm_set1_ps(w10);
              const auto w11v = _mm_set1_ps(w11);
              const auto w12v = _mm_set1_ps(w12);
              const auto w20v = _mm_set1_ps(w20);
              const auto w21v = _mm_set1_ps(w21);
              const auto w22v = _mm_set1_ps(w22);
              for (; ow + 4 <= interior_end; ow += 4) {
                const auto iw = ow - 1;
                auto value = _mm_loadu_ps(output_row + ow);
                if (row0 != nullptr) {
                  value = MultiplyAdd(value, _mm_loadu_ps(row0 + iw), w00v);
                  value = MultiplyAdd(value, _mm_loadu_ps(row0 + iw + 1), w01v);
                  value = MultiplyAdd(value, _mm_loadu_ps(row0 + iw + 2), w02v);
                }
                if (row1 != nullptr) {
                  value = MultiplyAdd(value, _mm_loadu_ps(row1 + iw), w10v);
                  value = MultiplyAdd(value, _mm_loadu_ps(row1 + iw + 1), w11v);
                  value = MultiplyAdd(value, _mm_loadu_ps(row1 + iw + 2), w12v);
                }
                if (row2 != nullptr) {
                  value = MultiplyAdd(value, _mm_loadu_ps(row2 + iw), w20v);
                  value = MultiplyAdd(value, _mm_loadu_ps(row2 + iw + 1), w21v);
                  value = MultiplyAdd(value, _mm_loadu_ps(row2 + iw + 2), w22v);
                }
                _mm_storeu_ps(output_row + ow, value);
              }
            }
#endif
            for (; ow < interior_end; ++ow) {
              const auto iw = static_cast<std::size_t>(static_cast<std::int64_t>(ow) * stride_w - 1);
              float value = output_row[ow];
              if (row0 != nullptr) {
                value += row0[iw] * w00 + row0[iw + 1] * w01 + row0[iw + 2] * w02;
              }
              if (row1 != nullptr) {
                value += row1[iw] * w10 + row1[iw + 1] * w11 + row1[iw + 2] * w12;
              }
              if (row2 != nullptr) {
                value += row2[iw] * w20 + row2[iw + 1] * w21 + row2[iw + 2] * w22;
              }
              output_row[ow] = value;
            }
            for (std::size_t ow = interior_end; ow < output_w; ++ow) {
              add_edge(ow);
            }
          }
        }

        if (post_op == ConvPostOp::kSiLU) {
          for (std::size_t i = 0; i < output_hw; ++i) {
            const auto value = output_plane[i];
            output_plane[i] = value * (1.0f / (1.0f + std::exp(-value)));
          }
        }
      }
    };

    const auto plane_work = c_in * kernel_hw * output_hw;
    std::size_t min_output_planes_per_thread = 4;
    if (plane_work < 200000) {
      min_output_planes_per_thread = 16;
    } else if (plane_work < 800000) {
      min_output_planes_per_thread = 8;
    }
    ParallelFor(n * c_out, min_output_planes_per_thread, run_output_planes);
    return output;
  }

  std::vector<std::int64_t> input_h_bases(k_h);
  std::vector<std::size_t> oh_begins(k_h);
  std::vector<std::size_t> oh_ends(k_h);
  for (std::size_t kh = 0; kh < k_h; ++kh) {
    const auto input_h_base = static_cast<std::int64_t>(kh) * dilation_h - pad_top;
    input_h_bases[kh] = input_h_base;
    oh_begins[kh] =
        input_h_base >= 0 ? 0 : static_cast<std::size_t>((-input_h_base + stride_h - 1) / stride_h);
    oh_ends[kh] = static_cast<std::size_t>(
        std::min<std::int64_t>(h_out, (static_cast<std::int64_t>(h_in) - 1 - input_h_base) / stride_h + 1));
  }
  std::vector<std::int64_t> input_w_bases(k_w);
  std::vector<std::size_t> ow_begins(k_w);
  std::vector<std::size_t> ow_ends(k_w);
  for (std::size_t kw = 0; kw < k_w; ++kw) {
    const auto input_w_base = static_cast<std::int64_t>(kw) * dilation_w - pad_left;
    input_w_bases[kw] = input_w_base;
    ow_begins[kw] =
        input_w_base >= 0 ? 0 : static_cast<std::size_t>((-input_w_base + stride_w - 1) / stride_w);
    ow_ends[kw] = static_cast<std::size_t>(
        std::min<std::int64_t>(w_out, (static_cast<std::int64_t>(w_in) - 1 - input_w_base) / stride_w + 1));
  }

  const auto run_output_planes = [&](std::size_t begin, std::size_t end) {
    for (std::size_t plane_index = begin; plane_index < end; ++plane_index) {
      const auto batch = plane_index / c_out;
      const auto oc = plane_index % c_out;
      const auto* batch_input = input_data.data() + batch * c_in * input_hw;
      auto* batch_output = output.float_data.data() + batch * c_out * output_hw;
      auto* output_plane = batch_output + oc * output_hw;
      const float bias_value = bias_data != nullptr ? (*bias_data)[oc] : 0.0f;
      std::fill_n(output_plane, output_hw, bias_value);

      const auto* weight_oc = weight_data.data() + oc * c_in * kernel_hw;
      for (std::size_t ic = 0; ic < c_in; ++ic) {
        const auto* input_plane = batch_input + ic * input_hw;
        const auto* weight_ic = weight_oc + ic * kernel_hw;

        for (std::size_t kh = 0; kh < k_h; ++kh) {
          const auto input_h_base = input_h_bases[kh];
          const auto oh_begin = oh_begins[kh];
          const auto oh_end = oh_ends[kh];
          if (oh_begin >= oh_end) {
            continue;
          }

          for (std::size_t kw = 0; kw < k_w; ++kw) {
            const auto input_w_base = input_w_bases[kw];
            const auto ow_begin = ow_begins[kw];
            const auto ow_end = ow_ends[kw];
            if (ow_begin >= ow_end) {
              continue;
            }

            const float weight_value = weight_ic[kh * k_w + kw];
            for (std::size_t oh = oh_begin; oh < oh_end; ++oh) {
              const auto ih = static_cast<std::size_t>(static_cast<std::int64_t>(oh) * stride_h + input_h_base);
              const auto* input_row = input_plane + ih * w_in;
              auto* output_row = output_plane + oh * output_w;
              for (std::size_t ow = ow_begin; ow < ow_end; ++ow) {
                const auto iw = static_cast<std::size_t>(static_cast<std::int64_t>(ow) * stride_w + input_w_base);
                output_row[ow] += input_row[iw] * weight_value;
              }
            }
          }
        }
      }

      if (post_op == ConvPostOp::kSiLU) {
        for (std::size_t i = 0; i < output_hw; ++i) {
          const auto value = output_plane[i];
          output_plane[i] = value * (1.0f / (1.0f + std::exp(-value)));
        }
      }
    }
  };

  // ORT's CPU thread-pool path uses a cost model to avoid over-parallelizing
  // light kernels. Keep MiniORT simple, but scale down thread fan-out when a
  // single output plane is relatively cheap.
  const auto plane_work = c_in * kernel_hw * output_hw;
  std::size_t min_output_planes_per_thread = 4;
  if (plane_work < 200000) {
    min_output_planes_per_thread = 16;
  } else if (plane_work < 800000) {
    min_output_planes_per_thread = 8;
  }
  ParallelFor(n * c_out, min_output_planes_per_thread, run_output_planes);

  return output;
}

Tensor RunLayerNormalization(const Node& node, const Tensor& input, const Tensor& scale, const Tensor& bias,
                             ExecutionContext& context) {
  const auto& input_data = RequireFloatData(input, "LayerNormalization");
  const auto& scale_data = RequireFloatData(scale, "LayerNormalization");
  const auto& bias_data = RequireFloatData(bias, "LayerNormalization");
  const auto axis = static_cast<std::size_t>(
      NormalizeAxis(ReadIntAttribute(node, "axis", -1), input.shape.size(), "LayerNormalization"));
  const auto epsilon_it = node.attributes.find("epsilon");
  const float epsilon = epsilon_it == node.attributes.end() ? 1e-5f : epsilon_it->second.float_value;

  std::size_t outer = 1;
  for (std::size_t i = 0; i < axis; ++i) {
    outer *= static_cast<std::size_t>(input.shape[i]);
  }
  std::size_t normalized_size = 1;
  for (std::size_t i = axis; i < input.shape.size(); ++i) {
    normalized_size *= static_cast<std::size_t>(input.shape[i]);
  }
  const float inv_normalized_size = 1.0f / static_cast<float>(normalized_size);

  if (scale_data.size() != normalized_size || bias_data.size() != normalized_size) {
    throw std::runtime_error("LayerNormalization scale/bias shape mismatch");
  }

  auto output = MakeOutputLikeWithReusedStorage(node.outputs.at(0), input, context);
  for (std::size_t outer_index = 0; outer_index < outer; ++outer_index) {
    const auto base = outer_index * normalized_size;
    const auto* input_row = input_data.data() + base;
    const auto* scale_row = scale_data.data();
    const auto* bias_row = bias_data.data();
    float mean = 0.0f;
    for (std::size_t i = 0; i < normalized_size; ++i) {
      mean += input_row[i];
    }
    mean *= inv_normalized_size;

    float variance = 0.0f;
    for (std::size_t i = 0; i < normalized_size; ++i) {
      const auto diff = input_row[i] - mean;
      variance += diff * diff;
    }
    variance *= inv_normalized_size;
    const auto inv_stddev = 1.0f / std::sqrt(variance + epsilon);

    for (std::size_t i = 0; i < normalized_size; ++i) {
      output.float_data[base + i] = ((input_row[i] - mean) * inv_stddev) * scale_row[i] + bias_row[i];
    }
  }

  return output;
}

}  // namespace

void RegisterNnKernels(KernelRegistry& registry) {
  registry.Register("MatMul", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& lhs = RequireTensor(context, node.inputs.at(0));
    const auto& rhs = RequireTensor(context, node.inputs.at(1));
    auto output = RunMatMul(node.outputs.at(0), lhs, rhs, context);
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel MatMul produced " << node.outputs.at(0) << "\n";
    }
  });

  registry.Register("Gemm", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& a = RequireTensor(context, node.inputs.at(0));
    const auto& b = RequireTensor(context, node.inputs.at(1));
    const Tensor* c = nullptr;
    if (node.inputs.size() > 2 && !node.inputs.at(2).empty()) {
      c = &RequireTensor(context, node.inputs.at(2));
    }
    auto output = RunGemm2D(node, a, b, c, context);
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Gemm produced " << node.outputs.at(0) << "\n";
    }
  });

  registry.Register("Conv", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& input = RequireTensor(context, node.inputs.at(0));
    const auto& weight = RequireTensor(context, node.inputs.at(1));
    const Tensor* bias = nullptr;
    if (node.inputs.size() > 2 && !node.inputs.at(2).empty()) {
      bias = &RequireTensor(context, node.inputs.at(2));
    }
    auto output = RunConv2D(node, input, weight, bias, context);
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Conv produced " << node.outputs.at(0) << "\n";
    }
  });

  registry.Register("ConvSiLU", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& input = RequireTensor(context, node.inputs.at(0));
    const auto& weight = RequireTensor(context, node.inputs.at(1));
    const Tensor* bias = nullptr;
    if (node.inputs.size() > 2 && !node.inputs.at(2).empty()) {
      bias = &RequireTensor(context, node.inputs.at(2));
    }

    auto output = RunConv2D(node, input, weight, bias, context, ConvPostOp::kSiLU);

    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel ConvSiLU produced " << node.outputs.at(0) << "\n";
    }
  });

  registry.Register("MaxPool", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& input = RequireTensor(context, node.inputs.at(0));
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
    const bool is_sppf_maxpool =
        k_h == 5 && k_w == 5 &&
        stride_h == 1 && stride_w == 1 &&
        dilation_h == 1 && dilation_w == 1 &&
        pad_top == 2 && pad_left == 2 && pad_bottom == 2 && pad_right == 2 &&
        static_cast<std::size_t>(h_out) == h_in && static_cast<std::size_t>(w_out) == w_in;
    if (is_sppf_maxpool) {
      std::vector<float> horizontal_max(n * c * input_hw);
      const auto run_planes = [&](std::size_t begin, std::size_t end) {
        for (std::size_t plane_index = begin; plane_index < end; ++plane_index) {
          const auto* input_plane = input_data.data() + plane_index * input_hw;
          auto* horizontal_plane = horizontal_max.data() + plane_index * input_hw;
          auto* output_plane = output.float_data.data() + plane_index * output_hw;

          for (std::size_t h = 0; h < h_in; ++h) {
            const auto* input_row = input_plane + h * w_in;
            auto* horizontal_row = horizontal_plane + h * w_in;
            for (std::size_t w = 0; w < w_in; ++w) {
              const auto w_begin = w > 2 ? w - 2 : 0;
              const auto w_end = std::min(w_in, w + 3);
              float best = -std::numeric_limits<float>::infinity();
              for (std::size_t iw = w_begin; iw < w_end; ++iw) {
                best = std::max(best, input_row[iw]);
              }
              horizontal_row[w] = best;
            }
          }

          for (std::size_t h = 0; h < h_in; ++h) {
            auto* output_row = output_plane + h * w_in;
            const auto h_begin = h > 2 ? h - 2 : 0;
            const auto h_end = std::min(h_in, h + 3);
            for (std::size_t w = 0; w < w_in; ++w) {
              float best = -std::numeric_limits<float>::infinity();
              for (std::size_t ih = h_begin; ih < h_end; ++ih) {
                best = std::max(best, horizontal_plane[ih * w_in + w]);
              }
              output_row[w] = best;
            }
          }
        }
      };
      ParallelFor(n * c, 4, run_planes);
      context.BindTensor(std::move(output));
      if (trace != nullptr) {
        *trace << "    kernel MaxPool produced " << node.outputs.at(0) << "\n";
      }
      return;
    }

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
                const auto input_index =
                    ((batch * c + channel) * input_hw) + static_cast<std::size_t>(ih) * w_in + static_cast<std::size_t>(iw);
                best = std::max(best, input_data[input_index]);
              }
            }

            const auto output_index =
                ((batch * c + channel) * output_hw) +
                static_cast<std::size_t>(oh) * static_cast<std::size_t>(w_out) +
                static_cast<std::size_t>(ow);
            output.float_data[output_index] = best;
          }
        }
      }
    }

    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel MaxPool produced " << node.outputs.at(0) << "\n";
    }
  });

  registry.Register("Resize", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& input = RequireTensor(context, node.inputs.at(0));
    const auto& input_data = RequireFloatData(input, "Resize");
    if (input.shape.size() != 4) {
      throw std::runtime_error("Resize currently only supports 4D NCHW tensors");
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
      throw std::runtime_error("Resize currently only supports nearest+asymmetric+floor");
    }

    if (node.inputs.size() < 3 || node.inputs.at(2).empty()) {
      throw std::runtime_error("Resize currently expects scales input");
    }
    const auto& scales_tensor = RequireTensor(context, node.inputs.at(2));
    const auto& scales = RequireFloatData(scales_tensor, "Resize");
    if (scales.size() != 4) {
      throw std::runtime_error("Resize currently expects 4D scales");
    }

    const auto n_out = static_cast<std::int64_t>(std::floor(static_cast<double>(input.shape[0]) * scales[0]));
    const auto c_out = static_cast<std::int64_t>(std::floor(static_cast<double>(input.shape[1]) * scales[1]));
    const auto h_out = static_cast<std::int64_t>(std::floor(static_cast<double>(input.shape[2]) * scales[2]));
    const auto w_out = static_cast<std::int64_t>(std::floor(static_cast<double>(input.shape[3]) * scales[3]));
    if (n_out != input.shape[0] || c_out != input.shape[1]) {
      throw std::runtime_error("Resize currently requires batch/channel scales to keep dimensions unchanged");
    }
    if (h_out <= 0 || w_out <= 0) {
      throw std::runtime_error("Resize output shape is invalid");
    }

    const auto n = static_cast<std::size_t>(input.shape[0]);
    const auto c = static_cast<std::size_t>(input.shape[1]);
    const auto h_in = static_cast<std::size_t>(input.shape[2]);
    const auto w_in = static_cast<std::size_t>(input.shape[3]);

    auto output = MakeFloatOutput(node.outputs.at(0),
                                  {input.shape[0], input.shape[1], h_out, w_out}, context);

    const auto input_hw = h_in * w_in;
    const auto output_hw = static_cast<std::size_t>(h_out) * static_cast<std::size_t>(w_out);
    const auto output_h = static_cast<std::size_t>(h_out);
    const auto output_w = static_cast<std::size_t>(w_out);

    const bool is_nearest_2x =
        std::fabs(scales[2] - 2.0f) < 1e-6f && std::fabs(scales[3] - 2.0f) < 1e-6f &&
        output_h == h_in * 2 && output_w == w_in * 2;
    if (is_nearest_2x) {
      const auto run_planes = [&](std::size_t begin, std::size_t end) {
        for (std::size_t plane_index = begin; plane_index < end; ++plane_index) {
          const auto batch = plane_index / c;
          const auto channel = plane_index % c;
          const auto* input_plane = input_data.data() + (batch * c + channel) * input_hw;
          auto* output_plane = output.float_data.data() + (batch * c + channel) * output_hw;
          for (std::size_t ih = 0; ih < h_in; ++ih) {
            const auto* input_row = input_plane + ih * w_in;
            auto* output_row0 = output_plane + (ih * 2) * output_w;
            auto* output_row1 = output_row0 + output_w;
            std::size_t iw = 0;
#if defined(MINIORT_ENABLE_SSE)
            for (; iw + 4 <= w_in; iw += 4) {
              const auto value = _mm_loadu_ps(input_row + iw);
              _mm_storeu_ps(output_row0 + iw * 2, _mm_unpacklo_ps(value, value));
              _mm_storeu_ps(output_row0 + iw * 2 + 4, _mm_unpackhi_ps(value, value));
            }
#endif
            for (; iw < w_in; ++iw) {
              const float value = input_row[iw];
              output_row0[iw * 2] = value;
              output_row0[iw * 2 + 1] = value;
            }
            std::copy_n(output_row0, output_w, output_row1);
          }
        }
      };
      ParallelFor(n * c, 4, run_planes);
      context.BindTensor(std::move(output));
      if (trace != nullptr) {
        *trace << "    kernel Resize produced " << node.outputs.at(0) << "\n";
      }
      return;
    }

    for (std::size_t batch = 0; batch < n; ++batch) {
      for (std::size_t channel = 0; channel < c; ++channel) {
        for (std::int64_t oh = 0; oh < h_out; ++oh) {
          const auto ih = std::min(static_cast<std::size_t>(std::floor(static_cast<double>(oh) / scales[2])), h_in - 1);
          for (std::int64_t ow = 0; ow < w_out; ++ow) {
            const auto iw = std::min(static_cast<std::size_t>(std::floor(static_cast<double>(ow) / scales[3])), w_in - 1);

            const auto input_index = ((batch * c + channel) * input_hw) + ih * w_in + iw;
            const auto output_index =
                ((batch * c + channel) * output_hw) +
                static_cast<std::size_t>(oh) * static_cast<std::size_t>(w_out) +
                static_cast<std::size_t>(ow);
            output.float_data[output_index] = input_data[input_index];
          }
        }
      }
    }

    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Resize produced " << node.outputs.at(0) << "\n";
    }
  });

  registry.Register("Softmax", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& input = RequireTensor(context, node.inputs.at(0));
    const auto& input_data = RequireFloatData(input, "Softmax");
    const auto axis = static_cast<std::size_t>(
        NormalizeAxis(ReadIntAttribute(node, "axis", 1), input.shape.size(), "Softmax"));

    std::size_t outer = 1;
    for (std::size_t i = 0; i < axis; ++i) {
      outer *= static_cast<std::size_t>(input.shape[i]);
    }
    const std::size_t axis_dim = static_cast<std::size_t>(input.shape[axis]);
    std::size_t inner = 1;
    for (std::size_t i = axis + 1; i < input.shape.size(); ++i) {
      inner *= static_cast<std::size_t>(input.shape[i]);
    }

  auto output = MakeOutputLikeWithReusedStorage(node.outputs.at(0), input, context);
  if (inner == 1) {
    std::vector<float> exp_values(axis_dim);
    std::vector<float> shifted(axis_dim);
    for (std::size_t outer_index = 0; outer_index < outer; ++outer_index) {
      const auto* row = input_data.data() + outer_index * axis_dim;
      auto* out_row = output.float_data.data() + outer_index * axis_dim;
      float max_value = -std::numeric_limits<float>::infinity();
      for (std::size_t i = 0; i < axis_dim; ++i) {
        max_value = std::max(max_value, row[i]);
      }
      for (std::size_t i = 0; i < axis_dim; ++i) {
        shifted[i] = row[i] - max_value;
      }
      for (std::size_t i = 0; i < axis_dim; ++i) {
        exp_values[i] = std::exp(shifted[i]);
      }
      float denom_sum = 0.0f;
      for (std::size_t i = 0; i < axis_dim; ++i) {
        denom_sum += exp_values[i];
      }
      const float inv_sum = 1.0f / denom_sum;
      for (std::size_t i = 0; i < axis_dim; ++i) {
        out_row[i] = exp_values[i] * inv_sum;
      }
    }
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Softmax produced " << node.outputs.at(0) << "\n";
    }
    return;
  }

  for (std::size_t outer_index = 0; outer_index < outer; ++outer_index) {
    for (std::size_t inner_index = 0; inner_index < inner; ++inner_index) {
      const auto row_base = (outer_index * axis_dim) * inner + inner_index;
      float max_value = -std::numeric_limits<float>::infinity();
      for (std::size_t axis_index = 0; axis_index < axis_dim; ++axis_index) {
        const auto offset = row_base + axis_index * inner;
        max_value = std::max(max_value, input_data[offset]);
      }

      float sum = 0.0f;
      for (std::size_t axis_index = 0; axis_index < axis_dim; ++axis_index) {
        const auto offset = row_base + axis_index * inner;
        const auto value = std::exp(input_data[offset] - max_value);
        output.float_data[offset] = value;
        sum += value;
      }

      for (std::size_t axis_index = 0; axis_index < axis_dim; ++axis_index) {
        const auto offset = row_base + axis_index * inner;
        output.float_data[offset] /= sum;
      }
    }
  }

    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Softmax produced " << node.outputs.at(0) << "\n";
    }
  });

  registry.Register("LayerNormalization", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& input = RequireTensor(context, node.inputs.at(0));
    const auto& scale = RequireTensor(context, node.inputs.at(1));
    const auto& bias = RequireTensor(context, node.inputs.at(2));
    auto output = RunLayerNormalization(node, input, scale, bias, context);
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel LayerNormalization produced " << node.outputs.at(0) << "\n";
    }
  });
}

}  // namespace miniort
