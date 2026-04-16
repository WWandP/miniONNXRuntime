#include "miniort/runtime/accelerate_execution_provider.h"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "kernel_utils.h"
#include "miniort/runtime/cpu_tensor_allocator.h"

namespace miniort {

namespace {

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

void ApplyGemmBias(Tensor& output, const Tensor* bias) {
  if (bias == nullptr) {
    return;
  }
  const auto& bias_data = RequireFloatData(*bias, "Gemm");
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

void ApplySiLUInPlaceAccelerate(Tensor& output) {
  const int element_count = static_cast<int>(output.float_data.size());
  std::vector<float> negated(output.float_data.size());
  std::vector<float> exp_values(output.float_data.size());
  std::vector<float> denom(output.float_data.size());
  std::vector<float> sigmoid(output.float_data.size());
  std::vector<float> ones(output.float_data.size(), 1.0f);
  float minus_one = -1.0f;

  vDSP_vsmul(output.float_data.data(), 1, &minus_one, negated.data(), 1, output.float_data.size());
  vvexpf(exp_values.data(), negated.data(), &element_count);
  vDSP_vadd(ones.data(), 1, exp_values.data(), 1, denom.data(), 1, output.float_data.size());
  vvrecf(sigmoid.data(), denom.data(), &element_count);
  vDSP_vmul(output.float_data.data(), 1, sigmoid.data(), 1, output.float_data.data(), 1, output.float_data.size());
}

template <typename FloatOp>
void ApplyFloatBroadcastOp(Tensor& output, const Tensor& lhs, const Tensor& rhs, FloatOp&& op) {
  if (lhs.dtype != "float32" || rhs.dtype != "float32") {
    throw std::runtime_error("Accelerate broadcast op currently supports float32 only");
  }

  const auto lhs_data = RequireFloatData(lhs, "Accelerate");
  const auto rhs_data = RequireFloatData(rhs, "Accelerate");
  const auto output_shape = ComputeBroadcastShape(lhs.shape, rhs.shape, "Accelerate");
  if (output_shape != output.shape) {
    throw std::runtime_error("broadcast output shape mismatch");
  }
  if (output.shape.empty()) {
    output.float_data[0] = op(lhs_data[0], rhs_data[0]);
    return;
  }

  const auto output_inner = static_cast<std::size_t>(output.shape.back());
  std::size_t output_outer = 1;
  for (std::size_t i = 0; i + 1 < output.shape.size(); ++i) {
    output_outer *= static_cast<std::size_t>(output.shape[i]);
  }

  const bool lhs_scalar = lhs.shape.empty();
  const bool rhs_scalar = rhs.shape.empty();
  const bool lhs_vector = lhs.shape.size() == 1 && static_cast<std::size_t>(lhs.shape[0]) == output_inner;
  const bool rhs_vector = rhs.shape.size() == 1 && static_cast<std::size_t>(rhs.shape[0]) == output_inner;

  if (lhs_scalar && rhs_scalar) {
    output.float_data[0] = op(lhs_data[0], rhs_data[0]);
    return;
  }

  if (lhs_scalar) {
    const float scalar = lhs_data[0];
    for (std::size_t i = 0; i < output.float_data.size(); ++i) {
      output.float_data[i] = op(scalar, rhs_data[i]);
    }
    return;
  }

  if (rhs_scalar) {
    const float scalar = rhs_data[0];
    for (std::size_t i = 0; i < output.float_data.size(); ++i) {
      output.float_data[i] = op(lhs_data[i], scalar);
    }
    return;
  }

  if (lhs.shape == rhs.shape && lhs.shape == output.shape) {
    for (std::size_t i = 0; i < output.float_data.size(); ++i) {
      output.float_data[i] = op(lhs_data[i], rhs_data[i]);
    }
    return;
  }

  if (lhs_vector && rhs.shape == output.shape) {
    for (std::size_t outer = 0; outer < output_outer; ++outer) {
      const auto base = outer * output_inner;
      for (std::size_t i = 0; i < output_inner; ++i) {
        output.float_data[base + i] = op(lhs_data[i], rhs_data[base + i]);
      }
    }
    return;
  }

  if (rhs_vector && lhs.shape == output.shape) {
    for (std::size_t outer = 0; outer < output_outer; ++outer) {
      const auto base = outer * output_inner;
      for (std::size_t i = 0; i < output_inner; ++i) {
        output.float_data[base + i] = op(lhs_data[base + i], rhs_data[i]);
      }
    }
    return;
  }

  const auto output_strides = ComputeStrides(output.shape);
  const auto lhs_strides = ComputeStrides(lhs.shape);
  const auto rhs_strides = ComputeStrides(rhs.shape);
  for (std::size_t i = 0; i < output.float_data.size(); ++i) {
    const auto output_index = UnravelIndex(i, output.shape, output_strides);
    const auto lhs_offset = ComputeBroadcastOffset(output_index, lhs.shape, lhs_strides);
    const auto rhs_offset = ComputeBroadcastOffset(output_index, rhs.shape, rhs_strides);
    output.float_data[i] = op(lhs_data[lhs_offset], rhs_data[rhs_offset]);
  }
}

Tensor RunLayerNormalizationAccelerate(const Node& node, const Tensor& input, const Tensor& scale, const Tensor& bias,
                                       ExecutionContext& context) {
  const auto& input_data = RequireFloatData(input, "LayerNormalization");
  const auto& scale_data = RequireFloatData(scale, "LayerNormalization");
  const auto& bias_data = RequireFloatData(bias, "LayerNormalization");
  const auto axis =
      static_cast<std::size_t>(NormalizeAxis(ReadIntAttribute(node, "axis", -1), input.shape.size(), "LayerNormalization"));
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
    const float inv_stddev = 1.0f / std::sqrt(variance + epsilon);

    for (std::size_t i = 0; i < normalized_size; ++i) {
      output.float_data[base + i] = ((input_row[i] - mean) * inv_stddev) * scale_row[i] + bias_row[i];
    }
  }

  return output;
}

Tensor RunMatMulAccelerate(const Node& node, const Tensor& lhs, const Tensor& rhs, ExecutionContext& context) {
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
  auto output = MakeFloatOutput(node.outputs.at(0), output_shape, context);

  const auto output_batch_strides = ComputeStrides(output_batch_shape);
  const auto lhs_full_strides = ComputeStrides(lhs.shape);
  const auto rhs_full_strides = ComputeStrides(rhs.shape);
  const auto batch_count = GetElementCount(output_batch_shape);

  for (std::size_t batch = 0; batch < batch_count; ++batch) {
    const auto batch_index = UnravelIndex(batch, output_batch_shape, output_batch_strides);
    const auto lhs_batch_offset =
        lhs_batch_shape.empty() ? 0 : ComputeBroadcastOffset(batch_index, lhs_batch_shape, lhs_full_strides);
    const auto rhs_batch_offset =
        rhs_batch_shape.empty() ? 0 : ComputeBroadcastOffset(batch_index, rhs_batch_shape, rhs_full_strides);

    const auto lhs_base = lhs_batch_shape.empty() ? 0 : lhs_batch_offset;
    const auto rhs_base = rhs_batch_shape.empty() ? 0 : rhs_batch_offset;
    const auto out_base = batch * m * n;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                1.0f,
                lhs_data.data() + lhs_base, static_cast<int>(k),
                rhs_data.data() + rhs_base, static_cast<int>(n),
                0.0f,
                output.float_data.data() + out_base, static_cast<int>(n));
  }

  return output;
}

Tensor RunGatherAccelerate(const Node& node, const Tensor& data, const Tensor& indices, ExecutionContext& context) {
  const auto axis = NormalizeAxis(ReadIntAttribute(node, "axis", 0), data.shape.size(), "Gather");
  const auto& index_data = RequireInt64Data(indices, "Gather");
  std::vector<std::int64_t> output_shape;
  output_shape.reserve(data.shape.size() + indices.shape.size());
  for (std::size_t i = 0; i < static_cast<std::size_t>(axis); ++i) {
    output_shape.push_back(data.shape[i]);
  }
  output_shape.insert(output_shape.end(), indices.shape.begin(), indices.shape.end());
  for (std::size_t i = static_cast<std::size_t>(axis) + 1; i < data.shape.size(); ++i) {
    output_shape.push_back(data.shape[i]);
  }

  std::size_t outer = 1;
  for (std::size_t i = 0; i < static_cast<std::size_t>(axis); ++i) {
    outer *= static_cast<std::size_t>(data.shape[i]);
  }
  const auto axis_dim = static_cast<std::size_t>(data.shape[static_cast<std::size_t>(axis)]);
  std::size_t inner = 1;
  for (std::size_t i = static_cast<std::size_t>(axis) + 1; i < data.shape.size(); ++i) {
    inner *= static_cast<std::size_t>(data.shape[i]);
  }

  const auto normalize_index = [&](std::int64_t index) -> std::size_t {
    if (index < 0) {
      index += static_cast<std::int64_t>(axis_dim);
    }
    if (index < 0 || index >= static_cast<std::int64_t>(axis_dim)) {
      throw std::runtime_error("Gather index is out of range");
    }
    return static_cast<std::size_t>(index);
  };

  Tensor output;
  output.name = node.outputs.at(0);
  output.dtype = data.dtype;
  output.shape = std::move(output_shape);
  output.is_placeholder = false;

  const auto output_size = GetElementCount(output.shape);
  if (data.dtype == "int64") {
    const auto& data_values = RequireInt64Data(data, "Gather");
    output.int64_data = context.AcquireInt64Buffer(output_size);
    output.int64_data.resize(output_size);
    for (std::size_t outer_index = 0; outer_index < outer; ++outer_index) {
      for (std::size_t index_pos = 0; index_pos < index_data.size(); ++index_pos) {
        const auto gather_index = normalize_index(index_data[index_pos]);
        const auto input_offset = (outer_index * axis_dim + gather_index) * inner;
        const auto output_offset = (outer_index * index_data.size() + index_pos) * inner;
        std::memcpy(output.int64_data.data() + output_offset, data_values.data() + input_offset,
                    inner * sizeof(std::int64_t));
      }
    }
  } else if (data.dtype == "float32") {
    const auto& data_values = RequireFloatData(data, "Gather");
    output.float_data = context.AcquireFloatBuffer(output_size);
    output.float_data.resize(output_size);
    for (std::size_t outer_index = 0; outer_index < outer; ++outer_index) {
      for (std::size_t index_pos = 0; index_pos < index_data.size(); ++index_pos) {
        const auto gather_index = normalize_index(index_data[index_pos]);
        const auto input_offset = (outer_index * axis_dim + gather_index) * inner;
        const auto output_offset = (outer_index * index_data.size() + index_pos) * inner;
        std::memcpy(output.float_data.data() + output_offset, data_values.data() + input_offset,
                    inner * sizeof(float));
      }
    }
  } else {
    throw std::runtime_error("Gather currently supports float32/int64 data only");
  }

  return output;
}

Tensor RunTransposeAccelerate(const Node& node, const Tensor& input, ExecutionContext& context) {
  std::vector<std::int64_t> perm;
  const auto perm_it = node.attributes.find("perm");
  if (perm_it == node.attributes.end() || perm_it->second.ints.empty()) {
    perm.resize(input.shape.size());
    for (std::size_t i = 0; i < perm.size(); ++i) {
      perm[i] = static_cast<std::int64_t>(perm.size() - 1 - i);
    }
  } else {
    perm = perm_it->second.ints;
  }
  if (perm.size() != input.shape.size()) {
    throw std::runtime_error("Transpose perm rank mismatch");
  }

  Tensor output;
  output.name = node.outputs.at(0);
  output.dtype = input.dtype;
  output.shape.resize(input.shape.size());
  for (std::size_t i = 0; i < perm.size(); ++i) {
    output.shape[i] = input.shape[static_cast<std::size_t>(perm[i])];
  }
  output.is_placeholder = false;

  const auto element_count = GetElementCount(output.shape);
  const auto rank = input.shape.size();
  const bool is_float = input.dtype == "float32";
  const bool is_int64 = input.dtype == "int64";
  if (!is_float && !is_int64) {
    throw std::runtime_error("Transpose currently supports float32/int64 only");
  }

  if (is_float) {
    const auto& input_data = RequireFloatData(input, "Transpose");
    output.float_data = context.AcquireFloatBuffer(element_count);
    output.float_data.resize(element_count);
  } else {
    const auto& input_data = RequireInt64Data(input, "Transpose");
    output.int64_data = context.AcquireInt64Buffer(element_count);
    output.int64_data.resize(element_count);
  }

  const auto perm_matches = [&](std::initializer_list<std::int64_t> expected) {
    if (perm.size() != expected.size()) {
      return false;
    }
    std::size_t i = 0;
    for (const auto value : expected) {
      if (perm[i++] != value) {
        return false;
      }
    }
    return true;
  };

  if (rank == 4 && perm_matches({0, 2, 1, 3})) {
    const auto n_dim = static_cast<std::size_t>(input.shape[0]);
    const auto h_dim = static_cast<std::size_t>(input.shape[1]);
    const auto c_dim = static_cast<std::size_t>(input.shape[2]);
    const auto w_dim = static_cast<std::size_t>(input.shape[3]);
    if (is_float) {
      const auto& input_data = RequireFloatData(input, "Transpose");
      for (std::size_t n = 0; n < n_dim; ++n) {
        for (std::size_t h = 0; h < h_dim; ++h) {
          for (std::size_t c = 0; c < c_dim; ++c) {
            const auto src = (((n * h_dim + h) * c_dim + c) * w_dim);
            const auto dst = (((n * c_dim + c) * h_dim + h) * w_dim);
            std::memcpy(output.float_data.data() + dst, input_data.data() + src, w_dim * sizeof(float));
          }
        }
      }
    } else {
      const auto& input_data = RequireInt64Data(input, "Transpose");
      for (std::size_t n = 0; n < n_dim; ++n) {
        for (std::size_t h = 0; h < h_dim; ++h) {
          for (std::size_t c = 0; c < c_dim; ++c) {
            const auto src = (((n * h_dim + h) * c_dim + c) * w_dim);
            const auto dst = (((n * c_dim + c) * h_dim + h) * w_dim);
            std::memcpy(output.int64_data.data() + dst, input_data.data() + src,
                        w_dim * sizeof(std::int64_t));
          }
        }
      }
    }
    return output;
  }

  const auto input_strides = ComputeStrides(input.shape);
  const auto output_strides = ComputeStrides(output.shape);
  if (is_float) {
    const auto& input_data = RequireFloatData(input, "Transpose");
    for (std::size_t i = 0; i < output.float_data.size(); ++i) {
      const auto output_index = UnravelIndex(i, output.shape, output_strides);
      std::size_t input_offset = 0;
      for (std::size_t j = 0; j < perm.size(); ++j) {
        input_offset += static_cast<std::size_t>(output_index[j]) *
                        input_strides[static_cast<std::size_t>(perm[j])];
      }
      output.float_data[i] = input_data[input_offset];
    }
  } else {
    const auto& input_data = RequireInt64Data(input, "Transpose");
    for (std::size_t i = 0; i < output.int64_data.size(); ++i) {
      const auto output_index = UnravelIndex(i, output.shape, output_strides);
      std::size_t input_offset = 0;
      for (std::size_t j = 0; j < perm.size(); ++j) {
        input_offset += static_cast<std::size_t>(output_index[j]) *
                        input_strides[static_cast<std::size_t>(perm[j])];
      }
      output.int64_data[i] = input_data[input_offset];
    }
  }

  return output;
}

Tensor RunWhereAccelerate(const Node& node, const Tensor& condition, const Tensor& x, const Tensor& y,
                          ExecutionContext& context) {
  const auto output_shape = ComputeBroadcastShape(ComputeBroadcastShape(condition.shape, x.shape, "Where"), y.shape,
                                                  "Where");
  const auto output_strides = ComputeStrides(output_shape);
  const auto condition_strides = ComputeStrides(condition.shape);
  const auto x_strides = ComputeStrides(x.shape);
  const auto y_strides = ComputeStrides(y.shape);
  const auto element_count = GetElementCount(output_shape);
  const auto* condition_int64_data =
      condition.dtype == "int64" ? &RequireInt64Data(condition, "Where") : nullptr;
  const auto* condition_float_data =
      condition.dtype == "float32" ? &RequireFloatData(condition, "Where") : nullptr;

  const auto read_condition = [&](std::size_t offset) {
    if (condition_int64_data != nullptr) {
      return (*condition_int64_data)[offset] != 0;
    }
    if (condition_float_data != nullptr) {
      return (*condition_float_data)[offset] != 0.0f;
    }
    throw std::runtime_error("Where condition currently supports int64/float32 only");
  };

  Tensor output;
  output.name = node.outputs.at(0);
  output.dtype = x.dtype == "int64" && y.dtype == "int64" ? "int64" : "float32";
  output.shape = output_shape;
  output.is_placeholder = false;

  if (output.dtype == "int64") {
    const auto& x_data = RequireInt64Data(x, "Where");
    const auto& y_data = RequireInt64Data(y, "Where");
    output.int64_data = context.AcquireInt64Buffer(element_count);
    output.int64_data.resize(element_count);
    for (std::size_t i = 0; i < element_count; ++i) {
      const auto output_index = UnravelIndex(i, output_shape, output_strides);
      const auto cond_offset = ComputeBroadcastOffset(output_index, condition.shape, condition_strides);
      const auto x_offset = ComputeBroadcastOffset(output_index, x.shape, x_strides);
      const auto y_offset = ComputeBroadcastOffset(output_index, y.shape, y_strides);
      output.int64_data[i] = read_condition(cond_offset) ? x_data[x_offset] : y_data[y_offset];
    }
  } else {
    const auto* x_float_data = x.dtype == "float32" ? &RequireFloatData(x, "Where") : nullptr;
    const auto* x_int_data = x.dtype == "int64" ? &RequireInt64Data(x, "Where") : nullptr;
    const auto* y_float_data = y.dtype == "float32" ? &RequireFloatData(y, "Where") : nullptr;
    const auto* y_int_data = y.dtype == "int64" ? &RequireInt64Data(y, "Where") : nullptr;
    output.float_data = context.AcquireFloatBuffer(element_count);
    output.float_data.resize(element_count);
    for (std::size_t i = 0; i < element_count; ++i) {
      const auto output_index = UnravelIndex(i, output_shape, output_strides);
      const auto cond_offset = ComputeBroadcastOffset(output_index, condition.shape, condition_strides);
      const auto x_offset = ComputeBroadcastOffset(output_index, x.shape, x_strides);
      const auto y_offset = ComputeBroadcastOffset(output_index, y.shape, y_strides);
      const auto x_value = x_float_data != nullptr ? (*x_float_data)[x_offset]
                                                   : static_cast<float>((*x_int_data)[x_offset]);
      const auto y_value = y_float_data != nullptr ? (*y_float_data)[y_offset]
                                                   : static_cast<float>((*y_int_data)[y_offset]);
      output.float_data[i] = read_condition(cond_offset) ? x_value : y_value;
    }
  }

  return output;
}

template <typename IntCompare, typename FloatCompare>
Tensor RunCompareAccelerate(const std::string& op_type, const Node& node, const Tensor& lhs, const Tensor& rhs,
                            ExecutionContext& context, IntCompare int_compare, FloatCompare float_compare) {
  const auto output_shape = ComputeBroadcastShape(lhs.shape, rhs.shape, op_type);
  const auto output_strides = ComputeStrides(output_shape);
  const auto lhs_strides = ComputeStrides(lhs.shape);
  const auto rhs_strides = ComputeStrides(rhs.shape);
  const auto element_count = GetElementCount(output_shape);
  auto output = MakeInt64Output(node.outputs.at(0), output_shape, context);

  if (lhs.dtype == "int64" && rhs.dtype == "int64") {
    const auto& lhs_data = RequireInt64Data(lhs, op_type);
    const auto& rhs_data = RequireInt64Data(rhs, op_type);
    for (std::size_t i = 0; i < element_count; ++i) {
      const auto output_index = UnravelIndex(i, output_shape, output_strides);
      const auto lhs_offset = ComputeBroadcastOffset(output_index, lhs.shape, lhs_strides);
      const auto rhs_offset = ComputeBroadcastOffset(output_index, rhs.shape, rhs_strides);
      output.int64_data[i] = int_compare(lhs_data[lhs_offset], rhs_data[rhs_offset]) ? 1 : 0;
    }
    return output;
  }

  const auto* lhs_float_data = lhs.dtype == "float32" ? &RequireFloatData(lhs, op_type) : nullptr;
  const auto* lhs_int_data = lhs.dtype == "int64" ? &RequireInt64Data(lhs, op_type) : nullptr;
  const auto* rhs_float_data = rhs.dtype == "float32" ? &RequireFloatData(rhs, op_type) : nullptr;
  const auto* rhs_int_data = rhs.dtype == "int64" ? &RequireInt64Data(rhs, op_type) : nullptr;
  for (std::size_t i = 0; i < element_count; ++i) {
    const auto output_index = UnravelIndex(i, output_shape, output_strides);
    const auto lhs_offset = ComputeBroadcastOffset(output_index, lhs.shape, lhs_strides);
    const auto rhs_offset = ComputeBroadcastOffset(output_index, rhs.shape, rhs_strides);
    const auto lhs_value =
        lhs_float_data != nullptr ? (*lhs_float_data)[lhs_offset] : static_cast<float>((*lhs_int_data)[lhs_offset]);
    const auto rhs_value =
        rhs_float_data != nullptr ? (*rhs_float_data)[rhs_offset] : static_cast<float>((*rhs_int_data)[rhs_offset]);
    output.int64_data[i] = float_compare(lhs_value, rhs_value) ? 1 : 0;
  }
  return output;
}

Tensor RunReduceMeanAccelerate(const Node& node, const Tensor& input, ExecutionContext& context) {
  const auto& input_data = RequireFloatData(input, "ReduceMean");
  std::vector<std::int64_t> axes;
  if (node.inputs.size() > 1 && !node.inputs.at(1).empty()) {
    axes = ReadVectorAsInt64(RequireTensor(context, node.inputs.at(1)), "ReduceMean");
  } else {
    axes = ReadIntsAttribute(node, "axes", {});
  }
  if (axes.empty()) {
    axes.resize(input.shape.size());
    for (std::size_t i = 0; i < axes.size(); ++i) {
      axes[i] = static_cast<std::int64_t>(i);
    }
  }
  const auto keepdims = ReadIntAttribute(node, "keepdims", 1);
  const auto normalized_axes = NormalizeAxes(axes, input.shape.size());
  std::vector<bool> is_reduced_axis(input.shape.size(), false);
  for (const auto axis : normalized_axes) {
    is_reduced_axis[static_cast<std::size_t>(axis)] = true;
  }

  std::vector<std::int64_t> output_shape;
  output_shape.reserve(input.shape.size());
  for (std::size_t i = 0; i < input.shape.size(); ++i) {
    if (is_reduced_axis[i]) {
      if (keepdims != 0) {
        output_shape.push_back(1);
      }
    } else {
      output_shape.push_back(input.shape[i]);
    }
  }

  auto output = MakeFloatOutput(node.outputs.at(0), output_shape, context);
  std::fill(output.float_data.begin(), output.float_data.end(), 0.0f);
  std::vector<std::int64_t> counts(output.float_data.size(), 0);
  const auto input_strides = ComputeStrides(input.shape);
  const auto output_strides = ComputeStrides(output_shape);

  for (std::size_t i = 0; i < input_data.size(); ++i) {
    const auto input_index = UnravelIndex(i, input.shape, input_strides);
    std::vector<std::int64_t> output_index;
    output_index.reserve(output_shape.size());
    for (std::size_t dim = 0; dim < input.shape.size(); ++dim) {
      if (is_reduced_axis[dim]) {
        if (keepdims != 0) {
          output_index.push_back(0);
        }
      } else {
        output_index.push_back(input_index[dim]);
      }
    }
    const auto output_offset = output_index.empty() ? 0 : ComputeOffset(output_index, output_strides);
    output.float_data[output_offset] += input_data[i];
    ++counts[output_offset];
  }

  for (std::size_t i = 0; i < output.float_data.size(); ++i) {
    if (counts[i] == 0) {
      throw std::runtime_error("ReduceMean encountered empty reduction bucket");
    }
    output.float_data[i] /= static_cast<float>(counts[i]);
  }
  return output;
}

Tensor RunTriluAccelerate(const Node& node, const Tensor& input, ExecutionContext& context) {
  if (input.shape.size() < 2) {
    throw std::runtime_error("Trilu requires rank >= 2");
  }
  const auto upper = ReadIntAttribute(node, "upper", 1) != 0;
  std::int64_t k = 0;
  if (node.inputs.size() > 1 && !node.inputs.at(1).empty()) {
    k = ReadScalarAsInt64(RequireTensor(context, node.inputs.at(1)), "Trilu");
  }

  const auto rows = static_cast<std::size_t>(input.shape[input.shape.size() - 2]);
  const auto cols = static_cast<std::size_t>(input.shape[input.shape.size() - 1]);
  const auto matrix_size = rows * cols;
  std::size_t batch = 1;
  for (std::size_t i = 0; i + 2 < input.shape.size(); ++i) {
    batch *= static_cast<std::size_t>(input.shape[i]);
  }

  if (input.dtype == "float32") {
    const auto& input_data = RequireFloatData(input, "Trilu");
    auto output = MakeFloatOutput(node.outputs.at(0), input.shape, context);
    std::fill(output.float_data.begin(), output.float_data.end(), 0.0f);
    for (std::size_t b = 0; b < batch; ++b) {
      const auto base = b * matrix_size;
      for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
          const auto row = static_cast<std::int64_t>(i);
          const auto col = static_cast<std::int64_t>(j);
          const bool keep = upper ? (col - row >= k) : (col - row <= k);
          if (keep) {
            const auto offset = base + i * cols + j;
            output.float_data[offset] = input_data[offset];
          }
        }
      }
    }
    return output;
  }

  if (input.dtype == "int64") {
    const auto& input_data = RequireInt64Data(input, "Trilu");
    auto output = MakeInt64Output(node.outputs.at(0), input.shape, context);
    std::fill(output.int64_data.begin(), output.int64_data.end(), 0);
    for (std::size_t b = 0; b < batch; ++b) {
      const auto base = b * matrix_size;
      for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
          const auto row = static_cast<std::int64_t>(i);
          const auto col = static_cast<std::int64_t>(j);
          const bool keep = upper ? (col - row >= k) : (col - row <= k);
          if (keep) {
            const auto offset = base + i * cols + j;
            output.int64_data[offset] = input_data[offset];
          }
        }
      }
    }
    return output;
  }

  throw std::runtime_error("Trilu currently supports float32/int64 only");
}

Tensor RunSoftmaxAccelerate(const Node& node, const Tensor& input, ExecutionContext& context) {
  const auto& input_data = RequireFloatData(input, "Softmax");
  const auto axis = static_cast<std::size_t>(NormalizeAxis(ReadIntAttribute(node, "axis", 1), input.shape.size(),
                                                           "Softmax"));
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
  if (axis == input.shape.size() - 1) {
    const int axis_count = static_cast<int>(axis_dim);
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
      vvexpf(exp_values.data(), shifted.data(), &axis_count);
      float denom_sum = 0.0f;
      for (std::size_t i = 0; i < axis_dim; ++i) {
        denom_sum += exp_values[i];
      }
      const float inv_sum = 1.0f / denom_sum;
      vDSP_vsmul(exp_values.data(), 1, &inv_sum, out_row, 1, axis_dim);
    }
    return output;
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

  return output;
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

Tensor RunConv2D(const Node& node, const Tensor& input, const Tensor& weight, const Tensor* bias,
                 ExecutionContext& context) {
  const auto& input_data = RequireFloatData(input, "Conv");
  const auto& weight_data = RequireFloatData(weight, "Conv");
  const std::vector<float>* bias_data = nullptr;
  if (bias != nullptr) {
    bias_data = &RequireFloatData(*bias, "Conv");
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

  for (std::size_t batch = 0; batch < params.n; ++batch) {
    const auto* batch_input = input_data.data() + batch * params.c_in * input_hw;
    auto* batch_output = output.float_data.data() + batch * params.c_out * output_hw;

    FillIm2ColBuffer(batch_input, params, columns);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(params.c_out), static_cast<int>(output_hw), static_cast<int>(kernel_dim),
                1.0f, weight_data.data(), static_cast<int>(kernel_dim),
                columns.data(), static_cast<int>(output_hw),
                0.0f, batch_output, static_cast<int>(output_hw));

    if (bias_data != nullptr) {
      for (std::size_t oc = 0; oc < params.c_out; ++oc) {
        float bias_value = (*bias_data)[oc];
        vDSP_vsadd(batch_output + oc * output_hw, 1, &bias_value, batch_output + oc * output_hw, 1, output_hw);
      }
    }
  }

  return output;
}

template <typename IntOp, typename FloatOp>
void RunBinaryNumericFallback(const std::string& op_type, const Node& node, ExecutionContext& context,
                              std::ostream* trace, IntOp eval_int, FloatOp eval_float) {
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
    context.BindTensor(std::move(output));
  } else {
    const auto* lhs_float_data = lhs.dtype == "float32" ? &RequireFloatData(lhs, op_type) : nullptr;
    const auto* lhs_int_data = lhs.dtype == "int64" ? &RequireInt64Data(lhs, op_type) : nullptr;
    const auto* rhs_float_data = rhs.dtype == "float32" ? &RequireFloatData(rhs, op_type) : nullptr;
    const auto* rhs_int_data = rhs.dtype == "int64" ? &RequireInt64Data(rhs, op_type) : nullptr;
    auto output = MakeFloatOutput(node.outputs.at(0), output_shape, context);
    for (std::size_t i = 0; i < element_count; ++i) {
      const auto output_index = UnravelIndex(i, output_shape, output_strides);
      const auto lhs_offset = ComputeBroadcastOffset(output_index, lhs.shape, lhs_strides);
      const auto rhs_offset = ComputeBroadcastOffset(output_index, rhs.shape, rhs_strides);
      const auto lhs_value = lhs_float_data != nullptr ? (*lhs_float_data)[lhs_offset]
                                                       : static_cast<float>((*lhs_int_data)[lhs_offset]);
      const auto rhs_value = rhs_float_data != nullptr ? (*rhs_float_data)[rhs_offset]
                                                       : static_cast<float>((*rhs_int_data)[rhs_offset]);
      output.float_data[i] = eval_float(lhs_value, rhs_value);
    }
    context.BindTensor(std::move(output));
  }

  if (trace != nullptr) {
    *trace << "    kernel " << op_type << " produced " << node.outputs.at(0) << " via Accelerate fallback\n";
  }
}

void RegisterAccelerateElementwiseKernels(KernelRegistry& registry) {
  registry.Register("Sigmoid", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& input = RequireTensor(context, node.inputs.at(0));
    const auto& input_data = RequireFloatData(input, "Sigmoid");
    auto output = MakeOutputLikeWithReusedStorage(node.outputs.at(0), input, context);

    const int element_count = static_cast<int>(input_data.size());
    std::vector<float> negated(input_data.size());
    std::vector<float> exp_values(input_data.size());
    std::vector<float> denom(input_data.size());
    std::vector<float> ones(input_data.size(), 1.0f);
    float minus_one = -1.0f;

    vDSP_vsmul(input_data.data(), 1, &minus_one, negated.data(), 1, input_data.size());
    vvexpf(exp_values.data(), negated.data(), &element_count);
    vDSP_vadd(ones.data(), 1, exp_values.data(), 1, denom.data(), 1, input_data.size());
    vvrecf(output.float_data.data(), denom.data(), &element_count);

    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Sigmoid produced " << node.outputs.at(0) << " via Accelerate\n";
    }
  });

  registry.Register("SiLU", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& input = RequireTensor(context, node.inputs.at(0));
    (void)RequireFloatData(input, "SiLU");
    auto output = MakeCopiedTensorWithReusedStorage(node.outputs.at(0), input, input.shape, context);
    ApplySiLUInPlaceAccelerate(output);

    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel SiLU produced " << node.outputs.at(0) << " via Accelerate\n";
    }
  });

  registry.Register("ConvSiLU", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& input = RequireTensor(context, node.inputs.at(0));
    const auto& weight = RequireTensor(context, node.inputs.at(1));
    const Tensor* bias = nullptr;
    if (node.inputs.size() > 2 && !node.inputs.at(2).empty()) {
      bias = &RequireTensor(context, node.inputs.at(2));
    }

    auto output = RunConv2D(node, input, weight, bias, context);
    ApplySiLUInPlaceAccelerate(output);
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel ConvSiLU produced " << node.outputs.at(0) << " via Accelerate\n";
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
      *trace << "    kernel Conv produced " << node.outputs.at(0) << " via Accelerate\n";
    }
  });

  registry.Register("Add", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& lhs = RequireTensor(context, node.inputs.at(0));
    const auto& rhs = RequireTensor(context, node.inputs.at(1));
    const auto output_shape = ComputeBroadcastShape(lhs.shape, rhs.shape, "Add");

    if (lhs.dtype == "float32" && rhs.dtype == "float32" && lhs.shape == rhs.shape && lhs.shape == output_shape) {
      const auto& lhs_data = RequireFloatData(lhs, "Add");
      const auto& rhs_data = RequireFloatData(rhs, "Add");
      auto output = MakeFloatOutput(node.outputs.at(0), output_shape, context);
      vDSP_vadd(lhs_data.data(), 1, rhs_data.data(), 1, output.float_data.data(), 1, lhs_data.size());
      context.BindTensor(std::move(output));
      if (trace != nullptr) {
        *trace << "    kernel Add produced " << node.outputs.at(0) << " via Accelerate\n";
      }
      return;
    }

    if (lhs.dtype == "float32" && rhs.dtype == "float32") {
      auto output = MakeFloatOutput(node.outputs.at(0), output_shape, context);
      ApplyFloatBroadcastOp(output, lhs, rhs, [](float a, float b) { return a + b; });
      context.BindTensor(std::move(output));
      if (trace != nullptr) {
        *trace << "    kernel Add produced " << node.outputs.at(0) << " via Accelerate\n";
      }
      return;
    }

    RunBinaryNumericFallback(
        "Add", node, context, trace, [](std::int64_t lhs, std::int64_t rhs) { return lhs + rhs; },
        [](float lhs, float rhs) { return lhs + rhs; });
  });

  registry.Register("Mul", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& lhs = RequireTensor(context, node.inputs.at(0));
    const auto& rhs = RequireTensor(context, node.inputs.at(1));
    const auto output_shape = ComputeBroadcastShape(lhs.shape, rhs.shape, "Mul");

    if (lhs.dtype == "float32" && rhs.dtype == "float32" && lhs.shape == rhs.shape && lhs.shape == output_shape) {
      const auto& lhs_data = RequireFloatData(lhs, "Mul");
      const auto& rhs_data = RequireFloatData(rhs, "Mul");
      auto output = MakeFloatOutput(node.outputs.at(0), output_shape, context);
      vDSP_vmul(lhs_data.data(), 1, rhs_data.data(), 1, output.float_data.data(), 1, lhs_data.size());
      context.BindTensor(std::move(output));
      if (trace != nullptr) {
        *trace << "    kernel Mul produced " << node.outputs.at(0) << " via Accelerate\n";
      }
      return;
    }

    if (lhs.dtype == "float32" && rhs.dtype == "float32") {
      auto output = MakeFloatOutput(node.outputs.at(0), output_shape, context);
      ApplyFloatBroadcastOp(output, lhs, rhs, [](float a, float b) { return a * b; });
      context.BindTensor(std::move(output));
      if (trace != nullptr) {
        *trace << "    kernel Mul produced " << node.outputs.at(0) << " via Accelerate\n";
      }
      return;
    }

    RunBinaryNumericFallback(
        "Mul", node, context, trace, [](std::int64_t lhs, std::int64_t rhs) { return lhs * rhs; },
        [](float lhs, float rhs) { return lhs * rhs; });
  });

  registry.Register("Sub", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& lhs = RequireTensor(context, node.inputs.at(0));
    const auto& rhs = RequireTensor(context, node.inputs.at(1));
    const auto output_shape = ComputeBroadcastShape(lhs.shape, rhs.shape, "Sub");

    if (lhs.dtype == "float32" && rhs.dtype == "float32" && lhs.shape == rhs.shape && lhs.shape == output_shape) {
      const auto& lhs_data = RequireFloatData(lhs, "Sub");
      const auto& rhs_data = RequireFloatData(rhs, "Sub");
      auto output = MakeFloatOutput(node.outputs.at(0), output_shape, context);
      vDSP_vsub(rhs_data.data(), 1, lhs_data.data(), 1, output.float_data.data(), 1, lhs_data.size());
      context.BindTensor(std::move(output));
      if (trace != nullptr) {
        *trace << "    kernel Sub produced " << node.outputs.at(0) << " via Accelerate\n";
      }
      return;
    }

    if (lhs.dtype == "float32" && rhs.dtype == "float32") {
      auto output = MakeFloatOutput(node.outputs.at(0), output_shape, context);
      ApplyFloatBroadcastOp(output, lhs, rhs, [](float a, float b) { return a - b; });
      context.BindTensor(std::move(output));
      if (trace != nullptr) {
        *trace << "    kernel Sub produced " << node.outputs.at(0) << " via Accelerate\n";
      }
      return;
    }

    RunBinaryNumericFallback(
        "Sub", node, context, trace, [](std::int64_t lhs, std::int64_t rhs) { return lhs - rhs; },
        [](float lhs, float rhs) { return lhs - rhs; });
  });

  registry.Register("Div", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& lhs = RequireTensor(context, node.inputs.at(0));
    const auto& rhs = RequireTensor(context, node.inputs.at(1));
    const auto output_shape = ComputeBroadcastShape(lhs.shape, rhs.shape, "Div");

    if (lhs.dtype == "float32" && rhs.dtype == "float32" && lhs.shape == rhs.shape && lhs.shape == output_shape) {
      const auto& lhs_data = RequireFloatData(lhs, "Div");
      const auto& rhs_data = RequireFloatData(rhs, "Div");
      auto output = MakeFloatOutput(node.outputs.at(0), output_shape, context);
      vDSP_vdiv(rhs_data.data(), 1, lhs_data.data(), 1, output.float_data.data(), 1, lhs_data.size());
      context.BindTensor(std::move(output));
      if (trace != nullptr) {
        *trace << "    kernel Div produced " << node.outputs.at(0) << " via Accelerate\n";
      }
      return;
    }

    if (lhs.dtype == "float32" && rhs.dtype == "float32") {
      auto output = MakeFloatOutput(node.outputs.at(0), output_shape, context);
      ApplyFloatBroadcastOp(output, lhs, rhs, [](float a, float b) {
        if (b == 0.0f) {
          throw std::runtime_error("Div divisor must not be zero");
        }
        return a / b;
      });
      context.BindTensor(std::move(output));
      if (trace != nullptr) {
        *trace << "    kernel Div produced " << node.outputs.at(0) << " via Accelerate\n";
      }
      return;
    }

    RunBinaryNumericFallback(
        "Div", node, context, trace,
        [](std::int64_t lhs, std::int64_t rhs) {
          if (rhs == 0) {
            throw std::runtime_error("Div divisor must not be zero");
          }
          return lhs / rhs;
        },
        [](float lhs, float rhs) {
          if (rhs == 0.0f) {
            throw std::runtime_error("Div divisor must not be zero");
          }
          return lhs / rhs;
        });
  });

  registry.Register("Gather", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& data = RequireTensor(context, node.inputs.at(0));
    const auto& indices = RequireTensor(context, node.inputs.at(1));
    auto output = RunGatherAccelerate(node, data, indices, context);
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Gather produced " << node.outputs.at(0) << " via Accelerate\n";
    }
  });

  registry.Register("Transpose", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& input = RequireTensor(context, node.inputs.at(0));
    auto output = RunTransposeAccelerate(node, input, context);
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Transpose produced " << node.outputs.at(0) << " via Accelerate\n";
    }
  });

  registry.Register("MatMul", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& lhs = RequireTensor(context, node.inputs.at(0));
    const auto& rhs = RequireTensor(context, node.inputs.at(1));
    auto output = RunMatMulAccelerate(node, lhs, rhs, context);
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel MatMul produced " << node.outputs.at(0) << " via Accelerate\n";
    }
  });

  registry.Register("Tanh", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& input = RequireTensor(context, node.inputs.at(0));
    const auto& input_data = RequireFloatData(input, "Tanh");
    auto output = MakeOutputLikeWithReusedStorage(node.outputs.at(0), input, context);
    const int element_count = static_cast<int>(input_data.size());
    vvtanhf(output.float_data.data(), input_data.data(), &element_count);
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Tanh produced " << node.outputs.at(0) << " via Accelerate\n";
    }
  });

  registry.Register("Neg", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& input = RequireTensor(context, node.inputs.at(0));
    if (input.dtype == "float32") {
      const auto& input_data = RequireFloatData(input, "Neg");
      auto output = MakeOutputLikeWithReusedStorage(node.outputs.at(0), input, context);
      float minus_one = -1.0f;
      vDSP_vsmul(input_data.data(), 1, &minus_one, output.float_data.data(), 1, input_data.size());
      context.BindTensor(std::move(output));
    } else if (input.dtype == "int64") {
      const auto& input_data = RequireInt64Data(input, "Neg");
      auto output = MakeOutputLikeWithReusedStorage(node.outputs.at(0), input, context);
      std::transform(input_data.begin(), input_data.end(), output.int64_data.begin(),
                     [](std::int64_t value) { return -value; });
      context.BindTensor(std::move(output));
    } else {
      throw std::runtime_error("Neg currently supports float32/int64 only");
    }
    if (trace != nullptr) {
      *trace << "    kernel Neg produced " << node.outputs.at(0) << " via Accelerate\n";
    }
  });

  registry.Register("Sqrt", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& input = RequireTensor(context, node.inputs.at(0));
    const auto& input_data = RequireFloatData(input, "Sqrt");
    auto output = MakeOutputLikeWithReusedStorage(node.outputs.at(0), input, context);
    const int element_count = static_cast<int>(input_data.size());
    vvsqrtf(output.float_data.data(), input_data.data(), &element_count);
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Sqrt produced " << node.outputs.at(0) << " via Accelerate\n";
    }
  });

  registry.Register("Pow", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& lhs = RequireTensor(context, node.inputs.at(0));
    const auto& rhs = RequireTensor(context, node.inputs.at(1));
    const auto output_shape = ComputeBroadcastShape(lhs.shape, rhs.shape, "Pow");
    auto output = MakeFloatOutput(node.outputs.at(0), output_shape, context);

    if (lhs.dtype != "float32" || rhs.dtype != "float32") {
      RunBinaryNumericFallback(
          "Pow", node, context, trace, [](std::int64_t lhs_value, std::int64_t rhs_value) {
            return static_cast<std::int64_t>(
                std::pow(static_cast<double>(lhs_value), static_cast<double>(rhs_value)));
          },
          [](float lhs_value, float rhs_value) { return std::pow(lhs_value, rhs_value); });
      return;
    }

    ApplyFloatBroadcastOp(output, lhs, rhs, [](float a, float b) { return std::pow(a, b); });
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Pow produced " << node.outputs.at(0) << " via Accelerate\n";
    }
  });

  registry.Register("Equal", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& lhs = RequireTensor(context, node.inputs.at(0));
    const auto& rhs = RequireTensor(context, node.inputs.at(1));
    auto output = RunCompareAccelerate("Equal", node, lhs, rhs, context,
                                       [](std::int64_t a, std::int64_t b) { return a == b; },
                                       [](float a, float b) { return a == b; });
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Equal produced " << node.outputs.at(0) << " via Accelerate\n";
    }
  });

  registry.Register("Less", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& lhs = RequireTensor(context, node.inputs.at(0));
    const auto& rhs = RequireTensor(context, node.inputs.at(1));
    auto output = RunCompareAccelerate("Less", node, lhs, rhs, context,
                                       [](std::int64_t a, std::int64_t b) { return a < b; },
                                       [](float a, float b) { return a < b; });
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Less produced " << node.outputs.at(0) << " via Accelerate\n";
    }
  });

  registry.Register("Where", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& condition = RequireTensor(context, node.inputs.at(0));
    const auto& x = RequireTensor(context, node.inputs.at(1));
    const auto& y = RequireTensor(context, node.inputs.at(2));
    auto output = RunWhereAccelerate(node, condition, x, y, context);
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Where produced " << node.outputs.at(0) << " via Accelerate\n";
    }
  });

  registry.Register("Cast", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& input = RequireTensor(context, node.inputs.at(0));
    const auto to_it = node.attributes.find("to");
    if (to_it == node.attributes.end()) {
      throw std::runtime_error("Cast missing to attribute");
    }

    const auto to_type = to_it->second.int_value;
    if (to_type == 1) {
      auto output = MakeFloatOutput(node.outputs.at(0), input.shape, context);
      if (input.dtype == "float32") {
        const auto& input_data = RequireFloatData(input, "Cast");
        std::copy(input_data.begin(), input_data.end(), output.float_data.begin());
      } else if (input.dtype == "int64") {
        const auto& input_data = RequireInt64Data(input, "Cast");
        for (std::size_t i = 0; i < input_data.size(); ++i) {
          output.float_data[i] = static_cast<float>(input_data[i]);
        }
      } else {
        throw std::runtime_error("Cast to float32 currently supports int64/float32 only");
      }
      context.BindTensor(std::move(output));
    } else if (to_type == 7 || to_type == 6) {
      auto output = MakeInt64Output(node.outputs.at(0), input.shape, context);
      if (input.dtype == "int64") {
        const auto& input_data = RequireInt64Data(input, "Cast");
        std::copy(input_data.begin(), input_data.end(), output.int64_data.begin());
      } else if (input.dtype == "float32") {
        const auto& input_data = RequireFloatData(input, "Cast");
        for (std::size_t i = 0; i < input_data.size(); ++i) {
          output.int64_data[i] = static_cast<std::int64_t>(input_data[i]);
        }
      } else {
        throw std::runtime_error("Cast to int64 currently supports int64/float32 only");
      }
      context.BindTensor(std::move(output));
    } else if (to_type == 9) {
      auto output = MakeInt64Output(node.outputs.at(0), input.shape, context);
      if (input.dtype == "int64") {
        const auto& input_data = RequireInt64Data(input, "Cast");
        for (std::size_t i = 0; i < input_data.size(); ++i) {
          output.int64_data[i] = input_data[i] != 0 ? 1 : 0;
        }
      } else if (input.dtype == "float32") {
        const auto& input_data = RequireFloatData(input, "Cast");
        for (std::size_t i = 0; i < input_data.size(); ++i) {
          output.int64_data[i] = input_data[i] != 0.0f ? 1 : 0;
        }
      } else {
        throw std::runtime_error("Cast to bool currently supports int64/float32 only");
      }
      context.BindTensor(std::move(output));
    } else {
      throw std::runtime_error("Cast currently supports only float32/int32/int64/bool outputs");
    }
    if (trace != nullptr) {
      *trace << "    kernel Cast produced " << node.outputs.at(0) << " via Accelerate\n";
    }
  });

  registry.Register("Softmax", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& input = RequireTensor(context, node.inputs.at(0));
    auto output = RunSoftmaxAccelerate(node, input, context);
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Softmax produced " << node.outputs.at(0) << " via Accelerate\n";
    }
  });

  registry.Register("ReduceMean", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& input = RequireTensor(context, node.inputs.at(0));
    auto output = RunReduceMeanAccelerate(node, input, context);
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel ReduceMean produced " << node.outputs.at(0) << " via Accelerate\n";
    }
  });

  registry.Register("Trilu", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& input = RequireTensor(context, node.inputs.at(0));
    auto output = RunTriluAccelerate(node, input, context);
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Trilu produced " << node.outputs.at(0) << " via Accelerate\n";
    }
  });

  registry.Register("Gemm", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& a = RequireTensor(context, node.inputs.at(0));
    const auto& b = RequireTensor(context, node.inputs.at(1));
    const Tensor* c = nullptr;
    if (node.inputs.size() > 2 && !node.inputs.at(2).empty()) {
      c = &RequireTensor(context, node.inputs.at(2));
    }

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
    cblas_sgemm(CblasRowMajor,
                trans_a ? CblasTrans : CblasNoTrans,
                trans_b ? CblasTrans : CblasNoTrans,
                static_cast<int>(m), static_cast<int>(n), static_cast<int>(k_a),
                alpha,
                a_data.data(), static_cast<int>(a_cols),
                b_data.data(), static_cast<int>(b_cols),
                0.0f,
                output.float_data.data(), static_cast<int>(n));

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

    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Gemm produced " << node.outputs.at(0) << " via Accelerate\n";
    }
  });

  registry.Register("LayerNormalization", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& input = RequireTensor(context, node.inputs.at(0));
    const auto& scale = RequireTensor(context, node.inputs.at(1));
    const auto& bias = RequireTensor(context, node.inputs.at(2));
    auto output = RunLayerNormalizationAccelerate(node, input, scale, bias, context);
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel LayerNormalization produced " << node.outputs.at(0) << " via Accelerate\n";
    }
  });
}

}  // namespace

std::string_view AccelerateExecutionProvider::Name() const {
  return "Accelerate";
}

void AccelerateExecutionProvider::RegisterKernels(KernelRegistry& registry) const {
  RegisterAccelerateElementwiseKernels(registry);
}

std::shared_ptr<TensorAllocator> AccelerateExecutionProvider::CreateTensorAllocator() const {
  return std::make_shared<CpuTensorAllocator>();
}

bool IsAccelerateAvailable() {
  return true;
}

}  // namespace miniort
