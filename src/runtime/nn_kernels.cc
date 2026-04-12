#include "builtin_kernel_groups.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

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

Tensor RunConv2D(const Node& node, const Tensor& input, const Tensor& weight, const Tensor* bias,
                 ExecutionContext& context) {
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

  for (std::size_t batch = 0; batch < n; ++batch) {
    const auto* batch_input = input_data.data() + batch * c_in * input_hw;
    auto* batch_output = output.float_data.data() + batch * c_out * output_hw;
    for (std::size_t oc = 0; oc < c_out; ++oc) {
      auto* output_plane = batch_output + oc * output_hw;
      const float bias_value = bias_data != nullptr ? (*bias_data)[oc] : 0.0f;
      std::fill_n(output_plane, output_hw, bias_value);

      const auto* weight_oc = weight_data.data() + oc * c_in * kernel_hw;
      for (std::size_t ic = 0; ic < c_in; ++ic) {
        const auto* input_plane = batch_input + ic * input_hw;
        const auto* weight_ic = weight_oc + ic * kernel_hw;

        for (std::size_t kh = 0; kh < k_h; ++kh) {
          const auto input_h_base = static_cast<std::int64_t>(kh) * dilation_h - pad_top;
          const auto oh_begin = input_h_base >= 0 ? 0 : static_cast<std::size_t>((-input_h_base + stride_h - 1) / stride_h);
          const auto oh_end = static_cast<std::size_t>(
              std::min<std::int64_t>(h_out, (static_cast<std::int64_t>(h_in) - 1 - input_h_base) / stride_h + 1));
          if (oh_begin >= oh_end) {
            continue;
          }

          for (std::size_t kw = 0; kw < k_w; ++kw) {
            const auto input_w_base = static_cast<std::int64_t>(kw) * dilation_w - pad_left;
            const auto ow_begin =
                input_w_base >= 0 ? 0 : static_cast<std::size_t>((-input_w_base + stride_w - 1) / stride_w);
            const auto ow_end = static_cast<std::size_t>(
                std::min<std::int64_t>(w_out, (static_cast<std::int64_t>(w_in) - 1 - input_w_base) / stride_w + 1));
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
    }
  }

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

    auto output = RunConv2D(node, input, weight, bias, context);
    for (auto& value : output.float_data) {
      value = value * (1.0f / (1.0f + std::exp(-value)));
    }

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
