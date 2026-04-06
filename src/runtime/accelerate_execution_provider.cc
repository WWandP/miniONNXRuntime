#include "miniort/runtime/accelerate_execution_provider.h"

#include <Accelerate/Accelerate.h>

#include <cstdint>
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
    auto output = MakeFloatOutput(node.outputs.at(0), output_shape, context);
    for (std::size_t i = 0; i < element_count; ++i) {
      const auto output_index = UnravelIndex(i, output_shape, output_strides);
      const auto lhs_offset = ComputeBroadcastOffset(output_index, lhs.shape, lhs_strides);
      const auto rhs_offset = ComputeBroadcastOffset(output_index, rhs.shape, rhs_strides);
      const auto lhs_value = lhs.dtype == "float32" ? RequireFloatData(lhs, op_type)[lhs_offset]
                                                    : static_cast<float>(RequireInt64Data(lhs, op_type)[lhs_offset]);
      const auto rhs_value = rhs.dtype == "float32" ? RequireFloatData(rhs, op_type)[rhs_offset]
                                                    : static_cast<float>(RequireInt64Data(rhs, op_type)[rhs_offset]);
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

  registry.Register("MatMul", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& lhs = RequireTensor(context, node.inputs.at(0));
    const auto& rhs = RequireTensor(context, node.inputs.at(1));
    const auto& lhs_data = RequireFloatData(lhs, "MatMul");
    const auto& rhs_data = RequireFloatData(rhs, "MatMul");
    if (lhs.shape.size() != 2 || rhs.shape.size() != 2) {
      throw std::runtime_error("MatMul currently only supports 2D float32 tensors");
    }

    const auto m = static_cast<std::size_t>(lhs.shape[0]);
    const auto k = static_cast<std::size_t>(lhs.shape[1]);
    const auto rhs_k = static_cast<std::size_t>(rhs.shape[0]);
    const auto n = static_cast<std::size_t>(rhs.shape[1]);
    if (k != rhs_k) {
      throw std::runtime_error("MatMul inner dimensions do not match");
    }

    auto output = MakeFloatOutput(node.outputs.at(0), {static_cast<std::int64_t>(m), static_cast<std::int64_t>(n)}, context);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                1.0f, lhs_data.data(), static_cast<int>(k),
                rhs_data.data(), static_cast<int>(n),
                0.0f, output.float_data.data(), static_cast<int>(n));
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel MatMul produced " << node.outputs.at(0) << " via Accelerate\n";
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
