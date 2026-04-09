#include "builtin_kernel_groups.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>

#include "kernel_utils.h"

namespace miniort {

void RegisterElementwiseKernels(KernelRegistry& registry) {
  registry.Register("Sigmoid", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& input = RequireTensor(context, node.inputs.at(0));
    const auto& input_data = RequireFloatData(input, "Sigmoid");
    auto output = MakeOutputLikeWithReusedStorage(node.outputs.at(0), input, context);
    std::transform(input_data.begin(), input_data.end(), output.float_data.begin(),
                   [](float value) { return 1.0f / (1.0f + std::exp(-value)); });
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Sigmoid produced " << node.outputs.at(0) << "\n";
    }
  });

  registry.Register("SiLU", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& input = RequireTensor(context, node.inputs.at(0));
    const auto& input_data = RequireFloatData(input, "SiLU");
    auto output = MakeOutputLikeWithReusedStorage(node.outputs.at(0), input, context);
    for (std::size_t i = 0; i < input_data.size(); ++i) {
      const auto value = input_data[i];
      output.float_data[i] = value * (1.0f / (1.0f + std::exp(-value)));
    }
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel SiLU produced " << node.outputs.at(0) << "\n";
    }
  });

  registry.Register("Tanh", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& input = RequireTensor(context, node.inputs.at(0));
    const auto& input_data = RequireFloatData(input, "Tanh");
    auto output = MakeOutputLikeWithReusedStorage(node.outputs.at(0), input, context);
    std::transform(input_data.begin(), input_data.end(), output.float_data.begin(),
                   [](float value) { return std::tanh(value); });
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Tanh produced " << node.outputs.at(0) << "\n";
    }
  });

  const auto register_binary_numeric_kernel =
      [&registry](const std::string& op_type, const std::function<float(float, float)>& eval_float,
                  const std::function<std::int64_t(std::int64_t, std::int64_t)>& eval_int) {
        registry.Register(op_type, [op_type, eval_float, eval_int](const Node& node, ExecutionContext& context,
                                                                   std::ostream* trace) {
          const auto& lhs = RequireTensor(context, node.inputs.at(0));
          const auto& rhs = RequireTensor(context, node.inputs.at(1));
          const auto output_shape = ComputeBroadcastShape(lhs.shape, rhs.shape, op_type);
          const auto output_strides = ComputeStrides(output_shape);
          const auto lhs_strides = ComputeStrides(lhs.shape);
          const auto rhs_strides = ComputeStrides(rhs.shape);

          Tensor output;
          const auto element_count = GetElementCount(output_shape);
          if (lhs.dtype == "int64" && rhs.dtype == "int64") {
            const auto& lhs_data = RequireInt64Data(lhs, op_type);
            const auto& rhs_data = RequireInt64Data(rhs, op_type);
            output = MakeInt64Output(node.outputs.at(0), output_shape, context);
            for (std::size_t i = 0; i < element_count; ++i) {
              const auto output_index = UnravelIndex(i, output_shape, output_strides);
              const auto lhs_offset = ComputeBroadcastOffset(output_index, lhs.shape, lhs_strides);
              const auto rhs_offset = ComputeBroadcastOffset(output_index, rhs.shape, rhs_strides);
              output.int64_data[i] = eval_int(lhs_data[lhs_offset], rhs_data[rhs_offset]);
            }
          } else {
            output = MakeFloatOutput(node.outputs.at(0), output_shape, context);
            for (std::size_t i = 0; i < element_count; ++i) {
              const auto output_index = UnravelIndex(i, output_shape, output_strides);
              const auto lhs_offset = ComputeBroadcastOffset(output_index, lhs.shape, lhs_strides);
              const auto rhs_offset = ComputeBroadcastOffset(output_index, rhs.shape, rhs_strides);
              const auto lhs_value =
                  lhs.dtype == "float32" ? RequireFloatData(lhs, op_type)[lhs_offset]
                                         : static_cast<float>(RequireInt64Data(lhs, op_type)[lhs_offset]);
              const auto rhs_value =
                  rhs.dtype == "float32" ? RequireFloatData(rhs, op_type)[rhs_offset]
                                         : static_cast<float>(RequireInt64Data(rhs, op_type)[rhs_offset]);
              output.float_data[i] = eval_float(lhs_value, rhs_value);
            }
          }

          context.BindTensor(std::move(output));
          if (trace != nullptr) {
            *trace << "    kernel " << op_type << " produced " << node.outputs.at(0) << "\n";
          }
        });
      };

  register_binary_numeric_kernel("Add", [](float lhs, float rhs) { return lhs + rhs; },
                                 [](std::int64_t lhs, std::int64_t rhs) { return lhs + rhs; });

  register_binary_numeric_kernel("Mul", [](float lhs, float rhs) { return lhs * rhs; },
                                 [](std::int64_t lhs, std::int64_t rhs) { return lhs * rhs; });

  register_binary_numeric_kernel(
      "Div",
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

  register_binary_numeric_kernel("Sub", [](float lhs, float rhs) { return lhs - rhs; },
                                 [](std::int64_t lhs, std::int64_t rhs) { return lhs - rhs; });

  registry.Register("Pow", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& lhs = RequireTensor(context, node.inputs.at(0));
    const auto& rhs = RequireTensor(context, node.inputs.at(1));
    const auto output_shape = ComputeBroadcastShape(lhs.shape, rhs.shape, "Pow");
    const auto output_strides = ComputeStrides(output_shape);
    const auto lhs_strides = ComputeStrides(lhs.shape);
    const auto rhs_strides = ComputeStrides(rhs.shape);
    const auto element_count = GetElementCount(output_shape);

    if (lhs.dtype == "int64" && rhs.dtype == "int64") {
      auto output = MakeInt64Output(node.outputs.at(0), output_shape, context);
      const auto& lhs_data = RequireInt64Data(lhs, "Pow");
      const auto& rhs_data = RequireInt64Data(rhs, "Pow");
      for (std::size_t i = 0; i < element_count; ++i) {
        const auto output_index = UnravelIndex(i, output_shape, output_strides);
        const auto lhs_offset = ComputeBroadcastOffset(output_index, lhs.shape, lhs_strides);
        const auto rhs_offset = ComputeBroadcastOffset(output_index, rhs.shape, rhs_strides);
        output.int64_data[i] =
            static_cast<std::int64_t>(std::pow(static_cast<double>(lhs_data[lhs_offset]),
                                               static_cast<double>(rhs_data[rhs_offset])));
      }
      context.BindTensor(std::move(output));
    } else {
      auto output = MakeFloatOutput(node.outputs.at(0), output_shape, context);
      for (std::size_t i = 0; i < element_count; ++i) {
        const auto output_index = UnravelIndex(i, output_shape, output_strides);
        const auto lhs_offset = ComputeBroadcastOffset(output_index, lhs.shape, lhs_strides);
        const auto rhs_offset = ComputeBroadcastOffset(output_index, rhs.shape, rhs_strides);
        const auto lhs_value =
            lhs.dtype == "float32" ? RequireFloatData(lhs, "Pow")[lhs_offset]
                                   : static_cast<float>(RequireInt64Data(lhs, "Pow")[lhs_offset]);
        const auto rhs_value =
            rhs.dtype == "float32" ? RequireFloatData(rhs, "Pow")[rhs_offset]
                                   : static_cast<float>(RequireInt64Data(rhs, "Pow")[rhs_offset]);
        output.float_data[i] = std::pow(lhs_value, rhs_value);
      }
      context.BindTensor(std::move(output));
    }

    if (trace != nullptr) {
      *trace << "    kernel Pow produced " << node.outputs.at(0) << "\n";
    }
  });

  registry.Register("Where", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& condition = RequireTensor(context, node.inputs.at(0));
    const auto& x = RequireTensor(context, node.inputs.at(1));
    const auto& y = RequireTensor(context, node.inputs.at(2));
    const auto output_shape = ComputeBroadcastShape(ComputeBroadcastShape(condition.shape, x.shape, "Where"),
                                                    y.shape, "Where");
    const auto output_strides = ComputeStrides(output_shape);
    const auto condition_strides = ComputeStrides(condition.shape);
    const auto x_strides = ComputeStrides(x.shape);
    const auto y_strides = ComputeStrides(y.shape);
    const auto element_count = GetElementCount(output_shape);

    const auto read_condition = [&](std::size_t offset) {
      if (condition.dtype == "int64") {
        return RequireInt64Data(condition, "Where")[offset] != 0;
      }
      if (condition.dtype == "float32") {
        return RequireFloatData(condition, "Where")[offset] != 0.0f;
      }
      throw std::runtime_error("Where condition currently supports int64/float32 only");
    };

    if (x.dtype == "int64" && y.dtype == "int64") {
      auto output = MakeInt64Output(node.outputs.at(0), output_shape, context);
      const auto& x_data = RequireInt64Data(x, "Where");
      const auto& y_data = RequireInt64Data(y, "Where");
      for (std::size_t i = 0; i < element_count; ++i) {
        const auto output_index = UnravelIndex(i, output_shape, output_strides);
        const auto cond_offset = ComputeBroadcastOffset(output_index, condition.shape, condition_strides);
        const auto x_offset = ComputeBroadcastOffset(output_index, x.shape, x_strides);
        const auto y_offset = ComputeBroadcastOffset(output_index, y.shape, y_strides);
        output.int64_data[i] = read_condition(cond_offset) ? x_data[x_offset] : y_data[y_offset];
      }
      context.BindTensor(std::move(output));
    } else {
      auto output = MakeFloatOutput(node.outputs.at(0), output_shape, context);
      for (std::size_t i = 0; i < element_count; ++i) {
        const auto output_index = UnravelIndex(i, output_shape, output_strides);
        const auto cond_offset = ComputeBroadcastOffset(output_index, condition.shape, condition_strides);
        const auto x_offset = ComputeBroadcastOffset(output_index, x.shape, x_strides);
        const auto y_offset = ComputeBroadcastOffset(output_index, y.shape, y_strides);
        const auto x_value = x.dtype == "float32" ? RequireFloatData(x, "Where")[x_offset]
                                                  : static_cast<float>(RequireInt64Data(x, "Where")[x_offset]);
        const auto y_value = y.dtype == "float32" ? RequireFloatData(y, "Where")[y_offset]
                                                  : static_cast<float>(RequireInt64Data(y, "Where")[y_offset]);
        output.float_data[i] = read_condition(cond_offset) ? x_value : y_value;
      }
      context.BindTensor(std::move(output));
    }

    if (trace != nullptr) {
      *trace << "    kernel Where produced " << node.outputs.at(0) << "\n";
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
        std::copy(RequireInt64Data(input, "Cast").begin(), RequireInt64Data(input, "Cast").end(),
                  output.int64_data.begin());
      } else if (input.dtype == "float32") {
        const auto& input_data = RequireFloatData(input, "Cast");
        for (std::size_t i = 0; i < input_data.size(); ++i) {
          output.int64_data[i] = static_cast<std::int64_t>(input_data[i]);
        }
      } else {
        throw std::runtime_error("Cast to int64 currently supports int64/float32 only");
      }
      context.BindTensor(std::move(output));
    } else {
      throw std::runtime_error("Cast currently supports only float32/int32/int64 outputs");
    }
    if (trace != nullptr) {
      *trace << "    kernel Cast produced " << node.outputs.at(0) << "\n";
    }
  });
}

}  // namespace miniort
