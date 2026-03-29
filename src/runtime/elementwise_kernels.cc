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
    auto output = MakeOutputLike(node.outputs.at(0), input);
    output.float_data.resize(input_data.size());
    std::transform(input_data.begin(), input_data.end(), output.float_data.begin(),
                   [](float value) { return 1.0f / (1.0f + std::exp(-value)); });
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Sigmoid produced " << node.outputs.at(0) << "\n";
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
          output.name = node.outputs.at(0);
          output.shape = output_shape;
          output.is_placeholder = false;

          const auto element_count = GetElementCount(output_shape);
          if (lhs.dtype == "int64" && rhs.dtype == "int64") {
            const auto& lhs_data = RequireInt64Data(lhs, op_type);
            const auto& rhs_data = RequireInt64Data(rhs, op_type);
            output.dtype = "int64";
            output.int64_data.resize(element_count);
            for (std::size_t i = 0; i < element_count; ++i) {
              const auto output_index = UnravelIndex(i, output_shape, output_strides);
              const auto lhs_offset = ComputeBroadcastOffset(output_index, lhs.shape, lhs_strides);
              const auto rhs_offset = ComputeBroadcastOffset(output_index, rhs.shape, rhs_strides);
              output.int64_data[i] = eval_int(lhs_data[lhs_offset], rhs_data[rhs_offset]);
            }
          } else {
            output.dtype = "float32";
            output.float_data.resize(element_count);
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

  registry.Register("Cast", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& input = RequireTensor(context, node.inputs.at(0));
    const auto to_it = node.attributes.find("to");
    if (to_it == node.attributes.end()) {
      throw std::runtime_error("Cast missing to attribute");
    }

    Tensor output;
    output.name = node.outputs.at(0);
    output.shape = input.shape;
    output.is_placeholder = false;

    const auto to_type = to_it->second.int_value;
    if (to_type == 1) {
      output.dtype = "float32";
      output.float_data.reserve(GetElementCount(input.shape));
      if (input.dtype == "float32") {
        output.float_data = RequireFloatData(input, "Cast");
      } else if (input.dtype == "int64") {
        const auto& input_data = RequireInt64Data(input, "Cast");
        for (const auto value : input_data) {
          output.float_data.push_back(static_cast<float>(value));
        }
      } else {
        throw std::runtime_error("Cast to float32 currently supports int64/float32 only");
      }
    } else if (to_type == 7 || to_type == 6) {
      output.dtype = "int64";
      output.int64_data.reserve(GetElementCount(input.shape));
      if (input.dtype == "int64") {
        output.int64_data = RequireInt64Data(input, "Cast");
      } else if (input.dtype == "float32") {
        const auto& input_data = RequireFloatData(input, "Cast");
        for (const auto value : input_data) {
          output.int64_data.push_back(static_cast<std::int64_t>(value));
        }
      } else {
        throw std::runtime_error("Cast to int64 currently supports int64/float32 only");
      }
    } else {
      throw std::runtime_error("Cast currently supports only float32/int32/int64 outputs");
    }

    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Cast produced " << node.outputs.at(0) << "\n";
    }
  });
}

}  // namespace miniort
