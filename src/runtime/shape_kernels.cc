#include "builtin_kernel_groups.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "kernel_utils.h"

namespace miniort {

void RegisterShapeKernels(KernelRegistry& registry) {
  registry.Register("ConstantOfShape", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& shape_tensor = RequireTensor(context, node.inputs.at(0));
    const auto shape = ReadVectorAsInt64(shape_tensor, "ConstantOfShape");

    Tensor output;
    output.name = node.outputs.at(0);
    output.shape = shape;
    output.is_placeholder = false;

    float fill_value = 0.0f;
    const auto value_it = node.attributes.find("value");
    if (value_it != node.attributes.end() && value_it->second.tensor.has_value()) {
      const auto& value = *value_it->second.tensor;
      if (value.dtype == "float32" && !value.float_data.empty()) {
        fill_value = value.float_data.front();
      } else if (value.dtype == "int64" && !value.int64_data.empty()) {
        fill_value = static_cast<float>(value.int64_data.front());
      }
    }

    output.dtype = "float32";
    output.float_data.assign(GetElementCount(output.shape), fill_value);
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel ConstantOfShape produced " << node.outputs.at(0) << "\n";
    }
  });

  registry.Register("Shape", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& input = RequireTensor(context, node.inputs.at(0));
    Tensor output;
    output.name = node.outputs.at(0);
    output.dtype = "int64";
    output.shape = {static_cast<std::int64_t>(input.shape.size())};
    output.int64_data = input.shape;
    output.is_placeholder = false;
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Shape produced " << node.outputs.at(0) << "\n";
    }
  });

  registry.Register("Gather", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& data = RequireTensor(context, node.inputs.at(0));
    const auto& indices = RequireTensor(context, node.inputs.at(1));
    const auto axis_it = node.attributes.find("axis");
    const auto axis = axis_it == node.attributes.end() ? 0 : axis_it->second.int_value;

    if (axis != 0 || data.shape.size() > 1) {
      throw std::runtime_error("Gather currently only supports axis=0 on scalar/1D tensors");
    }

    Tensor output;
    output.name = node.outputs.at(0);
    output.dtype = data.dtype;
    output.shape = indices.shape.empty() ? std::vector<std::int64_t>{} : indices.shape;
    output.is_placeholder = false;

    const auto& index_data = RequireInt64Data(indices, "Gather");
    if (data.dtype == "int64") {
      const auto& data_values = RequireInt64Data(data, "Gather");
      output.int64_data.reserve(index_data.size());
      for (auto index : index_data) {
        if (data.shape.empty() && index != 0) {
          throw std::runtime_error("Gather scalar data only supports index 0");
        }
        output.int64_data.push_back(data_values.at(static_cast<std::size_t>(index)));
      }
    } else if (data.dtype == "float32") {
      const auto& data_values = RequireFloatData(data, "Gather");
      output.float_data.reserve(index_data.size());
      for (auto index : index_data) {
        if (data.shape.empty() && index != 0) {
          throw std::runtime_error("Gather scalar data only supports index 0");
        }
        output.float_data.push_back(data_values.at(static_cast<std::size_t>(index)));
      }
    } else {
      throw std::runtime_error("Gather currently supports float32/int64 data only");
    }

    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Gather produced " << node.outputs.at(0) << "\n";
    }
  });

  registry.Register("Unsqueeze", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& data = RequireTensor(context, node.inputs.at(0));
    std::vector<std::int64_t> axes;
    if (node.inputs.size() > 1 && !node.inputs.at(1).empty()) {
      axes = RequireInt64Data(RequireTensor(context, node.inputs.at(1)), "Unsqueeze");
    } else {
      const auto axes_it = node.attributes.find("axes");
      if (axes_it == node.attributes.end()) {
        throw std::runtime_error("Unsqueeze missing axes");
      }
      axes = axes_it->second.ints;
    }

    const auto rank = data.shape.size() + axes.size();
    auto normalized = NormalizeAxes(axes, rank);
    auto output = data;
    output.name = node.outputs.at(0);
    output.shape.clear();
    output.shape.reserve(rank);
    std::size_t data_index = 0;
    std::size_t axis_index = 0;
    for (std::size_t i = 0; i < rank; ++i) {
      if (axis_index < normalized.size() && normalized[axis_index] == static_cast<std::int64_t>(i)) {
        output.shape.push_back(1);
        ++axis_index;
      } else {
        output.shape.push_back(data.shape[data_index++]);
      }
    }
    output.is_placeholder = false;
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Unsqueeze produced " << node.outputs.at(0) << "\n";
    }
  });

  registry.Register("Concat", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    std::vector<Tensor> inputs;
    inputs.reserve(node.inputs.size());
    for (const auto& input_name : node.inputs) {
      if (input_name.empty()) {
        continue;
      }
      inputs.push_back(RequireTensor(context, input_name));
    }
    if (inputs.empty()) {
      throw std::runtime_error("Concat requires at least one input");
    }

    Tensor output;
    if (inputs.front().dtype == "float32") {
      output = ConcatTensors<float>(node, trace, "float32", inputs,
                                    [](const Tensor& tensor) -> const std::vector<float>& {
                                      return RequireFloatData(tensor, "Concat");
                                    });
    } else if (inputs.front().dtype == "int64") {
      output = ConcatTensors<std::int64_t>(node, trace, "int64", inputs,
                                           [](const Tensor& tensor) -> const std::vector<std::int64_t>& {
                                             return RequireInt64Data(tensor, "Concat");
                                           });
    } else {
      throw std::runtime_error("Concat currently supports float32/int64 only");
    }
    context.BindTensor(std::move(output));
  });

  registry.Register("Reshape", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& data = RequireTensor(context, node.inputs.at(0));
    const auto& shape_tensor = RequireTensor(context, node.inputs.at(1));
    Tensor output = data;
    output.name = node.outputs.at(0);
    output.shape = ResolveReshapeDims(data, shape_tensor);
    output.is_placeholder = false;
    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Reshape produced " << node.outputs.at(0) << "\n";
    }
  });

  registry.Register("Range", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& start_tensor = RequireTensor(context, node.inputs.at(0));
    const auto& limit_tensor = RequireTensor(context, node.inputs.at(1));
    const auto& delta_tensor = RequireTensor(context, node.inputs.at(2));

    Tensor output;
    output.name = node.outputs.at(0);
    output.is_placeholder = false;

    if (start_tensor.dtype == "float32" || limit_tensor.dtype == "float32" || delta_tensor.dtype == "float32") {
      const auto start = ReadScalarAsFloat32(start_tensor, "Range");
      const auto limit = ReadScalarAsFloat32(limit_tensor, "Range");
      const auto delta = ReadScalarAsFloat32(delta_tensor, "Range");
      if (delta == 0.0f) {
        throw std::runtime_error("Range delta must not be zero");
      }
      output.dtype = "float32";
      for (float value = start; (delta > 0.0f) ? (value < limit) : (value > limit); value += delta) {
        output.float_data.push_back(value);
      }
      output.shape = {static_cast<std::int64_t>(output.float_data.size())};
    } else if (start_tensor.dtype == "int64" && limit_tensor.dtype == "int64" && delta_tensor.dtype == "int64") {
      const auto start = ReadScalarAsInt64(start_tensor, "Range");
      const auto limit = ReadScalarAsInt64(limit_tensor, "Range");
      const auto delta = ReadScalarAsInt64(delta_tensor, "Range");
      if (delta == 0) {
        throw std::runtime_error("Range delta must not be zero");
      }
      output.dtype = "int64";
      for (std::int64_t value = start; (delta > 0) ? (value < limit) : (value > limit); value += delta) {
        output.int64_data.push_back(value);
      }
      output.shape = {static_cast<std::int64_t>(output.int64_data.size())};
    }

    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Range produced " << node.outputs.at(0) << "\n";
    }
  });

  registry.Register("Split", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& data = RequireTensor(context, node.inputs.at(0));
    const auto& split_sizes = RequireTensor(context, node.inputs.at(1));
    const auto axis_it = node.attributes.find("axis");
    auto axis = axis_it == node.attributes.end() ? 0 : axis_it->second.int_value;
    const auto rank = static_cast<std::int64_t>(data.shape.size());
    if (axis < 0) {
      axis += rank;
    }
    if (axis < 0 || axis >= rank) {
      throw std::runtime_error("Split axis is out of range");
    }

    const auto& splits = RequireInt64Data(split_sizes, "Split");
    if (splits.size() != node.outputs.size()) {
      throw std::runtime_error("Split sizes/output count mismatch");
    }

    std::size_t outer = 1;
    for (std::size_t i = 0; i < static_cast<std::size_t>(axis); ++i) {
      outer *= static_cast<std::size_t>(data.shape[i]);
    }
    std::size_t inner = 1;
    for (std::size_t i = static_cast<std::size_t>(axis) + 1; i < data.shape.size(); ++i) {
      inner *= static_cast<std::size_t>(data.shape[i]);
    }

    const auto emit_split = [&](auto storage, const auto& input_data) {
      std::size_t axis_offset = 0;
      for (std::size_t output_index = 0; output_index < node.outputs.size(); ++output_index) {
        const auto split_dim = static_cast<std::size_t>(splits[output_index]);
        Tensor output;
        output.name = node.outputs[output_index];
        output.dtype = data.dtype;
        output.shape = data.shape;
        output.shape[static_cast<std::size_t>(axis)] = static_cast<std::int64_t>(split_dim);
        output.is_placeholder = false;
        storage(output).resize(GetElementCount(output.shape));

        std::size_t output_offset = 0;
        for (std::size_t outer_index = 0; outer_index < outer; ++outer_index) {
          const auto input_base =
              (outer_index * static_cast<std::size_t>(data.shape[static_cast<std::size_t>(axis)]) + axis_offset) * inner;
          const auto copy_count = split_dim * inner;
          std::copy_n(input_data.begin() + static_cast<std::ptrdiff_t>(input_base),
                      static_cast<std::ptrdiff_t>(copy_count),
                      storage(output).begin() + static_cast<std::ptrdiff_t>(output_offset));
          output_offset += copy_count;
        }
        axis_offset += split_dim;
        context.BindTensor(std::move(output));
        if (trace != nullptr) {
          *trace << "    kernel Split produced " << node.outputs[output_index] << "\n";
        }
      }
    };

    if (data.dtype == "float32") {
      emit_split([](Tensor& tensor) -> std::vector<float>& { return tensor.float_data; },
                 RequireFloatData(data, "Split"));
    } else if (data.dtype == "int64") {
      emit_split([](Tensor& tensor) -> std::vector<std::int64_t>& { return tensor.int64_data; },
                 RequireInt64Data(data, "Split"));
    } else {
      throw std::runtime_error("Split currently supports float32/int64 only");
    }
  });

  registry.Register("Expand", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& data = RequireTensor(context, node.inputs.at(0));
    const auto& shape_tensor = RequireTensor(context, node.inputs.at(1));
    const auto output_shape = ReadVectorAsInt64(shape_tensor, "Expand");
    const auto output_strides = ComputeStrides(output_shape);
    const auto input_strides = ComputeStrides(data.shape);

    if (output_shape.size() < data.shape.size()) {
      throw std::runtime_error("Expand output rank must be >= input rank");
    }
    for (std::size_t i = 0; i < data.shape.size(); ++i) {
      const auto input_dim = data.shape[data.shape.size() - 1 - i];
      const auto output_dim = output_shape[output_shape.size() - 1 - i];
      if (input_dim != output_dim && input_dim != 1) {
        throw std::runtime_error("Expand shape is incompatible with input");
      }
    }

    Tensor output;
    output.name = node.outputs.at(0);
    output.dtype = data.dtype;
    output.shape = output_shape;
    output.is_placeholder = false;

    if (data.dtype == "float32") {
      const auto& input_data = RequireFloatData(data, "Expand");
      output.float_data.resize(GetElementCount(output_shape));
      for (std::size_t i = 0; i < output.float_data.size(); ++i) {
        const auto output_index = UnravelIndex(i, output_shape, output_strides);
        const auto input_offset = ComputeBroadcastOffset(output_index, data.shape, input_strides);
        output.float_data[i] = input_data[input_offset];
      }
    } else if (data.dtype == "int64") {
      const auto& input_data = RequireInt64Data(data, "Expand");
      output.int64_data.resize(GetElementCount(output_shape));
      for (std::size_t i = 0; i < output.int64_data.size(); ++i) {
        const auto output_index = UnravelIndex(i, output_shape, output_strides);
        const auto input_offset = ComputeBroadcastOffset(output_index, data.shape, input_strides);
        output.int64_data[i] = input_data[input_offset];
      }
    } else {
      throw std::runtime_error("Expand currently supports float32/int64 only");
    }

    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Expand produced " << node.outputs.at(0) << "\n";
    }
  });

  registry.Register("Transpose", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& input = RequireTensor(context, node.inputs.at(0));
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

    const auto input_strides = ComputeStrides(input.shape);
    const auto output_strides = ComputeStrides(output.shape);
    if (input.dtype == "float32") {
      const auto& input_data = RequireFloatData(input, "Transpose");
      output.float_data.resize(GetElementCount(output.shape));
      for (std::size_t i = 0; i < output.float_data.size(); ++i) {
        const auto output_index = UnravelIndex(i, output.shape, output_strides);
        std::vector<std::int64_t> input_index(input.shape.size(), 0);
        for (std::size_t j = 0; j < perm.size(); ++j) {
          input_index[static_cast<std::size_t>(perm[j])] = output_index[j];
        }
        output.float_data[i] = input_data[ComputeOffset(input_index, input_strides)];
      }
    } else if (input.dtype == "int64") {
      const auto& input_data = RequireInt64Data(input, "Transpose");
      output.int64_data.resize(GetElementCount(output.shape));
      for (std::size_t i = 0; i < output.int64_data.size(); ++i) {
        const auto output_index = UnravelIndex(i, output.shape, output_strides);
        std::vector<std::int64_t> input_index(input.shape.size(), 0);
        for (std::size_t j = 0; j < perm.size(); ++j) {
          input_index[static_cast<std::size_t>(perm[j])] = output_index[j];
        }
        output.int64_data[i] = input_data[ComputeOffset(input_index, input_strides)];
      }
    } else {
      throw std::runtime_error("Transpose currently supports float32/int64 only");
    }

    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Transpose produced " << node.outputs.at(0) << "\n";
    }
  });

  registry.Register("Slice", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& data = RequireTensor(context, node.inputs.at(0));
    const auto& starts_tensor = RequireTensor(context, node.inputs.at(1));
    const auto& ends_tensor = RequireTensor(context, node.inputs.at(2));
    const auto starts = ReadVectorAsInt64(starts_tensor, "Slice");
    const auto ends = ReadVectorAsInt64(ends_tensor, "Slice");
    if (starts.size() != ends.size()) {
      throw std::runtime_error("Slice starts/ends size mismatch");
    }

    std::vector<std::int64_t> axes(starts.size());
    if (node.inputs.size() > 3 && !node.inputs.at(3).empty()) {
      axes = ReadVectorAsInt64(RequireTensor(context, node.inputs.at(3)), "Slice");
    } else {
      for (std::size_t i = 0; i < axes.size(); ++i) {
        axes[i] = static_cast<std::int64_t>(i);
      }
    }
    std::vector<std::int64_t> steps(starts.size(), 1);
    if (node.inputs.size() > 4 && !node.inputs.at(4).empty()) {
      steps = ReadVectorAsInt64(RequireTensor(context, node.inputs.at(4)), "Slice");
    }

    std::vector<std::int64_t> slice_starts(data.shape.size(), 0);
    std::vector<std::int64_t> slice_steps(data.shape.size(), 1);
    std::vector<std::int64_t> output_shape = data.shape;
    for (std::size_t i = 0; i < axes.size(); ++i) {
      const auto axis = static_cast<std::size_t>(NormalizeAxis(axes[i], data.shape.size(), "Slice"));
      const auto dim = data.shape[axis];
      const auto step = steps[i];
      if (step == 0) {
        throw std::runtime_error("Slice step must not be zero");
      }

      auto start = starts[i];
      auto end = ends[i];
      if (start < 0) {
        start += dim;
      }
      if (end < 0) {
        end += dim;
      }

      if (step > 0) {
        start = std::clamp(start, static_cast<std::int64_t>(0), dim);
        end = std::clamp(end, static_cast<std::int64_t>(0), dim);
        output_shape[axis] = start >= end ? 0 : ((end - start - 1) / step + 1);
      } else {
        start = std::clamp(start, static_cast<std::int64_t>(-1), dim - 1);
        end = std::clamp(end, static_cast<std::int64_t>(-1), dim - 1);
        output_shape[axis] = start <= end ? 0 : ((start - end - 1) / (-step) + 1);
      }

      slice_starts[axis] = start;
      slice_steps[axis] = step;
    }

    const auto input_strides = ComputeStrides(data.shape);
    const auto output_strides = ComputeStrides(output_shape);

    Tensor output;
    output.name = node.outputs.at(0);
    output.dtype = data.dtype;
    output.shape = output_shape;
    output.is_placeholder = false;

    const auto emit_slice = [&](auto& output_data, const auto& input_data) {
      output_data.resize(GetElementCount(output_shape));
      for (std::size_t i = 0; i < output_data.size(); ++i) {
        const auto output_index = UnravelIndex(i, output_shape, output_strides);
        auto input_index = output_index;
        for (std::size_t axis = 0; axis < input_index.size(); ++axis) {
          input_index[axis] = slice_starts[axis] + output_index[axis] * slice_steps[axis];
        }
        output_data[i] = input_data[ComputeOffset(input_index, input_strides)];
      }
    };

    if (data.dtype == "float32") {
      emit_slice(output.float_data, RequireFloatData(data, "Slice"));
    } else if (data.dtype == "int64") {
      emit_slice(output.int64_data, RequireInt64Data(data, "Slice"));
    } else {
      throw std::runtime_error("Slice currently supports float32/int64 only");
    }

    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Slice produced " << node.outputs.at(0) << "\n";
    }
  });

  registry.Register("ReduceMax", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& input = RequireTensor(context, node.inputs.at(0));
    const auto& input_data = RequireFloatData(input, "ReduceMax");
    const auto axes = NormalizeAxes(ReadIntsAttribute(node, "axes", {0}), input.shape.size());
    const auto keepdims = ReadIntAttribute(node, "keepdims", 1);
    if (axes.size() != 1) {
      throw std::runtime_error("ReduceMax currently supports a single axis");
    }

    const auto axis = static_cast<std::size_t>(axes.front());
    std::vector<std::int64_t> output_shape = input.shape;
    if (keepdims != 0) {
      output_shape[axis] = 1;
    } else {
      output_shape.erase(output_shape.begin() + static_cast<std::ptrdiff_t>(axis));
    }

    const auto input_strides = ComputeStrides(input.shape);
    const auto output_strides = ComputeStrides(output_shape);
    Tensor output;
    output.name = node.outputs.at(0);
    output.dtype = "float32";
    output.shape = output_shape;
    output.is_placeholder = false;
    output.float_data.resize(GetElementCount(output_shape));

    for (std::size_t i = 0; i < output.float_data.size(); ++i) {
      auto output_index = UnravelIndex(i, output_shape, output_strides);
      std::vector<std::int64_t> base_index = keepdims != 0 ? output_index : std::vector<std::int64_t>{};
      if (keepdims == 0) {
        base_index.reserve(input.shape.size());
        for (std::size_t dim = 0, j = 0; dim < input.shape.size(); ++dim) {
          if (dim == axis) {
            base_index.push_back(0);
          } else {
            base_index.push_back(output_index[j++]);
          }
        }
      }

      float best = -std::numeric_limits<float>::infinity();
      for (std::int64_t k = 0; k < input.shape[axis]; ++k) {
        base_index[axis] = k;
        best = std::max(best, input_data[ComputeOffset(base_index, input_strides)]);
      }
      output.float_data[i] = best;
    }

    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel ReduceMax produced " << node.outputs.at(0) << "\n";
    }
  });

  registry.Register("ArgMax", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& input = RequireTensor(context, node.inputs.at(0));
    const auto& input_data = RequireFloatData(input, "ArgMax");
    const auto axis = static_cast<std::size_t>(
        NormalizeAxis(ReadIntAttribute(node, "axis", 0), input.shape.size(), "ArgMax"));
    const auto keepdims = ReadIntAttribute(node, "keepdims", 1);

    std::vector<std::int64_t> output_shape = input.shape;
    if (keepdims != 0) {
      output_shape[axis] = 1;
    } else {
      output_shape.erase(output_shape.begin() + static_cast<std::ptrdiff_t>(axis));
    }

    const auto input_strides = ComputeStrides(input.shape);
    const auto output_strides = ComputeStrides(output_shape);
    Tensor output;
    output.name = node.outputs.at(0);
    output.dtype = "int64";
    output.shape = output_shape;
    output.is_placeholder = false;
    output.int64_data.resize(GetElementCount(output_shape));

    for (std::size_t i = 0; i < output.int64_data.size(); ++i) {
      auto output_index = UnravelIndex(i, output_shape, output_strides);
      std::vector<std::int64_t> base_index = keepdims != 0 ? output_index : std::vector<std::int64_t>{};
      if (keepdims == 0) {
        base_index.reserve(input.shape.size());
        for (std::size_t dim = 0, j = 0; dim < input.shape.size(); ++dim) {
          if (dim == axis) {
            base_index.push_back(0);
          } else {
            base_index.push_back(output_index[j++]);
          }
        }
      }

      std::int64_t best_index = 0;
      float best_value = -std::numeric_limits<float>::infinity();
      for (std::int64_t k = 0; k < input.shape[axis]; ++k) {
        base_index[axis] = k;
        const auto value = input_data[ComputeOffset(base_index, input_strides)];
        if (k == 0 || value > best_value) {
          best_value = value;
          best_index = k;
        }
      }
      output.int64_data[i] = best_index;
    }

    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel ArgMax produced " << node.outputs.at(0) << "\n";
    }
  });
}

}  // namespace miniort
