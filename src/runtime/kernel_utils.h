#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "miniort/runtime/execution_context.h"
#include "miniort/runtime/tensor.h"

namespace miniort {

inline const Tensor& RequireTensor(const ExecutionContext& context, const std::string& name) {
  const auto* tensor = context.FindTensor(name);
  if (tensor == nullptr) {
    throw std::runtime_error("missing input tensor: " + name);
  }
  return *tensor;
}

inline Tensor MakeOutputLike(const std::string& name, const Tensor& source) {
  Tensor tensor;
  tensor.name = name;
  tensor.dtype = source.dtype;
  tensor.shape = source.shape;
  tensor.is_placeholder = false;
  return tensor;
}

inline const std::vector<float>& RequireFloatData(const Tensor& tensor, const std::string& op_type);
inline const std::vector<std::int64_t>& RequireInt64Data(const Tensor& tensor, const std::string& op_type);

inline Tensor MakeFloatOutput(const std::string& name, const std::vector<std::int64_t>& shape,
                              ExecutionContext& context) {
  Tensor tensor;
  tensor.name = name;
  tensor.dtype = "float32";
  tensor.shape = shape;
  tensor.is_placeholder = false;
  tensor.float_data = context.AcquireFloatBuffer(GetElementCount(shape));
  tensor.float_data.resize(GetElementCount(shape));
  return tensor;
}

inline Tensor MakeInt64Output(const std::string& name, const std::vector<std::int64_t>& shape,
                              ExecutionContext& context) {
  Tensor tensor;
  tensor.name = name;
  tensor.dtype = "int64";
  tensor.shape = shape;
  tensor.is_placeholder = false;
  tensor.int64_data = context.AcquireInt64Buffer(GetElementCount(shape));
  tensor.int64_data.resize(GetElementCount(shape));
  return tensor;
}

inline Tensor MakeOutputLikeWithReusedStorage(const std::string& name, const Tensor& source,
                                              ExecutionContext& context) {
  Tensor tensor;
  tensor.name = name;
  tensor.dtype = source.dtype;
  tensor.shape = source.shape;
  tensor.is_placeholder = false;
  const auto element_count = GetElementCount(tensor.shape);
  if (tensor.dtype == "float32") {
    tensor.float_data = context.AcquireFloatBuffer(element_count);
    tensor.float_data.resize(element_count);
  } else if (tensor.dtype == "int64") {
    tensor.int64_data = context.AcquireInt64Buffer(element_count);
    tensor.int64_data.resize(element_count);
  }
  return tensor;
}

inline Tensor MakeTensorWithReusedStorage(const std::string& name, const std::string& dtype,
                                         const std::vector<std::int64_t>& shape, ExecutionContext& context) {
  Tensor tensor;
  tensor.name = name;
  tensor.dtype = dtype;
  tensor.shape = shape;
  tensor.is_placeholder = false;
  const auto element_count = GetElementCount(shape);
  if (dtype == "float32") {
    tensor.float_data = context.AcquireFloatBuffer(element_count);
    tensor.float_data.resize(element_count);
  } else if (dtype == "int64") {
    tensor.int64_data = context.AcquireInt64Buffer(element_count);
    tensor.int64_data.resize(element_count);
  }
  return tensor;
}

inline Tensor MakeCopiedTensorWithReusedStorage(const std::string& name, const Tensor& source,
                                                const std::vector<std::int64_t>& shape, ExecutionContext& context) {
  Tensor tensor = MakeTensorWithReusedStorage(name, source.dtype, shape, context);
  if (source.dtype == "float32") {
    const auto& input = RequireFloatData(source, "copy");
    if (tensor.float_data.size() != input.size()) {
      throw std::runtime_error("copied float tensor size mismatch: " + name);
    }
    std::copy(input.begin(), input.end(), tensor.float_data.begin());
  } else if (source.dtype == "int64") {
    const auto& input = RequireInt64Data(source, "copy");
    if (tensor.int64_data.size() != input.size()) {
      throw std::runtime_error("copied int64 tensor size mismatch: " + name);
    }
    std::copy(input.begin(), input.end(), tensor.int64_data.begin());
  }
  return tensor;
}

inline Tensor MakeTensorFromDataWithReusedStorage(const std::string& name, const TensorData& source,
                                                 ExecutionContext& context) {
  Tensor tensor;
  tensor.name = name;
  tensor.dtype = source.dtype.empty() ? "unknown" : source.dtype;
  tensor.shape = source.shape;
  tensor.is_placeholder = false;
  const auto element_count = GetElementCount(source.shape);

  if (tensor.dtype == "int32") {
    tensor.dtype = "int64";
    if (!source.int32_data.empty()) {
      tensor.int64_data = context.AcquireInt64Buffer(source.int32_data.size());
      tensor.int64_data.resize(source.int32_data.size());
      std::transform(source.int32_data.begin(), source.int32_data.end(), tensor.int64_data.begin(),
                     [](std::int32_t value) { return static_cast<std::int64_t>(value); });
      return tensor;
    }
    if (!source.raw_data.empty() && source.raw_data.size() % sizeof(std::int32_t) == 0) {
      const auto count = source.raw_data.size() / sizeof(std::int32_t);
      tensor.int64_data = context.AcquireInt64Buffer(count);
      tensor.int64_data.resize(count);
      for (std::size_t i = 0; i < count; ++i) {
        std::int32_t value = 0;
        std::memcpy(&value, source.raw_data.data() + i * sizeof(std::int32_t), sizeof(std::int32_t));
        tensor.int64_data[i] = static_cast<std::int64_t>(value);
      }
      return tensor;
    }
    if (element_count == 0) {
      tensor.int64_data.clear();
      return tensor;
    }
  }

  if (source.dtype == "float32" && !source.float_data.empty()) {
    tensor.float_data = context.AcquireFloatBuffer(source.float_data.size());
    tensor.float_data.resize(source.float_data.size());
    std::copy(source.float_data.begin(), source.float_data.end(), tensor.float_data.begin());
  } else if (source.dtype == "int64" && !source.int64_data.empty()) {
    tensor.int64_data = context.AcquireInt64Buffer(source.int64_data.size());
    tensor.int64_data.resize(source.int64_data.size());
    std::copy(source.int64_data.begin(), source.int64_data.end(), tensor.int64_data.begin());
  } else if (source.dtype == "int32" && !source.int32_data.empty()) {
    tensor.int64_data = context.AcquireInt64Buffer(source.int32_data.size());
    tensor.int64_data.resize(source.int32_data.size());
    std::transform(source.int32_data.begin(), source.int32_data.end(), tensor.int64_data.begin(),
                   [](std::int32_t value) { return static_cast<std::int64_t>(value); });
  } else if ((source.dtype == "float32" || source.dtype == "int64" || source.dtype == "int32") &&
             element_count == 0) {
    if (source.dtype == "int32") {
      tensor.dtype = "int64";
    }
  } else {
    tensor.is_placeholder = true;
  }
  return tensor;
}

inline std::vector<std::int64_t> ReadIntsAttribute(const Node& node, const std::string& name,
                                                   std::vector<std::int64_t> default_value) {
  const auto it = node.attributes.find(name);
  if (it == node.attributes.end()) {
    return default_value;
  }
  return it->second.ints.empty() ? default_value : it->second.ints;
}

inline std::int64_t ReadIntAttribute(const Node& node, const std::string& name, std::int64_t default_value) {
  const auto it = node.attributes.find(name);
  return it == node.attributes.end() ? default_value : it->second.int_value;
}

inline void EnsureSameShape(const Tensor& lhs, const Tensor& rhs, const std::string& op_type) {
  if (lhs.shape != rhs.shape) {
    throw std::runtime_error(op_type + " currently only supports identical shapes");
  }
}

inline const std::vector<float>& RequireFloatData(const Tensor& tensor, const std::string& op_type) {
  if (tensor.dtype != "float32") {
    throw std::runtime_error(op_type + " requires float32 tensor data: " + tensor.name);
  }
  if (tensor.float_data.empty() && GetElementCount(tensor.shape) != 0) {
    throw std::runtime_error(op_type + " requires float32 tensor data: " + tensor.name);
  }
  return tensor.float_data;
}

inline const std::vector<std::int64_t>& RequireInt64Data(const Tensor& tensor, const std::string& op_type) {
  if (tensor.dtype != "int64") {
    throw std::runtime_error(op_type + " requires int64 tensor data: " + tensor.name);
  }
  if (tensor.int64_data.empty() && GetElementCount(tensor.shape) != 0) {
    throw std::runtime_error(op_type + " requires int64 tensor data: " + tensor.name);
  }
  return tensor.int64_data;
}

inline std::vector<std::int64_t> NormalizeAxes(const std::vector<std::int64_t>& axes, std::size_t rank) {
  std::vector<std::int64_t> normalized;
  normalized.reserve(axes.size());
  for (auto axis : axes) {
    if (axis < 0) {
      axis += static_cast<std::int64_t>(rank);
    }
    normalized.push_back(axis);
  }
  std::sort(normalized.begin(), normalized.end());
  return normalized;
}

inline std::int64_t NormalizeAxis(std::int64_t axis, std::size_t rank, const std::string& op_type) {
  if (axis < 0) {
    axis += static_cast<std::int64_t>(rank);
  }
  if (axis < 0 || axis >= static_cast<std::int64_t>(rank)) {
    throw std::runtime_error(op_type + " axis is out of range");
  }
  return axis;
}

inline std::vector<std::size_t> ComputeStrides(const std::vector<std::int64_t>& shape) {
  std::vector<std::size_t> strides(shape.size(), 1);
  std::size_t running = 1;
  for (std::size_t i = shape.size(); i > 0; --i) {
    strides[i - 1] = running;
    running *= static_cast<std::size_t>(shape[i - 1]);
  }
  return strides;
}

inline std::vector<std::int64_t> UnravelIndex(std::size_t flat_index, const std::vector<std::int64_t>& shape,
                                              const std::vector<std::size_t>& strides) {
  std::vector<std::int64_t> index(shape.size(), 0);
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == 0) {
      continue;
    }
    index[i] = static_cast<std::int64_t>(flat_index / strides[i]);
    flat_index %= strides[i];
  }
  return index;
}

inline std::size_t ComputeOffset(const std::vector<std::int64_t>& index, const std::vector<std::size_t>& strides) {
  std::size_t offset = 0;
  for (std::size_t i = 0; i < index.size(); ++i) {
    offset += static_cast<std::size_t>(index[i]) * strides[i];
  }
  return offset;
}

inline std::vector<std::int64_t> ComputeBroadcastShape(const std::vector<std::int64_t>& lhs,
                                                       const std::vector<std::int64_t>& rhs,
                                                       const std::string& op_type) {
  const std::size_t rank = std::max(lhs.size(), rhs.size());
  std::vector<std::int64_t> shape(rank, 1);
  for (std::size_t i = 0; i < rank; ++i) {
    const auto lhs_dim = i < rank - lhs.size() ? 1 : lhs[i - (rank - lhs.size())];
    const auto rhs_dim = i < rank - rhs.size() ? 1 : rhs[i - (rank - rhs.size())];
    if (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1) {
      throw std::runtime_error(op_type + " broadcast shape mismatch");
    }
    shape[i] = std::max(lhs_dim, rhs_dim);
  }
  return shape;
}

inline std::size_t ComputeBroadcastOffset(const std::vector<std::int64_t>& output_index,
                                          const std::vector<std::int64_t>& input_shape,
                                          const std::vector<std::size_t>& input_strides) {
  const std::size_t rank = output_index.size();
  const std::size_t input_rank = input_shape.size();
  const std::size_t offset = rank - input_rank;
  std::size_t flat_index = 0;
  for (std::size_t i = 0; i < input_rank; ++i) {
    const auto dim = input_shape[i];
    const auto idx = dim == 1 ? 0 : output_index[offset + i];
    flat_index += static_cast<std::size_t>(idx) * input_strides[i];
  }
  return flat_index;
}

inline std::vector<std::int64_t> ResolveReshapeDims(const Tensor& data, const Tensor& shape_tensor) {
  const auto& requested_shape = RequireInt64Data(shape_tensor, "Reshape");
  std::vector<std::int64_t> output_shape;
  output_shape.reserve(requested_shape.size());

  const std::size_t input_count = GetElementCount(data.shape);
  std::int64_t infer_index = -1;
  std::size_t known_product = 1;

  for (std::size_t i = 0; i < requested_shape.size(); ++i) {
    const auto dim = requested_shape[i];
    if (dim == 0) {
      if (i >= data.shape.size()) {
        throw std::runtime_error("Reshape zero-copy dim is out of range");
      }
      output_shape.push_back(data.shape[i]);
      known_product *= static_cast<std::size_t>(data.shape[i]);
    } else if (dim == -1) {
      if (infer_index != -1) {
        throw std::runtime_error("Reshape currently supports at most one inferred dimension");
      }
      infer_index = static_cast<std::int64_t>(i);
      output_shape.push_back(-1);
    } else {
      output_shape.push_back(dim);
      known_product *= static_cast<std::size_t>(dim);
    }
  }

  if (infer_index != -1) {
    if (known_product == 0 || input_count % known_product != 0) {
      throw std::runtime_error("Reshape cannot infer output dimension");
    }
    output_shape[static_cast<std::size_t>(infer_index)] = static_cast<std::int64_t>(input_count / known_product);
  }

  if (GetElementCount(output_shape) != input_count) {
    throw std::runtime_error("Reshape element count mismatch");
  }

  return output_shape;
}

inline std::int64_t ReadScalarInt64(const Tensor& tensor, const std::string& op_type) {
  const auto& data = RequireInt64Data(tensor, op_type);
  if (data.size() != 1) {
    throw std::runtime_error(op_type + " requires scalar int64 input: " + tensor.name);
  }
  return data.front();
}

inline float ReadScalarFloat32(const Tensor& tensor, const std::string& op_type) {
  const auto& data = RequireFloatData(tensor, op_type);
  if (data.size() != 1) {
    throw std::runtime_error(op_type + " requires scalar float32 input: " + tensor.name);
  }
  return data.front();
}

inline float ReadScalarAsFloat32(const Tensor& tensor, const std::string& op_type) {
  if (tensor.dtype == "float32") {
    return ReadScalarFloat32(tensor, op_type);
  }
  if (tensor.dtype == "int64") {
    return static_cast<float>(ReadScalarInt64(tensor, op_type));
  }
  throw std::runtime_error(op_type + " currently supports scalar int64/float32 only: " + tensor.name);
}

inline std::int64_t ReadScalarAsInt64(const Tensor& tensor, const std::string& op_type) {
  if (tensor.dtype == "int64") {
    return ReadScalarInt64(tensor, op_type);
  }
  if (tensor.dtype == "float32") {
    return static_cast<std::int64_t>(ReadScalarFloat32(tensor, op_type));
  }
  throw std::runtime_error(op_type + " currently supports scalar int64/float32 only: " + tensor.name);
}

inline std::vector<std::int64_t> ReadVectorAsInt64(const Tensor& tensor, const std::string& op_type) {
  if (tensor.dtype == "int64") {
    if (tensor.int64_data.empty() && GetElementCount(tensor.shape) == 0) {
      return {};
    }
    return RequireInt64Data(tensor, op_type);
  }
  if (tensor.dtype == "float32") {
    if (tensor.float_data.empty() && GetElementCount(tensor.shape) == 0) {
      return {};
    }
    const auto& data = RequireFloatData(tensor, op_type);
    std::vector<std::int64_t> converted;
    converted.reserve(data.size());
    for (const auto value : data) {
      converted.push_back(static_cast<std::int64_t>(value));
    }
    return converted;
  }
  throw std::runtime_error(op_type + " currently supports int64/float32 vectors only: " + tensor.name);
}

template <typename T>
inline Tensor ConcatTensors(const Node& node, ExecutionContext* context, std::ostream* trace, const std::string& dtype,
                            const std::vector<const Tensor*>& inputs,
                            const std::function<const std::vector<T>&(const Tensor&)>& get_data) {
  const auto axis_it = node.attributes.find("axis");
  if (axis_it == node.attributes.end()) {
    throw std::runtime_error("Concat missing axis attribute");
  }

  if (inputs.empty()) {
    throw std::runtime_error("Concat requires at least one input");
  }

  auto axis = axis_it->second.int_value;
  const auto rank = static_cast<std::int64_t>(inputs.front()->shape.size());
  if (axis < 0) {
    axis += rank;
  }
  if (axis < 0 || axis >= rank) {
    throw std::runtime_error("Concat axis is out of range");
  }

  std::vector<std::int64_t> output_shape = inputs.front()->shape;
  output_shape[static_cast<std::size_t>(axis)] = 0;

  for (const auto& input : inputs) {
    if (input->dtype != dtype) {
      throw std::runtime_error("Concat input dtype mismatch");
    }
    if (input->shape.size() != output_shape.size()) {
      throw std::runtime_error("Concat rank mismatch");
    }
    for (std::size_t i = 0; i < input->shape.size(); ++i) {
      if (i == static_cast<std::size_t>(axis)) {
        output_shape[i] += input->shape[i];
      } else if (input->shape[i] < 0 || output_shape[i] < 0) {
        output_shape[i] = std::max(output_shape[i], input->shape[i]);
      } else if (input->shape[i] != output_shape[i]) {
        throw std::runtime_error("Concat currently requires matching non-axis dimensions");
      }
    }
  }

  std::size_t outer = 1;
  for (std::size_t i = 0; i < static_cast<std::size_t>(axis); ++i) {
    outer *= static_cast<std::size_t>(output_shape[i]);
  }
  std::size_t inner = 1;
  for (std::size_t i = static_cast<std::size_t>(axis) + 1; i < output_shape.size(); ++i) {
    inner *= static_cast<std::size_t>(output_shape[i]);
  }

  Tensor output;
  output.name = node.outputs.front();
  output.dtype = dtype;
  output.shape = output_shape;
  output.is_placeholder = false;

  if (context != nullptr) {
    output = MakeTensorWithReusedStorage(output.name, dtype, output_shape, *context);
  } else {
    if constexpr (std::is_same_v<T, float>) {
      output.float_data.resize(GetElementCount(output_shape));
    } else {
      output.int64_data.resize(GetElementCount(output_shape));
    }
  }

  std::size_t output_offset = 0;
  for (std::size_t outer_index = 0; outer_index < outer; ++outer_index) {
    for (const auto& input : inputs) {
      const auto axis_dim = static_cast<std::size_t>(input->shape[static_cast<std::size_t>(axis)]);
      const auto copy_count = axis_dim * inner;
      const auto& data = get_data(*input);
      const auto input_base = outer_index * axis_dim * inner;
      if constexpr (std::is_same_v<T, float>) {
        std::memcpy(output.float_data.data() + output_offset, data.data() + input_base,
                    copy_count * sizeof(float));
      } else {
        std::memcpy(output.int64_data.data() + output_offset, data.data() + input_base,
                    copy_count * sizeof(std::int64_t));
      }
      output_offset += copy_count;
    }
  }

  if (trace != nullptr) {
    *trace << "    kernel Concat produced " << output.name << "\n";
  }
  return output;
}

}  // namespace miniort
