#include "miniort/optimizer/graph_optimizer.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <functional>
#include <limits>
#include <onnx/onnx_pb.h>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "../runtime/kernel_utils.h"
#include "miniort/runtime/profiling.h"

namespace miniort {

namespace {

Tensor MakeRuntimeTensorFromTensorData(std::string name, const TensorData& data) {
  Tensor tensor;
  tensor.name = std::move(name);
  tensor.dtype = data.dtype;
  tensor.shape = data.shape;
  tensor.float_data = data.float_data;
  tensor.int64_data = data.int64_data;
  tensor.is_placeholder = false;
  tensor.is_initializer = true;
  return tensor;
}

TensorData MakeTensorDataFromRuntimeTensor(const Tensor& tensor) {
  TensorData data;
  data.dtype = tensor.dtype;
  data.shape = tensor.shape;
  data.float_data = tensor.float_data;
  data.int64_data = tensor.int64_data;
  return data;
}

TensorInfo MakeTensorInfoFromRuntimeTensor(const Tensor& tensor) {
  TensorInfo info;
  info.dtype = tensor.dtype;
  info.is_initializer = true;
  info.shape.reserve(tensor.shape.size());
  for (const auto dim : tensor.shape) {
    info.shape.push_back(std::to_string(dim));
  }
  return info;
}

Value MakeInitializerValueFromTensor(const Tensor& tensor) {
  Value value;
  value.name = tensor.name;
  value.info = MakeTensorInfoFromRuntimeTensor(tensor);
  value.data = MakeTensorDataFromRuntimeTensor(tensor);
  return value;
}

std::optional<Tensor> ResolveConstantTensor(const Graph& graph, const std::string& name) {
  if (name.empty()) {
    return std::nullopt;
  }
  const auto it = graph.initializers.find(name);
  if (it == graph.initializers.end() || !it->second.data.has_value()) {
    return std::nullopt;
  }
  return MakeRuntimeTensorFromTensorData(name, *it->second.data);
}

const std::vector<float>& RequireFloatDataAllowEmpty(const Tensor& tensor, const std::string& op_type) {
  if (tensor.dtype != "float32") {
    throw std::runtime_error(op_type + " requires float32 tensor data: " + tensor.name);
  }
  if (tensor.float_data.empty() && GetElementCount(tensor.shape) != 0) {
    throw std::runtime_error(op_type + " requires float32 tensor data: " + tensor.name);
  }
  return tensor.float_data;
}

const std::vector<std::int64_t>& RequireInt64DataAllowEmpty(const Tensor& tensor, const std::string& op_type) {
  if (tensor.dtype != "int64") {
    throw std::runtime_error(op_type + " requires int64 tensor data: " + tensor.name);
  }
  if (tensor.int64_data.empty() && GetElementCount(tensor.shape) != 0) {
    throw std::runtime_error(op_type + " requires int64 tensor data: " + tensor.name);
  }
  return tensor.int64_data;
}

std::vector<std::int64_t> ReadVectorAsInt64AllowEmpty(const Tensor& tensor, const std::string& op_type) {
  if (tensor.dtype == "int64") {
    return RequireInt64DataAllowEmpty(tensor, op_type);
  }
  if (tensor.dtype == "float32") {
    const auto& data = RequireFloatDataAllowEmpty(tensor, op_type);
    std::vector<std::int64_t> converted;
    converted.reserve(data.size());
    for (const auto value : data) {
      converted.push_back(static_cast<std::int64_t>(value));
    }
    return converted;
  }
  throw std::runtime_error(op_type + " currently supports int64/float32 vectors only: " + tensor.name);
}

std::int64_t ReadScalarInt64AllowEmpty(const Tensor& tensor, const std::string& op_type) {
  const auto& data = RequireInt64DataAllowEmpty(tensor, op_type);
  if (data.size() != 1) {
    throw std::runtime_error(op_type + " requires scalar int64 input: " + tensor.name);
  }
  return data.front();
}

float ReadScalarFloat32AllowEmpty(const Tensor& tensor, const std::string& op_type) {
  const auto& data = RequireFloatDataAllowEmpty(tensor, op_type);
  if (data.size() != 1) {
    throw std::runtime_error(op_type + " requires scalar float32 input: " + tensor.name);
  }
  return data.front();
}

void RebuildGraphDerivedState(Graph& graph) {
  graph.node_name_to_index.clear();
  graph.op_type_histogram.clear();
  for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
    const auto& node = graph.nodes[i];
    graph.node_name_to_index[node.name] = i;
    ++graph.op_type_histogram[node.op_type];
  }

  std::unordered_map<std::string, std::size_t> output_to_producer;
  for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
    for (const auto& output : graph.nodes[i].outputs) {
      output_to_producer[output] = i;
    }
  }

  std::vector<std::size_t> indegree(graph.nodes.size(), 0);
  std::vector<std::vector<std::size_t>> edges(graph.nodes.size());
  for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
    for (const auto& input : graph.nodes[i].inputs) {
      const auto it = output_to_producer.find(input);
      if (it == output_to_producer.end() || it->second == i) {
        continue;
      }
      edges[it->second].push_back(i);
      ++indegree[i];
    }
  }

  std::deque<std::size_t> ready;
  for (std::size_t i = 0; i < indegree.size(); ++i) {
    if (indegree[i] == 0) {
      ready.push_back(i);
    }
  }

  std::vector<std::size_t> order;
  while (!ready.empty()) {
    const auto index = ready.front();
    ready.pop_front();
    order.push_back(index);
    for (const auto consumer : edges[index]) {
      --indegree[consumer];
      if (indegree[consumer] == 0) {
        ready.push_back(consumer);
      }
    }
  }

  if (order.size() != graph.nodes.size()) {
    throw std::runtime_error("optimized graph contains a cycle or unsupported dependency structure");
  }

  graph.topological_order = std::move(order);
}

std::optional<Tensor> FoldBinaryBroadcastNode(const Node& node, const Tensor& lhs, const Tensor& rhs,
                                              const std::function<float(float, float)>& eval_float,
                                              const std::function<std::int64_t(std::int64_t, std::int64_t)>& eval_int,
                                              const std::string& op_type) {
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
    const auto& lhs_data = RequireInt64DataAllowEmpty(lhs, op_type);
    const auto& rhs_data = RequireInt64DataAllowEmpty(rhs, op_type);
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
      const auto lhs_value = lhs.dtype == "float32" ? RequireFloatDataAllowEmpty(lhs, op_type)[lhs_offset]
                                                    : static_cast<float>(RequireInt64DataAllowEmpty(lhs, op_type)[lhs_offset]);
      const auto rhs_value = rhs.dtype == "float32" ? RequireFloatDataAllowEmpty(rhs, op_type)[rhs_offset]
                                                    : static_cast<float>(RequireInt64DataAllowEmpty(rhs, op_type)[rhs_offset]);
      output.float_data[i] = eval_float(lhs_value, rhs_value);
    }
  }

  return output;
}

std::optional<Tensor> FoldConcatNode(const Node& node, const std::vector<Tensor>& inputs) {
  if (inputs.empty()) {
    return std::nullopt;
  }
  if (inputs.front().dtype == "float32") {
    auto output = ConcatTensors<float>(node, nullptr, "float32", inputs,
                                       [](const Tensor& tensor) -> const std::vector<float>& {
                                         return RequireFloatDataAllowEmpty(tensor, "Concat");
                                       });
    return output;
  }
  if (inputs.front().dtype == "int64") {
    auto output = ConcatTensors<std::int64_t>(node, nullptr, "int64", inputs,
                                              [](const Tensor& tensor) -> const std::vector<std::int64_t>& {
                                                return RequireInt64DataAllowEmpty(tensor, "Concat");
                                              });
    return output;
  }
  return std::nullopt;
}

std::optional<Tensor> FoldConstantNode(const Graph& graph, const Node& node) {
  const auto get_input = [&](std::size_t index) -> std::optional<Tensor> {
    if (index >= node.inputs.size()) {
      return std::nullopt;
    }
    return ResolveConstantTensor(graph, node.inputs[index]);
  };

  if (node.op_type == "Constant") {
    const auto attr_it = node.attributes.find("value");
    if (attr_it == node.attributes.end() || !attr_it->second.tensor.has_value()) {
      return std::nullopt;
    }
    const auto& tensor_attr = *attr_it->second.tensor;
    Tensor output;
    output.name = node.outputs.at(0);
    output.dtype = tensor_attr.dtype;
    output.shape = tensor_attr.shape;
    output.float_data = tensor_attr.float_data;
    output.int64_data = tensor_attr.int64_data;
    output.is_initializer = true;
    output.is_placeholder = false;
    return output;
  }

  if (node.inputs.empty()) {
    return std::nullopt;
  }

  if (node.op_type == "Shape") {
    const auto input = get_input(0);
    if (!input.has_value() || !HasConcreteShape(input->shape)) {
      return std::nullopt;
    }
    Tensor output;
    output.name = node.outputs.at(0);
    output.dtype = "int64";
    output.shape = {static_cast<std::int64_t>(input->shape.size())};
    output.int64_data = input->shape;
    return output;
  }

  if (node.op_type == "Gather") {
    const auto data = get_input(0);
    const auto indices = get_input(1);
    if (!data.has_value() || !indices.has_value()) {
      return std::nullopt;
    }
    const auto axis_it = node.attributes.find("axis");
    const auto axis = axis_it == node.attributes.end() ? 0 : axis_it->second.int_value;
    if (axis != 0 || data->shape.size() > 1) {
      return std::nullopt;
    }

    const auto index_data = ReadVectorAsInt64AllowEmpty(*indices, "Gather");
    Tensor output;
    output.name = node.outputs.at(0);
    output.dtype = data->dtype;
    output.shape = indices->shape.empty() ? std::vector<std::int64_t>{} : indices->shape;
    output.is_placeholder = false;

    if (data->dtype == "int64") {
      const auto& data_values = RequireInt64DataAllowEmpty(*data, "Gather");
      output.int64_data.reserve(index_data.size());
      for (const auto index : index_data) {
        if (data->shape.empty() && index != 0) {
          return std::nullopt;
        }
        output.int64_data.push_back(data_values.at(static_cast<std::size_t>(index)));
      }
    } else if (data->dtype == "float32") {
      const auto& data_values = RequireFloatDataAllowEmpty(*data, "Gather");
      output.float_data.reserve(index_data.size());
      for (const auto index : index_data) {
        if (data->shape.empty() && index != 0) {
          return std::nullopt;
        }
        output.float_data.push_back(data_values.at(static_cast<std::size_t>(index)));
      }
    } else {
      return std::nullopt;
    }
    return output;
  }

  if (node.op_type == "Unsqueeze") {
    const auto data = get_input(0);
    if (!data.has_value()) {
      return std::nullopt;
    }
    std::vector<std::int64_t> axes;
    if (node.inputs.size() > 1 && !node.inputs.at(1).empty()) {
      const auto axes_tensor = get_input(1);
      if (!axes_tensor.has_value()) {
        return std::nullopt;
      }
      axes = ReadVectorAsInt64AllowEmpty(*axes_tensor, "Unsqueeze");
    } else {
      const auto axes_it = node.attributes.find("axes");
      if (axes_it == node.attributes.end()) {
        return std::nullopt;
      }
      axes = axes_it->second.ints;
    }

    const auto rank = data->shape.size() + axes.size();
    auto normalized = NormalizeAxes(axes, rank);
    Tensor output = *data;
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
        output.shape.push_back(data->shape[data_index++]);
      }
    }
    return output;
  }

  if (node.op_type == "Concat") {
    std::vector<Tensor> inputs;
    inputs.reserve(node.inputs.size());
    for (const auto& input_name : node.inputs) {
      const auto tensor = ResolveConstantTensor(graph, input_name);
      if (!tensor.has_value()) {
        return std::nullopt;
      }
      inputs.push_back(*tensor);
    }
    return FoldConcatNode(node, inputs);
  }

  if (node.op_type == "Reshape") {
    const auto data = get_input(0);
    const auto shape_tensor = get_input(1);
    if (!data.has_value() || !shape_tensor.has_value()) {
      return std::nullopt;
    }
    const auto requested_shape = ReadVectorAsInt64AllowEmpty(*shape_tensor, "Reshape");
    std::vector<std::int64_t> output_shape;
    output_shape.reserve(requested_shape.size());

    const std::size_t input_count = GetElementCount(data->shape);
    std::int64_t infer_index = -1;
    std::size_t known_product = 1;

    for (std::size_t i = 0; i < requested_shape.size(); ++i) {
      const auto dim = requested_shape[i];
      if (dim == 0) {
        if (i >= data->shape.size()) {
          return std::nullopt;
        }
        output_shape.push_back(data->shape[i]);
        known_product *= static_cast<std::size_t>(data->shape[i]);
      } else if (dim == -1) {
        if (infer_index != -1) {
          return std::nullopt;
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
        return std::nullopt;
      }
      output_shape[static_cast<std::size_t>(infer_index)] = static_cast<std::int64_t>(input_count / known_product);
    }

    if (GetElementCount(output_shape) != input_count) {
      return std::nullopt;
    }

    Tensor output = *data;
    output.name = node.outputs.at(0);
    output.shape = output_shape;
    return output;
  }

  if (node.op_type == "Range") {
    const auto start = get_input(0);
    const auto limit = get_input(1);
    const auto delta = get_input(2);
    if (!start.has_value() || !limit.has_value() || !delta.has_value()) {
      return std::nullopt;
    }

    Tensor output;
    output.name = node.outputs.at(0);
    output.is_placeholder = false;

    if (start->dtype == "float32" || limit->dtype == "float32" || delta->dtype == "float32") {
      const auto start_value = start->dtype == "float32" ? ReadScalarFloat32AllowEmpty(*start, "Range")
                                                         : static_cast<float>(ReadScalarInt64AllowEmpty(*start, "Range"));
      const auto limit_value = limit->dtype == "float32" ? ReadScalarFloat32AllowEmpty(*limit, "Range")
                                                         : static_cast<float>(ReadScalarInt64AllowEmpty(*limit, "Range"));
      const auto delta_value = delta->dtype == "float32" ? ReadScalarFloat32AllowEmpty(*delta, "Range")
                                                         : static_cast<float>(ReadScalarInt64AllowEmpty(*delta, "Range"));
      if (delta_value == 0.0f) {
        return std::nullopt;
      }
      output.dtype = "float32";
      for (float value = start_value; (delta_value > 0.0f) ? (value < limit_value) : (value > limit_value);
           value += delta_value) {
        output.float_data.push_back(value);
      }
      output.shape = {static_cast<std::int64_t>(output.float_data.size())};
    } else if (start->dtype == "int64" && limit->dtype == "int64" && delta->dtype == "int64") {
      const auto start_value = ReadScalarInt64AllowEmpty(*start, "Range");
      const auto limit_value = ReadScalarInt64AllowEmpty(*limit, "Range");
      const auto delta_value = ReadScalarInt64AllowEmpty(*delta, "Range");
      if (delta_value == 0) {
        return std::nullopt;
      }
      output.dtype = "int64";
      for (std::int64_t value = start_value; (delta_value > 0) ? (value < limit_value) : (value > limit_value);
           value += delta_value) {
        output.int64_data.push_back(value);
      }
      output.shape = {static_cast<std::int64_t>(output.int64_data.size())};
    } else {
      return std::nullopt;
    }
    return output;
  }

  if (node.op_type == "Cast") {
    const auto input = get_input(0);
    if (!input.has_value()) {
      return std::nullopt;
    }
    const auto to_it = node.attributes.find("to");
    if (to_it == node.attributes.end()) {
      return std::nullopt;
    }

    Tensor output;
    output.name = node.outputs.at(0);
    output.shape = input->shape;
    output.is_placeholder = false;

    const auto to_type = to_it->second.int_value;
    if (to_type == onnx::TensorProto_DataType_FLOAT) {
      output.dtype = "float32";
      if (input->dtype == "float32") {
        output.float_data = RequireFloatDataAllowEmpty(*input, "Cast");
      } else if (input->dtype == "int64") {
        const auto& input_data = RequireInt64DataAllowEmpty(*input, "Cast");
        output.float_data.reserve(input_data.size());
        for (const auto value : input_data) {
          output.float_data.push_back(static_cast<float>(value));
        }
      } else {
        return std::nullopt;
      }
    } else if (to_type == onnx::TensorProto_DataType_INT32 || to_type == onnx::TensorProto_DataType_INT64) {
      output.dtype = "int64";
      if (input->dtype == "int64") {
        output.int64_data = RequireInt64DataAllowEmpty(*input, "Cast");
      } else if (input->dtype == "float32") {
        const auto& input_data = RequireFloatDataAllowEmpty(*input, "Cast");
        output.int64_data.reserve(input_data.size());
        for (const auto value : input_data) {
          output.int64_data.push_back(static_cast<std::int64_t>(value));
        }
      } else {
        return std::nullopt;
      }
    } else {
      return std::nullopt;
    }
    return output;
  }

  if (node.op_type == "ConstantOfShape") {
    const auto shape_tensor = get_input(0);
    if (!shape_tensor.has_value()) {
      return std::nullopt;
    }

    const auto shape = ReadVectorAsInt64AllowEmpty(*shape_tensor, "ConstantOfShape");
    Tensor output;
    output.name = node.outputs.at(0);
    output.dtype = "float32";
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

    output.float_data.assign(GetElementCount(output.shape), fill_value);
    return output;
  }

  if (node.op_type == "Expand") {
    const auto data = get_input(0);
    const auto shape_tensor = get_input(1);
    if (!data.has_value() || !shape_tensor.has_value()) {
      return std::nullopt;
    }
    const auto output_shape = ReadVectorAsInt64AllowEmpty(*shape_tensor, "Expand");
    const auto output_strides = ComputeStrides(output_shape);
    const auto input_strides = ComputeStrides(data->shape);

    if (output_shape.size() < data->shape.size()) {
      return std::nullopt;
    }
    for (std::size_t i = 0; i < data->shape.size(); ++i) {
      const auto input_dim = data->shape[data->shape.size() - 1 - i];
      const auto output_dim = output_shape[output_shape.size() - 1 - i];
      if (input_dim != output_dim && input_dim != 1) {
        return std::nullopt;
      }
    }

    Tensor output;
    output.name = node.outputs.at(0);
    output.dtype = data->dtype;
    output.shape = output_shape;
    output.is_placeholder = false;

    if (data->dtype == "float32") {
      const auto& input_data = RequireFloatDataAllowEmpty(*data, "Expand");
      output.float_data.resize(GetElementCount(output_shape));
      for (std::size_t i = 0; i < output.float_data.size(); ++i) {
        const auto output_index = UnravelIndex(i, output_shape, output_strides);
        const auto input_offset = ComputeBroadcastOffset(output_index, data->shape, input_strides);
        output.float_data[i] = input_data[input_offset];
      }
    } else if (data->dtype == "int64") {
      const auto& input_data = RequireInt64DataAllowEmpty(*data, "Expand");
      output.int64_data.resize(GetElementCount(output_shape));
      for (std::size_t i = 0; i < output.int64_data.size(); ++i) {
        const auto output_index = UnravelIndex(i, output_shape, output_strides);
        const auto input_offset = ComputeBroadcastOffset(output_index, data->shape, input_strides);
        output.int64_data[i] = input_data[input_offset];
      }
    } else {
      return std::nullopt;
    }
    return output;
  }

  if (node.op_type == "Transpose") {
    const auto input = get_input(0);
    if (!input.has_value()) {
      return std::nullopt;
    }
    std::vector<std::int64_t> perm;
    const auto perm_it = node.attributes.find("perm");
    if (perm_it == node.attributes.end() || perm_it->second.ints.empty()) {
      perm.resize(input->shape.size());
      for (std::size_t i = 0; i < perm.size(); ++i) {
        perm[i] = static_cast<std::int64_t>(perm.size() - 1 - i);
      }
    } else {
      perm = perm_it->second.ints;
    }
    if (perm.size() != input->shape.size()) {
      return std::nullopt;
    }

    Tensor output;
    output.name = node.outputs.at(0);
    output.dtype = input->dtype;
    output.shape.resize(input->shape.size());
    for (std::size_t i = 0; i < perm.size(); ++i) {
      output.shape[i] = input->shape[static_cast<std::size_t>(perm[i])];
    }
    output.is_placeholder = false;

    const auto input_strides = ComputeStrides(input->shape);
    const auto output_strides = ComputeStrides(output.shape);
    if (input->dtype == "float32") {
      const auto& input_data = RequireFloatDataAllowEmpty(*input, "Transpose");
      output.float_data.resize(GetElementCount(output.shape));
      for (std::size_t i = 0; i < output.float_data.size(); ++i) {
        const auto output_index = UnravelIndex(i, output.shape, output_strides);
        std::vector<std::int64_t> input_index(input->shape.size(), 0);
        for (std::size_t j = 0; j < perm.size(); ++j) {
          input_index[static_cast<std::size_t>(perm[j])] = output_index[j];
        }
        output.float_data[i] = input_data[ComputeOffset(input_index, input_strides)];
      }
    } else if (input->dtype == "int64") {
      const auto& input_data = RequireInt64DataAllowEmpty(*input, "Transpose");
      output.int64_data.resize(GetElementCount(output.shape));
      for (std::size_t i = 0; i < output.int64_data.size(); ++i) {
        const auto output_index = UnravelIndex(i, output.shape, output_strides);
        std::vector<std::int64_t> input_index(input->shape.size(), 0);
        for (std::size_t j = 0; j < perm.size(); ++j) {
          input_index[static_cast<std::size_t>(perm[j])] = output_index[j];
        }
        output.int64_data[i] = input_data[ComputeOffset(input_index, input_strides)];
      }
    } else {
      return std::nullopt;
    }
    return output;
  }

  if (node.op_type == "Slice") {
    const auto data = get_input(0);
    const auto starts_tensor = get_input(1);
    const auto ends_tensor = get_input(2);
    if (!data.has_value() || !starts_tensor.has_value() || !ends_tensor.has_value()) {
      return std::nullopt;
    }
    const auto starts = ReadVectorAsInt64AllowEmpty(*starts_tensor, "Slice");
    const auto ends = ReadVectorAsInt64AllowEmpty(*ends_tensor, "Slice");
    if (starts.size() != ends.size()) {
      return std::nullopt;
    }

    std::vector<std::int64_t> axes(starts.size());
    if (node.inputs.size() > 3 && !node.inputs.at(3).empty()) {
      const auto axes_tensor = get_input(3);
      if (!axes_tensor.has_value()) {
        return std::nullopt;
      }
      axes = ReadVectorAsInt64AllowEmpty(*axes_tensor, "Slice");
    } else {
      for (std::size_t i = 0; i < axes.size(); ++i) {
        axes[i] = static_cast<std::int64_t>(i);
      }
    }

    std::vector<std::int64_t> steps(starts.size(), 1);
    if (node.inputs.size() > 4 && !node.inputs.at(4).empty()) {
      const auto steps_tensor = get_input(4);
      if (!steps_tensor.has_value()) {
        return std::nullopt;
      }
      steps = ReadVectorAsInt64AllowEmpty(*steps_tensor, "Slice");
    }

    std::vector<std::int64_t> slice_starts(data->shape.size(), 0);
    std::vector<std::int64_t> slice_steps(data->shape.size(), 1);
    std::vector<std::int64_t> output_shape = data->shape;
    for (std::size_t i = 0; i < axes.size(); ++i) {
      const auto axis = static_cast<std::size_t>(NormalizeAxis(axes[i], data->shape.size(), "Slice"));
      const auto dim = data->shape[axis];
      const auto step = steps[i];
      if (step == 0) {
        return std::nullopt;
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

    const auto input_strides = ComputeStrides(data->shape);
    const auto output_strides = ComputeStrides(output_shape);
    Tensor output;
    output.name = node.outputs.at(0);
    output.dtype = data->dtype;
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

    if (data->dtype == "float32") {
      emit_slice(output.float_data, RequireFloatDataAllowEmpty(*data, "Slice"));
    } else if (data->dtype == "int64") {
      emit_slice(output.int64_data, RequireInt64DataAllowEmpty(*data, "Slice"));
    } else {
      return std::nullopt;
    }
    return output;
  }

  if (node.op_type == "ReduceMax") {
    const auto input = get_input(0);
    if (!input.has_value() || input->dtype != "float32") {
      return std::nullopt;
    }
    const auto& input_data = RequireFloatDataAllowEmpty(*input, "ReduceMax");
    const auto axes = NormalizeAxes(ReadIntsAttribute(node, "axes", {0}), input->shape.size());
    const auto keepdims = ReadIntAttribute(node, "keepdims", 1);
    if (axes.size() != 1) {
      return std::nullopt;
    }

    const auto axis = static_cast<std::size_t>(axes.front());
    std::vector<std::int64_t> output_shape = input->shape;
    if (keepdims != 0) {
      output_shape[axis] = 1;
    } else {
      output_shape.erase(output_shape.begin() + static_cast<std::ptrdiff_t>(axis));
    }

    const auto input_strides = ComputeStrides(input->shape);
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
        base_index.reserve(input->shape.size());
        for (std::size_t dim = 0, j = 0; dim < input->shape.size(); ++dim) {
          if (dim == axis) {
            base_index.push_back(0);
          } else {
            base_index.push_back(output_index[j++]);
          }
        }
      }

      float best = -std::numeric_limits<float>::infinity();
      for (std::int64_t k = 0; k < input->shape[axis]; ++k) {
        base_index[axis] = k;
        best = std::max(best, input_data[ComputeOffset(base_index, input_strides)]);
      }
      output.float_data[i] = best;
    }
    return output;
  }

  if (node.op_type == "ArgMax") {
    const auto input = get_input(0);
    if (!input.has_value() || input->dtype != "float32") {
      return std::nullopt;
    }
    const auto& input_data = RequireFloatDataAllowEmpty(*input, "ArgMax");
    const auto axis = static_cast<std::size_t>(
        NormalizeAxis(ReadIntAttribute(node, "axis", 0), input->shape.size(), "ArgMax"));
    const auto keepdims = ReadIntAttribute(node, "keepdims", 1);

    std::vector<std::int64_t> output_shape = input->shape;
    if (keepdims != 0) {
      output_shape[axis] = 1;
    } else {
      output_shape.erase(output_shape.begin() + static_cast<std::ptrdiff_t>(axis));
    }

    const auto input_strides = ComputeStrides(input->shape);
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
        base_index.reserve(input->shape.size());
        for (std::size_t dim = 0, j = 0; dim < input->shape.size(); ++dim) {
          if (dim == axis) {
            base_index.push_back(0);
          } else {
            base_index.push_back(output_index[j++]);
          }
        }
      }

      std::int64_t best_index = 0;
      float best_value = -std::numeric_limits<float>::infinity();
      for (std::int64_t k = 0; k < input->shape[axis]; ++k) {
        base_index[axis] = k;
        const auto value = input_data[ComputeOffset(base_index, input_strides)];
        if (k == 0 || value > best_value) {
          best_value = value;
          best_index = k;
        }
      }
      output.int64_data[i] = best_index;
    }
    return output;
  }

  if (node.op_type == "Sigmoid") {
    const auto input = get_input(0);
    if (!input.has_value()) {
      return std::nullopt;
    }

    Tensor output;
    output.name = node.outputs.at(0);
    output.shape = input->shape;
    output.dtype = "float32";
    output.is_placeholder = false;
    output.float_data.resize(GetElementCount(input->shape));
    if (input->dtype == "float32") {
      const auto& input_data = RequireFloatDataAllowEmpty(*input, "Sigmoid");
      std::transform(input_data.begin(), input_data.end(), output.float_data.begin(),
                     [](float value) { return 1.0f / (1.0f + std::exp(-value)); });
    } else if (input->dtype == "int64") {
      const auto& input_data = RequireInt64DataAllowEmpty(*input, "Sigmoid");
      for (std::size_t i = 0; i < input_data.size(); ++i) {
        output.float_data[i] = 1.0f / (1.0f + std::exp(-static_cast<float>(input_data[i])));
      }
    } else {
      return std::nullopt;
    }
    return output;
  }

  if (node.inputs.size() == 2 && (node.op_type == "Add" || node.op_type == "Mul" || node.op_type == "Sub" ||
                                  node.op_type == "Div")) {
    const auto lhs = get_input(0);
    const auto rhs = get_input(1);
    if (!lhs.has_value() || !rhs.has_value()) {
      return std::nullopt;
    }
    std::function<float(float, float)> eval_float;
    std::function<std::int64_t(std::int64_t, std::int64_t)> eval_int;
    if (node.op_type == "Add") {
      eval_float = [](float a, float b) { return a + b; };
      eval_int = [](std::int64_t a, std::int64_t b) { return a + b; };
    } else if (node.op_type == "Mul") {
      eval_float = [](float a, float b) { return a * b; };
      eval_int = [](std::int64_t a, std::int64_t b) { return a * b; };
    } else if (node.op_type == "Sub") {
      eval_float = [](float a, float b) { return a - b; };
      eval_int = [](std::int64_t a, std::int64_t b) { return a - b; };
    } else {
      eval_float = [](float a, float b) {
        if (b == 0.0f) {
          throw std::runtime_error("Div divisor must not be zero");
        }
        return a / b;
      };
      eval_int = [](std::int64_t a, std::int64_t b) {
        if (b == 0) {
          throw std::runtime_error("Div divisor must not be zero");
        }
        return a / b;
      };
    }
    return FoldBinaryBroadcastNode(
        node, *lhs, *rhs, eval_float, eval_int, node.op_type);
  }

  return std::nullopt;
}

bool RunConstantFolding(Graph& graph, std::ostream* trace) {
  std::vector<Node> kept_nodes;
  kept_nodes.reserve(graph.nodes.size());
  std::size_t folded_nodes = 0;
  std::size_t materialized_constants = 0;

  for (const auto& node : graph.nodes) {
    std::optional<Tensor> folded;
    try {
      folded = FoldConstantNode(graph, node);
    } catch (const std::exception&) {
      folded = std::nullopt;
    }

    if (!folded.has_value()) {
      kept_nodes.push_back(node);
      continue;
    }

    ++folded_nodes;
    graph.initializers[folded->name] = MakeInitializerValueFromTensor(*folded);
    graph.value_infos[folded->name] = MakeTensorInfoFromRuntimeTensor(*folded);
    ++materialized_constants;
  }

  if (folded_nodes == 0) {
    if (trace != nullptr) {
      *trace << "  [pass] ConstantFolding (no changes)\n";
    }
    return false;
  }

  graph.nodes = std::move(kept_nodes);
  RebuildGraphDerivedState(graph);

  if (trace != nullptr) {
    *trace << "  [pass] ConstantFolding removed " << folded_nodes << " nodes, materialized "
           << materialized_constants << " constants\n";
  }
  return true;
}

bool RunDeadNodeCleanup(Graph& graph, std::ostream* trace) {
  std::unordered_map<std::string, std::size_t> output_to_producer;
  for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
    for (const auto& output : graph.nodes[i].outputs) {
      output_to_producer[output] = i;
    }
  }

  std::unordered_set<std::string> live_tensors;
  std::deque<std::string> worklist;
  for (const auto& output : graph.outputs) {
    if (live_tensors.insert(output.name).second) {
      worklist.push_back(output.name);
    }
  }

  std::vector<bool> live_nodes(graph.nodes.size(), false);
  while (!worklist.empty()) {
    const auto tensor_name = worklist.front();
    worklist.pop_front();
    const auto producer_it = output_to_producer.find(tensor_name);
    if (producer_it == output_to_producer.end()) {
      continue;
    }

    const auto node_index = producer_it->second;
    if (live_nodes[node_index]) {
      continue;
    }
    live_nodes[node_index] = true;
    for (const auto& input : graph.nodes[node_index].inputs) {
      if (input.empty()) {
        continue;
      }
      if (live_tensors.insert(input).second) {
        worklist.push_back(input);
      }
    }
  }

  std::vector<Node> kept_nodes;
  kept_nodes.reserve(graph.nodes.size());
  std::size_t removed_nodes = 0;
  for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
    if (live_nodes[i]) {
      kept_nodes.push_back(graph.nodes[i]);
    } else {
      ++removed_nodes;
    }
  }

  if (removed_nodes == 0) {
    if (trace != nullptr) {
      *trace << "  [pass] DeadNodeCleanup (no changes)\n";
    }
    return false;
  }

  graph.nodes = std::move(kept_nodes);

  std::size_t removed_initializers = 0;
  for (auto it = graph.initializers.begin(); it != graph.initializers.end();) {
    if (!live_tensors.contains(it->first)) {
      it = graph.initializers.erase(it);
      ++removed_initializers;
    } else {
      ++it;
    }
  }

  std::size_t removed_value_infos = 0;
  for (auto it = graph.value_infos.begin(); it != graph.value_infos.end();) {
    const auto& name = it->first;
    const bool keep = live_tensors.contains(name) ||
                      std::any_of(graph.inputs.begin(), graph.inputs.end(),
                                  [&name](const Value& value) { return value.name == name; }) ||
                      std::any_of(graph.outputs.begin(), graph.outputs.end(),
                                  [&name](const Value& value) { return value.name == name; });
    if (!keep) {
      it = graph.value_infos.erase(it);
      ++removed_value_infos;
    } else {
      ++it;
    }
  }

  RebuildGraphDerivedState(graph);

  if (trace != nullptr) {
    *trace << "  [pass] DeadNodeCleanup removed " << removed_nodes << " nodes, pruned " << removed_initializers
           << " initializers and " << removed_value_infos << " value infos\n";
  }
  return true;
}

bool RunShapeSimplification(Graph& graph, std::ostream* trace) {
  if (trace != nullptr) {
    *trace << "  [pass] ShapeSimplification (scaffold, no-op for now)\n";
  }
  RebuildGraphDerivedState(graph);
  return false;
}

}  // namespace

Graph OptimizeGraph(Graph graph, const GraphOptimizationOptions& options,
                    std::ostream* trace, GraphOptimizationSummary* summary) {
  GraphOptimizationSummary local_summary;
  local_summary.nodes_before = graph.nodes.size();

  TimingMap timings;
  {
    ScopedTimer timer("graph_optimizer.total", trace, &timings["graph_optimizer.total"]);
    const auto run_pass = [&](const char* pass_name, bool enabled, const std::function<bool(Graph&, std::ostream*)>& fn) {
      if (!enabled) {
        return;
      }
      ++local_summary.passes_run;
      local_summary.applied_passes.emplace_back(pass_name);
      (void)fn(graph, trace);
    };

    run_pass("ConstantFolding", options.enable_constant_folding, RunConstantFolding);
    run_pass("DeadNodeCleanup", options.enable_dead_node_cleanup, RunDeadNodeCleanup);
    run_pass("ShapeSimplification", options.enable_shape_simplification, RunShapeSimplification);
  }

  local_summary.nodes_after = graph.nodes.size();

  if (trace != nullptr) {
    PrintTimingSummary(timings, *trace, "graph optimizer timing summary");
  }

  if (summary != nullptr) {
    *summary = local_summary;
  }

  return graph;
}

void PrintGraphOptimizationSummary(const GraphOptimizationSummary& summary, std::ostream& os) {
  os << "graph optimization summary\n";
  os << "  nodes_before: " << summary.nodes_before << "\n";
  os << "  nodes_after: " << summary.nodes_after << "\n";
  os << "  passes_run: " << summary.passes_run << "\n";
  os << "  applied_passes:";
  if (summary.applied_passes.empty()) {
    os << " <none>\n";
    return;
  }
  os << "\n";
  for (const auto& pass_name : summary.applied_passes) {
    os << "    - " << pass_name << "\n";
  }
}

}  // namespace miniort
