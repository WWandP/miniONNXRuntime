#include "miniort/loader/onnx_loader.h"

#include <deque>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <unordered_set>

#include <onnx/onnx_pb.h>

#include "miniort/runtime/profiling.h"

namespace miniort {

namespace {

std::string ToShapeDim(const onnx::TensorShapeProto_Dimension& dim) {
  if (dim.has_dim_value()) {
    return std::to_string(dim.dim_value());
  }
  if (dim.has_dim_param()) {
    return dim.dim_param();
  }
  return "?";
}

std::string ToTensorElemTypeString(int elem_type) {
  switch (elem_type) {
    case onnx::TensorProto_DataType_FLOAT:
      return "float32";
    case onnx::TensorProto_DataType_UINT8:
      return "uint8";
    case onnx::TensorProto_DataType_INT8:
      return "int8";
    case onnx::TensorProto_DataType_UINT16:
      return "uint16";
    case onnx::TensorProto_DataType_INT16:
      return "int16";
    case onnx::TensorProto_DataType_INT32:
      return "int32";
    case onnx::TensorProto_DataType_INT64:
      return "int64";
    case onnx::TensorProto_DataType_STRING:
      return "string";
    case onnx::TensorProto_DataType_BOOL:
      return "bool";
    case onnx::TensorProto_DataType_FLOAT16:
      return "float16";
    case onnx::TensorProto_DataType_DOUBLE:
      return "float64";
    case onnx::TensorProto_DataType_UINT32:
      return "uint32";
    case onnx::TensorProto_DataType_UINT64:
      return "uint64";
    default:
      return "unknown(" + std::to_string(elem_type) + ")";
  }
}

TensorInfo MakeTensorInfo(const onnx::ValueInfoProto& value_info, bool is_initializer = false) {
  TensorInfo info;
  info.is_initializer = is_initializer;
  if (!value_info.has_type() || !value_info.type().has_tensor_type()) {
    return info;
  }

  const auto& tensor_type = value_info.type().tensor_type();
  info.dtype = ToTensorElemTypeString(tensor_type.elem_type());

  if (tensor_type.has_shape()) {
    for (const auto& dim : tensor_type.shape().dim()) {
      info.shape.push_back(ToShapeDim(dim));
    }
  }
  return info;
}

TensorInfo MakeTensorInfo(const onnx::TensorProto& tensor_proto) {
  TensorInfo info;
  info.is_initializer = true;
  info.dtype = ToTensorElemTypeString(tensor_proto.data_type());
  for (const auto dim : tensor_proto.dims()) {
    info.shape.push_back(std::to_string(dim));
  }
  return info;
}

TensorData ParseTensorData(const onnx::TensorProto& tensor_proto) {
  TensorData data;
  data.dtype = ToTensorElemTypeString(tensor_proto.data_type());
  data.shape.assign(tensor_proto.dims().begin(), tensor_proto.dims().end());

  if (!tensor_proto.raw_data().empty()) {
    const auto& raw = tensor_proto.raw_data();
    data.raw_data.assign(raw.begin(), raw.end());
  }

  data.float_data.assign(tensor_proto.float_data().begin(), tensor_proto.float_data().end());
  data.double_data.assign(tensor_proto.double_data().begin(), tensor_proto.double_data().end());
  data.int32_data.assign(tensor_proto.int32_data().begin(), tensor_proto.int32_data().end());
  data.int64_data.assign(tensor_proto.int64_data().begin(), tensor_proto.int64_data().end());
  data.string_data.assign(tensor_proto.string_data().begin(), tensor_proto.string_data().end());

  if (!data.raw_data.empty()) {
    if (data.dtype == "float32" && data.float_data.empty() &&
        data.raw_data.size() % sizeof(float) == 0) {
      const auto count = data.raw_data.size() / sizeof(float);
      data.float_data.resize(count);
      std::memcpy(data.float_data.data(), data.raw_data.data(), data.raw_data.size());
    }
    if (data.dtype == "int64" && data.int64_data.empty() &&
        data.raw_data.size() % sizeof(std::int64_t) == 0) {
      const auto count = data.raw_data.size() / sizeof(std::int64_t);
      data.int64_data.resize(count);
      std::memcpy(data.int64_data.data(), data.raw_data.data(), data.raw_data.size());
    }
  }

  return data;
}

AttributeValue ParseAttributeValue(const onnx::AttributeProto& attr) {
  AttributeValue value;

  switch (attr.type()) {
    case onnx::AttributeProto_AttributeType_FLOAT:
      value.kind = AttributeValue::Kind::kFloat;
      value.float_value = attr.f();
      break;
    case onnx::AttributeProto_AttributeType_INT:
      value.kind = AttributeValue::Kind::kInt;
      value.int_value = attr.i();
      break;
    case onnx::AttributeProto_AttributeType_STRING:
      value.kind = AttributeValue::Kind::kString;
      value.string_value = attr.s();
      break;
    case onnx::AttributeProto_AttributeType_FLOATS:
      value.kind = AttributeValue::Kind::kFloats;
      value.floats.assign(attr.floats().begin(), attr.floats().end());
      break;
    case onnx::AttributeProto_AttributeType_INTS:
      value.kind = AttributeValue::Kind::kInts;
      value.ints.assign(attr.ints().begin(), attr.ints().end());
      break;
    case onnx::AttributeProto_AttributeType_STRINGS:
      value.kind = AttributeValue::Kind::kStrings;
      value.strings.assign(attr.strings().begin(), attr.strings().end());
      break;
    case onnx::AttributeProto_AttributeType_TENSOR:
      value.kind = AttributeValue::Kind::kTensor;
      value.tensor = ParseTensorData(attr.t());
      break;
    default:
      value.kind = AttributeValue::Kind::kUnknown;
      break;
  }

  return value;
}

void AddValueInfo(const onnx::ValueInfoProto& value_info, Graph& graph) {
  graph.value_infos[value_info.name()] = MakeTensorInfo(value_info);
}

std::vector<std::size_t> BuildTopologicalOrder(const std::vector<Node>& nodes) {
  std::unordered_map<std::string, std::size_t> output_to_producer;
  std::vector<std::size_t> indegree(nodes.size(), 0);
  std::vector<std::vector<std::size_t>> edges(nodes.size());

  for (std::size_t i = 0; i < nodes.size(); ++i) {
    for (const auto& output : nodes[i].outputs) {
      output_to_producer[output] = i;
    }
  }

  for (std::size_t i = 0; i < nodes.size(); ++i) {
    for (const auto& input : nodes[i].inputs) {
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

  if (order.size() != nodes.size()) {
    throw std::runtime_error("graph contains a cycle or unsupported dependency structure");
  }

  return order;
}

}  // namespace

Graph LoadOnnxGraph(const std::filesystem::path& model_path, std::ostream* trace) {
  TimingMap timings;
  std::ifstream input(model_path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open model: " + model_path.string());
  }

  onnx::ModelProto model;
  {
    ScopedTimer timer("loader.parse_model_proto", trace, &timings["loader.parse_model_proto"]);
    if (!model.ParseFromIstream(&input)) {
      throw std::runtime_error("failed to parse ONNX ModelProto: " + model_path.string());
    }
  }

  const auto& graph_proto = model.graph();
  Graph graph;

  {
    ScopedTimer timer("loader.read_metadata", trace, &timings["loader.read_metadata"]);
    // Copy high-level model metadata into the minimal internal graph.
    // This is mostly a direct transfer, with a few defaults filled in locally
    // (for example, graph name and the default ONNX domain).
    graph.name = graph_proto.name().empty() ? model_path.stem().string() : graph_proto.name();
    graph.metadata.model_path = model_path.string();
    graph.metadata.ir_version = model.ir_version();
    graph.metadata.producer_name = model.producer_name();
    graph.metadata.producer_version = model.producer_version();

    for (const auto& opset : model.opset_import()) {
      const auto domain = opset.domain().empty() ? "ai.onnx" : opset.domain();
      graph.metadata.opset_imports[domain] = opset.version();
    }
  }

  {
    ScopedTimer timer("loader.collect_value_infos", trace, &timings["loader.collect_value_infos"]);
    // Collect type/shape metadata for named values.
    // We keep only the TensorInfo needed by this project and intentionally drop
    // richer ONNX type details that are not used by the current mini runtime.
    for (const auto& value_info : graph_proto.input()) {
      AddValueInfo(value_info, graph);
    }
    for (const auto& value_info : graph_proto.output()) {
      AddValueInfo(value_info, graph);
    }
    for (const auto& value_info : graph_proto.value_info()) {
      AddValueInfo(value_info, graph);
    }
  }

  std::unordered_set<std::string> initializer_names;
  {
    ScopedTimer timer("loader.collect_initializers", trace, &timings["loader.collect_initializers"]);
    // Lift initializers into a dedicated map of constant values.
    // Here we keep basic tensor metadata and a minimal constant-data view for
    // simple scalar/list payloads and raw bytes. More advanced TensorProto
    // storage modes are still intentionally left out.
    for (const auto& initializer : graph_proto.initializer()) {
      Value value;
      value.name = initializer.name();
      value.info = MakeTensorInfo(initializer);
      value.data = ParseTensorData(initializer);
      graph.initializers.emplace(value.name, value);
      graph.value_infos[value.name] = value.info;
      initializer_names.insert(value.name);
    }
  }

  {
    ScopedTimer timer("loader.collect_inputs_outputs", trace, &timings["loader.collect_inputs_outputs"]);
    // Build the true graph input list by excluding entries that are also
    // initializers. ONNX may list weights in graph inputs, but for this project
    // we separate runtime feeds from constant parameters.
    for (const auto& input_value : graph_proto.input()) {
      if (initializer_names.contains(input_value.name())) {
        continue;
      }
      Value value;
      value.name = input_value.name();
      value.info = graph.value_infos[value.name];
      graph.inputs.push_back(std::move(value));
    }

    for (const auto& output_value : graph_proto.output()) {
      Value value;
      value.name = output_value.name();
      value.info = graph.value_infos[value.name];
      graph.outputs.push_back(std::move(value));
    }
  }

  {
    ScopedTimer timer("loader.collect_nodes", trace, &timings["loader.collect_nodes"]);
    // Convert ONNX nodes into the minimal internal node form.
    // We keep only name, op type, and input/output edges. Attributes, domains,
    // and doc strings are still trimmed, but basic attributes are now captured
    // in a compact internal form so later passes/runtime code can inspect them.
    graph.nodes.reserve(static_cast<std::size_t>(graph_proto.node_size()));
    for (int i = 0; i < graph_proto.node_size(); ++i) {
      const auto& node_proto = graph_proto.node(i);
      Node node;
      node.name = node_proto.name().empty() ? node_proto.op_type() + "_" + std::to_string(i) : node_proto.name();
      node.op_type = node_proto.op_type();
      node.inputs.assign(node_proto.input().begin(), node_proto.input().end());
      node.outputs.assign(node_proto.output().begin(), node_proto.output().end());
      for (const auto& attr : node_proto.attribute()) {
        node.attributes.emplace(attr.name(), ParseAttributeValue(attr));
      }
      graph.node_name_to_index[node.name] = graph.nodes.size();
      ++graph.op_type_histogram[node.op_type];
      graph.nodes.push_back(std::move(node));
    }
  }

  {
    ScopedTimer timer("loader.build_topological_order", trace, &timings["loader.build_topological_order"]);
    // Derive runtime-friendly helper structures that do not exist in the raw
    // ONNX graph: a name index, op histogram, and topological execution order.
    graph.topological_order = BuildTopologicalOrder(graph.nodes);
  }

  if (trace != nullptr) {
    PrintTimingSummary(timings, *trace, "loader timing summary");
  }
  return graph;
}

}  // namespace miniort
