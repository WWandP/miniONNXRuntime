#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "miniort/loader/onnx_loader.h"
#include "miniort/model/graph.h"
#include "miniort/optimizer/graph_optimizer.h"
#include "miniort/runtime/session.h"
#include "miniort/tools/phase_output.h"

namespace {

std::string FormatTensorShape(const std::vector<std::int64_t>& shape) {
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      oss << ", ";
    }
    oss << shape[i];
  }
  oss << "]";
  return oss.str();
}

template <typename T>
std::string FormatVectorPreview(const std::vector<T>& values, std::size_t limit = 6) {
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < std::min(limit, values.size()); ++i) {
    if (i != 0) {
      oss << ", ";
    }
    oss << values[i];
  }
  if (values.size() > limit) {
    oss << ", ...";
  }
  oss << "]";
  return oss.str();
}

std::string FormatStringVectorPreview(const std::vector<std::string>& values, std::size_t limit = 4) {
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < std::min(limit, values.size()); ++i) {
    if (i != 0) {
      oss << ", ";
    }
    oss << "\"" << values[i] << "\"";
  }
  if (values.size() > limit) {
    oss << ", ...";
  }
  oss << "]";
  return oss.str();
}

std::string FormatAttributeValue(const miniort::AttributeValue& value) {
  switch (value.kind) {
    case miniort::AttributeValue::Kind::kFloat:
      return std::to_string(value.float_value);
    case miniort::AttributeValue::Kind::kInt:
      return std::to_string(value.int_value);
    case miniort::AttributeValue::Kind::kString:
      return "\"" + value.string_value + "\"";
    case miniort::AttributeValue::Kind::kFloats:
      return FormatVectorPreview(value.floats);
    case miniort::AttributeValue::Kind::kInts:
      return FormatVectorPreview(value.ints);
    case miniort::AttributeValue::Kind::kStrings:
      return FormatStringVectorPreview(value.strings);
    case miniort::AttributeValue::Kind::kTensor:
      if (!value.tensor.has_value()) {
        return "<tensor: missing>";
      }
      return "<tensor dtype=" + value.tensor->dtype +
             " shape=" + FormatTensorShape(value.tensor->shape) +
             " raw_bytes=" + std::to_string(value.tensor->raw_data.size()) + ">";
    case miniort::AttributeValue::Kind::kUnknown:
    default:
      return "<unsupported>";
  }
}

void PrintNodeAttributes(const miniort::Node& node) {
  if (node.attributes.empty()) {
    return;
  }

  std::vector<std::pair<std::string, const miniort::AttributeValue*>> attrs;
  attrs.reserve(node.attributes.size());
  for (const auto& [name, value] : node.attributes) {
    attrs.push_back({name, &value});
  }

  std::sort(attrs.begin(), attrs.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

  for (const auto& [name, value] : attrs) {
    std::cout << "      attr " << name << " = " << FormatAttributeValue(*value) << "\n";
  }
}

struct Options {
  std::string model_path;
  bool graph_opt{false};
};

Options ParseArgs(int argc, char* argv[]) {
  if (argc < 2) {
    throw std::runtime_error("usage: miniort_inspect <model.onnx> [--graph-opt]");
  }

  Options options;
  options.model_path = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--graph-opt") {
      options.graph_opt = true;
      continue;
    }
    throw std::runtime_error("unknown argument: " + arg);
  }

  return options;
}

std::string ReadShapeKey(const miniort::Graph& graph, const std::string& value_name) {
  const auto initializer_it = graph.initializers.find(value_name);
  if (initializer_it != graph.initializers.end() && initializer_it->second.data.has_value()) {
    return FormatTensorShape(initializer_it->second.data->shape);
  }
  const auto value_info_it = graph.value_infos.find(value_name);
  if (value_info_it != graph.value_infos.end()) {
    return miniort::FormatShape(value_info_it->second.shape);
  }
  return "[]";
}

std::string ReadIntsAttributeKey(const miniort::Node& node, const std::string& name,
                                 const std::vector<std::int64_t>& default_values) {
  const auto it = node.attributes.find(name);
  if (it == node.attributes.end()) {
    return FormatVectorPreview(default_values, default_values.size());
  }
  return FormatAttributeValue(it->second);
}

void PrintConvShapeSummary(const miniort::Graph& graph) {
  std::unordered_map<std::string, std::size_t> groups;
  for (const auto& node : graph.nodes) {
    if ((node.op_type != "Conv" && node.op_type != "ConvSiLU") || node.inputs.size() < 2) {
      continue;
    }
    const auto input_shape = ReadShapeKey(graph, node.inputs.at(0));
    const auto weight_shape = ReadShapeKey(graph, node.inputs.at(1));
    const auto output_shape = node.outputs.empty() ? "[]" : ReadShapeKey(graph, node.outputs.at(0));
    const auto strides = ReadIntsAttributeKey(node, "strides", {1, 1});
    const auto pads = ReadIntsAttributeKey(node, "pads", {0, 0, 0, 0});
    const auto key = "op=" + node.op_type + " input=" + input_shape +
                     " weight=" + weight_shape + " output=" + output_shape +
                     " strides=" + strides + " pads=" + pads;
    ++groups[key];
  }
  if (groups.empty()) {
    return;
  }

  std::vector<std::pair<std::string, std::size_t>> entries(groups.begin(), groups.end());
  std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
    if (lhs.second != rhs.second) {
      return lhs.second > rhs.second;
    }
    return lhs.first < rhs.first;
  });

  std::cout << "conv_shape_summary:\n";
  for (const auto& [key, count] : entries) {
    std::cout << "  - count=" << count << " " << key << "\n";
  }
  std::cout << "\n";
}

void PrintMatMulShapeSummary(const miniort::Graph& graph) {
  std::unordered_map<std::string, std::size_t> groups;
  for (const auto& node : graph.nodes) {
    if (node.op_type != "MatMul" || node.inputs.size() < 2) {
      continue;
    }
    const auto lhs_shape = ReadShapeKey(graph, node.inputs.at(0));
    const auto rhs_shape = ReadShapeKey(graph, node.inputs.at(1));
    const auto output_shape = node.outputs.empty() ? "[]" : ReadShapeKey(graph, node.outputs.at(0));
    const auto key = "lhs=" + lhs_shape + " rhs=" + rhs_shape + " output=" + output_shape;
    ++groups[key];
  }
  if (groups.empty()) {
    return;
  }

  std::vector<std::pair<std::string, std::size_t>> entries(groups.begin(), groups.end());
  std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
    if (lhs.second != rhs.second) {
      return lhs.second > rhs.second;
    }
    return lhs.first < rhs.first;
  });

  std::cout << "matmul_shape_summary:\n";
  for (const auto& [key, count] : entries) {
    std::cout << "  - count=" << count << " " << key << "\n";
  }
  std::cout << "\n";
}

void PrintGraphSummary(const miniort::Graph& graph, const miniort::SessionAssignmentSummary& assignment_summary) {
  constexpr std::size_t kShowTopology = 10;
  constexpr std::size_t kShowInitializers = 5;
  std::cout << "graph: " << graph.name << "\n";
  std::cout << "model_path: " << graph.metadata.model_path << "\n";
  std::cout << "ir_version: " << graph.metadata.ir_version << "\n";
  std::cout << "producer: " << graph.metadata.producer_name << " " << graph.metadata.producer_version << "\n";
  std::cout << "opsets:";
  bool first = true;
  for (const auto& [domain, version] : graph.metadata.opset_imports) {
    std::cout << (first ? " " : ", ") << domain << "=" << version;
    first = false;
  }
  std::cout << "\n\n";

  std::cout << "inputs:\n";
  for (const auto& value : graph.inputs) {
    std::cout << "  - " << value.name << ": " << miniort::FormatTensorInfo(value.info) << "\n";
  }
  std::cout << "outputs:\n";
  for (const auto& value : graph.outputs) {
    std::cout << "  - " << value.name << ": " << miniort::FormatTensorInfo(value.info) << "\n";
  }
  std::cout << "\n";

  // node_count counts ONNX operator nodes, not high-level neural network
  // "layers". A real model often contains many shape/view/constant helper
  // nodes in addition to compute-heavy ops such as Conv.
  std::cout << "node_count: " << graph.nodes.size() << "\n";

  // initializer_count is the number of constant tensors embedded in the
  // model, typically weights, bias tensors, and other fixed parameters.
  std::cout << "initializer_count: " << graph.initializers.size() << "\n";

  // value_info_count is the number of named values for which the ONNX graph
  // explicitly provides dtype/shape metadata via input/output/value_info.
  // This is not necessarily the full count of all intermediate tensors.
  std::cout << "value_info_count: " << graph.value_infos.size() << "\n\n";
  miniort::PrintSessionAssignmentSummary(assignment_summary, std::cout);
  std::cout << "\n";

  std::vector<std::pair<std::string, std::size_t>> histogram(graph.op_type_histogram.begin(),
                                                              graph.op_type_histogram.end());
  std::sort(histogram.begin(), histogram.end(),
            [](const auto& lhs, const auto& rhs) {
              if (lhs.second != rhs.second) {
                return lhs.second > rhs.second;
              }
              return lhs.first < rhs.first;
            });

  std::cout << "op_type_histogram:\n";
  for (const auto& [op_type, count] : histogram) {
    std::cout << "  - " << op_type << ": " << count << "\n";
  }
  std::cout << "\n";
  PrintConvShapeSummary(graph);
  PrintMatMulShapeSummary(graph);

  std::cout << "initializers_preview: first " << kShowInitializers << "\n";
  std::size_t initializer_index = 0;
  for (const auto& [name, value] : graph.initializers) {
    if (initializer_index++ >= kShowInitializers) {
      break;
    }
    std::cout << "  - " << name << ": " << miniort::FormatTensorInfo(value.info) << "\n";
  }
  std::cout << "\n";

  std::cout << "topological_order_preview: first " << kShowTopology << "\n";

  std::size_t shown = 0;
  for (std::size_t i = 0; i < graph.topological_order.size() && shown < kShowTopology; ++i) {
    const auto node_index = graph.topological_order[i];
    const auto& node = graph.nodes[node_index];
    std::cout << "  - [" << i << "] " << node.name << ": " << node.op_type
              << " provider=" << (node.execution_provider.empty() ? "<unset>" : node.execution_provider)
              << " inputs=" << node.inputs.size()
              << " outputs=" << node.outputs.size()
              << " attrs=" << node.attributes.size() << "\n";
    PrintNodeAttributes(node);
    ++shown;
  }

}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    const Options options = ParseArgs(argc, argv);
    miniort::PrintPhaseBanner(std::cout, "phase1", "Inspect Graph Structure",
                              "只看模型图结构，不进入完整推理执行。");
    miniort::PrintPhaseStep(std::cout, 1, 3, "Load ONNX Graph", options.model_path);
    auto graph = miniort::LoadOnnxGraph(options.model_path);
    if (options.graph_opt) {
      graph = miniort::OptimizeGraph(std::move(graph),
                                     {.enable_constant_folding = true,
                                      .enable_dead_node_cleanup = true,
                                      .enable_shape_simplification = true},
                                     nullptr,
                                     nullptr);
    }
    miniort::PrintPhaseStep(std::cout, 2, 3, "Assign Execution Providers",
                            "构造 Session，查看节点当前会落到哪个 provider。");
    const miniort::Session session(graph);
    miniort::PrintPhaseStep(std::cout, 3, 3, "Print Graph Summary",
                            "重点看输入输出、op histogram 和拓扑顺序预览。");
    PrintGraphSummary(session.graph(), session.assignment_summary());
    miniort::PrintPhaseResult(std::cout, "phase1 complete", "你现在看到的是静态图视角。");
    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return EXIT_FAILURE;
  }
}
