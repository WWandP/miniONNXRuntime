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
  std::size_t show_topology{20};
  std::size_t show_initializers{10};
  std::string filter_op;
};

Options ParseArgs(int argc, char* argv[]) {
  if (argc < 2) {
    throw std::runtime_error(
        "usage: miniort_inspect <model.onnx> [--show-topology N] [--show-initializers N] [--filter-op OpType]");
  }

  Options options;
  options.model_path = argv[1];

  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--show-topology" && i + 1 < argc) {
      options.show_topology = static_cast<std::size_t>(std::stoul(argv[++i]));
      continue;
    }
    if (arg == "--show-initializers" && i + 1 < argc) {
      options.show_initializers = static_cast<std::size_t>(std::stoul(argv[++i]));
      continue;
    }
    if (arg == "--filter-op" && i + 1 < argc) {
      options.filter_op = argv[++i];
      continue;
    }
    throw std::runtime_error("unknown argument: " + arg);
  }

  return options;
}

void PrintGraphSummary(const miniort::Graph& graph, std::size_t show_topology,
                       std::size_t show_initializers, const std::string& filter_op) {
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

  std::cout << "initializers_preview: first " << show_initializers << "\n";
  std::size_t initializer_index = 0;
  for (const auto& [name, value] : graph.initializers) {
    if (initializer_index++ >= show_initializers) {
      break;
    }
    std::cout << "  - " << name << ": " << miniort::FormatTensorInfo(value.info) << "\n";
  }
  std::cout << "\n";

  std::cout << "topological_order_preview: first " << show_topology;
  if (!filter_op.empty()) {
    std::cout << " (filtered by op_type=" << filter_op << ")";
  }
  std::cout << "\n";

  std::size_t shown = 0;
  for (std::size_t i = 0; i < graph.topological_order.size() && shown < show_topology; ++i) {
    const auto node_index = graph.topological_order[i];
    const auto& node = graph.nodes[node_index];
    if (!filter_op.empty() && node.op_type != filter_op) {
      continue;
    }
    std::cout << "  - [" << i << "] " << node.name << ": " << node.op_type
              << " inputs=" << node.inputs.size()
              << " outputs=" << node.outputs.size()
              << " attrs=" << node.attributes.size() << "\n";
    PrintNodeAttributes(node);
    ++shown;
  }

  if (shown == 0 && !filter_op.empty()) {
    std::cout << "  - no nodes matched the requested op type\n";
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    const Options options = ParseArgs(argc, argv);
    const auto graph = miniort::LoadOnnxGraph(options.model_path);
    PrintGraphSummary(graph, options.show_topology, options.show_initializers, options.filter_op);
    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return EXIT_FAILURE;
  }
}
