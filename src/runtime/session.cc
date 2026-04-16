#include "miniort/runtime/accelerate_execution_provider.h"
#include "miniort/runtime/cuda_execution_provider.h"
#include "miniort/runtime/cpu_execution_provider.h"
#include "miniort/runtime/session.h"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "miniort/runtime/profiling.h"

namespace miniort {

namespace {

std::vector<std::shared_ptr<const ExecutionProvider>> MakeDefaultProviders() {
  std::vector<std::shared_ptr<const ExecutionProvider>> providers;
#if defined(MINIORT_BUILD_CUDA_EP)
  if (IsCudaExecutionProviderAvailable()) {
    providers.push_back(std::make_shared<CudaExecutionProvider>());
  }
#endif
#if defined(__APPLE__)
  if (IsAccelerateAvailable()) {
    providers.push_back(std::make_shared<AccelerateExecutionProvider>());
  }
#endif
  providers.push_back(std::make_shared<CpuExecutionProvider>());
  return providers;
}

std::string JoinProviderNames(const std::vector<std::shared_ptr<const ExecutionProvider>>& providers) {
  std::ostringstream oss;
  for (std::size_t i = 0; i < providers.size(); ++i) {
    if (i != 0) {
      oss << ",";
    }
    oss << providers[i]->Name();
  }
  return oss.str();
}

std::vector<std::int64_t> ResolveTransposePerm2D(const Node& node, const std::vector<std::int64_t>& input_shape) {
  if (input_shape.size() != 2) {
    return {};
  }
  const auto perm_it = node.attributes.find("perm");
  if (perm_it == node.attributes.end()) {
    return {1, 0};
  }
  const auto& attr = perm_it->second;
  if (attr.kind != AttributeValue::Kind::kInts || attr.ints.size() != 2) {
    return {};
  }
  if (attr.ints[0] == 1 && attr.ints[1] == 0) {
    return {1, 0};
  }
  return {};
}

bool TryFoldInitializerTranspose2D(Graph& graph, Node& node) {
  if (node.op_type != "Transpose" || node.inputs.size() != 1 || node.outputs.empty()) {
    return false;
  }

  const auto init_it = graph.initializers.find(node.inputs[0]);
  if (init_it == graph.initializers.end() || !init_it->second.data.has_value()) {
    return false;
  }

  const auto& source_value = init_it->second;
  const auto& source_data = *source_value.data;
  const auto perm = ResolveTransposePerm2D(node, source_data.shape);
  if (perm.empty()) {
    return false;
  }

  const auto rows = static_cast<std::size_t>(source_data.shape[0]);
  const auto cols = static_cast<std::size_t>(source_data.shape[1]);
  const auto expected_count = rows * cols;
  if (expected_count == 0) {
    return false;
  }

  TensorData folded_data;
  folded_data.dtype = source_data.dtype;
  folded_data.shape = {source_data.shape[1], source_data.shape[0]};

  if (source_data.dtype == "float32") {
    if (source_data.float_data.size() != expected_count) {
      return false;
    }
    folded_data.float_data.resize(expected_count);
    for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
        folded_data.float_data[j * rows + i] = source_data.float_data[i * cols + j];
      }
    }
  } else if (source_data.dtype == "int64") {
    if (source_data.int64_data.size() != expected_count) {
      return false;
    }
    folded_data.int64_data.resize(expected_count);
    for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
        folded_data.int64_data[j * rows + i] = source_data.int64_data[i * cols + j];
      }
    }
  } else {
    return false;
  }

  Value folded_value;
  folded_value.name = node.outputs[0];
  folded_value.info.dtype = folded_data.dtype;
  folded_value.info.is_initializer = true;
  for (const auto dim : folded_data.shape) {
    folded_value.info.shape.push_back(std::to_string(dim));
  }
  folded_value.data = folded_data;
  graph.initializers[folded_value.name] = folded_value;

  if (auto info_it = graph.value_infos.find(folded_value.name); info_it != graph.value_infos.end()) {
    info_it->second.dtype = folded_value.info.dtype;
    info_it->second.is_initializer = true;
    info_it->second.shape = folded_value.info.shape;
  }

  node.op_type = "Constant";
  node.inputs.clear();
  node.attributes.clear();
  AttributeValue value_attr;
  value_attr.kind = AttributeValue::Kind::kTensor;
  value_attr.tensor = folded_data;
  node.attributes.emplace("value", std::move(value_attr));
  return true;
}

void FoldInitializerTransposeNodes(Graph& graph) {
  std::size_t folded_count = 0;
  for (auto& node : graph.nodes) {
    if (TryFoldInitializerTranspose2D(graph, node)) {
      ++folded_count;
    }
  }
  if (folded_count == 0) {
    return;
  }

  const auto transpose_it = graph.op_type_histogram.find("Transpose");
  if (transpose_it != graph.op_type_histogram.end()) {
    if (transpose_it->second > folded_count) {
      transpose_it->second -= folded_count;
    } else {
      graph.op_type_histogram.erase(transpose_it);
    }
  }
  graph.op_type_histogram["Constant"] += folded_count;
}

}  // namespace

Session::Session(Graph graph, SessionOptions options)
    : Session(std::move(graph), MakeDefaultProviders(), options) {}

Session::Session(Graph graph, std::vector<std::shared_ptr<const ExecutionProvider>> providers, SessionOptions options)
    : graph_(std::move(graph)), options_(options), providers_(std::move(providers)) {
  FoldInitializerTransposeNodes(graph_);
  providers_.erase(std::remove(providers_.begin(), providers_.end(), nullptr), providers_.end());
  if (providers_.empty()) {
    providers_ = MakeDefaultProviders();
  }
  for (const auto& provider : providers_) {
    KernelRegistry provider_registry;
    provider->RegisterKernels(provider_registry);
    for (const auto& [op_type, fn] : provider_registry.Entries()) {
      if (!kernel_registry_.Has(op_type)) {
        kernel_registry_.Register(op_type, fn);
      }
    }
  }
  AssignExecutionProviders();
  ValidateAssignmentSummary();

  for (const auto& initializer : graph_.initializers) {
    tensor_is_persistent_[initializer.first] = true;
  }
  for (const auto& input : graph_.inputs) {
    tensor_is_persistent_[input.name] = true;
  }
  for (const auto& output : graph_.outputs) {
    tensor_is_persistent_[output.name] = true;
  }

  for (std::size_t topo_index = 0; topo_index < graph_.topological_order.size(); ++topo_index) {
    const auto node_index = graph_.topological_order[topo_index];
    const auto& node = graph_.nodes[node_index];
    for (const auto& input : node.inputs) {
      if (input.empty() || tensor_is_persistent_.contains(input)) {
        continue;
      }
      tensor_last_use_topo_index_[input] = topo_index;
    }
    for (const auto& output : node.outputs) {
      if (output.empty()) {
        continue;
      }
      tensor_last_use_topo_index_.try_emplace(output, topo_index);
      if (graph_.value_infos.contains(output) &&
          tensor_is_persistent_.contains(output) == false &&
          std::any_of(graph_.outputs.begin(), graph_.outputs.end(),
                      [&output](const Value& value) { return value.name == output; })) {
        tensor_is_persistent_[output] = true;
      }
    }
  }
}

Graph& Session::graph() {
  return graph_;
}

const Graph& Session::graph() const {
  return graph_;
}

KernelRegistry& Session::kernel_registry() {
  return kernel_registry_;
}

const KernelRegistry& Session::kernel_registry() const {
  return kernel_registry_;
}

const SessionAssignmentSummary& Session::assignment_summary() const {
  return assignment_summary_;
}

std::string Session::ResolveExecutionProviderForNode(const Node& node) const {
  switch (options_.provider_assignment_policy) {
    case ProviderAssignmentPolicy::kFirstMatch:
      break;
  }

  for (const auto& provider : providers_) {
    KernelRegistry provider_registry;
    provider->RegisterKernels(provider_registry);
    if (provider_registry.Has(node.op_type)) {
      return std::string(provider->Name());
    }
  }
  return "<unassigned>";
}

void Session::AssignExecutionProviders() {
  assignment_summary_ = {};
  assignment_summary_.total_nodes = graph_.nodes.size();
  std::set<std::string> unassigned_op_types;

  for (auto& node : graph_.nodes) {
    node.execution_provider = ResolveExecutionProviderForNode(node);
    if (node.execution_provider == "<unassigned>") {
      ++assignment_summary_.unassigned_nodes;
      unassigned_op_types.insert(node.op_type);
    } else {
      ++assignment_summary_.assigned_nodes;
    }
    ++assignment_summary_.provider_node_counts[node.execution_provider];
  }

  assignment_summary_.unassigned_op_types.assign(unassigned_op_types.begin(), unassigned_op_types.end());
}

void Session::ValidateAssignmentSummary() const {
  if (!options_.allow_unassigned_nodes && assignment_summary_.unassigned_nodes != 0) {
    std::ostringstream oss;
    oss << "provider assignment left " << assignment_summary_.unassigned_nodes << " unassigned nodes";
    if (!assignment_summary_.unassigned_op_types.empty()) {
      oss << " (op_types=";
      for (std::size_t i = 0; i < assignment_summary_.unassigned_op_types.size(); ++i) {
        if (i != 0) {
          oss << ",";
        }
        oss << assignment_summary_.unassigned_op_types[i];
      }
      oss << ")";
    }
    throw std::runtime_error(oss.str());
  }
}

std::shared_ptr<TensorAllocator> Session::MakeDefaultAllocator() const {
  for (const auto& provider : providers_) {
    if (provider == nullptr) {
      continue;
    }
    if (auto allocator = provider->CreateTensorAllocator(); allocator != nullptr) {
      return allocator;
    }
  }
  return nullptr;
}

RunSummary Session::Run(const std::unordered_map<std::string, Tensor>& feeds, ExecutionContext& context,
                        std::ostream* trace) const {
  TimingMap timings;
  RunSummary summary;
  {
    ScopedTimer session_timer("session.run.total", trace, &timings["session.run.total"]);
    {
      ScopedTimer timer("session.load_initializers", trace, &timings["session.load_initializers"]);
      context.LoadInitializers(graph_);
    }

    {
      ScopedTimer timer("session.bind_feeds", trace, &timings["session.bind_feeds"]);
      for (const auto& [name, tensor] : feeds) {
        (void)name;
        context.BindTensor(tensor);
      }
    }

    {
      ScopedTimer timer("session.bind_placeholders", trace, &timings["session.bind_placeholders"]);
      if (!context.HasAllocator()) {
        context.SetAllocator(MakeDefaultAllocator());
      }
      MaybeBindPlaceholderInputs(context, trace);
    }

    if (trace != nullptr) {
      *trace << "session.run begin\n";
      *trace << "  graph=" << graph_.name << "\n";
      *trace << "  nodes=" << graph_.nodes.size() << "\n";
      *trace << "  providers=" << JoinProviderNames(providers_) << "\n";
      *trace << "  registered_kernels=" << kernel_registry_.RegisteredOps().size() << "\n";
      PrintSessionAssignmentSummary(assignment_summary_, *trace);
    }

    const std::size_t start_node = std::min(options_.start_node, graph_.topological_order.size());
    const std::size_t end_node =
        options_.max_nodes == 0 ? graph_.topological_order.size()
                                : std::min(graph_.topological_order.size(), start_node + options_.max_nodes);

    if (trace != nullptr && start_node != 0) {
      *trace << "session.run start_node=" << start_node << "\n";
    }

    for (std::size_t topo_index = start_node; topo_index < graph_.topological_order.size(); ++topo_index) {
      if (options_.max_nodes != 0 && topo_index >= end_node) {
        if (trace != nullptr) {
          *trace << "session.run stopped early at start_node=" << start_node
                 << " max_nodes=" << options_.max_nodes << "\n";
        }
        break;
      }

      const auto node_index = graph_.topological_order[topo_index];
      const auto& node = graph_.nodes[node_index];
      ++summary.provider_visited_node_counts[node.execution_provider];

      if (trace != nullptr && options_.verbose) {
        *trace << "node[" << topo_index << "] " << node.name << " op=" << node.op_type
               << " provider=" << node.execution_provider << "\n";
        *trace << "  inputs:";
        if (node.inputs.empty()) {
          *trace << " <none>";
        }
        *trace << "\n";
        for (const auto& input_name : node.inputs) {
          *trace << "    - " << input_name;
          if (const auto* tensor = context.FindTensor(input_name); tensor != nullptr) {
            *trace << " -> " << FormatTensorSummary(*tensor);
          } else {
            *trace << " -> <missing>";
          }
          *trace << "\n";
        }
      }

      if (options_.before_node) {
        options_.before_node(topo_index, node, context, trace);
      }

      const auto* kernel = kernel_registry_.Lookup(node.op_type);
      if (kernel != nullptr) {
        const auto node_start = Clock::now();
        try {
          auto* kernel_trace = options_.verbose ? trace : nullptr;
          (*kernel)(node, context, kernel_trace);
          ++summary.executed_nodes;
          ++summary.provider_executed_node_counts[node.execution_provider];
          AddTiming(timings, "kernel." + node.op_type, DurationMs(node_start, Clock::now()));
        } catch (const std::exception& ex) {
          if (!options_.allow_missing_kernels) {
            throw;
          }
          ++summary.skipped_nodes;
          ++summary.provider_skipped_node_counts[node.execution_provider];
        if (trace != nullptr) {
          *trace << "    kernel execution failed for op=" << node.op_type
                 << " provider=" << node.execution_provider
                 << " reason=" << ex.what() << "\n";
        }
          AddTiming(timings, "kernel." + node.op_type, DurationMs(node_start, Clock::now()));
          MaterializeOutputsFromMetadata(node, context, trace, summary);
        }
      } else {
        ++summary.skipped_nodes;
        ++summary.provider_skipped_node_counts[node.execution_provider];
        if (trace != nullptr) {
          *trace << "    no kernel registered for op=" << node.op_type
                 << " provider=" << node.execution_provider << "\n";
        }
        MaterializeOutputsFromMetadata(node, context, trace, summary);
        if (!options_.allow_missing_kernels) {
          throw std::runtime_error("missing kernel for op_type: " + node.op_type);
        }
      }

      if (trace != nullptr && options_.verbose) {
        const auto timing_key = "kernel." + node.op_type;
        const auto timing_it = timings.find(timing_key);
        if (timing_it != timings.end()) {
          *trace << "    kernel_time_ms=" << std::fixed << std::setprecision(3) << timing_it->second << "\n";
        }
      }

      if (trace != nullptr && options_.verbose) {
        *trace << "  outputs:\n";
        for (const auto& output_name : node.outputs) {
          *trace << "    - " << output_name;
          if (const auto* tensor = context.FindTensor(output_name); tensor != nullptr) {
            *trace << " -> " << FormatTensorSummary(*tensor);
          } else {
            *trace << " -> <missing>";
          }
          *trace << "\n";
        }
      }

      if (options_.evict_dead_tensors) {
        EvictDeadTensors(topo_index, node, context, trace, summary);
      }

      if (options_.after_node) {
        options_.after_node(topo_index, node, context, trace);
      }
    }

  }

  if (trace != nullptr) {
    PrintRunSummary(summary, *trace);
    PrintTimingSummary(timings, *trace, "session timing summary");
  }

  return summary;
}

void PrintSessionAssignmentSummary(const SessionAssignmentSummary& summary, std::ostream& os) {
  os << "provider assignment summary\n";
  os << "  total_nodes=" << summary.total_nodes << "\n";
  os << "  assigned_nodes=" << summary.assigned_nodes << "\n";
  os << "  unassigned_nodes=" << summary.unassigned_nodes << "\n";
  if (!summary.provider_node_counts.empty()) {
    os << "  provider_counts:\n";
    std::vector<std::pair<std::string, std::size_t>> counts(summary.provider_node_counts.begin(),
                                                            summary.provider_node_counts.end());
    std::sort(counts.begin(), counts.end(),
              [](const auto& lhs, const auto& rhs) {
                if (lhs.second != rhs.second) {
                  return lhs.second > rhs.second;
                }
                return lhs.first < rhs.first;
              });
    for (const auto& [provider_name, count] : counts) {
      os << "    - " << provider_name << ": " << count << "\n";
    }
  }
  if (!summary.unassigned_op_types.empty()) {
    os << "  unassigned_op_types:\n";
    for (const auto& op_type : summary.unassigned_op_types) {
      os << "    - " << op_type << "\n";
    }
  }
}

void Session::MaybeBindPlaceholderInputs(ExecutionContext& context, std::ostream* trace) const {
  if (!options_.auto_bind_placeholder_inputs) {
    return;
  }

  for (const auto& input : graph_.inputs) {
    if (context.HasTensor(input.name)) {
      continue;
    }
    auto placeholder = MakePlaceholderTensor(input.name, input.info);
    context.BindTensor(placeholder);
    if (trace != nullptr) {
      *trace << "  auto-bound placeholder input " << FormatTensorSummary(placeholder) << "\n";
    }
  }
}

void Session::EvictDeadTensors(std::size_t topo_index, const Node& node, ExecutionContext& context,
                               std::ostream* trace, RunSummary& summary) const {
  std::vector<std::string> candidates;
  candidates.reserve(node.inputs.size() + node.outputs.size());
  candidates.insert(candidates.end(), node.inputs.begin(), node.inputs.end());
  candidates.insert(candidates.end(), node.outputs.begin(), node.outputs.end());

  std::size_t released_tensors = 0;
  for (const auto& name : candidates) {
    if (name.empty()) {
      continue;
    }
    const auto last_use_it = tensor_last_use_topo_index_.find(name);
    if (last_use_it == tensor_last_use_topo_index_.end() || last_use_it->second != topo_index) {
      continue;
    }
    if (tensor_is_persistent_.contains(name)) {
      continue;
    }
    if (context.EraseTensor(name)) {
      ++released_tensors;
      if (trace != nullptr) {
        *trace << "    evicted dead tensor " << name << "\n";
      }
    }
  }

  summary.released_tensors += released_tensors;
}

void Session::MaterializeOutputsFromMetadata(const Node& node, ExecutionContext& context, std::ostream* trace,
                                             RunSummary& summary) const {
  for (const auto& output_name : node.outputs) {
    if (output_name.empty() || context.HasTensor(output_name)) {
      continue;
    }

    const auto info_it = graph_.value_infos.find(output_name);
    Tensor tensor;
    if (info_it != graph_.value_infos.end()) {
      tensor = MakePlaceholderTensor(output_name, info_it->second);
    } else {
      tensor.name = output_name;
      tensor.dtype = "unknown";
      tensor.is_placeholder = true;
    }
    context.BindTensor(tensor);
    ++summary.materialized_outputs;
    ++summary.provider_materialized_output_counts[node.execution_provider];
    if (trace != nullptr) {
      *trace << "    materialized placeholder output " << FormatTensorSummary(tensor) << "\n";
    }
  }
}

void PrintRunSummary(const RunSummary& summary, std::ostream& os) {
  os << "session.run end executed=" << summary.executed_nodes
     << " skipped=" << summary.skipped_nodes
     << " materialized_outputs=" << summary.materialized_outputs
     << " released_tensors=" << summary.released_tensors << "\n";

  std::unordered_map<std::string, bool> seen;
  std::vector<std::string> provider_names;
  provider_names.reserve(summary.provider_visited_node_counts.size() + summary.provider_executed_node_counts.size() +
                         summary.provider_skipped_node_counts.size() +
                         summary.provider_materialized_output_counts.size());

  const auto collect = [&](const auto& counts) {
    for (const auto& [provider_name, count] : counts) {
      (void)count;
      if (!seen.contains(provider_name)) {
        seen[provider_name] = true;
        provider_names.push_back(provider_name);
      }
    }
  };
  collect(summary.provider_visited_node_counts);
  collect(summary.provider_executed_node_counts);
  collect(summary.provider_skipped_node_counts);
  collect(summary.provider_materialized_output_counts);
  std::sort(provider_names.begin(), provider_names.end());

  if (!provider_names.empty()) {
    os << "provider execution summary\n";
    for (const auto& provider_name : provider_names) {
      const auto visited = summary.provider_visited_node_counts.contains(provider_name)
                               ? summary.provider_visited_node_counts.at(provider_name)
                               : 0;
      const auto executed = summary.provider_executed_node_counts.contains(provider_name)
                                ? summary.provider_executed_node_counts.at(provider_name)
                                : 0;
      const auto skipped = summary.provider_skipped_node_counts.contains(provider_name)
                               ? summary.provider_skipped_node_counts.at(provider_name)
                               : 0;
      const auto materialized = summary.provider_materialized_output_counts.contains(provider_name)
                                    ? summary.provider_materialized_output_counts.at(provider_name)
                                    : 0;
      os << "  - " << provider_name
         << ": visited=" << visited
         << " executed=" << executed
         << " skipped=" << skipped
         << " materialized_outputs=" << materialized << "\n";
    }
  }
}

}  // namespace miniort
