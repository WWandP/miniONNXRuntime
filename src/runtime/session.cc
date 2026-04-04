#include "miniort/runtime/builtin_kernels.h"
#include "miniort/runtime/session.h"

#include <algorithm>
#include <iomanip>
#include <unordered_map>
#include <stdexcept>
#include <vector>

#include "miniort/runtime/profiling.h"

namespace miniort {

Session::Session(Graph graph, SessionOptions options)
    : graph_(std::move(graph)), options_(options) {
  RegisterBuiltinKernels(kernel_registry_);

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
      MaybeBindPlaceholderInputs(context, trace);
    }

    if (trace != nullptr) {
      *trace << "session.run begin\n";
      *trace << "  graph=" << graph_.name << "\n";
      *trace << "  nodes=" << graph_.nodes.size() << "\n";
      *trace << "  registered_kernels=" << kernel_registry_.RegisteredOps().size() << "\n";
    }

    for (std::size_t topo_index = 0; topo_index < graph_.topological_order.size(); ++topo_index) {
      if (options_.max_nodes != 0 && topo_index >= options_.max_nodes) {
        if (trace != nullptr) {
          *trace << "session.run stopped early at max_nodes=" << options_.max_nodes << "\n";
        }
        break;
      }

      const auto node_index = graph_.topological_order[topo_index];
      const auto& node = graph_.nodes[node_index];

      if (trace != nullptr && options_.verbose) {
        *trace << "node[" << topo_index << "] " << node.name << " op=" << node.op_type << "\n";
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
          AddTiming(timings, "kernel." + node.op_type, DurationMs(node_start, Clock::now()));
        } catch (const std::exception& ex) {
          if (!options_.allow_missing_kernels) {
            throw;
          }
          ++summary.skipped_nodes;
          if (trace != nullptr) {
            *trace << "    kernel execution failed for op=" << node.op_type
                   << " reason=" << ex.what() << "\n";
          }
          AddTiming(timings, "kernel." + node.op_type, DurationMs(node_start, Clock::now()));
          MaterializeOutputsFromMetadata(node, context, trace, summary);
        }
      } else {
        ++summary.skipped_nodes;
        if (trace != nullptr) {
          *trace << "    no kernel registered for op=" << node.op_type << "\n";
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
    *trace << "session.run end executed=" << summary.executed_nodes
           << " skipped=" << summary.skipped_nodes
           << " materialized_outputs=" << summary.materialized_outputs
           << " released_tensors=" << summary.released_tensors << "\n";
    PrintTimingSummary(timings, *trace, "session timing summary");
  }

  return summary;
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
    if (trace != nullptr) {
      *trace << "    materialized placeholder output " << FormatTensorSummary(tensor) << "\n";
    }
  }
}

}  // namespace miniort
