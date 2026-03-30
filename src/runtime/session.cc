#include "miniort/runtime/builtin_kernels.h"
#include "miniort/runtime/session.h"

#include <iomanip>
#include <unordered_map>
#include <stdexcept>

#include "miniort/runtime/profiling.h"

namespace miniort {

Session::Session(Graph graph, SessionOptions options)
    : graph_(std::move(graph)), options_(options) {
  RegisterBuiltinKernels(kernel_registry_);
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
    }

  }

  if (trace != nullptr) {
    *trace << "session.run end executed=" << summary.executed_nodes
           << " skipped=" << summary.skipped_nodes
           << " materialized_outputs=" << summary.materialized_outputs << "\n";
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
