#pragma once

#include <cstddef>
#include <functional>
#include <ostream>
#include <string>
#include <unordered_map>

#include "miniort/model/graph.h"
#include "miniort/runtime/execution_context.h"
#include "miniort/runtime/kernel_registry.h"
#include "miniort/runtime/tensor.h"

namespace miniort {

struct SessionOptions {
  bool verbose{false};
  bool allow_missing_kernels{true};
  bool auto_bind_placeholder_inputs{true};
  bool evict_dead_tensors{false};
  std::size_t max_nodes{0};
  std::function<void(std::size_t topo_index, const Node& node, const ExecutionContext& context, std::ostream* trace)>
      before_node;
  std::function<void(std::size_t topo_index, const Node& node, const ExecutionContext& context, std::ostream* trace)>
      after_node;
};

struct RunSummary {
  std::size_t executed_nodes{0};
  std::size_t skipped_nodes{0};
  std::size_t materialized_outputs{0};
  std::size_t released_tensors{0};
};

class Session {
 public:
  Session(Graph graph, SessionOptions options = {});

  Graph& graph();
  const Graph& graph() const;
  KernelRegistry& kernel_registry();
  const KernelRegistry& kernel_registry() const;

  RunSummary Run(const std::unordered_map<std::string, Tensor>& feeds, ExecutionContext& context,
                 std::ostream* trace = nullptr) const;

 private:
  void MaybeBindPlaceholderInputs(ExecutionContext& context, std::ostream* trace) const;
  void EvictDeadTensors(std::size_t topo_index, const Node& node, ExecutionContext& context, std::ostream* trace,
                        RunSummary& summary) const;
  void MaterializeOutputsFromMetadata(const Node& node, ExecutionContext& context, std::ostream* trace,
                                      RunSummary& summary) const;

  Graph graph_;
  KernelRegistry kernel_registry_;
  SessionOptions options_;
  std::unordered_map<std::string, std::size_t> tensor_last_use_topo_index_;
  std::unordered_map<std::string, bool> tensor_is_persistent_;
};

}  // namespace miniort
