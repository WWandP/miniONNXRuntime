#pragma once

#include <cstddef>
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
  std::size_t max_nodes{0};
};

struct RunSummary {
  std::size_t executed_nodes{0};
  std::size_t skipped_nodes{0};
  std::size_t materialized_outputs{0};
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
  void MaterializeOutputsFromMetadata(const Node& node, ExecutionContext& context, std::ostream* trace,
                                      RunSummary& summary) const;

  Graph graph_;
  KernelRegistry kernel_registry_;
  SessionOptions options_;
};

}  // namespace miniort
