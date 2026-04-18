#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "miniort/model/graph.h"
#include "miniort/runtime/execution_context.h"
#include "miniort/runtime/execution_provider.h"
#include "miniort/runtime/kernel_registry.h"
#include "miniort/runtime/tensor.h"

namespace miniort {

enum class ProviderAssignmentPolicy {
  kFirstMatch,
};

struct SessionOptions {
  bool verbose{false};
  bool allow_missing_kernels{true};
  bool allow_unassigned_nodes{true};
  bool auto_bind_placeholder_inputs{true};
  bool evict_dead_tensors{false};
  std::size_t start_node{0};
  std::size_t max_nodes{0};
  ProviderAssignmentPolicy provider_assignment_policy{ProviderAssignmentPolicy::kFirstMatch};
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
  std::unordered_map<std::string, std::size_t> provider_visited_node_counts;
  std::unordered_map<std::string, std::size_t> provider_executed_node_counts;
  std::unordered_map<std::string, std::size_t> provider_skipped_node_counts;
  std::unordered_map<std::string, std::size_t> provider_materialized_output_counts;
};

struct SessionAssignmentSummary {
  std::size_t total_nodes{0};
  std::size_t assigned_nodes{0};
  std::size_t unassigned_nodes{0};
  std::unordered_map<std::string, std::size_t> provider_node_counts;
  std::vector<std::string> unassigned_op_types;
};

class Session {
 public:
  Session(Graph graph, SessionOptions options = {});
  Session(Graph graph, std::vector<std::shared_ptr<const ExecutionProvider>> providers, SessionOptions options = {});

  Graph& graph();
  const Graph& graph() const;
  KernelRegistry& kernel_registry();
  const KernelRegistry& kernel_registry() const;
  const SessionAssignmentSummary& assignment_summary() const;

  RunSummary Run(const std::unordered_map<std::string, Tensor>& feeds, ExecutionContext& context,
                 std::ostream* trace = nullptr) const;

 private:
  void AssignExecutionProviders();
  std::string ResolveExecutionProviderForNode(const Node& node) const;
  void ValidateAssignmentSummary() const;
  std::shared_ptr<TensorAllocator> MakeDefaultAllocator() const;
  void MaybeBindPlaceholderInputs(ExecutionContext& context, std::ostream* trace) const;
  void EvictDeadTensors(std::size_t topo_index, const Node& node, ExecutionContext& context, std::ostream* trace,
                        RunSummary& summary) const;
  void MaterializeOutputsFromMetadata(const Node& node, ExecutionContext& context, std::ostream* trace,
                                      RunSummary& summary) const;

  Graph graph_;
  KernelRegistry kernel_registry_;
  SessionOptions options_;
  std::vector<std::shared_ptr<const ExecutionProvider>> providers_;
  std::vector<std::unordered_set<std::string>> provider_supported_ops_;
  SessionAssignmentSummary assignment_summary_;
  std::unordered_map<std::string, std::size_t> tensor_last_use_topo_index_;
  std::unordered_map<std::string, bool> tensor_is_persistent_;
};

void PrintSessionAssignmentSummary(const SessionAssignmentSummary& summary, std::ostream& os);
void PrintRunSummary(const RunSummary& summary, std::ostream& os);

}  // namespace miniort
