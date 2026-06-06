#pragma once

#include "miniort/runtime/execution_provider.h"

namespace miniort {

class ExecutionContext;
struct Graph;
struct Node;

class CudaExecutionProvider final : public ExecutionProvider {
 public:
  std::string_view Name() const override;
  void RegisterKernels(KernelRegistry& registry) const override;
  std::shared_ptr<TensorAllocator> CreateTensorAllocator() const override;
};

bool IsCudaExecutionProviderAvailable();
void MaterializeCudaInputsForNode(const Node& node, ExecutionContext& context);
void MaterializeCudaTensor(const std::string& name, ExecutionContext& context);
std::size_t PrepareCudaInitializersForGraph(const Graph& graph, ExecutionContext& context);

}  // namespace miniort
