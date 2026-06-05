#pragma once

#include "miniort/runtime/execution_provider.h"

namespace miniort {

class ExecutionContext;
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

}  // namespace miniort
