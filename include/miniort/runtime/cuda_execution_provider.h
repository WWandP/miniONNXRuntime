#pragma once

#include "miniort/runtime/execution_provider.h"

namespace miniort {

class CudaExecutionProvider final : public ExecutionProvider {
 public:
  std::string_view Name() const override;
  void RegisterKernels(KernelRegistry& registry) const override;
  std::shared_ptr<TensorAllocator> CreateTensorAllocator() const override;
};

bool IsCudaExecutionProviderAvailable();

}  // namespace miniort
