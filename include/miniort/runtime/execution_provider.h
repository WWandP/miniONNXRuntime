#pragma once

#include <memory>
#include <string_view>

#include "miniort/runtime/kernel_registry.h"
#include "miniort/runtime/tensor_allocator.h"

namespace miniort {

class ExecutionProvider {
 public:
  virtual ~ExecutionProvider() = default;

  virtual std::string_view Name() const = 0;
  virtual void RegisterKernels(KernelRegistry& registry) const = 0;
  virtual std::shared_ptr<TensorAllocator> CreateTensorAllocator() const = 0;
};

}  // namespace miniort
