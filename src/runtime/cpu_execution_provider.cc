#include "miniort/runtime/cpu_execution_provider.h"

#include "miniort/runtime/builtin_kernels.h"
#include "miniort/runtime/cpu_tensor_allocator.h"

namespace miniort {

std::string_view CpuExecutionProvider::Name() const {
  return "CPU";
}

void CpuExecutionProvider::RegisterKernels(KernelRegistry& registry) const {
  RegisterBuiltinKernels(registry);
}

std::shared_ptr<TensorAllocator> CpuExecutionProvider::CreateTensorAllocator() const {
  return std::make_shared<CpuTensorAllocator>();
}

}  // namespace miniort
