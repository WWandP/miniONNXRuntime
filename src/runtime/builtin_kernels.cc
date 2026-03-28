#include "miniort/runtime/builtin_kernels.h"

#include "builtin_kernel_groups.h"

namespace miniort {

void RegisterBuiltinKernels(KernelRegistry& registry) {
  RegisterBasicKernels(registry);
  RegisterElementwiseKernels(registry);
  RegisterNnKernels(registry);
  RegisterShapeKernels(registry);
}

}  // namespace miniort
