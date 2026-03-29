#pragma once

#include "miniort/runtime/kernel_registry.h"

namespace miniort {

void RegisterBasicKernels(KernelRegistry& registry);
void RegisterElementwiseKernels(KernelRegistry& registry);
void RegisterNnKernels(KernelRegistry& registry);
void RegisterShapeKernels(KernelRegistry& registry);

}  // namespace miniort
