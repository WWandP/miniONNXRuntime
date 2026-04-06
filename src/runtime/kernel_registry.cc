#include "miniort/runtime/kernel_registry.h"

#include <algorithm>

namespace miniort {

void KernelRegistry::Register(std::string op_type, KernelFn fn) {
  kernels_[std::move(op_type)] = std::move(fn);
}

bool KernelRegistry::Has(std::string_view op_type) const {
  return kernels_.contains(std::string(op_type));
}

const KernelFn* KernelRegistry::Lookup(std::string_view op_type) const {
  const auto it = kernels_.find(std::string(op_type));
  return it == kernels_.end() ? nullptr : &it->second;
}

std::vector<std::pair<std::string, KernelFn>> KernelRegistry::Entries() const {
  std::vector<std::pair<std::string, KernelFn>> entries;
  entries.reserve(kernels_.size());
  for (const auto& [op_type, fn] : kernels_) {
    entries.emplace_back(op_type, fn);
  }
  std::sort(entries.begin(), entries.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
  return entries;
}

std::vector<std::string> KernelRegistry::RegisteredOps() const {
  std::vector<std::string> ops;
  ops.reserve(kernels_.size());
  for (const auto& [op_type, fn] : kernels_) {
    (void)fn;
    ops.push_back(op_type);
  }
  std::sort(ops.begin(), ops.end());
  return ops;
}

}  // namespace miniort
