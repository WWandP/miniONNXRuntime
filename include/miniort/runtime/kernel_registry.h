#pragma once

#include <functional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "miniort/model/graph.h"
#include "miniort/runtime/execution_context.h"

namespace miniort {

using KernelFn = std::function<void(const Node& node, ExecutionContext& context, std::ostream* trace)>;

class KernelRegistry {
 public:
  void Register(std::string op_type, KernelFn fn);
  bool Has(std::string_view op_type) const;
  const KernelFn* Lookup(std::string_view op_type) const;
  std::vector<std::pair<std::string, KernelFn>> Entries() const;
  std::vector<std::string> RegisteredOps() const;

 private:
  std::unordered_map<std::string, KernelFn> kernels_;
};

}  // namespace miniort
