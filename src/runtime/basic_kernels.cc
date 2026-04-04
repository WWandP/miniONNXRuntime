#include "builtin_kernel_groups.h"

#include <stdexcept>

#include "kernel_utils.h"

namespace miniort {

void RegisterBasicKernels(KernelRegistry& registry) {
  registry.Register("Constant", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto attr_it = node.attributes.find("value");
    if (attr_it == node.attributes.end() || !attr_it->second.tensor.has_value()) {
      throw std::runtime_error("Constant node missing tensor value attribute: " + node.name);
    }

    const auto& tensor_attr = *attr_it->second.tensor;
    for (const auto& output_name : node.outputs) {
      Tensor tensor = MakeTensorFromDataWithReusedStorage(output_name, tensor_attr, context);
      if (tensor.is_placeholder && (!HasConcreteShape(tensor.shape) || GetElementCount(tensor.shape) == 0)) {
        tensor.is_placeholder = true;
      }
      context.BindTensor(std::move(tensor));
      if (trace != nullptr) {
        *trace << "    kernel Constant produced " << output_name << "\n";
      }
    }
  });
}

}  // namespace miniort
