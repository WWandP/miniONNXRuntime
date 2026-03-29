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
      Tensor tensor;
      tensor.name = output_name;
      tensor.dtype = tensor_attr.dtype.empty() ? "unknown" : tensor_attr.dtype;
      tensor.shape = tensor_attr.shape;
      tensor.is_placeholder = false;
      if (tensor_attr.dtype == "float32" && !tensor_attr.float_data.empty()) {
        tensor.float_data = tensor_attr.float_data;
      } else if (tensor_attr.dtype == "int64" && !tensor_attr.int64_data.empty()) {
        tensor.int64_data = tensor_attr.int64_data;
      } else if (!tensor_attr.raw_data.empty()) {
        tensor.is_placeholder = true;
      } else if (!HasConcreteShape(tensor.shape) || GetElementCount(tensor.shape) == 0) {
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
