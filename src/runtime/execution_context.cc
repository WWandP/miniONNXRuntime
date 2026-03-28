#include "miniort/runtime/execution_context.h"

#include <algorithm>

namespace miniort {

namespace {

Tensor TensorFromValue(const Value& value) {
  Tensor tensor = MakePlaceholderTensor(value.name, value.info);
  tensor.is_initializer = value.info.is_initializer;
  tensor.is_placeholder = !value.data.has_value();
  if (value.data.has_value() && value.data->dtype == "float32" && !value.data->float_data.empty()) {
    tensor.float_data = value.data->float_data;
    tensor.is_placeholder = false;
  }
  if (value.data.has_value() && value.data->dtype == "int64" && !value.data->int64_data.empty()) {
    tensor.int64_data = value.data->int64_data;
    tensor.is_placeholder = false;
  }
  return tensor;
}

}  // namespace

void ExecutionContext::BindTensor(const Tensor& tensor) {
  tensors_[tensor.name] = tensor;
}

void ExecutionContext::BindTensor(Tensor&& tensor) {
  tensors_[tensor.name] = std::move(tensor);
}

bool ExecutionContext::HasTensor(const std::string& name) const {
  return tensors_.contains(name);
}

const Tensor* ExecutionContext::FindTensor(const std::string& name) const {
  const auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

Tensor* ExecutionContext::FindTensor(const std::string& name) {
  const auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

const std::unordered_map<std::string, Tensor>& ExecutionContext::tensors() const {
  return tensors_;
}

void ExecutionContext::LoadInitializers(const Graph& graph) {
  for (const auto& [name, value] : graph.initializers) {
    (void)name;
    BindTensor(TensorFromValue(value));
  }
}

void ExecutionContext::Dump(std::ostream& os, std::size_t limit) const {
  std::vector<const Tensor*> values;
  values.reserve(tensors_.size());
  for (const auto& [name, tensor] : tensors_) {
    (void)name;
    values.push_back(&tensor);
  }

  std::sort(values.begin(), values.end(),
            [](const Tensor* lhs, const Tensor* rhs) { return lhs->name < rhs->name; });

  os << "context_tensors: " << tensors_.size() << "\n";
  for (std::size_t i = 0; i < values.size() && i < limit; ++i) {
    os << "  - " << FormatTensorSummary(*values[i]) << "\n";
  }
  if (values.size() > limit) {
    os << "  - ...\n";
  }
}

}  // namespace miniort
