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

ExecutionContext::ExecutionContext(std::shared_ptr<TensorAllocator> allocator)
    : allocator_(std::move(allocator)) {}

void ExecutionContext::BindTensor(const Tensor& tensor) {
  if (auto it = tensors_.find(tensor.name); it != tensors_.end()) {
    RecycleTensorStorage(std::move(it->second));
    it->second = tensor;
    return;
  }
  tensors_[tensor.name] = tensor;
}

void ExecutionContext::BindTensor(Tensor&& tensor) {
  if (auto it = tensors_.find(tensor.name); it != tensors_.end()) {
    RecycleTensorStorage(std::move(it->second));
    it->second = std::move(tensor);
    return;
  }
  tensors_[tensor.name] = std::move(tensor);
}

bool ExecutionContext::EraseTensor(const std::string& name) {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    return false;
  }
  RecycleTensorStorage(std::move(it->second));
  tensors_.erase(it);
  return true;
}

bool ExecutionContext::HasTensor(const std::string& name) const {
  if (tensors_.contains(name)) {
    return true;
  }
  return graph_ != nullptr && graph_->initializers.contains(name);
}

const Tensor* ExecutionContext::FindTensor(const std::string& name) const {
  const auto it = tensors_.find(name);
  if (it != tensors_.end()) {
    return &it->second;
  }
  return MaterializeInitializer(name);
}

Tensor* ExecutionContext::FindTensor(const std::string& name) {
  const auto it = tensors_.find(name);
  if (it != tensors_.end()) {
    return &it->second;
  }
  if (graph_ == nullptr) {
    return nullptr;
  }
  const auto init_it = graph_->initializers.find(name);
  if (init_it == graph_->initializers.end()) {
    return nullptr;
  }
  auto tensor = TensorFromValue(init_it->second);
  auto [inserted_it, inserted] = tensors_.emplace(name, std::move(tensor));
  (void)inserted;
  return &inserted_it->second;
}

const std::unordered_map<std::string, Tensor>& ExecutionContext::tensors() const {
  return tensors_;
}

void ExecutionContext::LoadInitializers(const Graph& graph) {
  graph_ = &graph;
}

void ExecutionContext::SetAllocator(std::shared_ptr<TensorAllocator> allocator) {
  allocator_ = std::move(allocator);
}

bool ExecutionContext::HasAllocator() const {
  return allocator_ != nullptr;
}

const Tensor* ExecutionContext::MaterializeInitializer(const std::string& name) const {
  if (graph_ == nullptr) {
    return nullptr;
  }
  const auto init_it = graph_->initializers.find(name);
  if (init_it == graph_->initializers.end()) {
    return nullptr;
  }

  const auto it = tensors_.find(name);
  if (it != tensors_.end()) {
    return &it->second;
  }

  auto tensor = TensorFromValue(init_it->second);
  auto [inserted_it, inserted] = tensors_.emplace(name, std::move(tensor));
  (void)inserted;
  return &inserted_it->second;
}

std::vector<float> ExecutionContext::AcquireFloatBuffer(std::size_t element_count) {
  if (allocator_ != nullptr) {
    return allocator_->AcquireFloatBuffer(element_count);
  }
  std::vector<float> buffer;
  buffer.reserve(element_count);
  return buffer;
}

std::vector<std::int64_t> ExecutionContext::AcquireInt64Buffer(std::size_t element_count) {
  if (allocator_ != nullptr) {
    return allocator_->AcquireInt64Buffer(element_count);
  }
  std::vector<std::int64_t> buffer;
  buffer.reserve(element_count);
  return buffer;
}

void ExecutionContext::RecycleTensorStorage(Tensor&& tensor) {
  if (allocator_ != nullptr) {
    allocator_->RecycleTensorStorage(std::move(tensor));
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
