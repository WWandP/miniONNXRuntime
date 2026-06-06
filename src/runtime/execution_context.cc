#include "miniort/runtime/execution_context.h"

#include <algorithm>

namespace miniort {

namespace {

Tensor TensorFromValue(const Value& value) {
  constexpr std::size_t kExternalFloatInitializerThreshold = 4096;
  Tensor tensor = MakePlaceholderTensor(value.name, value.info);
  tensor.is_initializer = value.info.is_initializer;
  tensor.is_placeholder = !value.data.has_value();
  if (value.data.has_value() && !value.data->dtype.empty()) {
    tensor.dtype = value.data->dtype;
  }
  if (value.data.has_value() && value.data->dtype == "float32" && !value.data->float_data.empty()) {
    if (value.info.is_initializer && value.data->float_data.size() > kExternalFloatInitializerThreshold) {
      tensor.external_float_data = &value.data->float_data;
    } else {
      tensor.float_data = value.data->float_data;
    }
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
    RecycleTensorStorage(tensor.name, std::move(it->second));
    it->second = tensor;
    return;
  }
  tensors_[tensor.name] = tensor;
}

void ExecutionContext::BindTensor(Tensor&& tensor) {
  if (auto it = tensors_.find(tensor.name); it != tensors_.end()) {
    RecycleTensorStorage(tensor.name, std::move(it->second));
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
  RecycleTensorStorage(name, std::move(it->second));
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
  tensor.is_initializer = true;
  auto [inserted_it, inserted] = tensors_.emplace(name, std::move(tensor));
  (void)inserted;
  return &inserted_it->second;
}

const std::unordered_map<std::string, Tensor>& ExecutionContext::tensors() const {
  return tensors_;
}

const Graph* ExecutionContext::CurrentGraph() const {
  return graph_;
}

void ExecutionContext::LoadInitializers(const Graph& graph) {
  if (graph_ != nullptr && graph_ != &graph) {
    for (auto it = tensors_.begin(); it != tensors_.end();) {
      if (it->second.is_initializer && graph.initializers.contains(it->first)) {
        RecycleTensorStorage(it->first, std::move(it->second));
        it = tensors_.erase(it);
      } else {
        ++it;
      }
    }
  }
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
  tensor.is_initializer = true;
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

std::vector<float> ExecutionContext::AcquireFloatBufferForTensor(const std::string& name,
                                                                 std::size_t element_count) {
  if (const auto it = planned_float_buffers_.find(name); it != planned_float_buffers_.end() &&
                                                         it->second.capacity() >= element_count) {
    std::vector<float> buffer = std::move(it->second);
    planned_float_buffers_.erase(it);
    buffer.clear();
    buffer.reserve(element_count);
    return buffer;
  }
  return AcquireFloatBuffer(element_count);
}

std::vector<std::int64_t> ExecutionContext::AcquireInt64BufferForTensor(const std::string& name,
                                                                        std::size_t element_count) {
  if (const auto it = planned_int64_buffers_.find(name); it != planned_int64_buffers_.end() &&
                                                          it->second.capacity() >= element_count) {
    std::vector<std::int64_t> buffer = std::move(it->second);
    planned_int64_buffers_.erase(it);
    buffer.clear();
    buffer.reserve(element_count);
    return buffer;
  }
  return AcquireInt64Buffer(element_count);
}

void ExecutionContext::SetPlannedBufferReuse(std::unordered_map<std::string, std::string> source_to_target) {
  planned_reuse_source_to_target_ = std::move(source_to_target);
  planned_float_buffers_.clear();
  planned_int64_buffers_.clear();
}

void ExecutionContext::RecycleTensorStorage(const std::string& name, Tensor&& tensor) {
  const auto reuse_it = planned_reuse_source_to_target_.find(name);
  if (reuse_it != planned_reuse_source_to_target_.end()) {
    const auto& target = reuse_it->second;
    if (!tensor.float_data.empty()) {
      planned_float_buffers_[target] = std::move(tensor.float_data);
    }
    if (!tensor.int64_data.empty()) {
      planned_int64_buffers_[target] = std::move(tensor.int64_data);
    }
    return;
  }
  RecycleTensorStorage(std::move(tensor));
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
