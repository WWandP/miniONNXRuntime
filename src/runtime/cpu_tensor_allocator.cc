#include "miniort/runtime/cpu_tensor_allocator.h"

namespace miniort {

std::vector<float> CpuTensorAllocator::AcquireFloatBuffer(std::size_t element_count) {
  auto best_it = float_buffer_pool_.end();
  for (auto it = float_buffer_pool_.begin(); it != float_buffer_pool_.end(); ++it) {
    if (it->capacity() < element_count) {
      continue;
    }
    if (best_it == float_buffer_pool_.end() || it->capacity() < best_it->capacity()) {
      best_it = it;
    }
  }
  if (best_it != float_buffer_pool_.end()) {
    std::vector<float> buffer = std::move(*best_it);
    float_buffer_pool_.erase(best_it);
    buffer.clear();
    buffer.reserve(element_count);
    return buffer;
  }

  std::vector<float> buffer;
  buffer.reserve(element_count);
  return buffer;
}

std::vector<std::int64_t> CpuTensorAllocator::AcquireInt64Buffer(std::size_t element_count) {
  auto best_it = int64_buffer_pool_.end();
  for (auto it = int64_buffer_pool_.begin(); it != int64_buffer_pool_.end(); ++it) {
    if (it->capacity() < element_count) {
      continue;
    }
    if (best_it == int64_buffer_pool_.end() || it->capacity() < best_it->capacity()) {
      best_it = it;
    }
  }
  if (best_it != int64_buffer_pool_.end()) {
    std::vector<std::int64_t> buffer = std::move(*best_it);
    int64_buffer_pool_.erase(best_it);
    buffer.clear();
    buffer.reserve(element_count);
    return buffer;
  }

  std::vector<std::int64_t> buffer;
  buffer.reserve(element_count);
  return buffer;
}

void CpuTensorAllocator::RecycleTensorStorage(Tensor&& tensor) {
  if (!tensor.float_data.empty()) {
    float_buffer_pool_.push_back(std::move(tensor.float_data));
  }
  if (!tensor.int64_data.empty()) {
    int64_buffer_pool_.push_back(std::move(tensor.int64_data));
  }
}

}  // namespace miniort
