#pragma once

#include <cstdint>
#include <vector>

#include "miniort/runtime/tensor_allocator.h"

namespace miniort {

class CpuTensorAllocator final : public TensorAllocator {
 public:
  std::vector<float> AcquireFloatBuffer(std::size_t element_count) override;
  std::vector<std::int64_t> AcquireInt64Buffer(std::size_t element_count) override;
  void RecycleTensorStorage(Tensor&& tensor) override;

 private:
  std::vector<std::vector<float>> float_buffer_pool_;
  std::vector<std::vector<std::int64_t>> int64_buffer_pool_;
};

}  // namespace miniort
