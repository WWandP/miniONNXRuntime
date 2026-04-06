#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "miniort/runtime/tensor.h"

namespace miniort {

class TensorAllocator {
 public:
  virtual ~TensorAllocator() = default;

  virtual std::vector<float> AcquireFloatBuffer(std::size_t element_count) = 0;
  virtual std::vector<std::int64_t> AcquireInt64Buffer(std::size_t element_count) = 0;
  virtual void RecycleTensorStorage(Tensor&& tensor) = 0;
};

}  // namespace miniort
