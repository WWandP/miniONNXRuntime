#pragma once

#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>

#include "miniort/model/graph.h"
#include "miniort/runtime/tensor.h"
#include "miniort/runtime/tensor_allocator.h"

namespace miniort {

class ExecutionContext {
 public:
  ExecutionContext() = default;
  explicit ExecutionContext(std::shared_ptr<TensorAllocator> allocator);

  void BindTensor(const Tensor& tensor);
  void BindTensor(Tensor&& tensor);
  bool EraseTensor(const std::string& name);
  bool HasTensor(const std::string& name) const;
  const Tensor* FindTensor(const std::string& name) const;
  Tensor* FindTensor(const std::string& name);
  const std::unordered_map<std::string, Tensor>& tensors() const;
  void LoadInitializers(const Graph& graph);
  void Dump(std::ostream& os, std::size_t limit = 16) const;
  void SetAllocator(std::shared_ptr<TensorAllocator> allocator);
  bool HasAllocator() const;
  std::vector<float> AcquireFloatBuffer(std::size_t element_count);
  std::vector<std::int64_t> AcquireInt64Buffer(std::size_t element_count);

 private:
  const Tensor* MaterializeInitializer(const std::string& name) const;
  void RecycleTensorStorage(Tensor&& tensor);

  const Graph* graph_{nullptr};
  std::shared_ptr<TensorAllocator> allocator_;
  mutable std::unordered_map<std::string, Tensor> tensors_;
};

}  // namespace miniort
