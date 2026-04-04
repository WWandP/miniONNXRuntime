#pragma once

#include <ostream>
#include <string>
#include <unordered_map>

#include "miniort/model/graph.h"
#include "miniort/runtime/tensor.h"

namespace miniort {

class ExecutionContext {
 public:
  void BindTensor(const Tensor& tensor);
  void BindTensor(Tensor&& tensor);
  bool EraseTensor(const std::string& name);
  bool HasTensor(const std::string& name) const;
  const Tensor* FindTensor(const std::string& name) const;
  Tensor* FindTensor(const std::string& name);
  const std::unordered_map<std::string, Tensor>& tensors() const;
  void LoadInitializers(const Graph& graph);
  void Dump(std::ostream& os, std::size_t limit = 16) const;
  std::vector<float> AcquireFloatBuffer(std::size_t element_count);
  std::vector<std::int64_t> AcquireInt64Buffer(std::size_t element_count);

 private:
  const Tensor* MaterializeInitializer(const std::string& name) const;
  void RecycleTensorStorage(Tensor&& tensor);

  const Graph* graph_{nullptr};
  mutable std::unordered_map<std::string, Tensor> tensors_;
  std::vector<std::vector<float>> float_buffer_pool_;
  std::vector<std::vector<std::int64_t>> int64_buffer_pool_;
};

}  // namespace miniort
