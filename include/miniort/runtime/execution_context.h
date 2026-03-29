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
  bool HasTensor(const std::string& name) const;
  const Tensor* FindTensor(const std::string& name) const;
  Tensor* FindTensor(const std::string& name);
  const std::unordered_map<std::string, Tensor>& tensors() const;
  void LoadInitializers(const Graph& graph);
  void Dump(std::ostream& os, std::size_t limit = 16) const;

 private:
  std::unordered_map<std::string, Tensor> tensors_;
};

}  // namespace miniort
