#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "miniort/model/graph.h"

namespace miniort {

struct Tensor {
  std::string name;
  std::string dtype;
  std::vector<std::int64_t> shape;
  std::vector<float> float_data;
  std::vector<std::int64_t> int64_data;
  bool is_placeholder{false};
  bool is_initializer{false};
};

Tensor MakePlaceholderTensor(std::string name, const TensorInfo& info);
std::size_t GetElementCount(const std::vector<std::int64_t>& shape);
bool HasConcreteShape(const std::vector<std::int64_t>& shape);
bool HasAnyData(const Tensor& tensor);
std::string FormatRuntimeShape(const std::vector<std::int64_t>& shape);
std::string FormatTensorSummary(const Tensor& tensor);

}  // namespace miniort
