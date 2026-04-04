#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "miniort/model/graph.h"
#include "miniort/runtime/tensor.h"

namespace miniort {

struct TensorMemoryProfile {
  std::string name;
  std::string dtype;
  std::vector<std::int64_t> shape;
  std::size_t bytes{0};
  std::size_t producer_topo_index{static_cast<std::size_t>(-1)};
  std::size_t first_use_topo_index{static_cast<std::size_t>(-1)};
  std::size_t last_use_topo_index{static_cast<std::size_t>(-1)};
  bool is_initializer{false};
  bool is_input{false};
  bool is_output{false};
  bool is_reusable{false};
};

struct MemoryProfile {
  std::vector<TensorMemoryProfile> tensors;
  std::unordered_map<std::string, std::size_t> tensor_to_index;
  std::size_t initializer_count{0};
  std::size_t initializer_bytes{0};
  std::size_t estimated_peak_bytes{0};
  std::size_t estimated_peak_live_tensors{0};
};

std::size_t EstimateTensorBytes(const Tensor& tensor);
std::size_t EstimateTensorBytes(const TensorInfo& info);
std::string FormatBytes(std::size_t bytes);

MemoryProfile BuildMemoryProfile(const Graph& graph);

}  // namespace miniort
