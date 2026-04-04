#include "miniort/runtime/memory_profile.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <optional>

namespace miniort {

namespace {

constexpr std::size_t kNoIndex = std::numeric_limits<std::size_t>::max();

std::optional<std::int64_t> ParseConcreteDim(const std::string& dim) {
  if (dim.empty() || dim == "?") {
    return std::nullopt;
  }

  std::size_t parsed = 0;
  try {
    const auto value = std::stoll(dim, &parsed);
    if (parsed == dim.size()) {
      return value;
    }
  } catch (...) {
  }
  return std::nullopt;
}

std::size_t BytesPerElement(const std::string& dtype) {
  if (dtype == "float32" || dtype == "int32" || dtype == "uint32") {
    return 4;
  }
  if (dtype == "float64" || dtype == "int64" || dtype == "uint64") {
    return 8;
  }
  if (dtype == "float16" || dtype == "int16" || dtype == "uint16") {
    return 2;
  }
  if (dtype == "bool" || dtype == "uint8" || dtype == "int8") {
    return 1;
  }
  return 0;
}

std::size_t StaticElementCount(const std::vector<std::int64_t>& shape) {
  if (shape.empty()) {
    return 1;
  }

  std::size_t count = 1;
  for (const auto dim : shape) {
    if (dim < 0) {
      return 0;
    }
    count *= static_cast<std::size_t>(dim);
  }
  return count;
}

std::optional<std::vector<std::int64_t>> ResolveConcreteShape(const TensorInfo& info) {
  std::vector<std::int64_t> shape;
  shape.reserve(info.shape.size());
  for (const auto& dim : info.shape) {
    const auto parsed = ParseConcreteDim(dim);
    if (!parsed.has_value()) {
      return std::nullopt;
    }
    shape.push_back(*parsed);
  }
  return shape;
}

std::size_t EstimateFromShapeAndDtype(const std::vector<std::int64_t>& shape, const std::string& dtype) {
  const auto bytes_per_elem = BytesPerElement(dtype);
  if (bytes_per_elem == 0) {
    return 0;
  }
  const auto count = StaticElementCount(shape);
  if (count == 0) {
    return 0;
  }
  return count * bytes_per_elem;
}

bool IsPersistent(const TensorMemoryProfile& profile) {
  return profile.is_initializer || profile.is_input || profile.is_output;
}

std::size_t EstimateTensorDataBytes(const TensorData& data) {
  if (!data.raw_data.empty()) {
    return data.raw_data.size();
  }
  if (data.dtype == "float32") {
    if (!data.float_data.empty()) {
      return data.float_data.size() * sizeof(float);
    }
    return EstimateFromShapeAndDtype(data.shape, data.dtype);
  }
  if (data.dtype == "int64") {
    if (!data.int64_data.empty()) {
      return data.int64_data.size() * sizeof(std::int64_t);
    }
    return EstimateFromShapeAndDtype(data.shape, data.dtype);
  }
  if (!data.double_data.empty()) {
    return data.double_data.size() * sizeof(double);
  }
  if (!data.int32_data.empty()) {
    return data.int32_data.size() * sizeof(std::int32_t);
  }
  if (!data.string_data.empty()) {
    return data.string_data.size() * sizeof(std::string);
  }
  return EstimateFromShapeAndDtype(data.shape, data.dtype);
}

}  // namespace

std::size_t EstimateTensorBytes(const Tensor& tensor) {
  if (!tensor.float_data.empty()) {
    return tensor.float_data.size() * sizeof(float);
  }
  if (!tensor.int64_data.empty()) {
    return tensor.int64_data.size() * sizeof(std::int64_t);
  }
  return EstimateFromShapeAndDtype(tensor.shape, tensor.dtype);
}

std::size_t EstimateTensorBytes(const TensorInfo& info) {
  const auto concrete_shape = ResolveConcreteShape(info);
  if (!concrete_shape.has_value()) {
    return 0;
  }
  return EstimateFromShapeAndDtype(*concrete_shape, info.dtype);
}

std::string FormatBytes(std::size_t bytes) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  const double kb = 1024.0;
  const double mb = kb * 1024.0;
  const double gb = mb * 1024.0;

  if (bytes >= static_cast<std::size_t>(gb)) {
    oss << static_cast<double>(bytes) / gb << " GiB";
  } else if (bytes >= static_cast<std::size_t>(mb)) {
    oss << static_cast<double>(bytes) / mb << " MiB";
  } else if (bytes >= static_cast<std::size_t>(kb)) {
    oss << static_cast<double>(bytes) / kb << " KiB";
  } else {
    oss << bytes << " B";
  }
  return oss.str();
}

MemoryProfile BuildMemoryProfile(const Graph& graph) {
  MemoryProfile profile;
  if (graph.topological_order.empty()) {
    return profile;
  }

  std::unordered_map<std::string, TensorMemoryProfile> profiles_by_name;
  profiles_by_name.reserve(graph.initializers.size() + graph.inputs.size() + graph.outputs.size() + graph.nodes.size());

  const auto ensure_profile = [&](const std::string& name) -> TensorMemoryProfile& {
    auto [it, inserted] = profiles_by_name.emplace(name, TensorMemoryProfile{});
    if (inserted) {
      it->second.name = name;
      it->second.producer_topo_index = kNoIndex;
      it->second.first_use_topo_index = kNoIndex;
      it->second.last_use_topo_index = kNoIndex;
    }
    return it->second;
  };

  for (const auto& [name, value] : graph.initializers) {
    auto& item = ensure_profile(name);
    item.dtype = value.info.dtype;
    item.shape.clear();
    if (value.data.has_value()) {
      item.shape = value.data->shape;
      item.bytes = EstimateTensorDataBytes(*value.data);
    } else {
      item.bytes = EstimateTensorBytes(value.info);
    }
    item.is_initializer = true;
    ++profile.initializer_count;
    profile.initializer_bytes += item.bytes;
  }

  for (const auto& input : graph.inputs) {
    auto& item = ensure_profile(input.name);
    item.dtype = input.info.dtype;
    item.shape = ResolveConcreteShape(input.info).value_or(std::vector<std::int64_t>{});
    item.bytes = EstimateTensorBytes(input.info);
    item.is_input = true;
  }

  for (const auto& output : graph.outputs) {
    auto& item = ensure_profile(output.name);
    item.dtype = output.info.dtype;
    item.shape = ResolveConcreteShape(output.info).value_or(std::vector<std::int64_t>{});
    item.bytes = EstimateTensorBytes(output.info);
    item.is_output = true;
  }

  for (std::size_t topo_index = 0; topo_index < graph.topological_order.size(); ++topo_index) {
    const auto node_index = graph.topological_order[topo_index];
    const auto& node = graph.nodes[node_index];
    for (const auto& output_name : node.outputs) {
      auto& item = ensure_profile(output_name);
      item.producer_topo_index = topo_index;
      if (const auto info_it = graph.value_infos.find(output_name); info_it != graph.value_infos.end()) {
        item.dtype = info_it->second.dtype;
        item.shape = ResolveConcreteShape(info_it->second).value_or(std::vector<std::int64_t>{});
        if (item.bytes == 0) {
          item.bytes = EstimateTensorBytes(info_it->second);
        }
      }
    }
  }

  for (std::size_t topo_index = 0; topo_index < graph.topological_order.size(); ++topo_index) {
    const auto node_index = graph.topological_order[topo_index];
    const auto& node = graph.nodes[node_index];
    for (const auto& input_name : node.inputs) {
      if (input_name.empty()) {
        continue;
      }
      auto& item = ensure_profile(input_name);
      if (item.first_use_topo_index == kNoIndex) {
        item.first_use_topo_index = topo_index;
      }
      item.last_use_topo_index = topo_index;
      if (item.bytes == 0) {
        if (const auto init_it = graph.initializers.find(input_name); init_it != graph.initializers.end()) {
          item.bytes = init_it->second.data.has_value() ? EstimateTensorDataBytes(*init_it->second.data)
                                                        : EstimateTensorBytes(init_it->second.info);
        } else if (const auto input_it = std::find_if(graph.inputs.begin(), graph.inputs.end(),
                                                      [&input_name](const Value& value) { return value.name == input_name; });
                   input_it != graph.inputs.end()) {
          item.bytes = EstimateTensorBytes(input_it->info);
        } else if (const auto info_it = graph.value_infos.find(input_name); info_it != graph.value_infos.end()) {
          item.bytes = EstimateTensorBytes(info_it->second);
        }
      }
    }
  }

  const auto final_topo_index = graph.topological_order.size();
  for (const auto& output : graph.outputs) {
    auto& item = ensure_profile(output.name);
    if (item.first_use_topo_index == kNoIndex) {
      item.first_use_topo_index = final_topo_index;
    }
    item.last_use_topo_index = final_topo_index;
  }

  for (auto& [name, item] : profiles_by_name) {
    (void)name;
    if (item.producer_topo_index != kNoIndex && item.last_use_topo_index == kNoIndex && !IsPersistent(item)) {
      item.last_use_topo_index = item.producer_topo_index;
    }
  }

  std::vector<TensorMemoryProfile> tensors;
  tensors.reserve(profiles_by_name.size());
  for (auto& [name, item] : profiles_by_name) {
    (void)name;
    if (item.bytes == 0) {
      if (const auto init_it = graph.initializers.find(item.name); init_it != graph.initializers.end() && init_it->second.data.has_value()) {
        item.bytes = EstimateTensorDataBytes(*init_it->second.data);
      } else if (const auto info_it = graph.value_infos.find(item.name); info_it != graph.value_infos.end()) {
        item.bytes = EstimateTensorBytes(info_it->second);
      }
    }
    item.is_reusable = !IsPersistent(item) && item.bytes != 0;
    tensors.push_back(std::move(item));
  }

  std::sort(tensors.begin(), tensors.end(),
            [](const TensorMemoryProfile& lhs, const TensorMemoryProfile& rhs) { return lhs.name < rhs.name; });

  profile.tensors = std::move(tensors);
  profile.tensor_to_index.reserve(profile.tensors.size());
  for (std::size_t i = 0; i < profile.tensors.size(); ++i) {
    profile.tensor_to_index[profile.tensors[i].name] = i;
  }

  std::size_t live_bytes = 0;
  std::size_t live_tensors = 0;
  std::unordered_map<std::string, std::size_t> live_set;
  live_set.reserve(profile.tensors.size());
  for (std::size_t topo_index = 0; topo_index < graph.topological_order.size(); ++topo_index) {
    const auto node_index = graph.topological_order[topo_index];
    const auto& node = graph.nodes[node_index];
    for (const auto& input : node.inputs) {
      if (input.empty()) {
        continue;
      }
      const auto it = profile.tensor_to_index.find(input);
      if (it == profile.tensor_to_index.end()) {
        continue;
      }
      const auto& tensor = profile.tensors[it->second];
      if (tensor.bytes == 0) {
        continue;
      }
      if (live_set.emplace(input, 1).second) {
        live_bytes += tensor.bytes;
        ++live_tensors;
      }
    }
    for (const auto& output : node.outputs) {
      const auto it = profile.tensor_to_index.find(output);
      if (it == profile.tensor_to_index.end()) {
        continue;
      }
      const auto& tensor = profile.tensors[it->second];
      if (tensor.bytes == 0) {
        continue;
      }
      if (live_set.emplace(output, 1).second) {
        live_bytes += tensor.bytes;
        ++live_tensors;
      }
    }
    profile.estimated_peak_bytes = std::max(profile.estimated_peak_bytes, live_bytes);
    profile.estimated_peak_live_tensors = std::max(profile.estimated_peak_live_tensors, live_tensors);

    for (auto it = live_set.begin(); it != live_set.end();) {
      const auto idx_it = profile.tensor_to_index.find(it->first);
      if (idx_it == profile.tensor_to_index.end()) {
        ++it;
        continue;
      }
      const auto& tensor = profile.tensors[idx_it->second];
      if (tensor.last_use_topo_index != kNoIndex && tensor.last_use_topo_index <= topo_index && !IsPersistent(tensor)) {
        live_bytes -= tensor.bytes;
        --live_tensors;
        it = live_set.erase(it);
      } else {
        ++it;
      }
    }
  }

  return profile;
}

}  // namespace miniort
