#include "miniort/runtime/tensor.h"

#include <sstream>

namespace miniort {

namespace {

std::int64_t ToConcreteDim(const std::string& dim) {
  if (dim.empty() || dim == "?") {
    return -1;
  }

  std::size_t parsed_chars = 0;
  try {
    const auto value = std::stoll(dim, &parsed_chars);
    if (parsed_chars == dim.size()) {
      return value;
    }
  } catch (...) {
  }

  return -1;
}

}  // namespace

Tensor MakePlaceholderTensor(std::string name, const TensorInfo& info) {
  Tensor tensor;
  tensor.name = std::move(name);
  tensor.dtype = info.dtype.empty() ? "unknown" : info.dtype;
  tensor.is_placeholder = true;
  tensor.is_initializer = info.is_initializer;
  tensor.shape.reserve(info.shape.size());
  for (const auto& dim : info.shape) {
    tensor.shape.push_back(ToConcreteDim(dim));
  }
  return tensor;
}

std::size_t GetElementCount(const std::vector<std::int64_t>& shape) {
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

bool HasConcreteShape(const std::vector<std::int64_t>& shape) {
  for (const auto dim : shape) {
    if (dim < 0) {
      return false;
    }
  }
  return true;
}

bool HasAnyData(const Tensor& tensor) {
  return !tensor.float_data.empty() || !tensor.int64_data.empty();
}

std::string FormatRuntimeShape(const std::vector<std::int64_t>& shape) {
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      oss << ", ";
    }
    if (shape[i] < 0) {
      oss << "?";
    } else {
      oss << shape[i];
    }
  }
  oss << "]";
  return oss.str();
}

std::string FormatTensorSummary(const Tensor& tensor) {
  std::ostringstream oss;
  oss << tensor.name << " dtype=" << (tensor.dtype.empty() ? "unknown" : tensor.dtype)
      << " shape=" << FormatRuntimeShape(tensor.shape);
  if (tensor.is_initializer) {
    oss << " initializer";
  }
  if (tensor.is_placeholder) {
    oss << " placeholder";
  }
  if (!tensor.float_data.empty()) {
    oss << " float_data=" << tensor.float_data.size();
  }
  if (!tensor.int64_data.empty()) {
    oss << " int64_data=" << tensor.int64_data.size();
  }
  return oss.str();
}

}  // namespace miniort
