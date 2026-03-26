#include "miniort/model/graph.h"

#include <sstream>

namespace miniort {

std::string FormatShape(const std::vector<std::string>& shape) {
  std::ostringstream oss;
  oss << "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      oss << ", ";
    }
    oss << shape[i];
  }
  oss << "]";
  return oss.str();
}

std::string FormatTensorInfo(const TensorInfo& info) {
  std::ostringstream oss;
  oss << "dtype=" << (info.dtype.empty() ? "unknown" : info.dtype);
  oss << " shape=" << FormatShape(info.shape);
  if (info.is_initializer) {
    oss << " initializer";
  }
  return oss.str();
}

}  // namespace miniort
