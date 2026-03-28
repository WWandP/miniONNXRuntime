#pragma once

#include <filesystem>
#include <ostream>
#include <string>

#include "miniort/model/graph.h"
#include "miniort/runtime/tensor.h"

namespace miniort {

Tensor LoadImageAsNchwTensor(const std::filesystem::path& image_path, const std::string& input_name,
                             const TensorInfo& input_info, std::ostream* trace = nullptr);

}  // namespace miniort
