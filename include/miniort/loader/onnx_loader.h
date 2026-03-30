#pragma once

#include <filesystem>
#include <ostream>

#include "miniort/model/graph.h"

namespace miniort {

Graph LoadOnnxGraph(const std::filesystem::path& model_path, std::ostream* trace = nullptr);

}  // namespace miniort
