#pragma once

#include <filesystem>

#include "miniort/model/graph.h"

namespace miniort {

Graph LoadOnnxGraph(const std::filesystem::path& model_path);

}  // namespace miniort
