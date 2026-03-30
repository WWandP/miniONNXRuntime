#pragma once

#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

#include "miniort/model/graph.h"

namespace miniort {

struct GraphOptimizationOptions {
  bool verbose{false};
  bool enable_constant_folding{true};
  bool enable_dead_node_cleanup{true};
  bool enable_shape_simplification{true};
};

struct GraphOptimizationSummary {
  std::size_t nodes_before{0};
  std::size_t nodes_after{0};
  std::size_t passes_run{0};
  std::vector<std::string> applied_passes;
};

Graph OptimizeGraph(Graph graph, const GraphOptimizationOptions& options = {},
                    std::ostream* trace = nullptr, GraphOptimizationSummary* summary = nullptr);

void PrintGraphOptimizationSummary(const GraphOptimizationSummary& summary, std::ostream& os);

}  // namespace miniort

