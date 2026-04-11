#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "miniort/model/graph.h"
#include "miniort/runtime/execution_context.h"
#include "miniort/runtime/tensor.h"

namespace miniort {

struct GptCacheTensorBinding {
  std::string prefill_output_name;
  std::string decode_input_name;
  std::string decode_output_name;
};

struct GptCacheBinding {
  std::vector<GptCacheTensorBinding> tensors;
};

enum class GptCacheStateSource {
  kPrefill,
  kDecode,
};

GptCacheBinding BuildCacheBinding(const Graph& prefill_graph, const Graph& decode_graph);
void CollectCacheState(const ExecutionContext& source_context, const GptCacheBinding& binding,
                       GptCacheStateSource source, std::unordered_map<std::string, Tensor>& cache_state);

}  // namespace miniort
