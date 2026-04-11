#include "miniort/tools/gpt2_cache_binding.h"

#include <sstream>
#include <stdexcept>

namespace miniort {

namespace {

std::string MakeCacheTensorName(const char* prefix, std::size_t layer_index, const char* suffix) {
  std::ostringstream oss;
  oss << prefix << layer_index << "." << suffix;
  return oss.str();
}

void ValidateTensorName(const std::string& actual, const std::string& expected, const char* role,
                        std::size_t layer_index, const char* kind) {
  if (actual == expected) {
    return;
  }

  std::ostringstream oss;
  oss << "KV cache " << role << " mismatch at layer " << layer_index
      << " (" << kind << "): expected '" << expected << "' but got '" << actual << "'";
  throw std::runtime_error(oss.str());
}

const std::string& SelectSourceName(const GptCacheTensorBinding& binding, GptCacheStateSource source) {
  switch (source) {
    case GptCacheStateSource::kPrefill:
      return binding.prefill_output_name;
    case GptCacheStateSource::kDecode:
      return binding.decode_output_name;
  }
  throw std::runtime_error("unknown KV cache state source");
}

}  // namespace

GptCacheBinding BuildCacheBinding(const Graph& prefill_graph, const Graph& decode_graph) {
  if (prefill_graph.inputs.empty() || decode_graph.inputs.empty()) {
    throw std::runtime_error("KV cache models must expose at least one input");
  }
  if (prefill_graph.outputs.size() < 2 || decode_graph.inputs.size() < 2 || decode_graph.outputs.size() < 2) {
    throw std::runtime_error("KV cache models must expose logits plus cache tensors");
  }

  const std::size_t prefill_cache_count = prefill_graph.outputs.size() - 1;
  const std::size_t decode_input_count = decode_graph.inputs.size() - 1;
  const std::size_t decode_output_count = decode_graph.outputs.size() - 1;
  if (prefill_cache_count % 2 != 0 || decode_input_count % 2 != 0 || decode_output_count % 2 != 0) {
    throw std::runtime_error("KV cache tensors must come in key/value pairs");
  }
  if (prefill_cache_count != decode_input_count || decode_input_count != decode_output_count) {
    throw std::runtime_error("KV cache input/output counts do not match");
  }

  GptCacheBinding binding;
  binding.tensors.reserve(prefill_cache_count);
  const std::size_t layer_count = prefill_cache_count / 2;
  for (std::size_t layer_index = 0; layer_index < layer_count; ++layer_index) {
    const std::size_t key_index = 1 + layer_index * 2;
    const std::size_t value_index = key_index + 1;

    const auto expected_prefill_key = MakeCacheTensorName("present.", layer_index, "key");
    const auto expected_prefill_value = MakeCacheTensorName("present.", layer_index, "value");
    const auto expected_decode_input_key = MakeCacheTensorName("past_key_values.", layer_index, "key");
    const auto expected_decode_input_value = MakeCacheTensorName("past_key_values.", layer_index, "value");

    ValidateTensorName(prefill_graph.outputs[key_index].name, expected_prefill_key, "prefill output", layer_index,
                       "key");
    ValidateTensorName(prefill_graph.outputs[value_index].name, expected_prefill_value, "prefill output", layer_index,
                       "value");
    ValidateTensorName(decode_graph.inputs[key_index].name, expected_decode_input_key, "decode input", layer_index,
                       "key");
    ValidateTensorName(decode_graph.inputs[value_index].name, expected_decode_input_value, "decode input",
                       layer_index, "value");
    ValidateTensorName(decode_graph.outputs[key_index].name, expected_prefill_key, "decode output", layer_index,
                       "key");
    ValidateTensorName(decode_graph.outputs[value_index].name, expected_prefill_value, "decode output", layer_index,
                       "value");

    binding.tensors.push_back(
        {.prefill_output_name = expected_prefill_key,
         .decode_input_name = expected_decode_input_key,
         .decode_output_name = expected_prefill_key});
    binding.tensors.push_back(
        {.prefill_output_name = expected_prefill_value,
         .decode_input_name = expected_decode_input_value,
         .decode_output_name = expected_prefill_value});
  }

  return binding;
}

void CollectCacheState(const ExecutionContext& source_context, const GptCacheBinding& binding,
                       GptCacheStateSource source, std::unordered_map<std::string, Tensor>& cache_state) {
  cache_state.clear();
  for (const auto& tensor_binding : binding.tensors) {
    const auto& source_name = SelectSourceName(tensor_binding, source);
    const auto* tensor = source_context.FindTensor(source_name);
    if (tensor == nullptr) {
      std::ostringstream oss;
      oss << "KV cache output was not produced: " << source_name;
      throw std::runtime_error(oss.str());
    }
    auto mapped = *tensor;
    mapped.name = tensor_binding.decode_input_name;
    cache_state[tensor_binding.decode_input_name] = std::move(mapped);
  }
}

}  // namespace miniort
