#include "miniort/tools/gpt2_cache_binding.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace miniort {

namespace {

enum class CacheTensorKind {
  kKey,
  kValue,
};

struct CacheTensorSlot {
  std::size_t layer_index{0};
  CacheTensorKind kind{CacheTensorKind::kKey};
};

struct CacheTensorEndpoint {
  CacheTensorSlot slot;
  std::string name;
};

std::vector<std::string> TokenizeName(std::string_view name) {
  std::vector<std::string> tokens;
  std::string current;
  for (const unsigned char raw_ch : name) {
    const auto ch = static_cast<char>(raw_ch);
    if (std::isalnum(raw_ch) != 0) {
      current.push_back(static_cast<char>(std::tolower(raw_ch)));
      continue;
    }
    if (!current.empty()) {
      tokens.push_back(current);
      current.clear();
    }
  }
  if (!current.empty()) {
    tokens.push_back(current);
  }
  return tokens;
}

std::optional<CacheTensorKind> DetectKindFromTokens(const std::vector<std::string>& tokens) {
  for (auto it = tokens.rbegin(); it != tokens.rend(); ++it) {
    if (*it == "key" || *it == "k") {
      return CacheTensorKind::kKey;
    }
    if (*it == "value" || *it == "values" || *it == "v") {
      return CacheTensorKind::kValue;
    }
  }
  return std::nullopt;
}

std::optional<std::size_t> DetectLayerIndexFromTokens(const std::vector<std::string>& tokens) {
  for (const auto& token : tokens) {
    if (token.empty()) {
      continue;
    }
    if (std::all_of(token.begin(), token.end(), [](unsigned char ch) { return std::isdigit(ch) != 0; })) {
      return static_cast<std::size_t>(std::stoull(token));
    }
  }
  return std::nullopt;
}

std::optional<CacheTensorEndpoint> TryParseCacheTensorEndpoint(const std::string& name) {
  const auto tokens = TokenizeName(name);
  if (tokens.empty()) {
    return std::nullopt;
  }

  const auto kind = DetectKindFromTokens(tokens);
  const auto layer_index = DetectLayerIndexFromTokens(tokens);
  if (!kind.has_value() || !layer_index.has_value()) {
    return std::nullopt;
  }

  CacheTensorEndpoint endpoint;
  endpoint.slot.layer_index = *layer_index;
  endpoint.slot.kind = *kind;
  endpoint.name = name;
  return endpoint;
}

std::string SlotKey(const CacheTensorSlot& slot) {
  std::ostringstream oss;
  oss << slot.layer_index << ":" << (slot.kind == CacheTensorKind::kKey ? "key" : "value");
  return oss.str();
}

using SlotMap = std::map<std::pair<std::size_t, CacheTensorKind>, std::string>;

SlotMap BuildEndpointSlotMap(const std::vector<Value>& values, const char* role_name) {
  SlotMap slots;
  for (const auto& value : values) {
    const auto parsed = TryParseCacheTensorEndpoint(value.name);
    if (!parsed.has_value()) {
      continue;
    }
    const auto key = std::make_pair(parsed->slot.layer_index, parsed->slot.kind);
    auto [it, inserted] = slots.emplace(key, parsed->name);
    if (!inserted) {
      std::ostringstream oss;
      oss << "KV cache " << role_name << " has duplicate tensor slot for " << SlotKey(parsed->slot)
          << " ('" << it->second << "' vs '" << parsed->name << "')";
      throw std::runtime_error(oss.str());
    }
  }
  return slots;
}

std::vector<std::size_t> CollectLayersWithBothKeyAndValue(const SlotMap& slots) {
  std::unordered_set<std::size_t> key_layers;
  std::unordered_set<std::size_t> value_layers;
  for (const auto& [slot, name] : slots) {
    (void)name;
    if (slot.second == CacheTensorKind::kKey) {
      key_layers.insert(slot.first);
    } else {
      value_layers.insert(slot.first);
    }
  }
  std::vector<std::size_t> layers;
  for (const auto layer : key_layers) {
    if (value_layers.contains(layer)) {
      layers.push_back(layer);
    }
  }
  std::sort(layers.begin(), layers.end());
  return layers;
}

std::string LookupRequired(const SlotMap& slots, std::size_t layer_index, CacheTensorKind kind,
                           const char* role_name) {
  const auto it = slots.find({layer_index, kind});
  if (it != slots.end()) {
    return it->second;
  }

  std::ostringstream oss;
  oss << "KV cache " << role_name << " missing tensor for layer " << layer_index << " "
      << (kind == CacheTensorKind::kKey ? "key" : "value");
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

  const auto prefill_slots = BuildEndpointSlotMap(prefill_graph.outputs, "prefill outputs");
  const auto decode_input_slots = BuildEndpointSlotMap(decode_graph.inputs, "decode inputs");
  const auto decode_output_slots = BuildEndpointSlotMap(decode_graph.outputs, "decode outputs");
  if (prefill_slots.empty() || decode_input_slots.empty() || decode_output_slots.empty()) {
    throw std::runtime_error("KV cache tensors were not detected from model inputs/outputs");
  }

  GptCacheBinding binding;
  const auto prefill_layers = CollectLayersWithBothKeyAndValue(prefill_slots);
  const auto decode_input_layers = CollectLayersWithBothKeyAndValue(decode_input_slots);
  const auto decode_output_layers = CollectLayersWithBothKeyAndValue(decode_output_slots);
  if (prefill_layers.empty() || decode_input_layers.empty() || decode_output_layers.empty()) {
    std::ostringstream oss;
    oss << "KV cache tensors must expose key/value pairs for at least one layer"
        << " (prefill_slots=" << prefill_slots.size() << " prefill_layers=" << prefill_layers.size()
        << ", decode_input_slots=" << decode_input_slots.size() << " decode_input_layers=" << decode_input_layers.size()
        << ", decode_output_slots=" << decode_output_slots.size() << " decode_output_layers="
        << decode_output_layers.size() << ")";
    throw std::runtime_error(oss.str());
  }

  std::vector<std::size_t> common_layers;
  for (const auto layer : prefill_layers) {
    if (std::binary_search(decode_input_layers.begin(), decode_input_layers.end(), layer) &&
        std::binary_search(decode_output_layers.begin(), decode_output_layers.end(), layer)) {
      common_layers.push_back(layer);
    }
  }
  if (common_layers.empty()) {
    throw std::runtime_error("KV cache mismatch: no common layers found across prefill/decode tensors");
  }

  binding.tensors.reserve(common_layers.size() * 2);
  for (const auto layer_index : common_layers) {
    const auto prefill_key = LookupRequired(prefill_slots, layer_index, CacheTensorKind::kKey, "prefill outputs");
    const auto prefill_value = LookupRequired(prefill_slots, layer_index, CacheTensorKind::kValue, "prefill outputs");
    const auto decode_input_key = LookupRequired(decode_input_slots, layer_index, CacheTensorKind::kKey, "decode inputs");
    const auto decode_input_value =
        LookupRequired(decode_input_slots, layer_index, CacheTensorKind::kValue, "decode inputs");
    const auto decode_output_key =
        LookupRequired(decode_output_slots, layer_index, CacheTensorKind::kKey, "decode outputs");
    const auto decode_output_value =
        LookupRequired(decode_output_slots, layer_index, CacheTensorKind::kValue, "decode outputs");

    binding.tensors.push_back(
        {.prefill_output_name = prefill_key, .decode_input_name = decode_input_key, .decode_output_name = decode_output_key});
    binding.tensors.push_back({.prefill_output_name = prefill_value,
                               .decode_input_name = decode_input_value,
                               .decode_output_name = decode_output_value});
  }

  return binding;
}

void CollectCacheState(const ExecutionContext& source_context, const GptCacheBinding& binding,
                       GptCacheStateSource source, std::unordered_map<std::string, Tensor>& cache_state) {
  cache_state.reserve(binding.tensors.size());
  for (const auto& tensor_binding : binding.tensors) {
    const auto& source_name = SelectSourceName(tensor_binding, source);
    const auto* tensor = source_context.FindTensor(source_name);
    if (tensor == nullptr) {
      std::ostringstream oss;
      oss << "KV cache output was not produced: " << source_name;
      throw std::runtime_error(oss.str());
    }
    auto& mapped = cache_state[tensor_binding.decode_input_name];
    mapped = *tensor;
    mapped.name = tensor_binding.decode_input_name;
  }
}

}  // namespace miniort
