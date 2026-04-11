#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "miniort/loader/onnx_loader.h"
#include "miniort/runtime/cpu_execution_provider.h"
#include "miniort/runtime/execution_context.h"
#include "miniort/runtime/session.h"
#include "miniort/tools/gpt2_tokenizer.h"

namespace {

struct Options {
  std::string model_path;
  std::string kv_cache_prefill_model_path;
  std::string kv_cache_decode_model_path;
  std::string tokens;
  std::string prompt;
  std::string model_dir;
  std::size_t start_node{0};
  std::size_t max_nodes{0};
  std::size_t top_k{5};
  std::size_t generate{0};
  bool kv_cache{false};
  bool strict{false};
  bool cpu_only{false};
  bool verbose{false};
  bool quiet{false};
};

std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open file: " + path.string());
  }
  std::ostringstream oss;
  oss << input.rdbuf();
  return oss.str();
}

std::string Trim(std::string_view value) {
  std::size_t begin = 0;
  while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
    ++begin;
  }
  std::size_t end = value.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
    --end;
  }
  return std::string(value.substr(begin, end - begin));
}

std::string Unquote(std::string value) {
  if (value.size() >= 2 &&
      ((value.front() == '"' && value.back() == '"') || (value.front() == '\'' && value.back() == '\''))) {
    return value.substr(1, value.size() - 2);
  }
  return value;
}

bool ParseBool(const std::string& value) {
  if (value == "1" || value == "true" || value == "on" || value == "yes") {
    return true;
  }
  if (value == "0" || value == "false" || value == "off" || value == "no") {
    return false;
  }
  throw std::runtime_error("invalid boolean value: " + value);
}

void ApplyConfigEntry(Options& options, const std::string& key, const std::string& value,
                      const std::filesystem::path& base_dir) {
  if (key == "model" || key == "model_path") {
    options.model_path = value;
    return;
  }
  if (key == "kv_cache_prefill_model" || key == "kv_cache_prefill_model_path") {
    options.kv_cache_prefill_model_path = value;
    return;
  }
  if (key == "kv_cache_decode_model" || key == "kv_cache_decode_model_path") {
    options.kv_cache_decode_model_path = value;
    return;
  }
  if (key == "model_dir") {
    options.model_dir = value;
    return;
  }
  if (key == "tokens") {
    options.tokens = value;
    return;
  }
  if (key == "prompt") {
    options.prompt = value;
    return;
  }
  if (key == "prompt_file") {
    const auto prompt_path = base_dir / value;
    options.prompt = ReadTextFile(prompt_path);
    return;
  }
  if (key == "start_node") {
    options.start_node = static_cast<std::size_t>(std::stoull(value));
    return;
  }
  if (key == "max_nodes") {
    options.max_nodes = static_cast<std::size_t>(std::stoull(value));
    return;
  }
  if (key == "top_k") {
    options.top_k = static_cast<std::size_t>(std::stoull(value));
    return;
  }
  if (key == "generate" || key == "max_new_tokens") {
    options.generate = static_cast<std::size_t>(std::stoull(value));
    return;
  }
  if (key == "kv_cache") {
    options.kv_cache = ParseBool(value);
    return;
  }
  if (key == "strict") {
    options.strict = ParseBool(value);
    return;
  }
  if (key == "cpu_only") {
    options.cpu_only = ParseBool(value);
    return;
  }
  if (key == "verbose") {
    options.verbose = ParseBool(value);
    return;
  }
  if (key == "quiet") {
    options.quiet = ParseBool(value);
    return;
  }
  throw std::runtime_error("unknown config key: " + key);
}

void LoadConfigFile(Options& options, const std::filesystem::path& config_path) {
  std::ifstream input(config_path);
  if (!input) {
    throw std::runtime_error("failed to open config: " + config_path.string());
  }
  const auto base_dir = config_path.parent_path();
  std::string line;
  std::size_t line_no = 0;
  while (std::getline(input, line)) {
    ++line_no;
    const auto trimmed = Trim(line);
    if (trimmed.empty() || trimmed[0] == '#') {
      continue;
    }
    const auto eq_pos = trimmed.find('=');
    if (eq_pos == std::string::npos) {
      throw std::runtime_error("invalid config line " + std::to_string(line_no) + ": missing '='");
    }
    const auto key = Trim(std::string_view(trimmed).substr(0, eq_pos));
    const auto value = Unquote(Trim(std::string_view(trimmed).substr(eq_pos + 1)));
    ApplyConfigEntry(options, key, value, base_dir);
  }
}

std::vector<std::int64_t> ParseTokenIds(const std::string& tokens_arg) {
  if (tokens_arg.empty()) {
    throw std::runtime_error("--tokens requires a comma-separated token list");
  }

  std::vector<std::int64_t> token_ids;
  std::stringstream ss(tokens_arg);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token.empty()) {
      continue;
    }
    token_ids.push_back(std::stoll(token));
  }

  if (token_ids.empty()) {
    throw std::runtime_error("--tokens did not contain any token ids");
  }

  return token_ids;
}

miniort::Tensor MakeTokenTensor(const miniort::Value& input, const std::vector<std::int64_t>& token_ids) {
  miniort::Tensor tensor;
  tensor.name = input.name;
  tensor.dtype = "int64";
  tensor.int64_data = token_ids;
  tensor.shape = {1, static_cast<std::int64_t>(tensor.int64_data.size())};
  return tensor;
}

Options ParseArgs(int argc, char* argv[]) {
  if (argc < 2) {
    throw std::runtime_error(
        "usage: miniort_run_gpt [model.onnx] [--config path] (--tokens 1,2,3 | --prompt \"text\") [--model-dir path] [--start-node N] [--max-nodes N] [--top-k N] [--generate N] [--kv-cache] [--kv-cache-prefill-model path] [--kv-cache-decode-model path] [--strict] [--cpu-only] [--verbose] [--quiet]");
  }

  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (!arg.empty() && arg[0] != '-') {
      if (!options.model_path.empty()) {
        throw std::runtime_error("multiple model paths provided");
      }
      options.model_path = arg;
      continue;
    }
    if (arg == "--config" && i + 1 < argc) {
      LoadConfigFile(options, argv[++i]);
      continue;
    }
    if (arg == "--tokens" && i + 1 < argc) {
      options.tokens = argv[++i];
      continue;
    }
    if (arg == "--prompt" && i + 1 < argc) {
      options.prompt = argv[++i];
      continue;
    }
    if (arg == "--model-dir" && i + 1 < argc) {
      options.model_dir = argv[++i];
      continue;
    }
    if (arg == "--start-node" && i + 1 < argc) {
      options.start_node = static_cast<std::size_t>(std::stoull(argv[++i]));
      continue;
    }
    if (arg == "--max-nodes" && i + 1 < argc) {
      options.max_nodes = static_cast<std::size_t>(std::stoull(argv[++i]));
      continue;
    }
    if (arg == "--top-k" && i + 1 < argc) {
      options.top_k = static_cast<std::size_t>(std::stoull(argv[++i]));
      continue;
    }
    if (arg == "--generate" && i + 1 < argc) {
      options.generate = static_cast<std::size_t>(std::stoull(argv[++i]));
      continue;
    }
    if (arg == "--kv-cache") {
      options.kv_cache = true;
      continue;
    }
    if (arg == "--kv-cache-prefill-model" && i + 1 < argc) {
      options.kv_cache_prefill_model_path = argv[++i];
      continue;
    }
    if (arg == "--kv-cache-decode-model" && i + 1 < argc) {
      options.kv_cache_decode_model_path = argv[++i];
      continue;
    }
    if (arg == "--strict") {
      options.strict = true;
      continue;
    }
    if (arg == "--cpu-only") {
      options.cpu_only = true;
      continue;
    }
    if (arg == "--verbose") {
      options.verbose = true;
      continue;
    }
    if (arg == "--quiet") {
      options.quiet = true;
      continue;
    }
    throw std::runtime_error("unknown argument: " + arg);
  }

  if (options.model_path.empty()) {
    throw std::runtime_error("miniort_run_gpt requires a model path");
  }
  if (options.tokens.empty() == options.prompt.empty()) {
    throw std::runtime_error("miniort_run_gpt requires exactly one of --tokens or --prompt");
  }
  if (!options.prompt.empty() && options.model_dir.empty()) {
    options.model_dir = std::filesystem::path(options.model_path).parent_path().string();
  }
  return options;
}

struct GptCacheBinding {
  std::vector<std::string> prefill_output_names;
  std::vector<std::string> decode_input_names;
  std::vector<std::string> decode_output_names;
};

GptCacheBinding BuildCacheBinding(const miniort::Graph& prefill_graph, const miniort::Graph& decode_graph) {
  if (prefill_graph.inputs.empty() || decode_graph.inputs.empty()) {
    throw std::runtime_error("KV cache models must expose at least one input");
  }
  if (prefill_graph.outputs.size() < 2 || decode_graph.inputs.size() < 2 || decode_graph.outputs.size() < 2) {
    throw std::runtime_error("KV cache models must expose logits plus cache tensors");
  }

  GptCacheBinding binding;
  binding.prefill_output_names.reserve(prefill_graph.outputs.size() - 1);
  binding.decode_input_names.reserve(decode_graph.inputs.size() - 1);
  binding.decode_output_names.reserve(decode_graph.outputs.size() - 1);

  for (std::size_t i = 1; i < prefill_graph.outputs.size(); ++i) {
    binding.prefill_output_names.push_back(prefill_graph.outputs[i].name);
  }
  for (std::size_t i = 1; i < decode_graph.inputs.size(); ++i) {
    binding.decode_input_names.push_back(decode_graph.inputs[i].name);
  }
  for (std::size_t i = 1; i < decode_graph.outputs.size(); ++i) {
    binding.decode_output_names.push_back(decode_graph.outputs[i].name);
  }

  if (binding.prefill_output_names.size() != binding.decode_input_names.size() ||
      binding.decode_input_names.size() != binding.decode_output_names.size()) {
    throw std::runtime_error("KV cache input/output counts do not match");
  }
  return binding;
}

std::vector<std::pair<float, std::size_t>> RankLastTokenLogits(const miniort::Tensor& logits) {
  if (logits.dtype != "float32" || logits.float_data.empty() || logits.shape.size() != 3) {
    return {};
  }

  const auto batch = static_cast<std::size_t>(logits.shape[0]);
  const auto sequence = static_cast<std::size_t>(logits.shape[1]);
  const auto vocab = static_cast<std::size_t>(logits.shape[2]);
  if (batch == 0 || sequence == 0 || vocab == 0) {
    return {};
  }

  const auto offset = (sequence - 1) * vocab;
  std::vector<std::pair<float, std::size_t>> ranked;
  ranked.reserve(vocab);
  for (std::size_t token_id = 0; token_id < vocab; ++token_id) {
    ranked.emplace_back(logits.float_data[offset + token_id], token_id);
  }

  std::sort(ranked.begin(), ranked.end(), [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });
  return ranked;
}

void PrintTopKFromLogits(const miniort::Tensor& logits, std::size_t top_k, std::ostream& os) {
  auto ranked = RankLastTokenLogits(logits);
  if (ranked.empty()) {
    os << "logits_summary unavailable\n";
    return;
  }

  os << "last_token_topk\n";
  for (std::size_t i = 0; i < std::min(top_k, ranked.size()); ++i) {
    os << "  - token_id=" << ranked[i].second << " logit=" << ranked[i].first << "\n";
  }
}

std::int64_t SelectGreedyNextToken(const miniort::Tensor& logits) {
  auto ranked = RankLastTokenLogits(logits);
  if (ranked.empty()) {
    throw std::runtime_error("failed to rank logits for greedy generation");
  }
  return static_cast<std::int64_t>(ranked.front().second);
}

void PrintTokenIds(const std::vector<std::int64_t>& token_ids, const std::string& label, std::ostream& os) {
  if (!label.empty()) {
    os << label << "\n";
  }
  os << "[";
  for (std::size_t i = 0; i < token_ids.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << token_ids[i];
  }
  os << "]\n";
}

void AccumulateSummary(const miniort::RunSummary& src, miniort::RunSummary& dst) {
  dst.executed_nodes += src.executed_nodes;
  dst.skipped_nodes += src.skipped_nodes;
  dst.materialized_outputs += src.materialized_outputs;
  dst.released_tensors += src.released_tensors;
  for (const auto& [provider, count] : src.provider_visited_node_counts) {
    dst.provider_visited_node_counts[provider] += count;
  }
  for (const auto& [provider, count] : src.provider_executed_node_counts) {
    dst.provider_executed_node_counts[provider] += count;
  }
  for (const auto& [provider, count] : src.provider_skipped_node_counts) {
    dst.provider_skipped_node_counts[provider] += count;
  }
  for (const auto& [provider, count] : src.provider_materialized_output_counts) {
    dst.provider_materialized_output_counts[provider] += count;
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    const auto options = ParseArgs(argc, argv);
    const auto prefill_model_path = options.kv_cache && !options.kv_cache_prefill_model_path.empty()
                                        ? options.kv_cache_prefill_model_path
                                        : options.model_path;
    const auto decode_model_path = options.kv_cache ? options.kv_cache_decode_model_path : std::string{};

    std::unique_ptr<miniort::Gpt2Tokenizer> tokenizer;
    std::vector<std::int64_t> token_ids;
    if (!options.prompt.empty()) {
      tokenizer = std::make_unique<miniort::Gpt2Tokenizer>(options.model_dir);
      token_ids = tokenizer->Encode(options.prompt);
      if (token_ids.empty()) {
        throw std::runtime_error("prompt encoded to an empty token sequence");
      }
    } else {
      token_ids = ParseTokenIds(options.tokens);
    }
    const std::vector<std::int64_t> input_token_ids = token_ids;

    miniort::SessionOptions session_options;
    session_options.verbose = options.verbose;
    session_options.allow_missing_kernels = !options.strict;
    session_options.allow_unassigned_nodes = !options.strict;
    session_options.auto_bind_placeholder_inputs = true;
    session_options.start_node = options.start_node;
    session_options.max_nodes = options.max_nodes;

    auto make_session = [&](miniort::Graph graph) {
      if (options.cpu_only) {
        std::vector<std::shared_ptr<const miniort::ExecutionProvider>> providers;
        providers.push_back(std::make_shared<miniort::CpuExecutionProvider>());
        return std::make_unique<miniort::Session>(std::move(graph), std::move(providers), session_options);
      }
      return std::make_unique<miniort::Session>(std::move(graph), session_options);
    };

    auto prefill_graph = miniort::LoadOnnxGraph(prefill_model_path, options.quiet ? nullptr : &std::cout);
    if (prefill_graph.inputs.empty()) {
      throw std::runtime_error("graph has no runtime inputs");
    }
    auto prefill_session = make_session(std::move(prefill_graph));

    std::unique_ptr<miniort::Session> decode_session;
    GptCacheBinding cache_binding;
    if (options.kv_cache) {
      if (decode_model_path.empty()) {
        throw std::runtime_error("kv-cache mode requires --kv-cache-decode-model");
      }
      auto decode_graph = miniort::LoadOnnxGraph(decode_model_path, options.quiet ? nullptr : &std::cout);
      cache_binding = BuildCacheBinding(prefill_session->graph(), decode_graph);
      decode_session = make_session(std::move(decode_graph));
    }

    miniort::RunSummary summary;
    std::unordered_map<std::string, miniort::Tensor> cache_state;
    std::vector<std::int64_t> step_tokens = token_ids;

    const auto run_step = [&](miniort::Session& session, const miniort::Graph& graph,
                              const std::vector<std::int64_t>& input_ids,
                              const std::unordered_map<std::string, miniort::Tensor>& extra_feeds,
                              miniort::ExecutionContext& context) -> const miniort::Tensor* {
      std::unordered_map<std::string, miniort::Tensor> feeds = extra_feeds;
      feeds.emplace(graph.inputs.front().name, MakeTokenTensor(graph.inputs.front(), input_ids));
      const auto step_summary = session.Run(feeds, context, options.quiet ? nullptr : &std::cout);
      AccumulateSummary(step_summary, summary);
      const auto* logits = context.FindTensor(graph.outputs.front().name);
      if (logits == nullptr) {
        throw std::runtime_error("logits output was not produced");
      }
      return logits;
    };

    if (!options.kv_cache) {
      miniort::ExecutionContext context;
      for (std::size_t step = 0; step <= options.generate; ++step) {
        const auto* logits = run_step(*prefill_session, prefill_session->graph(), step_tokens, {}, context);
        if (step == options.generate) {
          if (!options.quiet) {
            std::cout << "\nfinal_context\n";
            context.Dump(std::cout, 12);
          }
          std::cout << "\n";
          PrintTopKFromLogits(*logits, options.top_k, std::cout);
        } else {
          const auto next_token_id = SelectGreedyNextToken(*logits);
          token_ids.push_back(next_token_id);
          step_tokens.push_back(next_token_id);
          if (options.verbose && !options.quiet) {
            std::cout << "generation_step[" << step << "] next_token_id=" << next_token_id << "\n";
          }
        }
      }
    } else {
      miniort::ExecutionContext context;
      const auto collect_cache_state = [&](const miniort::ExecutionContext& cache_context,
                                           const std::vector<std::string>& output_names,
                                           const std::vector<std::string>& input_names) {
        cache_state.clear();
        for (std::size_t i = 0; i < output_names.size(); ++i) {
          const auto* tensor = cache_context.FindTensor(output_names[i]);
          if (tensor == nullptr) {
            throw std::runtime_error("KV cache output was not produced: " + output_names[i]);
          }
          auto mapped = *tensor;
          mapped.name = input_names[i];
          cache_state[input_names[i]] = std::move(mapped);
        }
      };

      const auto prefill_logits = run_step(*prefill_session, prefill_session->graph(), step_tokens, {}, context);
      collect_cache_state(context, cache_binding.prefill_output_names, cache_binding.decode_input_names);

      if (options.generate == 0) {
        if (!options.quiet) {
          std::cout << "\nfinal_context\n";
          context.Dump(std::cout, 12);
        }
        std::cout << "\n";
        PrintTopKFromLogits(*prefill_logits, options.top_k, std::cout);
      } else {
        const auto next_token_id = SelectGreedyNextToken(*prefill_logits);
        token_ids.push_back(next_token_id);
        step_tokens = {next_token_id};
        if (options.verbose && !options.quiet) {
          std::cout << "generation_step[0] next_token_id=" << next_token_id << "\n";
        }

        for (std::size_t step = 1; step <= options.generate; ++step) {
          context = miniort::ExecutionContext();
          std::vector<std::int64_t> decode_tokens = step_tokens;
          const auto* logits = run_step(*decode_session, decode_session->graph(), decode_tokens, cache_state, context);
          collect_cache_state(context, cache_binding.decode_output_names, cache_binding.decode_input_names);

          if (step == options.generate) {
            if (!options.quiet) {
              std::cout << "\nfinal_context\n";
              context.Dump(std::cout, 12);
            }
            std::cout << "\n";
            PrintTopKFromLogits(*logits, options.top_k, std::cout);
          } else {
            const auto next_token_id = SelectGreedyNextToken(*logits);
            token_ids.push_back(next_token_id);
            step_tokens = {next_token_id};
            if (options.verbose && !options.quiet) {
              std::cout << "generation_step[" << step << "] next_token_id=" << next_token_id << "\n";
            }
          }
        }
      }
    }

    if (options.generate != 0) {
      std::cout << "\n";
      PrintTokenIds(token_ids, "full_token_ids:", std::cout);
    }
    if (tokenizer != nullptr) {
      std::cout << "\ninput_text:\n" << options.prompt << "\n";
      PrintTokenIds(input_token_ids, "\ninput_token_ids:", std::cout);
      std::cout << "\noutput_text:\n" << tokenizer->Decode(token_ids) << "\n";
    }
    std::cout << "\nsummary executed=" << summary.executed_nodes
              << " skipped=" << summary.skipped_nodes
              << " materialized_outputs=" << summary.materialized_outputs << "\n";
    miniort::PrintRunSummary(summary, std::cout);
    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return EXIT_FAILURE;
  }
}
