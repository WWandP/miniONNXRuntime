#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "miniort/loader/onnx_loader.h"
#include "miniort/optimizer/graph_optimizer.h"
#include "miniort/runtime/cuda_execution_provider.h"
#include "miniort/runtime/execution_context.h"
#include "miniort/runtime/session.h"
#include "miniort/tools/gpt2_cache_binding.h"

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::string prefill_model{"models/gpt2/model.kv_prefill.onnx"};
  std::string decode_model{"models/gpt2/model.kv_decode.onnx"};
  std::string tokens{"40,2883,6155,351,616,13779,3290"};
  std::size_t generate{48};
  std::size_t warmup{1};
  std::size_t repeat{3};
  bool strict{true};
  bool graph_opt{false};
  bool shared_context{false};
  bool print_steps{false};
  bool print_cache_residency{false};
  bool trace_measured{false};
};

double ElapsedMs(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

std::vector<std::int64_t> ParseTokenIds(const std::string& text) {
  std::vector<std::int64_t> ids;
  std::stringstream ss(text);
  std::string part;
  while (std::getline(ss, part, ',')) {
    if (!part.empty()) {
      ids.push_back(std::stoll(part));
    }
  }
  if (ids.empty()) {
    throw std::runtime_error("empty token list");
  }
  return ids;
}

miniort::Tensor MakeTokenTensor(const miniort::Value& input, const std::vector<std::int64_t>& token_ids) {
  miniort::Tensor tensor;
  tensor.name = input.name;
  tensor.dtype = "int64";
  tensor.int64_data = token_ids;
  tensor.shape = {1, static_cast<std::int64_t>(token_ids.size())};
  return tensor;
}

std::int64_t SelectGreedyNextToken(const miniort::Tensor& logits) {
  if (logits.dtype != "float32" || logits.float_data.empty() || logits.shape.size() != 3) {
    throw std::runtime_error("invalid logits tensor");
  }
  const auto sequence = static_cast<std::size_t>(logits.shape[1]);
  const auto vocab = static_cast<std::size_t>(logits.shape[2]);
  const auto offset = (sequence - 1) * vocab;
  std::size_t best_token = 0;
  float best_logit = logits.float_data[offset];
  for (std::size_t token = 1; token < vocab; ++token) {
    const auto logit = logits.float_data[offset + token];
    if (logit > best_logit) {
      best_logit = logit;
      best_token = token;
    }
  }
  return static_cast<std::int64_t>(best_token);
}

double Percentile(std::vector<double> values, double percentile) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const auto index = static_cast<std::size_t>(
      std::llround((percentile / 100.0) * static_cast<double>(values.size() - 1)));
  return values[index];
}

void PrintStats(const std::string& name, const std::vector<double>& values) {
  if (values.empty()) {
    std::cout << name << " mean=0 min=0 p50=0 p95=0 max=0\n";
    return;
  }
  const auto sum = std::accumulate(values.begin(), values.end(), 0.0);
  const auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
  std::cout << name
            << " mean=" << (sum / static_cast<double>(values.size()))
            << " min=" << *min_it
            << " p50=" << Percentile(values, 50.0)
            << " p95=" << Percentile(values, 95.0)
            << " max=" << *max_it << "\n";
}

Options ParseArgs(int argc, char* argv[]) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--prefill-model" && i + 1 < argc) {
      options.prefill_model = argv[++i];
    } else if (arg == "--decode-model" && i + 1 < argc) {
      options.decode_model = argv[++i];
    } else if (arg == "--tokens" && i + 1 < argc) {
      options.tokens = argv[++i];
    } else if (arg == "--generate" && i + 1 < argc) {
      options.generate = static_cast<std::size_t>(std::stoull(argv[++i]));
    } else if (arg == "--warmup" && i + 1 < argc) {
      options.warmup = static_cast<std::size_t>(std::stoull(argv[++i]));
    } else if (arg == "--repeat" && i + 1 < argc) {
      options.repeat = static_cast<std::size_t>(std::stoull(argv[++i]));
    } else if (arg == "--allow-missing") {
      options.strict = false;
    } else if (arg == "--graph-opt") {
      options.graph_opt = true;
    } else if (arg == "--shared-context") {
      options.shared_context = true;
    } else if (arg == "--print-steps") {
      options.print_steps = true;
    } else if (arg == "--print-cache-residency") {
      options.print_cache_residency = true;
    } else if (arg == "--trace-measured") {
      options.trace_measured = true;
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  return options;
}

miniort::Graph MaybeOptimize(miniort::Graph graph, bool enabled) {
  if (!enabled) {
    return graph;
  }
  return miniort::OptimizeGraph(std::move(graph),
                                {.enable_constant_folding = true,
                                 .enable_dead_node_cleanup = true,
                                 .enable_shape_simplification = true});
}

struct RunTiming {
  double prefill_ms{0.0};
  std::vector<double> decode_ms;
  std::vector<std::int64_t> token_ids;
};

RunTiming RunOnce(miniort::Session& prefill_session, miniort::Session& decode_session,
                  const miniort::GptCacheBinding& cache_binding,
                  const std::vector<std::int64_t>& prompt_tokens, std::size_t generate,
                  bool shared_context, bool print_steps, bool print_cache_residency,
                  std::ostream* trace) {
  RunTiming timing;
  timing.token_ids = prompt_tokens;

  miniort::ExecutionContext prefill_context;
  miniort::ExecutionContext separate_decode_context;
  miniort::ExecutionContext& decode_context = shared_context ? prefill_context : separate_decode_context;
  std::unordered_map<std::string, miniort::Tensor> feeds;
  feeds.emplace(prefill_session.graph().inputs.front().name,
                MakeTokenTensor(prefill_session.graph().inputs.front(), prompt_tokens));

  const auto prefill_start = Clock::now();
  (void)prefill_session.Run(feeds, prefill_context, trace);
  miniort::MaterializeCudaTensor(prefill_session.graph().outputs.front().name, prefill_context);
  timing.prefill_ms = ElapsedMs(prefill_start, Clock::now());
  const auto* prefill_logits = prefill_context.FindTensor(prefill_session.graph().outputs.front().name);
  if (prefill_logits == nullptr) {
    throw std::runtime_error("missing prefill logits");
  }
  miniort::CollectCacheState(prefill_context, cache_binding, miniort::GptCacheStateSource::kPrefill, feeds);
  if (print_cache_residency) {
    std::size_t float_cache = 0;
    std::size_t cuda_cache = 0;
    for (const auto& [name, tensor] : feeds) {
      if (name.find("past_key_values.") == std::string::npos) {
        continue;
      }
      if (tensor.dtype == "float32") {
        ++float_cache;
        if (tensor.cuda_data != nullptr) {
          ++cuda_cache;
        }
      }
    }
    std::cout << "prefill_cache_cuda=" << cuda_cache << "/" << float_cache << "\n";
  }

  if (generate == 0) {
    return timing;
  }

  auto next_token = SelectGreedyNextToken(*prefill_logits);
  timing.token_ids.push_back(next_token);
  std::vector<std::int64_t> step_tokens{next_token};

  for (std::size_t step = 1; step <= generate; ++step) {
    feeds[decode_session.graph().inputs.front().name] =
        MakeTokenTensor(decode_session.graph().inputs.front(), step_tokens);
    const auto decode_start = Clock::now();
    (void)decode_session.Run(feeds, decode_context, trace);
    miniort::MaterializeCudaTensor(decode_session.graph().outputs.front().name, decode_context);
    const auto decode_ms = ElapsedMs(decode_start, Clock::now());
    timing.decode_ms.push_back(decode_ms);
    if (print_steps) {
      std::cout << "decode_step[" << step << "]=" << decode_ms << "\n";
    }
    const auto* logits = decode_context.FindTensor(decode_session.graph().outputs.front().name);
    if (logits == nullptr) {
      throw std::runtime_error("missing decode logits");
    }
    miniort::CollectCacheState(decode_context, cache_binding, miniort::GptCacheStateSource::kDecode, feeds);
    if (step != generate) {
      next_token = SelectGreedyNextToken(*logits);
      timing.token_ids.push_back(next_token);
      step_tokens = {next_token};
    }
  }

  return timing;
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    const auto options = ParseArgs(argc, argv);
    auto prefill_graph = MaybeOptimize(miniort::LoadOnnxGraph(options.prefill_model, nullptr), options.graph_opt);
    auto decode_graph = MaybeOptimize(miniort::LoadOnnxGraph(options.decode_model, nullptr), options.graph_opt);
    const auto cache_binding = miniort::BuildCacheBinding(prefill_graph, decode_graph);

    miniort::SessionOptions session_options;
    session_options.allow_missing_kernels = !options.strict;
    session_options.allow_unassigned_nodes = !options.strict;
    session_options.evict_dead_tensors = true;
    session_options.evict_dead_cuda_tensors_only = true;
    session_options.materialize_cuda_graph_outputs = false;

    miniort::Session prefill_session(std::move(prefill_graph), session_options);
    miniort::Session decode_session(std::move(decode_graph), session_options);
    const auto prompt_tokens = ParseTokenIds(options.tokens);

    std::vector<double> prefill_runs;
    std::vector<double> decode_step_runs;
    std::vector<double> total_runs;
    std::vector<std::int64_t> final_tokens;

    for (std::size_t run = 0; run < options.warmup + options.repeat; ++run) {
      auto timing = RunOnce(prefill_session, decode_session, cache_binding, prompt_tokens, options.generate,
                            options.shared_context, options.print_steps && run >= options.warmup,
                            options.print_cache_residency && run >= options.warmup,
                            options.trace_measured && run >= options.warmup ? &std::cout : nullptr);
      if (run >= options.warmup) {
        prefill_runs.push_back(timing.prefill_ms);
        decode_step_runs.insert(decode_step_runs.end(), timing.decode_ms.begin(), timing.decode_ms.end());
        total_runs.push_back(timing.prefill_ms + std::accumulate(timing.decode_ms.begin(), timing.decode_ms.end(), 0.0));
        final_tokens = std::move(timing.token_ids);
      }
    }

    std::cout << "miniort_gpt2_kv_benchmark\n";
    std::cout << "graph_opt=" << (options.graph_opt ? "enabled" : "disabled") << "\n";
    std::cout << "shared_context=" << (options.shared_context ? "enabled" : "disabled") << "\n";
    std::cout << "warmup=" << options.warmup << "\n";
    std::cout << "repeat=" << options.repeat << "\n";
    std::cout << "prompt_tokens=" << prompt_tokens.size() << "\n";
    std::cout << "generated_tokens=" << options.generate << "\n";
    PrintStats("prefill_ms", prefill_runs);
    PrintStats("decode_step_ms", decode_step_runs);
    PrintStats("total_generation_ms", total_runs);
    std::cout << "full_token_ids:\n[";
    for (std::size_t i = 0; i < final_tokens.size(); ++i) {
      if (i != 0) {
        std::cout << ", ";
      }
      std::cout << final_tokens[i];
    }
    std::cout << "]\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return EXIT_FAILURE;
  }
}
