#include <algorithm>
#include <cstdlib>
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

namespace {

struct Options {
  std::string model_path;
  std::string tokens;
  std::size_t start_node{0};
  std::size_t max_nodes{0};
  std::size_t top_k{5};
  std::size_t generate{0};
  bool strict{false};
  bool cpu_only{false};
  bool verbose{false};
  bool quiet{false};
};

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
        "usage: miniort_run_gpt <model.onnx> --tokens 1,2,3 [--start-node N] [--max-nodes N] [--top-k N] [--generate N] [--strict] [--cpu-only] [--verbose] [--quiet]");
  }

  Options options;
  options.model_path = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--tokens" && i + 1 < argc) {
      options.tokens = argv[++i];
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

  if (options.tokens.empty()) {
    throw std::runtime_error("miniort_run_gpt requires --tokens");
  }
  return options;
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
  os << label << "\n[";
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
    auto graph = miniort::LoadOnnxGraph(options.model_path, options.quiet ? nullptr : &std::cout);
    if (graph.inputs.empty()) {
      throw std::runtime_error("graph has no runtime inputs");
    }
    const auto input_name = graph.inputs.front().name;
    std::vector<std::int64_t> token_ids = ParseTokenIds(options.tokens);

    miniort::SessionOptions session_options;
    session_options.verbose = options.verbose;
    session_options.allow_missing_kernels = !options.strict;
    session_options.allow_unassigned_nodes = !options.strict;
    session_options.auto_bind_placeholder_inputs = true;
    session_options.start_node = options.start_node;
    session_options.max_nodes = options.max_nodes;

    miniort::ExecutionContext context;
    std::unique_ptr<miniort::Session> session;
    if (options.cpu_only) {
      std::vector<std::shared_ptr<const miniort::ExecutionProvider>> providers;
      providers.push_back(std::make_shared<miniort::CpuExecutionProvider>());
      session = std::make_unique<miniort::Session>(std::move(graph), std::move(providers), session_options);
    } else {
      session = std::make_unique<miniort::Session>(std::move(graph), session_options);
    }

    miniort::RunSummary summary;
    for (std::size_t step = 0; step <= options.generate; ++step) {
      context = miniort::ExecutionContext();
      std::unordered_map<std::string, miniort::Tensor> feeds;
      feeds.emplace(input_name, MakeTokenTensor(session->graph().inputs.front(), token_ids));
      const auto step_summary = session->Run(feeds, context, options.quiet ? nullptr : &std::cout);
      AccumulateSummary(step_summary, summary);

      const auto* logits = context.FindTensor("logits");
      if (logits == nullptr) {
        throw std::runtime_error("logits output was not produced");
      }

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
        if (options.verbose && !options.quiet) {
          std::cout << "generation_step[" << step << "] next_token_id=" << next_token_id << "\n";
        }
      }
    }

    if (options.generate != 0) {
      std::cout << "\n";
      PrintTokenIds(token_ids, "full_token_ids:", std::cout);
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
