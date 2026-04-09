#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "miniort/loader/onnx_loader.h"
#include "miniort/runtime/cpu_execution_provider.h"
#include "miniort/runtime/execution_context.h"
#include "miniort/runtime/session.h"
#include "miniort/tools/image_loader.h"

namespace {

struct Options {
  std::string model_path;
  std::string image_path;
  std::string tokens;
  std::size_t start_node{0};
  std::size_t max_nodes{16};
  bool strict{false};
  bool cpu_only{false};
};

miniort::Tensor MakeTokenTensor(const miniort::Value& input, const std::string& tokens_arg) {
  if (tokens_arg.empty()) {
    throw std::runtime_error("--tokens requires a comma-separated token list");
  }

  miniort::Tensor tensor;
  tensor.name = input.name;
  tensor.dtype = "int64";
  std::stringstream ss(tokens_arg);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token.empty()) {
      continue;
    }
    tensor.int64_data.push_back(std::stoll(token));
  }

  if (tensor.int64_data.empty()) {
    throw std::runtime_error("--tokens did not contain any token ids");
  }

  tensor.shape = {1, static_cast<std::int64_t>(tensor.int64_data.size())};
  return tensor;
}

Options ParseArgs(int argc, char* argv[]) {
  if (argc < 2) {
    throw std::runtime_error(
        "usage: miniort_session_trace <model.onnx> [--image path] [--tokens 1,2,3] [--start-node N] [--max-nodes N] [--strict] [--cpu-only]");
  }

  Options options;
  options.model_path = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--image" && i + 1 < argc) {
      options.image_path = argv[++i];
      continue;
    }
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
    if (arg == "--strict") {
      options.strict = true;
      continue;
    }
    if (arg == "--cpu-only") {
      options.cpu_only = true;
      continue;
    }
    throw std::runtime_error("unknown argument: " + arg);
  }

  return options;
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    const auto options = ParseArgs(argc, argv);
    auto graph = miniort::LoadOnnxGraph(options.model_path, &std::cout);
    std::unordered_map<std::string, miniort::Tensor> feeds;
    if (!options.image_path.empty()) {
      if (graph.inputs.empty()) {
        throw std::runtime_error("graph has no runtime inputs for --image");
      }
      const auto& input = graph.inputs.front();
      feeds.emplace(input.name,
                    miniort::LoadImageAsNchwTensor(std::filesystem::path(options.image_path), input.name, input.info,
                                                   &std::cout));
    }
    if (!options.tokens.empty()) {
      if (graph.inputs.empty()) {
        throw std::runtime_error("graph has no runtime inputs for --tokens");
      }
      const auto& input = graph.inputs.front();
      feeds.emplace(input.name, MakeTokenTensor(input, options.tokens));
    }

    miniort::SessionOptions session_options;
    session_options.verbose = true;
    session_options.allow_missing_kernels = !options.strict;
    session_options.allow_unassigned_nodes = !options.strict;
    session_options.auto_bind_placeholder_inputs = true;
    session_options.start_node = options.start_node;
    session_options.max_nodes = options.max_nodes;

    std::vector<std::shared_ptr<const miniort::ExecutionProvider>> providers;
    if (options.cpu_only) {
      providers.push_back(std::make_shared<miniort::CpuExecutionProvider>());
    }

    miniort::Session session =
        providers.empty() ? miniort::Session(std::move(graph), session_options)
                          : miniort::Session(std::move(graph), std::move(providers), session_options);

    miniort::ExecutionContext context;
    const auto summary = session.Run(feeds, context, &std::cout);

    std::cout << "\nfinal_context\n";
    context.Dump(std::cout, 12);
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
