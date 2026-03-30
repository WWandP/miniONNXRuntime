#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "miniort/loader/onnx_loader.h"
#include "miniort/runtime/execution_context.h"
#include "miniort/runtime/session.h"
#include "miniort/tools/image_loader.h"

namespace {

struct Options {
  std::string model_path;
  std::string image_path;
  bool verbose{false};
  bool profile{false};
  bool allow_missing_kernels{true};
  std::size_t context_dump_limit{20};
  std::size_t max_nodes{0};
};

Options ParseArgs(int argc, char* argv[]) {
  if (argc < 2) {
    throw std::runtime_error(
        "usage: miniort_run <model.onnx> [--image path] [--verbose] [--profile] [--strict-kernel] [--context-dump-limit N]");
  }

  Options options;
  options.model_path = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--verbose") {
      options.verbose = true;
      continue;
    }
    if (arg == "--profile") {
      options.profile = true;
      continue;
    }
    if (arg == "--image" && i + 1 < argc) {
      options.image_path = argv[++i];
      continue;
    }
    if (arg == "--strict-kernel") {
      options.allow_missing_kernels = false;
      continue;
    }
    if (arg == "--context-dump-limit" && i + 1 < argc) {
      options.context_dump_limit = static_cast<std::size_t>(std::stoul(argv[++i]));
      continue;
    }
    if (arg == "--max-nodes" && i + 1 < argc) {
      options.max_nodes = static_cast<std::size_t>(std::stoul(argv[++i]));
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
    std::ostream* trace = (options.verbose || options.profile) ? &std::cout : nullptr;
    auto graph = miniort::LoadOnnxGraph(options.model_path, trace);
    std::unordered_map<std::string, miniort::Tensor> feeds;
    if (!options.image_path.empty()) {
      if (graph.inputs.empty()) {
        throw std::runtime_error("graph has no runtime inputs for --image");
      }
      const auto& input = graph.inputs.front();
      feeds.emplace(input.name,
                    miniort::LoadImageAsNchwTensor(std::filesystem::path(options.image_path), input.name, input.info,
                                                   trace));
    }

    miniort::Session session(std::move(graph),
                             {.verbose = options.verbose,
                              .allow_missing_kernels = options.allow_missing_kernels,
                              .auto_bind_placeholder_inputs = true,
                              .max_nodes = options.max_nodes});

    miniort::ExecutionContext context;
    const auto summary = session.Run(feeds, context, trace);

    std::cout << "\nfinal_context\n";
    context.Dump(std::cout, options.context_dump_limit);
    std::cout << "\nsummary executed=" << summary.executed_nodes
              << " skipped=" << summary.skipped_nodes
              << " materialized_outputs=" << summary.materialized_outputs << "\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return EXIT_FAILURE;
  }
}
