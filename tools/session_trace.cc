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
  std::size_t start_node{0};
  std::size_t max_nodes{16};
};

Options ParseArgs(int argc, char* argv[]) {
  if (argc < 2) {
    throw std::runtime_error(
        "usage: miniort_session_trace <model.onnx> [--image path] [--start-node N] [--max-nodes N]");
  }

  Options options;
  options.model_path = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--image" && i + 1 < argc) {
      options.image_path = argv[++i];
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

    miniort::Session session(std::move(graph),
                             {.verbose = true,
                              .auto_bind_placeholder_inputs = true,
                              .start_node = options.start_node,
                              .max_nodes = options.max_nodes});

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
