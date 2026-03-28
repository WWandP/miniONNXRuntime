#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "miniort/loader/onnx_loader.h"
#include "miniort/runtime/execution_context.h"
#include "miniort/runtime/session.h"

namespace {

struct Options {
  std::string model_path;
  bool verbose{true};
  bool allow_missing_kernels{true};
  std::size_t context_dump_limit{20};
  std::size_t max_nodes{24};
};

Options ParseArgs(int argc, char* argv[]) {
  if (argc < 2) {
    throw std::runtime_error(
        "usage: miniort_session_trace <model.onnx> [--quiet] [--strict-kernel] [--context-dump-limit N] [--max-nodes N]");
  }

  Options options;
  options.model_path = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--quiet") {
      options.verbose = false;
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
    auto graph = miniort::LoadOnnxGraph(options.model_path);

    miniort::Session session(std::move(graph),
                             {.verbose = options.verbose,
                              .allow_missing_kernels = options.allow_missing_kernels,
                              .auto_bind_placeholder_inputs = true,
                              .max_nodes = options.max_nodes});

    miniort::ExecutionContext context;
    const auto summary = session.Run({}, context, &std::cout);

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
