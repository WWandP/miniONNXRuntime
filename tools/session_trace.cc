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
};

Options ParseArgs(int argc, char* argv[]) {
  if (argc < 2) {
    throw std::runtime_error("usage: miniort_session_trace <model.onnx>");
  }

  Options options;
  options.model_path = argv[1];

  return options;
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    const auto options = ParseArgs(argc, argv);
    auto graph = miniort::LoadOnnxGraph(options.model_path, &std::cout);

    miniort::Session session(std::move(graph),
                             {.verbose = true,
                              .auto_bind_placeholder_inputs = true,
                              .max_nodes = 16});

    miniort::ExecutionContext context;
    const auto summary = session.Run({}, context, &std::cout);

    std::cout << "\nfinal_context\n";
    context.Dump(std::cout, 12);
    std::cout << "\nsummary executed=" << summary.executed_nodes
              << " skipped=" << summary.skipped_nodes
              << " materialized_outputs=" << summary.materialized_outputs << "\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return EXIT_FAILURE;
  }
}
