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
#include "miniort/tools/phase_output.h"

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
    miniort::PrintPhaseBanner(std::cout, "phase2", "Trace Minimal Execution Pipeline",
                              "看 runtime 从加载、喂数据到逐节点执行的主线。");
    miniort::PrintPhaseStep(std::cout, 1, 4, "Load ONNX Graph", options.model_path);
    auto graph = miniort::LoadOnnxGraph(options.model_path, &std::cout);
    std::unordered_map<std::string, miniort::Tensor> feeds;
    if (!options.image_path.empty()) {
      if (graph.inputs.empty()) {
        throw std::runtime_error("graph has no runtime inputs for --image");
      }
      const auto& input = graph.inputs.front();
      miniort::PrintPhaseStep(std::cout, 2, 4, "Prepare Runtime Input", options.image_path);
      feeds.emplace(input.name,
                    miniort::LoadImageAsNchwTensor(std::filesystem::path(options.image_path), input.name, input.info,
                                                   &std::cout));
    }

    miniort::PrintPhaseStep(std::cout, 3, 4, "Create Session",
                            "开启 verbose trace，按拓扑顺序观察节点执行。");
    miniort::Session session(std::move(graph),
                             {.verbose = true,
                              .auto_bind_placeholder_inputs = true,
                              .start_node = options.start_node,
                              .max_nodes = options.max_nodes});

    miniort::PrintPhaseStep(std::cout, 4, 4, "Run Selected Nodes",
                            "关注每个节点的 inputs / outputs / kernel_time_ms。");
    miniort::ExecutionContext context;
    const auto summary = session.Run(feeds, context, &std::cout);

    std::cout << "\nfinal_context\n";
    context.Dump(std::cout, 12);
    std::cout << "\nsummary executed=" << summary.executed_nodes
              << " skipped=" << summary.skipped_nodes
              << " materialized_outputs=" << summary.materialized_outputs << "\n";
    miniort::PrintRunSummary(summary, std::cout);
    miniort::PrintPhaseResult(std::cout, "phase2 complete", "你现在看到的是最小执行主线。");
    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return EXIT_FAILURE;
  }
}
