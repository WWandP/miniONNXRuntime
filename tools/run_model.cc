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
};

Options ParseArgs(int argc, char* argv[]) {
  if (argc < 2) {
    throw std::runtime_error("usage: miniort_run <model.onnx> [--image path]");
  }

  Options options;
  options.model_path = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--image" && i + 1 < argc) {
      options.image_path = argv[++i];
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
    miniort::PrintPhaseBanner(std::cout, "phase3", "Run CPU Inference End to End",
                              "跑通一次完整推理，看整图执行和最终上下文。");
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
                            "使用默认 provider，准备整图执行。");
    miniort::Session session(std::move(graph), {.auto_bind_placeholder_inputs = true});

    miniort::PrintPhaseStep(std::cout, 4, 4, "Run Full Graph",
                            "关注 summary、provider 统计和 final_context。");
    miniort::ExecutionContext context;
    const auto summary = session.Run(feeds, context, &std::cout);

    std::cout << "\nfinal_context\n";
    context.Dump(std::cout, 12);
    std::cout << "\nsummary executed=" << summary.executed_nodes
              << " skipped=" << summary.skipped_nodes
              << " materialized_outputs=" << summary.materialized_outputs << "\n";
    miniort::PrintPhaseResult(std::cout, "phase3 complete", "你现在看到的是完整 CPU 推理结果。");
    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return EXIT_FAILURE;
  }
}
