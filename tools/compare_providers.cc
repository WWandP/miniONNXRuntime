#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "miniort/loader/onnx_loader.h"
#include "miniort/runtime/cpu_execution_provider.h"
#include "miniort/runtime/execution_context.h"
#include "miniort/runtime/session.h"
#include "miniort/tools/image_loader.h"
#include "miniort/tools/phase_output.h"

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::string model_path;
  std::string image_path;
  std::size_t repeat{1};
};

Options ParseArgs(int argc, char* argv[]) {
  if (argc < 4) {
    throw std::runtime_error("usage: miniort_compare_providers <model.onnx> --image path [--repeat N]");
  }

  Options options;
  options.model_path = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--image" && i + 1 < argc) {
      options.image_path = argv[++i];
      continue;
    }
    if (arg == "--repeat" && i + 1 < argc) {
      options.repeat = static_cast<std::size_t>(std::stoull(argv[++i]));
      continue;
    }
    throw std::runtime_error("unknown argument: " + arg);
  }
  if (options.image_path.empty()) {
    throw std::runtime_error("--image is required");
  }
  return options;
}

double RunOnce(miniort::Session& session, const std::unordered_map<std::string, miniort::Tensor>& feeds) {
  miniort::ExecutionContext context;
  const auto start = Clock::now();
  const auto summary = session.Run(feeds, context, nullptr);
  const auto end = Clock::now();
  if (summary.executed_nodes == 0) {
    throw std::runtime_error("session executed zero nodes");
  }
  return std::chrono::duration<double, std::milli>(end - start).count();
}

double RunAverage(miniort::Session& session, const std::unordered_map<std::string, miniort::Tensor>& feeds,
                  std::size_t repeat) {
  double total_ms = 0.0;
  for (std::size_t i = 0; i < repeat; ++i) {
    total_ms += RunOnce(session, feeds);
  }
  return total_ms / static_cast<double>(repeat);
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    const auto options = ParseArgs(argc, argv);
    miniort::PrintPhaseBanner(std::cout, "phase5", "Compare Execution Providers",
                              "看默认 provider 路径和纯 CPU 路径的差异。");
    miniort::PrintPhaseStep(std::cout, 1, 4, "Load ONNX Graph", options.model_path);
    auto graph = miniort::LoadOnnxGraph(options.model_path, nullptr);
    if (graph.inputs.empty()) {
      throw std::runtime_error("graph has no inputs");
    }

    const auto& input = graph.inputs.front();
    std::unordered_map<std::string, miniort::Tensor> feeds;
    miniort::PrintPhaseStep(std::cout, 2, 4, "Prepare Runtime Input", options.image_path);
    feeds.emplace(input.name,
                  miniort::LoadImageAsNchwTensor(std::filesystem::path(options.image_path), input.name, input.info,
                                                 nullptr));

    miniort::SessionOptions session_options;
    session_options.auto_bind_placeholder_inputs = true;

    miniort::PrintPhaseStep(std::cout, 3, 4, "Create Sessions",
                            "分别构造默认 provider 路径和 CPU-only 路径。");
    miniort::Session mixed_session(graph, session_options);
    miniort::Session cpu_only_session(
        graph, std::vector<std::shared_ptr<const miniort::ExecutionProvider>>{
                   std::make_shared<miniort::CpuExecutionProvider>()},
        session_options);

    miniort::PrintPhaseStep(std::cout, 4, 4, "Run And Compare",
                            "关注 mixed_ms、cpu_only_ms 和 speedup_pct。");
    const auto mixed_ms = RunAverage(mixed_session, feeds, options.repeat);
    const auto cpu_ms = RunAverage(cpu_only_session, feeds, options.repeat);

    std::cout << "provider_compare\n";
    std::cout << "  repeat=" << options.repeat << "\n";
    std::cout << "  mixed_ms=" << mixed_ms << "\n";
    std::cout << "  cpu_only_ms=" << cpu_ms << "\n";
    std::cout << "  delta_ms=" << (cpu_ms - mixed_ms) << "\n";
    std::cout << "  speedup_pct=" << ((cpu_ms - mixed_ms) / cpu_ms * 100.0) << "\n";
    miniort::PrintPhaseResult(std::cout, "phase5 complete", "你现在看到的是 provider 对比视角。");
    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return EXIT_FAILURE;
  }
}
