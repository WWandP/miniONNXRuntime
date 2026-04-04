#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <unordered_map>

#include "miniort/loader/onnx_loader.h"
#include "miniort/runtime/execution_context.h"
#include "miniort/runtime/session.h"
#include "miniort/optimizer/graph_optimizer.h"
#include "miniort/tools/image_loader.h"
#include "miniort/tools/yolo_detection.h"

namespace {

struct Options {
  std::string model_path;
  std::string image_path;
};

Options ParseArgs(int argc, char* argv[]) {
  if (argc < 2) {
    throw std::runtime_error("usage: miniort_optimize_model <model.onnx> [--image path]");
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

void PrintGraphSnapshot(const miniort::Graph& graph, const char* title) {
  std::cout << title << "\n";
  std::cout << "  graph=" << graph.name << "\n";
  std::cout << "  nodes=" << graph.nodes.size() << "\n";
  std::cout << "  initializers=" << graph.initializers.size() << "\n";
  std::cout << "  value_infos=" << graph.value_infos.size() << "\n";
  std::cout << "  inputs=" << graph.inputs.size() << "\n";
  std::cout << "  outputs=" << graph.outputs.size() << "\n";
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    const auto options = ParseArgs(argc, argv);
    auto graph = miniort::LoadOnnxGraph(options.model_path, nullptr);

    PrintGraphSnapshot(graph, "before optimization");

    miniort::GraphOptimizationSummary summary;
    graph = miniort::OptimizeGraph(std::move(graph),
                                   {.enable_constant_folding = true,
                                    .enable_dead_node_cleanup = true,
                                    .enable_shape_simplification = true},
                                   nullptr,
                                   &summary);

    PrintGraphSnapshot(graph, "after optimization");
    miniort::PrintGraphOptimizationSummary(summary, std::cout);

    if (!options.image_path.empty()) {
      if (graph.inputs.empty()) {
        throw std::runtime_error("graph has no runtime inputs for --image");
      }

      const auto image = miniort::LoadRgbImage(std::filesystem::path(options.image_path));
      const auto output_name = graph.outputs.front().name;
      std::unordered_map<std::string, miniort::Tensor> feeds;
      const auto& input = graph.inputs.front();
      feeds.emplace(input.name,
                    miniort::LoadImageAsNchwTensor(std::filesystem::path(options.image_path), input.name, input.info,
                                                   nullptr));

      miniort::Session session(std::move(graph),
                               {.auto_bind_placeholder_inputs = true, .evict_dead_tensors = true});
      miniort::ExecutionContext context;
      const auto run_summary = session.Run(feeds, context, nullptr);
      const auto* output = context.FindTensor(output_name);
      if (output == nullptr) {
        throw std::runtime_error("missing graph output tensor: " + output_name);
      }

      const auto detections = miniort::DecodeYolov8Detections(*output, image.width, image.height, 0.25f, 0.45f);

      std::cout << "yolov8n detections: " << detections.size() << "\n";
      for (const auto& det : detections) {
        std::cout << "  - " << miniort::kCocoClasses[static_cast<std::size_t>(det.class_id)]
                  << " score=" << det.score
                  << " box=[" << det.x1 << ", " << det.y1 << ", " << det.x2 << ", " << det.y2 << "]\n";
      }

      std::cout << "run summary\n";
      std::cout << "  executed=" << run_summary.executed_nodes << "\n";
      std::cout << "  skipped=" << run_summary.skipped_nodes << "\n";
      std::cout << "  materialized_outputs=" << run_summary.materialized_outputs << "\n";
    }

    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return EXIT_FAILURE;
  }
}
