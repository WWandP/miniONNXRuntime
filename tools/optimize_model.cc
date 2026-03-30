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
  std::string save_vis_path;
  std::string dump_json_path;
  float score_threshold{0.25f};
  float iou_threshold{0.45f};
  bool verbose{false};
  bool profile{false};
  bool allow_missing_kernels{true};
  std::size_t max_nodes{0};
};

Options ParseArgs(int argc, char* argv[]) {
  if (argc < 2) {
    throw std::runtime_error(
        "usage: miniort_optimize_model <model.onnx> [--image path] [--save-vis out.png] [--dump-json out.json] "
        "[--score-threshold 0.25] [--iou-threshold 0.45] [--verbose] [--profile] [--strict-kernel] [--max-nodes N]");
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
    if (arg == "--save-vis" && i + 1 < argc) {
      options.save_vis_path = argv[++i];
      continue;
    }
    if (arg == "--dump-json" && i + 1 < argc) {
      options.dump_json_path = argv[++i];
      continue;
    }
    if (arg == "--score-threshold" && i + 1 < argc) {
      options.score_threshold = std::stof(argv[++i]);
      continue;
    }
    if (arg == "--iou-threshold" && i + 1 < argc) {
      options.iou_threshold = std::stof(argv[++i]);
      continue;
    }
    if (arg == "--strict-kernel") {
      options.allow_missing_kernels = false;
      continue;
    }
    if (arg == "--max-nodes" && i + 1 < argc) {
      options.max_nodes = static_cast<std::size_t>(std::stoul(argv[++i]));
      continue;
    }
    throw std::runtime_error("unknown argument: " + arg);
  }

  if (!options.image_path.empty()) {
    const auto image_stem = std::filesystem::path(options.image_path).stem().string();
    const auto output_dir = std::filesystem::path("outputs");
    if (options.save_vis_path.empty()) {
      options.save_vis_path = (output_dir / (image_stem + "_yolov8n.png")).string();
    }
    if (options.dump_json_path.empty()) {
      options.dump_json_path = (output_dir / (image_stem + "_yolov8n.json")).string();
    }
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
    std::ostream* trace = (options.verbose || options.profile) ? &std::cout : nullptr;
    auto graph = miniort::LoadOnnxGraph(options.model_path, trace);

    PrintGraphSnapshot(graph, "before optimization");

    miniort::GraphOptimizationSummary summary;
    graph = miniort::OptimizeGraph(std::move(graph),
                                   {.verbose = options.verbose,
                                    .enable_constant_folding = true,
                                    .enable_dead_node_cleanup = true,
                                    .enable_shape_simplification = true},
                                   trace,
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
                                                   trace));

      miniort::Session session(std::move(graph),
                               {.verbose = options.verbose,
                                .allow_missing_kernels = options.allow_missing_kernels,
                                .auto_bind_placeholder_inputs = true,
                                .max_nodes = options.max_nodes});
      miniort::ExecutionContext context;
      const auto run_summary = session.Run(feeds, context, trace);
      const auto* output = context.FindTensor(output_name);
      if (output == nullptr) {
        throw std::runtime_error("missing graph output tensor: " + output_name);
      }

      const auto detections = miniort::DecodeYolov8Detections(*output, image.width, image.height,
                                                              options.score_threshold, options.iou_threshold);

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

      if (!options.dump_json_path.empty()) {
        miniort::DumpDetectionsJson(options.dump_json_path, detections);
        std::cout << "json saved to " << options.dump_json_path << "\n";
      }

      if (!options.save_vis_path.empty()) {
        auto vis = image;
        for (const auto& det : detections) {
          miniort::DrawRect(vis,
                            static_cast<int>(std::floor(det.x1)),
                            static_cast<int>(std::floor(det.y1)),
                            static_cast<int>(std::ceil(det.x2)),
                            static_cast<int>(std::ceil(det.y2)),
                            miniort::ColorForClass(det.class_id));
        }
        miniort::SaveImage(options.save_vis_path, vis);
        std::cout << "visualization saved to " << options.save_vis_path << "\n";
      }
    }

    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return EXIT_FAILURE;
  }
}
