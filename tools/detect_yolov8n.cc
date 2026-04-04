#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "miniort/loader/onnx_loader.h"
#include "miniort/runtime/execution_context.h"
#include "miniort/runtime/profiling.h"
#include "miniort/runtime/session.h"
#include "miniort/tools/image_loader.h"
#include "miniort/tools/yolo_detection.h"

namespace {

struct Options {
  std::string model_path;
  std::string image_path;
  std::string save_vis_path;
  std::string dump_json_path;
};

Options ParseArgs(int argc, char* argv[]) {
  if (argc < 3) {
    throw std::runtime_error("usage: miniort_detect_yolov8n <model.onnx> --image path");
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

  if (options.image_path.empty()) {
    throw std::runtime_error("--image is required");
  }

  const auto image_stem = std::filesystem::path(options.image_path).stem().string();
  const auto output_dir = std::filesystem::path("outputs");
  if (options.save_vis_path.empty()) {
    options.save_vis_path = (output_dir / (image_stem + "_yolov8n.png")).string();
  }
  if (options.dump_json_path.empty()) {
    options.dump_json_path = (output_dir / (image_stem + "_yolov8n.json")).string();
  }
  return options;
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    const auto options = ParseArgs(argc, argv);
    constexpr float kScoreThreshold = 0.25f;
    constexpr float kIouThreshold = 0.45f;
    auto graph = miniort::LoadOnnxGraph(options.model_path, nullptr);
    if (graph.inputs.empty() || graph.outputs.empty()) {
      throw std::runtime_error("graph must have at least one input and one output");
    }

    miniort::YoloImage image;
    {
      miniort::ScopedTimer original_image_timer("detect.load_original_image", nullptr);
      image = miniort::LoadRgbImage(options.image_path);
    }

    std::unordered_map<std::string, miniort::Tensor> feeds;
    const auto& input = graph.inputs.front();
    feeds.emplace(input.name,
                  miniort::LoadImageAsNchwTensor(std::filesystem::path(options.image_path), input.name, input.info,
                                                 nullptr));

    const auto output_name = graph.outputs.front().name;
    miniort::Session session(std::move(graph),
                             {.verbose = false,
                              .allow_missing_kernels = false,
                              .auto_bind_placeholder_inputs = false,
                              .max_nodes = 0});
    miniort::ExecutionContext context;
    const auto summary = session.Run(feeds, context, nullptr);
    const auto* output = context.FindTensor(output_name);
    if (output == nullptr) {
      throw std::runtime_error("missing graph output tensor: " + output_name);
    }

    const auto detections = miniort::DecodeYolov8Detections(*output, image.width, image.height,
                                                            kScoreThreshold, kIouThreshold);

    std::cout << "yolov8n detections: " << detections.size() << "\n";
    for (const auto& det : detections) {
      std::cout << "  - " << miniort::kCocoClasses[static_cast<std::size_t>(det.class_id)]
                << " score=" << det.score
                << " box=[" << det.x1 << ", " << det.y1 << ", " << det.x2 << ", " << det.y2 << "]\n";
    }
    std::cout << "summary executed=" << summary.executed_nodes
              << " skipped=" << summary.skipped_nodes
              << " materialized_outputs=" << summary.materialized_outputs << "\n";

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
    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return EXIT_FAILURE;
  }
}
