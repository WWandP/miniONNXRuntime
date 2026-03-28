#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "miniort/loader/onnx_loader.h"
#include "miniort/runtime/execution_context.h"
#include "miniort/runtime/session.h"
#include "miniort/tools/image_loader.h"

#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

namespace {

constexpr std::array<const char*, 80> kCocoClasses = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"};

struct Options {
  std::string model_path;
  std::string image_path;
  std::string save_vis_path;
  std::string dump_json_path;
  float score_threshold{0.25f};
  float iou_threshold{0.45f};
  bool verbose{false};
};

struct Image {
  int width{0};
  int height{0};
  std::vector<unsigned char> rgb;
};

struct Detection {
  int class_id{0};
  float score{0.0f};
  float x1{0.0f};
  float y1{0.0f};
  float x2{0.0f};
  float y2{0.0f};
};

Options ParseArgs(int argc, char* argv[]) {
  if (argc < 3) {
    throw std::runtime_error(
        "usage: miniort_detect_yolov8n <model.onnx> --image path [--save-vis out.png] [--dump-json out.json] "
        "[--score-threshold 0.25] [--iou-threshold 0.45] [--verbose]");
  }

  Options options;
  options.model_path = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
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
    if (arg == "--verbose") {
      options.verbose = true;
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

Image LoadRgbImage(const std::filesystem::path& path) {
  int width = 0;
  int height = 0;
  int channels = 0;
  stbi_uc* pixels = stbi_load(path.string().c_str(), &width, &height, &channels, 3);
  if (pixels == nullptr) {
    throw std::runtime_error("failed to load image for visualization: " + path.string());
  }
  Image image;
  image.width = width;
  image.height = height;
  image.rgb.assign(pixels, pixels + static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3);
  stbi_image_free(pixels);
  return image;
}

void SaveImage(const std::filesystem::path& path, const Image& image) {
  std::filesystem::create_directories(path.parent_path());
  auto ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });

  const auto* data = reinterpret_cast<const unsigned char*>(image.rgb.data());
  const auto path_string = path.string();
  if (ext == ".png") {
    if (stbi_write_png(path_string.c_str(), image.width, image.height, 3, data, image.width * 3) == 0) {
      throw std::runtime_error("failed to write png: " + path.string());
    }
    return;
  }
  if (ext == ".jpg" || ext == ".jpeg") {
    if (stbi_write_jpg(path_string.c_str(), image.width, image.height, 3, data, 95) == 0) {
      throw std::runtime_error("failed to write jpg: " + path.string());
    }
    return;
  }
  if (ext == ".bmp") {
    if (stbi_write_bmp(path_string.c_str(), image.width, image.height, 3, data) == 0) {
      throw std::runtime_error("failed to write bmp: " + path.string());
    }
    return;
  }

  std::ofstream os(path, std::ios::binary);
  if (!os) {
    throw std::runtime_error("failed to open output image: " + path.string());
  }
  os << "P6\n" << image.width << " " << image.height << "\n255\n";
  os.write(reinterpret_cast<const char*>(image.rgb.data()), static_cast<std::streamsize>(image.rgb.size()));
}

void DrawPixel(Image& image, int x, int y, const std::array<unsigned char, 3>& color) {
  if (x < 0 || y < 0 || x >= image.width || y >= image.height) {
    return;
  }
  const auto index = (static_cast<std::size_t>(y) * static_cast<std::size_t>(image.width) +
                      static_cast<std::size_t>(x)) * 3;
  image.rgb[index] = color[0];
  image.rgb[index + 1] = color[1];
  image.rgb[index + 2] = color[2];
}

void DrawRect(Image& image, int x1, int y1, int x2, int y2, const std::array<unsigned char, 3>& color) {
  const int thickness = 2;
  for (int t = 0; t < thickness; ++t) {
    for (int x = x1; x <= x2; ++x) {
      DrawPixel(image, x, y1 + t, color);
      DrawPixel(image, x, y2 - t, color);
    }
    for (int y = y1; y <= y2; ++y) {
      DrawPixel(image, x1 + t, y, color);
      DrawPixel(image, x2 - t, y, color);
    }
  }
}

float ComputeIoU(const Detection& lhs, const Detection& rhs) {
  const auto ix1 = std::max(lhs.x1, rhs.x1);
  const auto iy1 = std::max(lhs.y1, rhs.y1);
  const auto ix2 = std::min(lhs.x2, rhs.x2);
  const auto iy2 = std::min(lhs.y2, rhs.y2);
  const auto iw = std::max(0.0f, ix2 - ix1);
  const auto ih = std::max(0.0f, iy2 - iy1);
  const auto inter = iw * ih;
  const auto lhs_area = std::max(0.0f, lhs.x2 - lhs.x1) * std::max(0.0f, lhs.y2 - lhs.y1);
  const auto rhs_area = std::max(0.0f, rhs.x2 - rhs.x1) * std::max(0.0f, rhs.y2 - rhs.y1);
  const auto denom = lhs_area + rhs_area - inter;
  return denom <= 0.0f ? 0.0f : inter / denom;
}

std::vector<Detection> NonMaximumSuppression(std::vector<Detection> detections, float iou_threshold) {
  std::sort(detections.begin(), detections.end(),
            [](const Detection& lhs, const Detection& rhs) { return lhs.score > rhs.score; });

  std::vector<Detection> kept;
  std::vector<bool> removed(detections.size(), false);
  for (std::size_t i = 0; i < detections.size(); ++i) {
    if (removed[i]) {
      continue;
    }
    kept.push_back(detections[i]);
    for (std::size_t j = i + 1; j < detections.size(); ++j) {
      if (removed[j] || detections[i].class_id != detections[j].class_id) {
        continue;
      }
      if (ComputeIoU(detections[i], detections[j]) > iou_threshold) {
        removed[j] = true;
      }
    }
  }
  return kept;
}

std::pair<std::size_t, std::size_t> ResolveYoloLayout(const miniort::Tensor& output) {
  if (output.shape.size() != 3 || output.dtype != "float32" || output.float_data.empty()) {
    throw std::runtime_error("yolov8n output must be float32 rank-3 tensor");
  }

  if (output.shape[2] == 6) {
    return {6, static_cast<std::size_t>(output.shape[1])};
  }
  if (output.shape[1] == 84) {
    return {84, static_cast<std::size_t>(output.shape[2])};
  }
  if (output.shape[2] == 84) {
    return {static_cast<std::size_t>(output.shape[2]), static_cast<std::size_t>(output.shape[1])};
  }
  throw std::runtime_error("unexpected yolov8n output shape: " + miniort::FormatRuntimeShape(output.shape));
}

float ReadOutputValue(const miniort::Tensor& output, std::size_t channels, std::size_t boxes,
                      std::size_t channel_index, std::size_t box_index) {
  if (channels == 6 && output.shape[2] == 6) {
    return output.float_data[box_index * channels + channel_index];
  }
  if (output.shape[1] == static_cast<std::int64_t>(channels)) {
    return output.float_data[channel_index * boxes + box_index];
  }
  return output.float_data[box_index * channels + channel_index];
}

std::vector<Detection> DecodeYolov8Detections(const miniort::Tensor& output, int image_width, int image_height,
                                              float score_threshold, float iou_threshold) {
  const auto [channels, boxes] = ResolveYoloLayout(output);
  if (channels == 6) {
    std::vector<Detection> raw;
    raw.reserve(boxes);
    const float sx = static_cast<float>(image_width) / 640.0f;
    const float sy = static_cast<float>(image_height) / 640.0f;
    for (std::size_t box = 0; box < boxes; ++box) {
      const auto score = ReadOutputValue(output, channels, boxes, 4, box);
      const auto class_id = static_cast<int>(std::round(ReadOutputValue(output, channels, boxes, 5, box)));
      if (score < score_threshold || class_id < 0 || class_id >= static_cast<int>(kCocoClasses.size())) {
        continue;
      }

      Detection det;
      det.class_id = class_id;
      det.score = score;
      det.x1 = std::clamp(ReadOutputValue(output, channels, boxes, 0, box) * sx,
                          0.0f, static_cast<float>(image_width - 1));
      det.y1 = std::clamp(ReadOutputValue(output, channels, boxes, 1, box) * sy,
                          0.0f, static_cast<float>(image_height - 1));
      det.x2 = std::clamp(ReadOutputValue(output, channels, boxes, 2, box) * sx,
                          0.0f, static_cast<float>(image_width - 1));
      det.y2 = std::clamp(ReadOutputValue(output, channels, boxes, 3, box) * sy,
                          0.0f, static_cast<float>(image_height - 1));
      raw.push_back(det);
    }
    return NonMaximumSuppression(std::move(raw), iou_threshold);
  }

  if (channels != 84) {
    throw std::runtime_error("yolov8n output channel count must be 84 or 6");
  }

  std::vector<Detection> raw;
  raw.reserve(boxes);
  const float sx = static_cast<float>(image_width) / 640.0f;
  const float sy = static_cast<float>(image_height) / 640.0f;
  for (std::size_t box = 0; box < boxes; ++box) {
    float best_score = 0.0f;
    int best_class = -1;
    for (std::size_t cls = 4; cls < channels; ++cls) {
      const auto score = ReadOutputValue(output, channels, boxes, cls, box);
      if (score > best_score) {
        best_score = score;
        best_class = static_cast<int>(cls - 4);
      }
    }
    if (best_class < 0 || best_score < score_threshold) {
      continue;
    }

    const auto cx = ReadOutputValue(output, channels, boxes, 0, box) * sx;
    const auto cy = ReadOutputValue(output, channels, boxes, 1, box) * sy;
    const auto w = ReadOutputValue(output, channels, boxes, 2, box) * sx;
    const auto h = ReadOutputValue(output, channels, boxes, 3, box) * sy;

    Detection det;
    det.class_id = best_class;
    det.score = best_score;
    det.x1 = std::clamp(cx - w * 0.5f, 0.0f, static_cast<float>(image_width - 1));
    det.y1 = std::clamp(cy - h * 0.5f, 0.0f, static_cast<float>(image_height - 1));
    det.x2 = std::clamp(cx + w * 0.5f, 0.0f, static_cast<float>(image_width - 1));
    det.y2 = std::clamp(cy + h * 0.5f, 0.0f, static_cast<float>(image_height - 1));
    raw.push_back(det);
  }

  return NonMaximumSuppression(std::move(raw), iou_threshold);
}

void DumpDetectionsJson(const std::filesystem::path& path, const std::vector<Detection>& detections) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream os(path);
  if (!os) {
    throw std::runtime_error("failed to open json output: " + path.string());
  }

  os << "{\n  \"model\": \"yolov8n\",\n  \"detections\": [\n";
  for (std::size_t i = 0; i < detections.size(); ++i) {
    const auto& det = detections[i];
    os << "    {\"class_id\": " << det.class_id
       << ", \"class_name\": \"" << kCocoClasses[static_cast<std::size_t>(det.class_id)] << "\""
       << ", \"score\": " << det.score
       << ", \"box\": [" << det.x1 << ", " << det.y1 << ", " << det.x2 << ", " << det.y2 << "]}";
    if (i + 1 != detections.size()) {
      os << ",";
    }
    os << "\n";
  }
  os << "  ]\n}\n";
}

std::array<unsigned char, 3> ColorForClass(int class_id) {
  static constexpr std::array<std::array<unsigned char, 3>, 6> kColors = {{
      {255, 80, 80}, {80, 200, 120}, {80, 160, 255},
      {255, 200, 80}, {220, 120, 255}, {80, 220, 220},
  }};
  return kColors[static_cast<std::size_t>(class_id) % kColors.size()];
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    const auto options = ParseArgs(argc, argv);
    auto graph = miniort::LoadOnnxGraph(options.model_path);
    if (graph.inputs.empty() || graph.outputs.empty()) {
      throw std::runtime_error("graph must have at least one input and one output");
    }

    const auto original = LoadRgbImage(options.image_path);
    std::unordered_map<std::string, miniort::Tensor> feeds;
    const auto& input = graph.inputs.front();
    feeds.emplace(input.name,
                  miniort::LoadImageAsNchwTensor(std::filesystem::path(options.image_path), input.name, input.info,
                                                 options.verbose ? &std::cout : nullptr));

    const auto output_name = graph.outputs.front().name;
    miniort::Session session(std::move(graph),
                             {.verbose = false,
                              .allow_missing_kernels = false,
                              .auto_bind_placeholder_inputs = false,
                              .max_nodes = 0});
    miniort::ExecutionContext context;
    const auto summary = session.Run(feeds, context, options.verbose ? &std::cout : nullptr);
    const auto* output = context.FindTensor(output_name);
    if (output == nullptr) {
      throw std::runtime_error("missing graph output tensor: " + output_name);
    }

    const auto detections = DecodeYolov8Detections(*output, original.width, original.height,
                                                   options.score_threshold, options.iou_threshold);

    std::cout << "yolov8n detections: " << detections.size() << "\n";
    for (const auto& det : detections) {
      std::cout << "  - " << kCocoClasses[static_cast<std::size_t>(det.class_id)]
                << " score=" << det.score
                << " box=[" << det.x1 << ", " << det.y1 << ", " << det.x2 << ", " << det.y2 << "]\n";
    }
    std::cout << "summary executed=" << summary.executed_nodes
              << " skipped=" << summary.skipped_nodes
              << " materialized_outputs=" << summary.materialized_outputs << "\n";

    if (!options.dump_json_path.empty()) {
      DumpDetectionsJson(options.dump_json_path, detections);
      std::cout << "json saved to " << options.dump_json_path << "\n";
    }

    if (!options.save_vis_path.empty()) {
      auto vis = original;
      for (const auto& det : detections) {
        DrawRect(vis,
                 static_cast<int>(std::floor(det.x1)),
                 static_cast<int>(std::floor(det.y1)),
                 static_cast<int>(std::ceil(det.x2)),
                 static_cast<int>(std::ceil(det.y2)),
                 ColorForClass(det.class_id));
      }
      SaveImage(options.save_vis_path, vis);
      std::cout << "visualization saved to " << options.save_vis_path << "\n";
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return EXIT_FAILURE;
  }
}
