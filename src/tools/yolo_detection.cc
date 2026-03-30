#include "miniort/tools/yolo_detection.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

#include "miniort/runtime/profiling.h"

#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

namespace miniort {

namespace {

float ComputeIoU(const YoloDetection& lhs, const YoloDetection& rhs) {
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

std::vector<YoloDetection> NonMaximumSuppression(std::vector<YoloDetection> detections, float iou_threshold) {
  std::sort(detections.begin(), detections.end(),
            [](const YoloDetection& lhs, const YoloDetection& rhs) { return lhs.score > rhs.score; });

  std::vector<YoloDetection> kept;
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

std::pair<std::size_t, std::size_t> ResolveYoloLayout(const Tensor& output) {
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
  throw std::runtime_error("unexpected yolov8n output shape: " + FormatRuntimeShape(output.shape));
}

float ReadOutputValue(const Tensor& output, std::size_t channels, std::size_t boxes,
                      std::size_t channel_index, std::size_t box_index) {
  if (channels == 6 && output.shape[2] == 6) {
    return output.float_data[box_index * channels + channel_index];
  }
  if (output.shape[1] == static_cast<std::int64_t>(channels)) {
    return output.float_data[channel_index * boxes + box_index];
  }
  return output.float_data[box_index * channels + channel_index];
}

void DrawPixel(YoloImage& image, int x, int y, const std::array<unsigned char, 3>& color) {
  if (x < 0 || y < 0 || x >= image.width || y >= image.height) {
    return;
  }
  const auto index = (static_cast<std::size_t>(y) * static_cast<std::size_t>(image.width) +
                      static_cast<std::size_t>(x)) * 3;
  image.rgb[index] = color[0];
  image.rgb[index + 1] = color[1];
  image.rgb[index + 2] = color[2];
}

}  // namespace

YoloImage LoadRgbImage(const std::filesystem::path& path) {
  int width = 0;
  int height = 0;
  int channels = 0;
  stbi_uc* pixels = stbi_load(path.string().c_str(), &width, &height, &channels, 3);
  if (pixels == nullptr) {
    throw std::runtime_error("failed to load image for visualization: " + path.string());
  }
  YoloImage image;
  image.width = width;
  image.height = height;
  image.rgb.assign(pixels, pixels + static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3);
  stbi_image_free(pixels);
  return image;
}

void SaveImage(const std::filesystem::path& path, const YoloImage& image) {
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

void DrawRect(YoloImage& image, int x1, int y1, int x2, int y2, const std::array<unsigned char, 3>& color) {
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

std::array<unsigned char, 3> ColorForClass(int class_id) {
  static constexpr std::array<std::array<unsigned char, 3>, 6> kColors = {{
      {255, 80, 80}, {80, 200, 120}, {80, 160, 255},
      {255, 200, 80}, {220, 120, 255}, {80, 220, 220},
  }};
  return kColors[static_cast<std::size_t>(class_id) % kColors.size()];
}

std::vector<YoloDetection> DecodeYolov8Detections(const Tensor& output, int image_width, int image_height,
                                                  float score_threshold, float iou_threshold) {
  const auto [channels, boxes] = ResolveYoloLayout(output);
  if (channels == 6) {
    std::vector<YoloDetection> raw;
    raw.reserve(boxes);
    const float sx = static_cast<float>(image_width) / 640.0f;
    const float sy = static_cast<float>(image_height) / 640.0f;
    for (std::size_t box = 0; box < boxes; ++box) {
      const auto score = ReadOutputValue(output, channels, boxes, 4, box);
      const auto class_id = static_cast<int>(std::round(ReadOutputValue(output, channels, boxes, 5, box)));
      if (score < score_threshold || class_id < 0 || class_id >= static_cast<int>(kCocoClasses.size())) {
        continue;
      }

      YoloDetection det;
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

  std::vector<YoloDetection> raw;
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

    YoloDetection det;
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

void DumpDetectionsJson(const std::filesystem::path& path, const std::vector<YoloDetection>& detections) {
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

}  // namespace miniort

