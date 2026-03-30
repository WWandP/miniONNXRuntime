#pragma once

#include <array>
#include <filesystem>
#include <ostream>
#include <string>
#include <vector>

#include "miniort/runtime/tensor.h"

namespace miniort {

inline constexpr std::array<const char*, 80> kCocoClasses = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"};

struct YoloImage {
  int width{0};
  int height{0};
  std::vector<unsigned char> rgb;
};

struct YoloDetection {
  int class_id{0};
  float score{0.0f};
  float x1{0.0f};
  float y1{0.0f};
  float x2{0.0f};
  float y2{0.0f};
};

YoloImage LoadRgbImage(const std::filesystem::path& path);
void SaveImage(const std::filesystem::path& path, const YoloImage& image);
void DrawRect(YoloImage& image, int x1, int y1, int x2, int y2, const std::array<unsigned char, 3>& color);
std::array<unsigned char, 3> ColorForClass(int class_id);
std::vector<YoloDetection> DecodeYolov8Detections(const Tensor& output, int image_width, int image_height,
                                                  float score_threshold, float iou_threshold);
void DumpDetectionsJson(const std::filesystem::path& path, const std::vector<YoloDetection>& detections);

}  // namespace miniort

