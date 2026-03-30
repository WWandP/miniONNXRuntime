#include "miniort/tools/image_loader.h"

#include <stdexcept>
#include <vector>

#include "miniort/runtime/profiling.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize2.h"

namespace miniort {

namespace {

std::int64_t ResolvePositiveDim(const std::vector<std::string>& shape, std::size_t index, std::int64_t fallback) {
  if (index >= shape.size()) {
    return fallback;
  }

  std::size_t parsed_chars = 0;
  try {
    const auto value = std::stoll(shape[index], &parsed_chars);
    if (parsed_chars == shape[index].size() && value > 0) {
      return value;
    }
  } catch (...) {
  }
  return fallback;
}

}  // namespace

Tensor LoadImageAsNchwTensor(const std::filesystem::path& image_path, const std::string& input_name,
                             const TensorInfo& input_info, std::ostream* trace) {
  TimingMap timings;
  Tensor tensor;
  {
    ScopedTimer total_timer("image_loader.total", trace, &timings["image_loader.total"]);
    int src_w = 0;
    int src_h = 0;
    int src_channels = 0;
    stbi_uc* pixels = nullptr;
    {
      ScopedTimer timer("image_loader.load", trace, &timings["image_loader.load"]);
      pixels = stbi_load(image_path.string().c_str(), &src_w, &src_h, &src_channels, 3);
      if (pixels == nullptr) {
        throw std::runtime_error("failed to load image: " + image_path.string());
      }
    }

    const int dst_h = static_cast<int>(ResolvePositiveDim(input_info.shape, 2, 640));
    const int dst_w = static_cast<int>(ResolvePositiveDim(input_info.shape, 3, 640));
    std::vector<unsigned char> resized(static_cast<std::size_t>(dst_h) * static_cast<std::size_t>(dst_w) * 3);

    {
      ScopedTimer timer("image_loader.resize", trace, &timings["image_loader.resize"]);
      if (stbir_resize_uint8_linear(pixels, src_w, src_h, 0, resized.data(), dst_w, dst_h, 0, STBIR_RGB) == nullptr) {
        stbi_image_free(pixels);
        throw std::runtime_error("failed to resize image: " + image_path.string());
      }
    }

    tensor.name = input_name;
    tensor.dtype = input_info.dtype.empty() ? "float32" : input_info.dtype;
    tensor.shape = {1, 3, dst_h, dst_w};
    tensor.float_data.resize(static_cast<std::size_t>(dst_h) * static_cast<std::size_t>(dst_w) * 3);
    tensor.is_placeholder = false;

    {
      ScopedTimer timer("image_loader.normalize_and_pack", trace, &timings["image_loader.normalize_and_pack"]);
      const std::size_t plane_size = static_cast<std::size_t>(dst_h) * static_cast<std::size_t>(dst_w);
      for (int y = 0; y < dst_h; ++y) {
        for (int x = 0; x < dst_w; ++x) {
          const std::size_t hw_index = static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_w) + static_cast<std::size_t>(x);
          const std::size_t src_index = hw_index * 3;
          tensor.float_data[hw_index] = static_cast<float>(resized[src_index]) / 255.0f;
          tensor.float_data[plane_size + hw_index] = static_cast<float>(resized[src_index + 1]) / 255.0f;
          tensor.float_data[plane_size * 2 + hw_index] = static_cast<float>(resized[src_index + 2]) / 255.0f;
        }
      }
    }

    stbi_image_free(pixels);

    if (trace != nullptr) {
      *trace << "loaded image input " << image_path.string()
             << " src=[" << src_h << ", " << src_w << ", 3]"
             << " dst=" << FormatTensorSummary(tensor) << "\n";
    }
  }

  if (trace != nullptr) {
    PrintTimingSummary(timings, *trace, "image loader timing summary");
  }

  return tensor;
}

}  // namespace miniort
