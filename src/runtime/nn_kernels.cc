#include "builtin_kernel_groups.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "kernel_utils.h"

namespace miniort {

void RegisterNnKernels(KernelRegistry& registry) {
  registry.Register("Conv", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& input = RequireTensor(context, node.inputs.at(0));
    const auto& weight = RequireTensor(context, node.inputs.at(1));
    const Tensor* bias = nullptr;
    if (node.inputs.size() > 2 && !node.inputs.at(2).empty()) {
      bias = &RequireTensor(context, node.inputs.at(2));
    }

    const auto& input_data = RequireFloatData(input, "Conv");
    const auto& weight_data = RequireFloatData(weight, "Conv");
    const std::vector<float>* bias_data = nullptr;
    if (bias != nullptr) {
      bias_data = &RequireFloatData(*bias, "Conv");
    }

    if (input.shape.size() != 4 || weight.shape.size() != 4) {
      throw std::runtime_error("Conv currently only supports 2D NCHW tensors");
    }

    const auto group = ReadIntAttribute(node, "group", 1);
    if (group != 1) {
      throw std::runtime_error("Conv currently only supports group=1");
    }

    const auto dilations = ReadIntsAttribute(node, "dilations", {1, 1});
    const auto strides = ReadIntsAttribute(node, "strides", {1, 1});
    const auto pads = ReadIntsAttribute(node, "pads", {0, 0, 0, 0});
    if (dilations.size() != 2 || strides.size() != 2 || pads.size() != 4) {
      throw std::runtime_error("Conv attribute rank is not supported");
    }

    const auto n = static_cast<std::size_t>(input.shape[0]);
    const auto c_in = static_cast<std::size_t>(input.shape[1]);
    const auto h_in = static_cast<std::size_t>(input.shape[2]);
    const auto w_in = static_cast<std::size_t>(input.shape[3]);
    const auto c_out = static_cast<std::size_t>(weight.shape[0]);
    const auto w_c_in = static_cast<std::size_t>(weight.shape[1]);
    const auto k_h = static_cast<std::size_t>(weight.shape[2]);
    const auto k_w = static_cast<std::size_t>(weight.shape[3]);

    if (c_in != w_c_in) {
      throw std::runtime_error("Conv input channel count does not match weight");
    }
    if (bias_data != nullptr && bias_data->size() != c_out) {
      throw std::runtime_error("Conv bias size does not match output channels");
    }

    const auto pad_top = pads[0];
    const auto pad_left = pads[1];
    const auto pad_bottom = pads[2];
    const auto pad_right = pads[3];
    const auto dilation_h = dilations[0];
    const auto dilation_w = dilations[1];
    const auto stride_h = strides[0];
    const auto stride_w = strides[1];

    const auto effective_kh = static_cast<std::int64_t>((k_h - 1) * dilation_h + 1);
    const auto effective_kw = static_cast<std::int64_t>((k_w - 1) * dilation_w + 1);
    const auto h_out = (static_cast<std::int64_t>(h_in) + pad_top + pad_bottom - effective_kh) / stride_h + 1;
    const auto w_out = (static_cast<std::int64_t>(w_in) + pad_left + pad_right - effective_kw) / stride_w + 1;
    if (h_out <= 0 || w_out <= 0) {
      throw std::runtime_error("Conv output shape is invalid");
    }

    Tensor output;
    output.name = node.outputs.at(0);
    output.dtype = "float32";
    output.shape = {static_cast<std::int64_t>(n), static_cast<std::int64_t>(c_out), h_out, w_out};
    output.is_placeholder = false;
    output.float_data.assign(GetElementCount(output.shape), 0.0f);

    const auto input_hw = h_in * w_in;
    const auto output_hw = static_cast<std::size_t>(h_out) * static_cast<std::size_t>(w_out);
    const auto kernel_hw = k_h * k_w;
    const auto output_w = static_cast<std::size_t>(w_out);
    const auto output_h = static_cast<std::size_t>(h_out);

    for (std::size_t batch = 0; batch < n; ++batch) {
      const auto* batch_input = input_data.data() + batch * c_in * input_hw;
      auto* batch_output = output.float_data.data() + batch * c_out * output_hw;
      for (std::size_t oc = 0; oc < c_out; ++oc) {
        auto* output_plane = batch_output + oc * output_hw;
        const float bias_value = bias_data != nullptr ? (*bias_data)[oc] : 0.0f;
        std::fill_n(output_plane, output_hw, bias_value);

        const auto* weight_oc = weight_data.data() + oc * c_in * kernel_hw;
        for (std::size_t ic = 0; ic < c_in; ++ic) {
          const auto* input_plane = batch_input + ic * input_hw;
          const auto* weight_ic = weight_oc + ic * kernel_hw;

          for (std::size_t kh = 0; kh < k_h; ++kh) {
            const auto input_h_base = static_cast<std::int64_t>(kh) * dilation_h - pad_top;
            const auto oh_begin = input_h_base >= 0 ? 0 : static_cast<std::size_t>((-input_h_base + stride_h - 1) / stride_h);
            const auto oh_end = static_cast<std::size_t>(
                std::min<std::int64_t>(h_out, (static_cast<std::int64_t>(h_in) - 1 - input_h_base) / stride_h + 1));
            if (oh_begin >= oh_end) {
              continue;
            }

            for (std::size_t kw = 0; kw < k_w; ++kw) {
              const auto input_w_base = static_cast<std::int64_t>(kw) * dilation_w - pad_left;
              const auto ow_begin =
                  input_w_base >= 0 ? 0 : static_cast<std::size_t>((-input_w_base + stride_w - 1) / stride_w);
              const auto ow_end = static_cast<std::size_t>(
                  std::min<std::int64_t>(w_out, (static_cast<std::int64_t>(w_in) - 1 - input_w_base) / stride_w + 1));
              if (ow_begin >= ow_end) {
                continue;
              }

              const float weight_value = weight_ic[kh * k_w + kw];
              for (std::size_t oh = oh_begin; oh < oh_end; ++oh) {
                const auto ih = static_cast<std::size_t>(static_cast<std::int64_t>(oh) * stride_h + input_h_base);
                const auto* input_row = input_plane + ih * w_in;
                auto* output_row = output_plane + oh * output_w;
                for (std::size_t ow = ow_begin; ow < ow_end; ++ow) {
                  const auto iw = static_cast<std::size_t>(static_cast<std::int64_t>(ow) * stride_w + input_w_base);
                  output_row[ow] += input_row[iw] * weight_value;
                }
              }
            }
          }
        }
      }
    }

    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Conv produced " << node.outputs.at(0) << "\n";
    }
  });

  registry.Register("MaxPool", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& input = RequireTensor(context, node.inputs.at(0));
    const auto& input_data = RequireFloatData(input, "MaxPool");
    if (input.shape.size() != 4) {
      throw std::runtime_error("MaxPool currently only supports 2D NCHW tensors");
    }

    const auto kernel_shape = ReadIntsAttribute(node, "kernel_shape", {});
    const auto strides = ReadIntsAttribute(node, "strides", {1, 1});
    const auto pads = ReadIntsAttribute(node, "pads", {0, 0, 0, 0});
    const auto dilations = ReadIntsAttribute(node, "dilations", {1, 1});
    const auto ceil_mode = ReadIntAttribute(node, "ceil_mode", 0);
    if (kernel_shape.size() != 2 || strides.size() != 2 || pads.size() != 4 || dilations.size() != 2) {
      throw std::runtime_error("MaxPool attribute rank is not supported");
    }
    if (ceil_mode != 0) {
      throw std::runtime_error("MaxPool currently only supports ceil_mode=0");
    }

    const auto n = static_cast<std::size_t>(input.shape[0]);
    const auto c = static_cast<std::size_t>(input.shape[1]);
    const auto h_in = static_cast<std::size_t>(input.shape[2]);
    const auto w_in = static_cast<std::size_t>(input.shape[3]);
    const auto k_h = static_cast<std::size_t>(kernel_shape[0]);
    const auto k_w = static_cast<std::size_t>(kernel_shape[1]);
    const auto stride_h = strides[0];
    const auto stride_w = strides[1];
    const auto dilation_h = dilations[0];
    const auto dilation_w = dilations[1];
    const auto pad_top = pads[0];
    const auto pad_left = pads[1];
    const auto pad_bottom = pads[2];
    const auto pad_right = pads[3];

    const auto effective_kh = static_cast<std::int64_t>((k_h - 1) * dilation_h + 1);
    const auto effective_kw = static_cast<std::int64_t>((k_w - 1) * dilation_w + 1);
    const auto h_out = (static_cast<std::int64_t>(h_in) + pad_top + pad_bottom - effective_kh) / stride_h + 1;
    const auto w_out = (static_cast<std::int64_t>(w_in) + pad_left + pad_right - effective_kw) / stride_w + 1;
    if (h_out <= 0 || w_out <= 0) {
      throw std::runtime_error("MaxPool output shape is invalid");
    }

    Tensor output;
    output.name = node.outputs.at(0);
    output.dtype = "float32";
    output.shape = {static_cast<std::int64_t>(n), static_cast<std::int64_t>(c), h_out, w_out};
    output.is_placeholder = false;
    output.float_data.assign(GetElementCount(output.shape), 0.0f);

    const auto input_hw = h_in * w_in;
    const auto output_hw = static_cast<std::size_t>(h_out) * static_cast<std::size_t>(w_out);
    for (std::size_t batch = 0; batch < n; ++batch) {
      for (std::size_t channel = 0; channel < c; ++channel) {
        for (std::int64_t oh = 0; oh < h_out; ++oh) {
          for (std::int64_t ow = 0; ow < w_out; ++ow) {
            float best = -std::numeric_limits<float>::infinity();
            for (std::size_t kh = 0; kh < k_h; ++kh) {
              for (std::size_t kw = 0; kw < k_w; ++kw) {
                const auto ih = oh * stride_h - pad_top + static_cast<std::int64_t>(kh) * dilation_h;
                const auto iw = ow * stride_w - pad_left + static_cast<std::int64_t>(kw) * dilation_w;
                if (ih < 0 || iw < 0 || ih >= static_cast<std::int64_t>(h_in) ||
                    iw >= static_cast<std::int64_t>(w_in)) {
                  continue;
                }
                const auto input_index =
                    ((batch * c + channel) * input_hw) + static_cast<std::size_t>(ih) * w_in + static_cast<std::size_t>(iw);
                best = std::max(best, input_data[input_index]);
              }
            }

            const auto output_index =
                ((batch * c + channel) * output_hw) +
                static_cast<std::size_t>(oh) * static_cast<std::size_t>(w_out) +
                static_cast<std::size_t>(ow);
            output.float_data[output_index] = best;
          }
        }
      }
    }

    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel MaxPool produced " << node.outputs.at(0) << "\n";
    }
  });

  registry.Register("Resize", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& input = RequireTensor(context, node.inputs.at(0));
    const auto& input_data = RequireFloatData(input, "Resize");
    if (input.shape.size() != 4) {
      throw std::runtime_error("Resize currently only supports 4D NCHW tensors");
    }

    const auto mode_it = node.attributes.find("mode");
    const auto coord_it = node.attributes.find("coordinate_transformation_mode");
    const auto nearest_it = node.attributes.find("nearest_mode");
    const auto mode = mode_it == node.attributes.end() ? std::string("nearest") : mode_it->second.string_value;
    const auto coord_mode =
        coord_it == node.attributes.end() ? std::string("asymmetric") : coord_it->second.string_value;
    const auto nearest_mode =
        nearest_it == node.attributes.end() ? std::string("floor") : nearest_it->second.string_value;
    if (mode != "nearest" || coord_mode != "asymmetric" || nearest_mode != "floor") {
      throw std::runtime_error("Resize currently only supports nearest+asymmetric+floor");
    }

    if (node.inputs.size() < 3 || node.inputs.at(2).empty()) {
      throw std::runtime_error("Resize currently expects scales input");
    }
    const auto& scales_tensor = RequireTensor(context, node.inputs.at(2));
    const auto& scales = RequireFloatData(scales_tensor, "Resize");
    if (scales.size() != 4) {
      throw std::runtime_error("Resize currently expects 4D scales");
    }

    const auto n_out = static_cast<std::int64_t>(std::floor(static_cast<double>(input.shape[0]) * scales[0]));
    const auto c_out = static_cast<std::int64_t>(std::floor(static_cast<double>(input.shape[1]) * scales[1]));
    const auto h_out = static_cast<std::int64_t>(std::floor(static_cast<double>(input.shape[2]) * scales[2]));
    const auto w_out = static_cast<std::int64_t>(std::floor(static_cast<double>(input.shape[3]) * scales[3]));
    if (n_out != input.shape[0] || c_out != input.shape[1]) {
      throw std::runtime_error("Resize currently requires batch/channel scales to keep dimensions unchanged");
    }
    if (h_out <= 0 || w_out <= 0) {
      throw std::runtime_error("Resize output shape is invalid");
    }

    const auto n = static_cast<std::size_t>(input.shape[0]);
    const auto c = static_cast<std::size_t>(input.shape[1]);
    const auto h_in = static_cast<std::size_t>(input.shape[2]);
    const auto w_in = static_cast<std::size_t>(input.shape[3]);

    Tensor output;
    output.name = node.outputs.at(0);
    output.dtype = "float32";
    output.shape = {input.shape[0], input.shape[1], h_out, w_out};
    output.is_placeholder = false;
    output.float_data.assign(GetElementCount(output.shape), 0.0f);

    const auto input_hw = h_in * w_in;
    const auto output_hw = static_cast<std::size_t>(h_out) * static_cast<std::size_t>(w_out);
    for (std::size_t batch = 0; batch < n; ++batch) {
      for (std::size_t channel = 0; channel < c; ++channel) {
        for (std::int64_t oh = 0; oh < h_out; ++oh) {
          const auto ih = std::min(static_cast<std::size_t>(std::floor(static_cast<double>(oh) / scales[2])), h_in - 1);
          for (std::int64_t ow = 0; ow < w_out; ++ow) {
            const auto iw = std::min(static_cast<std::size_t>(std::floor(static_cast<double>(ow) / scales[3])), w_in - 1);

            const auto input_index = ((batch * c + channel) * input_hw) + ih * w_in + iw;
            const auto output_index =
                ((batch * c + channel) * output_hw) +
                static_cast<std::size_t>(oh) * static_cast<std::size_t>(w_out) +
                static_cast<std::size_t>(ow);
            output.float_data[output_index] = input_data[input_index];
          }
        }
      }
    }

    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Resize produced " << node.outputs.at(0) << "\n";
    }
  });

  registry.Register("Softmax", [](const Node& node, ExecutionContext& context, std::ostream* trace) {
    const auto& input = RequireTensor(context, node.inputs.at(0));
    const auto& input_data = RequireFloatData(input, "Softmax");
    const auto axis = static_cast<std::size_t>(
        NormalizeAxis(ReadIntAttribute(node, "axis", 1), input.shape.size(), "Softmax"));

    std::size_t outer = 1;
    for (std::size_t i = 0; i < axis; ++i) {
      outer *= static_cast<std::size_t>(input.shape[i]);
    }
    const std::size_t axis_dim = static_cast<std::size_t>(input.shape[axis]);
    std::size_t inner = 1;
    for (std::size_t i = axis + 1; i < input.shape.size(); ++i) {
      inner *= static_cast<std::size_t>(input.shape[i]);
    }

    Tensor output = MakeOutputLike(node.outputs.at(0), input);
    output.float_data.resize(input_data.size());
    for (std::size_t outer_index = 0; outer_index < outer; ++outer_index) {
      for (std::size_t inner_index = 0; inner_index < inner; ++inner_index) {
        float max_value = -std::numeric_limits<float>::infinity();
        for (std::size_t axis_index = 0; axis_index < axis_dim; ++axis_index) {
          const auto offset = (outer_index * axis_dim + axis_index) * inner + inner_index;
          max_value = std::max(max_value, input_data[offset]);
        }

        float sum = 0.0f;
        for (std::size_t axis_index = 0; axis_index < axis_dim; ++axis_index) {
          const auto offset = (outer_index * axis_dim + axis_index) * inner + inner_index;
          const auto value = std::exp(input_data[offset] - max_value);
          output.float_data[offset] = value;
          sum += value;
        }

        for (std::size_t axis_index = 0; axis_index < axis_dim; ++axis_index) {
          const auto offset = (outer_index * axis_dim + axis_index) * inner + inner_index;
          output.float_data[offset] /= sum;
        }
      }
    }

    context.BindTensor(std::move(output));
    if (trace != nullptr) {
      *trace << "    kernel Softmax produced " << node.outputs.at(0) << "\n";
    }
  });
}

}  // namespace miniort
