#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace miniort {

struct TensorInfo {
  std::vector<std::string> shape;
  std::string dtype;
  bool is_initializer{false};
};

struct TensorData {
  std::string dtype;
  std::vector<std::int64_t> shape;
  std::vector<std::uint8_t> raw_data;
  std::vector<float> float_data;
  std::vector<double> double_data;
  std::vector<std::int32_t> int32_data;
  std::vector<std::int64_t> int64_data;
  std::vector<std::string> string_data;
};

struct AttributeValue {
  enum class Kind {
    kUnknown,
    kFloat,
    kInt,
    kString,
    kFloats,
    kInts,
    kStrings,
    kTensor,
  };

  Kind kind{Kind::kUnknown};
  float float_value{0.0f};
  std::int64_t int_value{0};
  std::string string_value;
  std::vector<float> floats;
  std::vector<std::int64_t> ints;
  std::vector<std::string> strings;
  std::optional<TensorData> tensor;
};

struct Value {
  std::string name;
  TensorInfo info;
  std::optional<TensorData> data;
};

struct Node {
  std::string name;
  std::string op_type;
  std::string execution_provider;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::unordered_map<std::string, AttributeValue> attributes;
};

struct GraphMetadata {
  std::string model_path;
  std::string producer_name;
  std::string producer_version;
  std::int64_t ir_version{0};
  std::unordered_map<std::string, std::int64_t> opset_imports;
};

struct Graph {
  std::string name;
  std::vector<Node> nodes;
  std::vector<Value> inputs;
  std::vector<Value> outputs;
  std::unordered_map<std::string, Value> initializers;
  std::unordered_map<std::string, TensorInfo> value_infos;
  std::unordered_map<std::string, std::size_t> node_name_to_index;
  std::vector<std::size_t> topological_order;
  std::unordered_map<std::string, std::size_t> op_type_histogram;
  GraphMetadata metadata;
};

std::string FormatShape(const std::vector<std::string>& shape);
std::string FormatTensorInfo(const TensorInfo& info);

}  // namespace miniort
