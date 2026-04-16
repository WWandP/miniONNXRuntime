#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "miniort/model/graph.h"
#include "miniort/runtime/cuda_execution_provider.h"
#include "miniort/runtime/cpu_execution_provider.h"
#include "miniort/runtime/execution_context.h"
#include "miniort/runtime/tensor.h"
#include "miniort/runtime/session.h"
#include "miniort/tools/gpt2_cache_binding.h"

namespace {

using miniort::Graph;
using miniort::Node;
using miniort::ProviderAssignmentPolicy;
using miniort::Session;
using miniort::SessionAssignmentSummary;
using miniort::SessionOptions;

Session MakeCpuSession(Graph graph, SessionOptions options = {}) {
  std::vector<std::shared_ptr<const miniort::ExecutionProvider>> providers;
  providers.push_back(std::make_shared<miniort::CpuExecutionProvider>());
  return Session(std::move(graph), std::move(providers), options);
}

#if defined(MINIORT_BUILD_CUDA_EP)
Session MakeCudaThenCpuSession(Graph graph, SessionOptions options = {}) {
  std::vector<std::shared_ptr<const miniort::ExecutionProvider>> providers;
  providers.push_back(std::make_shared<miniort::CudaExecutionProvider>());
  providers.push_back(std::make_shared<miniort::CpuExecutionProvider>());
  return Session(std::move(graph), std::move(providers), options);
}
#endif

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

Graph MakeGraphWithOps(std::initializer_list<std::string> op_types) {
  Graph graph;
  graph.name = "test_graph";

  std::size_t index = 0;
  for (const auto& op_type : op_types) {
    Node node;
    node.name = "node_" + std::to_string(index);
    node.op_type = op_type;
    node.outputs.push_back("out_" + std::to_string(index));
    graph.node_name_to_index[node.name] = graph.nodes.size();
    graph.topological_order.push_back(graph.nodes.size());
    ++graph.op_type_histogram[op_type];
    graph.nodes.push_back(std::move(node));
    ++index;
  }

  return graph;
}

Graph MakeCacheGraph(std::initializer_list<std::string> output_names,
                     std::initializer_list<std::string> input_names = {"input_ids"}) {
  Graph graph;
  graph.name = "cache_graph";

  for (const auto& name : input_names) {
    miniort::Value value;
    value.name = name;
    graph.inputs.push_back(std::move(value));
  }
  for (const auto& name : output_names) {
    miniort::Value value;
    value.name = name;
    graph.outputs.push_back(std::move(value));
  }

  return graph;
}

void TestAssignmentSummaryMarksSupportedAndUnsupportedOps() {
  auto graph = MakeGraphWithOps({"Constant", "NotSupportedYet"});

  Session session(std::move(graph),
                  {.allow_unassigned_nodes = true,
                   .provider_assignment_policy = ProviderAssignmentPolicy::kFirstMatch});

  const SessionAssignmentSummary& summary = session.assignment_summary();
  Expect(summary.total_nodes == 2, "expected total_nodes=2");
  Expect(summary.assigned_nodes == 1, "expected assigned_nodes=1");
  Expect(summary.unassigned_nodes == 1, "expected unassigned_nodes=1");
  Expect(summary.provider_node_counts.contains("CPU"), "expected CPU provider count");
  Expect(summary.provider_node_counts.at("CPU") == 1, "expected CPU to own 1 node");
  Expect(summary.provider_node_counts.contains("<unassigned>"), "expected unassigned provider count");
  Expect(summary.provider_node_counts.at("<unassigned>") == 1, "expected unassigned to own 1 node");
  Expect(summary.unassigned_op_types.size() == 1 && summary.unassigned_op_types.front() == "NotSupportedYet",
         "expected unassigned op type to be recorded");
  Expect(session.graph().nodes[0].execution_provider == "CPU", "expected Constant to assign to CPU");
  Expect(session.graph().nodes[1].execution_provider == "<unassigned>",
         "expected unsupported op to remain unassigned");
}

void TestSessionRejectsUnassignedNodesWhenConfigured() {
  auto graph = MakeGraphWithOps({"DefinitelyUnsupported"});

  bool threw = false;
  try {
    Session session(std::move(graph),
                    {.allow_unassigned_nodes = false,
                     .provider_assignment_policy = ProviderAssignmentPolicy::kFirstMatch});
    (void)session;
  } catch (const std::exception& ex) {
    threw = true;
    Expect(std::string(ex.what()).find("unassigned nodes") != std::string::npos,
           "expected unassigned node validation failure");
  }

  Expect(threw, "expected session construction to reject unassigned nodes");
}

void TestRunInjectsAllocatorIntoExecutionContext() {
  Graph graph;
  graph.name = "allocator_injection_graph";

  Session session = MakeCpuSession(std::move(graph), SessionOptions{});
  miniort::ExecutionContext context;

  Expect(!context.HasAllocator(), "expected context to start without allocator");
  const auto summary = session.Run({}, context, nullptr);
  Expect(summary.executed_nodes == 0, "expected empty graph to execute zero nodes");
  Expect(context.HasAllocator(), "expected session to inject allocator during run");
}

void TestEmptyInt64ConstantSurvivesAndFeedsConstantOfShape() {
  Graph graph;
  graph.name = "empty_constant_graph";

  Node constant;
  constant.name = "empty_shape";
  constant.op_type = "Constant";
  constant.outputs = {"empty_shape"};
  miniort::TensorData shape_attr;
  shape_attr.dtype = "int64";
  shape_attr.shape = {0};
  constant.attributes["value"].tensor = shape_attr;

  Node cos;
  cos.name = "filled";
  cos.op_type = "ConstantOfShape";
  cos.inputs = {"empty_shape"};
  cos.outputs = {"filled"};

  graph.node_name_to_index[constant.name] = 0;
  graph.node_name_to_index[cos.name] = 1;
  graph.topological_order = {0, 1};
  graph.nodes.push_back(std::move(constant));
  graph.nodes.push_back(std::move(cos));

  Session session = MakeCpuSession(std::move(graph), SessionOptions{});
  miniort::ExecutionContext context;
  const auto summary = session.Run({}, context, nullptr);

  Expect(summary.executed_nodes == 2, "expected constant and ConstantOfShape to execute");

  const auto* shape_tensor = context.FindTensor("empty_shape");
  Expect(shape_tensor != nullptr, "expected empty_shape tensor to exist");
  Expect(shape_tensor->dtype == "int64", "expected empty_shape dtype=int64");
  Expect(shape_tensor->shape.size() == 1 && shape_tensor->shape.front() == 0,
         "expected empty_shape to keep its zero-length shape");
  Expect(!shape_tensor->is_placeholder, "expected empty_shape to be materialized");
  Expect(shape_tensor->int64_data.empty(), "expected empty_shape to have no int64 payload");

  const auto* output = context.FindTensor("filled");
  Expect(output != nullptr, "expected ConstantOfShape output to exist");
  Expect(output->dtype == "float32", "expected ConstantOfShape output dtype=float32");
  Expect(output->shape.empty(), "expected ConstantOfShape([]) to produce a scalar");
  Expect(output->float_data.size() == 1, "expected ConstantOfShape scalar payload");
  Expect(output->float_data.front() == 0.0f, "expected ConstantOfShape default fill value");
}

void TestIdentityExecutionPassesThroughTensorData() {
  auto graph = MakeGraphWithOps({"Identity"});
  graph.nodes[0].inputs = {"x"};
  graph.nodes[0].outputs = {"y"};

  miniort::Tensor input;
  input.name = "x";
  input.dtype = "float32";
  input.shape = {2, 2};
  input.float_data = {1.f, 2.f, 3.f, 4.f};

  Session session = MakeCpuSession(std::move(graph), SessionOptions{});
  miniort::ExecutionContext context;
  std::unordered_map<std::string, miniort::Tensor> feeds;
  feeds.emplace(input.name, input);

  const auto summary = session.Run(feeds, context, nullptr);
  Expect(summary.executed_nodes == 1, "expected Identity graph to execute one node");

  const auto* output = context.FindTensor("y");
  Expect(output != nullptr, "expected Identity output tensor");
  Expect(output->dtype == "float32", "expected Identity output dtype=float32");
  Expect(output->shape == std::vector<std::int64_t>({2, 2}), "expected Identity output shape");
  Expect(output->float_data == input.float_data, "expected Identity to preserve float payload");
}

void TestGptCacheBindingMatchesExpectedSchema() {
  auto prefill_graph = MakeCacheGraph({
      "logits",
      "present.0.key",
      "present.0.value",
      "present.1.key",
      "present.1.value",
  });
  auto decode_graph = MakeCacheGraph({
      "logits",
      "present.0.key",
      "present.0.value",
      "present.1.key",
      "present.1.value",
  }, {
      "input_ids",
      "past_key_values.0.key",
      "past_key_values.0.value",
      "past_key_values.1.key",
      "past_key_values.1.value",
  });

  const auto binding = miniort::BuildCacheBinding(prefill_graph, decode_graph);
  Expect(binding.tensors.size() == 4, "expected four cache tensor bindings");
  Expect(binding.tensors[0].prefill_output_name == "present.0.key", "unexpected first prefill output");
  Expect(binding.tensors[0].decode_input_name == "past_key_values.0.key", "unexpected first decode input");
  Expect(binding.tensors[1].prefill_output_name == "present.0.value", "unexpected second prefill output");
  Expect(binding.tensors[2].prefill_output_name == "present.1.key", "unexpected third prefill output");
  Expect(binding.tensors[3].decode_output_name == "present.1.value", "unexpected fourth decode output");

  miniort::ExecutionContext context;
  miniort::Tensor tensor;
  tensor.dtype = "float32";
  tensor.shape = {1};

  tensor.name = "present.0.key";
  tensor.float_data = {1.f};
  context.BindTensor(tensor);

  tensor.name = "present.0.value";
  tensor.float_data = {2.f};
  context.BindTensor(tensor);

  tensor.name = "present.1.key";
  tensor.float_data = {3.f};
  context.BindTensor(tensor);

  tensor.name = "present.1.value";
  tensor.float_data = {4.f};
  context.BindTensor(tensor);

  std::unordered_map<std::string, miniort::Tensor> cache_state;
  miniort::CollectCacheState(context, binding, miniort::GptCacheStateSource::kPrefill, cache_state);
  Expect(cache_state.size() == 4, "expected four cache state tensors after prefill collection");
  Expect(cache_state.contains("past_key_values.0.key"), "expected mapped cache key");
  Expect(cache_state.at("past_key_values.0.key").float_data.front() == 1.f, "unexpected mapped key payload");
  Expect(cache_state.at("past_key_values.1.value").float_data.front() == 4.f, "unexpected mapped value payload");

  tensor.name = "present.0.key";
  tensor.float_data = {11.f};
  context.BindTensor(tensor);
  tensor.name = "present.0.value";
  tensor.float_data = {12.f};
  context.BindTensor(tensor);
  tensor.name = "present.1.key";
  tensor.float_data = {13.f};
  context.BindTensor(tensor);
  tensor.name = "present.1.value";
  tensor.float_data = {14.f};
  context.BindTensor(tensor);

  miniort::CollectCacheState(context, binding, miniort::GptCacheStateSource::kDecode, cache_state);
  Expect(cache_state.at("past_key_values.0.key").float_data.front() == 11.f,
         "expected decode collection to refresh key payload");
  Expect(cache_state.at("past_key_values.1.value").float_data.front() == 14.f,
         "expected decode collection to refresh value payload");
}

void TestGptCacheBindingRejectsMalformedSchemas() {
  auto prefill_graph = MakeCacheGraph({
      "logits",
      "present.0.query",
      "present.0.value",
  });
  auto decode_graph = MakeCacheGraph({
      "logits",
      "present.0.key",
      "present.0.value",
  }, {
      "input_ids",
      "past_key_values.0.key",
      "past_key_values.0.value",
  });

  bool threw = false;
  try {
    (void)miniort::BuildCacheBinding(prefill_graph, decode_graph);
  } catch (const std::exception& ex) {
    threw = true;
    Expect(std::string(ex.what()).find("KV cache") != std::string::npos,
           "expected cache binding schema validation failure");
  }
  Expect(threw, "expected malformed cache schema to be rejected");
}

void TestGptCacheBindingSupportsGenericTensorNames() {
  auto prefill_graph = MakeCacheGraph({
      "logits",
      "layer_cache.0.key_state",
      "layer_cache.0.value_state",
      "layer_cache.1.key_state",
      "layer_cache.1.value_state",
  });
  auto decode_graph = MakeCacheGraph({
      "logits",
      "present_layer_0_k",
      "present_layer_0_v",
      "present_layer_1_k",
      "present_layer_1_v",
  }, {
      "input_ids",
      "past.0.k",
      "past.0.v",
      "past.1.k",
      "past.1.v",
  });

  const auto binding = miniort::BuildCacheBinding(prefill_graph, decode_graph);
  Expect(binding.tensors.size() == 4, "expected generic naming to bind four cache tensors");
  Expect(binding.tensors[0].prefill_output_name == "layer_cache.0.key_state", "unexpected generic layer0 key");
  Expect(binding.tensors[0].decode_input_name == "past.0.k", "unexpected generic decode input key");
  Expect(binding.tensors[0].decode_output_name == "present_layer_0_k", "unexpected generic decode output key");
  Expect(binding.tensors[1].prefill_output_name == "layer_cache.0.value_state", "unexpected generic layer0 value");
  Expect(binding.tensors[2].prefill_output_name == "layer_cache.1.key_state", "unexpected generic layer1 key");
  Expect(binding.tensors[3].decode_output_name == "present_layer_1_v", "unexpected generic layer1 value");
}

void TestMatMulExecutionProducesExpectedOutput() {
  auto graph = MakeGraphWithOps({"MatMul"});
  graph.nodes[0].inputs = {"a", "b"};

  miniort::Tensor lhs;
  lhs.name = "a";
  lhs.dtype = "float32";
  lhs.shape = {2, 3};
  lhs.float_data = {1.f, 2.f, 3.f,
                    4.f, 5.f, 6.f};

  miniort::Tensor rhs;
  rhs.name = "b";
  rhs.dtype = "float32";
  rhs.shape = {3, 2};
  rhs.float_data = {7.f, 8.f,
                    9.f, 10.f,
                    11.f, 12.f};

  Session session = MakeCpuSession(std::move(graph), SessionOptions{});
  miniort::ExecutionContext context;
  std::unordered_map<std::string, miniort::Tensor> feeds;
  feeds.emplace(lhs.name, lhs);
  feeds.emplace(rhs.name, rhs);

  const auto summary = session.Run(feeds, context, nullptr);
  Expect(summary.executed_nodes == 1, "expected MatMul graph to execute one node");

  const auto* output = context.FindTensor("out_0");
  Expect(output != nullptr, "expected MatMul output tensor");
  Expect(output->shape == std::vector<std::int64_t>({2, 2}), "expected MatMul output shape [2,2]");
  Expect(output->float_data.size() == 4, "expected MatMul output size 4");
  Expect(std::fabs(output->float_data[0] - 58.f) < 1e-5f, "unexpected MatMul output[0]");
  Expect(std::fabs(output->float_data[1] - 64.f) < 1e-5f, "unexpected MatMul output[1]");
  Expect(std::fabs(output->float_data[2] - 139.f) < 1e-5f, "unexpected MatMul output[2]");
  Expect(std::fabs(output->float_data[3] - 154.f) < 1e-5f, "unexpected MatMul output[3]");
}

#if defined(MINIORT_BUILD_CUDA_EP)
void TestCudaProviderPrefersMatMulAndFallsBackForUnsupportedOps() {
  auto graph = MakeGraphWithOps({"MatMul", "Gemm", "Sigmoid", "Add", "Mul", "SiLU"});
  Session session = MakeCudaThenCpuSession(std::move(graph), SessionOptions{});

  Expect(session.graph().nodes.size() == 6, "expected six-node graph");
  Expect(session.graph().nodes[0].execution_provider == "CUDA", "expected MatMul to assign to CUDA");
  Expect(session.graph().nodes[1].execution_provider == "CUDA", "expected Gemm to assign to CUDA");
  Expect(session.graph().nodes[2].execution_provider == "CUDA", "expected Sigmoid to assign to CUDA");
  Expect(session.graph().nodes[3].execution_provider == "CUDA", "expected Add to assign to CUDA");
  Expect(session.graph().nodes[4].execution_provider == "CUDA", "expected Mul to assign to CUDA");
  Expect(session.graph().nodes[5].execution_provider == "CUDA", "expected SiLU to assign to CUDA");
}

void TestCudaMatMulExecutionProducesExpectedOutput() {
  auto graph = MakeGraphWithOps({"MatMul"});
  graph.nodes[0].inputs = {"a", "b"};

  miniort::Tensor lhs;
  lhs.name = "a";
  lhs.dtype = "float32";
  lhs.shape = {2, 3};
  lhs.float_data = {1.f, 2.f, 3.f,
                    4.f, 5.f, 6.f};

  miniort::Tensor rhs;
  rhs.name = "b";
  rhs.dtype = "float32";
  rhs.shape = {3, 2};
  rhs.float_data = {7.f, 8.f,
                    9.f, 10.f,
                    11.f, 12.f};

  Session session = MakeCudaThenCpuSession(std::move(graph), SessionOptions{});
  miniort::ExecutionContext context;
  std::unordered_map<std::string, miniort::Tensor> feeds;
  feeds.emplace(lhs.name, lhs);
  feeds.emplace(rhs.name, rhs);

  const auto summary = session.Run(feeds, context, nullptr);
  Expect(summary.executed_nodes == 1, "expected CUDA MatMul graph to execute one node");
  Expect(summary.provider_executed_node_counts.contains("CUDA"), "expected CUDA execution count");
  Expect(summary.provider_executed_node_counts.at("CUDA") == 1, "expected CUDA to execute one node");

  const auto* output = context.FindTensor("out_0");
  Expect(output != nullptr, "expected CUDA MatMul output tensor");
  Expect(output->shape == std::vector<std::int64_t>({2, 2}), "expected CUDA MatMul output shape [2,2]");
  Expect(output->float_data.size() == 4, "expected CUDA MatMul output size 4");
  Expect(std::fabs(output->float_data[0] - 58.f) < 1e-5f, "unexpected CUDA MatMul output[0]");
  Expect(std::fabs(output->float_data[1] - 64.f) < 1e-5f, "unexpected CUDA MatMul output[1]");
  Expect(std::fabs(output->float_data[2] - 139.f) < 1e-5f, "unexpected CUDA MatMul output[2]");
  Expect(std::fabs(output->float_data[3] - 154.f) < 1e-5f, "unexpected CUDA MatMul output[3]");
}

void TestCudaGemmExecutionProducesExpectedOutput() {
  auto graph = MakeGraphWithOps({"Gemm"});
  graph.nodes[0].inputs = {"a", "b", "c"};

  miniort::Tensor a;
  a.name = "a";
  a.dtype = "float32";
  a.shape = {2, 2};
  a.float_data = {1.f, 2.f,
                  3.f, 4.f};

  miniort::Tensor b;
  b.name = "b";
  b.dtype = "float32";
  b.shape = {2, 2};
  b.float_data = {5.f, 6.f,
                  7.f, 8.f};

  miniort::Tensor c;
  c.name = "c";
  c.dtype = "float32";
  c.shape = {2};
  c.float_data = {1.f, 2.f};

  Session session = MakeCudaThenCpuSession(std::move(graph), SessionOptions{});
  miniort::ExecutionContext context;
  std::unordered_map<std::string, miniort::Tensor> feeds;
  feeds.emplace(a.name, a);
  feeds.emplace(b.name, b);
  feeds.emplace(c.name, c);

  const auto summary = session.Run(feeds, context, nullptr);
  Expect(summary.executed_nodes == 1, "expected CUDA Gemm graph to execute one node");
  Expect(summary.provider_executed_node_counts.contains("CUDA"), "expected CUDA Gemm execution count");
  Expect(summary.provider_executed_node_counts.at("CUDA") == 1, "expected CUDA to execute one Gemm node");

  const auto* output = context.FindTensor("out_0");
  Expect(output != nullptr, "expected CUDA Gemm output tensor");
  Expect(output->shape == std::vector<std::int64_t>({2, 2}), "expected CUDA Gemm output shape [2,2]");
  Expect(output->float_data.size() == 4, "expected CUDA Gemm output size 4");
  Expect(std::fabs(output->float_data[0] - 20.f) < 1e-5f, "unexpected CUDA Gemm output[0]");
  Expect(std::fabs(output->float_data[1] - 24.f) < 1e-5f, "unexpected CUDA Gemm output[1]");
  Expect(std::fabs(output->float_data[2] - 44.f) < 1e-5f, "unexpected CUDA Gemm output[2]");
  Expect(std::fabs(output->float_data[3] - 52.f) < 1e-5f, "unexpected CUDA Gemm output[3]");
}

void TestCudaElementwiseExecutionProducesExpectedOutput() {
  auto graph = MakeGraphWithOps({"Sigmoid", "Tanh", "Add", "Mul", "Div", "SiLU"});
  graph.nodes[0].inputs = {"x"};
  graph.nodes[1].inputs = {"x"};
  graph.nodes[2].inputs = {"a", "b"};
  graph.nodes[3].inputs = {"a", "b"};
  graph.nodes[4].inputs = {"a", "scalar_two"};
  graph.nodes[5].inputs = {"x"};

  miniort::Tensor x;
  x.name = "x";
  x.dtype = "float32";
  x.shape = {3};
  x.float_data = {-1.f, 0.f, 1.f};

  miniort::Tensor a;
  a.name = "a";
  a.dtype = "float32";
  a.shape = {3};
  a.float_data = {2.f, 4.f, 6.f};

  miniort::Tensor b;
  b.name = "b";
  b.dtype = "float32";
  b.shape = {3};
  b.float_data = {1.f, 3.f, 5.f};

  miniort::Tensor scalar_two;
  scalar_two.name = "scalar_two";
  scalar_two.dtype = "float32";
  scalar_two.shape = {};
  scalar_two.float_data = {2.f};

  Session session = MakeCudaThenCpuSession(std::move(graph), SessionOptions{});
  miniort::ExecutionContext context;
  std::unordered_map<std::string, miniort::Tensor> feeds;
  feeds.emplace(x.name, x);
  feeds.emplace(a.name, a);
  feeds.emplace(b.name, b);
  feeds.emplace(scalar_two.name, scalar_two);

  const auto summary = session.Run(feeds, context, nullptr);
  Expect(summary.executed_nodes == 6, "expected CUDA elementwise graph to execute six nodes");
  Expect(summary.provider_executed_node_counts.contains("CUDA"), "expected CUDA elementwise execution count");
  Expect(summary.provider_executed_node_counts.at("CUDA") == 6, "expected CUDA to execute all elementwise nodes");

  const auto* sigmoid_output = context.FindTensor("out_0");
  Expect(sigmoid_output != nullptr, "expected CUDA Sigmoid output tensor");
  const std::vector<float> expected_sigmoid = {
      1.0f / (1.0f + std::exp(1.0f)),
      0.5f,
      1.0f / (1.0f + std::exp(-1.0f)),
  };
  for (std::size_t i = 0; i < expected_sigmoid.size(); ++i) {
    Expect(std::fabs(sigmoid_output->float_data[i] - expected_sigmoid[i]) < 1e-5f,
           "unexpected CUDA Sigmoid output");
  }

  const auto* tanh_output = context.FindTensor("out_1");
  Expect(tanh_output != nullptr, "expected CUDA Tanh output tensor");
  for (std::size_t i = 0; i < x.float_data.size(); ++i) {
    Expect(std::fabs(tanh_output->float_data[i] - std::tanh(x.float_data[i])) < 1e-5f,
           "unexpected CUDA Tanh output");
  }

  const auto* add_output = context.FindTensor("out_2");
  Expect(add_output != nullptr, "expected CUDA Add output tensor");
  const std::vector<float> expected_add = {3.f, 7.f, 11.f};
  Expect(add_output->float_data == expected_add, "unexpected CUDA Add output");

  const auto* mul_output = context.FindTensor("out_3");
  Expect(mul_output != nullptr, "expected CUDA Mul output tensor");
  const std::vector<float> expected_mul = {2.f, 12.f, 30.f};
  Expect(mul_output->float_data == expected_mul, "unexpected CUDA Mul output");

  const auto* div_output = context.FindTensor("out_4");
  Expect(div_output != nullptr, "expected CUDA Div output tensor");
  const std::vector<float> expected_div = {1.f, 2.f, 3.f};
  for (std::size_t i = 0; i < expected_div.size(); ++i) {
    Expect(std::fabs(div_output->float_data[i] - expected_div[i]) < 1e-5f,
           "unexpected CUDA Div output");
  }

  const auto* silu_output = context.FindTensor("out_5");
  Expect(silu_output != nullptr, "expected CUDA SiLU output tensor");
  for (std::size_t i = 0; i < x.float_data.size(); ++i) {
    const auto value = x.float_data[i];
    const auto expected = value * (1.0f / (1.0f + std::exp(-value)));
    Expect(std::fabs(silu_output->float_data[i] - expected) < 1e-5f,
           "unexpected CUDA SiLU output");
  }
}

void TestCudaConvExecutionProducesExpectedOutput() {
  auto graph = MakeGraphWithOps({"Conv"});
  graph.nodes[0].inputs = {"x", "w", "b"};

  miniort::Tensor x;
  x.name = "x";
  x.dtype = "float32";
  x.shape = {1, 1, 2, 2};
  x.float_data = {1.f, 2.f,
                  3.f, 4.f};

  miniort::Tensor w;
  w.name = "w";
  w.dtype = "float32";
  w.shape = {1, 1, 1, 1};
  w.float_data = {2.f};

  miniort::Tensor b;
  b.name = "b";
  b.dtype = "float32";
  b.shape = {1};
  b.float_data = {1.f};

  Session session = MakeCudaThenCpuSession(std::move(graph), SessionOptions{});
  miniort::ExecutionContext context;
  std::unordered_map<std::string, miniort::Tensor> feeds;
  feeds.emplace(x.name, x);
  feeds.emplace(w.name, w);
  feeds.emplace(b.name, b);

  const auto summary = session.Run(feeds, context, nullptr);
  Expect(summary.executed_nodes == 1, "expected CUDA Conv graph to execute one node");
  Expect(summary.provider_executed_node_counts.contains("CUDA"), "expected CUDA Conv execution count");
  Expect(summary.provider_executed_node_counts.at("CUDA") == 1, "expected CUDA to execute one Conv node");

  const auto* output = context.FindTensor("out_0");
  Expect(output != nullptr, "expected CUDA Conv output tensor");
  Expect(output->shape == std::vector<std::int64_t>({1, 1, 2, 2}), "expected CUDA Conv output shape [1,1,2,2]");
  const std::vector<float> expected = {3.f, 5.f, 7.f, 9.f};
  for (std::size_t i = 0; i < expected.size(); ++i) {
    Expect(std::fabs(output->float_data[i] - expected[i]) < 1e-5f, "unexpected CUDA Conv output value");
  }
}

void TestCudaMaxPoolExecutionProducesExpectedOutput() {
  auto graph = MakeGraphWithOps({"MaxPool"});
  graph.nodes[0].inputs = {"x"};
  graph.nodes[0].attributes["kernel_shape"].kind = miniort::AttributeValue::Kind::kInts;
  graph.nodes[0].attributes["kernel_shape"].ints = {2, 2};
  graph.nodes[0].attributes["strides"].kind = miniort::AttributeValue::Kind::kInts;
  graph.nodes[0].attributes["strides"].ints = {2, 2};
  graph.nodes[0].attributes["pads"].kind = miniort::AttributeValue::Kind::kInts;
  graph.nodes[0].attributes["pads"].ints = {0, 0, 0, 0};

  miniort::Tensor x;
  x.name = "x";
  x.dtype = "float32";
  x.shape = {1, 1, 4, 4};
  x.float_data = {
      1.f, 2.f, 3.f, 4.f,
      5.f, 6.f, 7.f, 8.f,
      9.f, 10.f, 11.f, 12.f,
      13.f, 14.f, 15.f, 16.f,
  };

  Session session = MakeCudaThenCpuSession(std::move(graph), SessionOptions{});
  miniort::ExecutionContext context;
  std::unordered_map<std::string, miniort::Tensor> feeds;
  feeds.emplace(x.name, x);

  const auto summary = session.Run(feeds, context, nullptr);
  Expect(summary.executed_nodes == 1, "expected CUDA MaxPool graph to execute one node");
  Expect(summary.provider_executed_node_counts.contains("CUDA"), "expected CUDA MaxPool execution count");
  Expect(summary.provider_executed_node_counts.at("CUDA") == 1, "expected CUDA to execute one MaxPool node");

  const auto* output = context.FindTensor("out_0");
  Expect(output != nullptr, "expected CUDA MaxPool output tensor");
  Expect(output->shape == std::vector<std::int64_t>({1, 1, 2, 2}), "expected CUDA MaxPool output shape [1,1,2,2]");
  const std::vector<float> expected = {6.f, 8.f, 14.f, 16.f};
  for (std::size_t i = 0; i < expected.size(); ++i) {
    Expect(std::fabs(output->float_data[i] - expected[i]) < 1e-5f, "unexpected CUDA MaxPool output value");
  }
}
#endif

void TestConvExecutionProducesExpectedOutput() {
  auto graph = MakeGraphWithOps({"Conv"});
  graph.nodes[0].inputs = {"x", "w", "b"};

  miniort::Tensor x;
  x.name = "x";
  x.dtype = "float32";
  x.shape = {1, 1, 2, 2};
  x.float_data = {1.f, 2.f,
                  3.f, 4.f};

  miniort::Tensor w;
  w.name = "w";
  w.dtype = "float32";
  w.shape = {1, 1, 1, 1};
  w.float_data = {2.f};

  miniort::Tensor b;
  b.name = "b";
  b.dtype = "float32";
  b.shape = {1};
  b.float_data = {1.f};

  Session session = MakeCpuSession(std::move(graph), SessionOptions{});
  miniort::ExecutionContext context;
  std::unordered_map<std::string, miniort::Tensor> feeds;
  feeds.emplace(x.name, x);
  feeds.emplace(w.name, w);
  feeds.emplace(b.name, b);

  const auto summary = session.Run(feeds, context, nullptr);
  Expect(summary.executed_nodes == 1, "expected Conv graph to execute one node");
  const auto* output = context.FindTensor("out_0");
  Expect(output != nullptr, "expected Conv output tensor");
  Expect(output->shape == std::vector<std::int64_t>({1, 1, 2, 2}), "expected Conv output shape [1,1,2,2]");
  const std::vector<float> expected = {3.f, 5.f, 7.f, 9.f};
  for (std::size_t i = 0; i < expected.size(); ++i) {
    Expect(std::fabs(output->float_data[i] - expected[i]) < 1e-5f, "unexpected Conv output value");
  }
}

void TestGemmExecutionProducesExpectedOutput() {
  auto graph = MakeGraphWithOps({"Gemm"});
  graph.nodes[0].inputs = {"a", "b", "c"};

  miniort::Tensor a;
  a.name = "a";
  a.dtype = "float32";
  a.shape = {2, 2};
  a.float_data = {1.f, 2.f,
                  3.f, 4.f};

  miniort::Tensor b;
  b.name = "b";
  b.dtype = "float32";
  b.shape = {2, 2};
  b.float_data = {5.f, 6.f,
                  7.f, 8.f};

  miniort::Tensor c;
  c.name = "c";
  c.dtype = "float32";
  c.shape = {2};
  c.float_data = {1.f, 2.f};

  Session session = MakeCpuSession(std::move(graph), SessionOptions{});
  miniort::ExecutionContext context;
  std::unordered_map<std::string, miniort::Tensor> feeds;
  feeds.emplace(a.name, a);
  feeds.emplace(b.name, b);
  feeds.emplace(c.name, c);

  const auto summary = session.Run(feeds, context, nullptr);
  Expect(summary.executed_nodes == 1, "expected Gemm graph to execute one node");

  const auto* output = context.FindTensor("out_0");
  Expect(output != nullptr, "expected Gemm output tensor");
  Expect(output->shape == std::vector<std::int64_t>({2, 2}), "expected Gemm output shape [2,2]");
  Expect(output->float_data.size() == 4, "expected Gemm output size 4");
  Expect(std::fabs(output->float_data[0] - 20.f) < 1e-5f, "unexpected Gemm output[0]");
  Expect(std::fabs(output->float_data[1] - 24.f) < 1e-5f, "unexpected Gemm output[1]");
  Expect(std::fabs(output->float_data[2] - 44.f) < 1e-5f, "unexpected Gemm output[2]");
  Expect(std::fabs(output->float_data[3] - 52.f) < 1e-5f, "unexpected Gemm output[3]");
}

void TestSiLUExecutionProducesExpectedOutput() {
  auto graph = MakeGraphWithOps({"SiLU"});
  graph.nodes[0].inputs = {"x"};

  miniort::Tensor x;
  x.name = "x";
  x.dtype = "float32";
  x.shape = {4};
  x.float_data = {-1.f, 0.f, 1.f, 2.f};

  Session session = MakeCpuSession(std::move(graph), SessionOptions{});
  miniort::ExecutionContext context;
  std::unordered_map<std::string, miniort::Tensor> feeds;
  feeds.emplace(x.name, x);

  const auto summary = session.Run(feeds, context, nullptr);
  Expect(summary.executed_nodes == 1, "expected SiLU graph to execute one node");
  const auto* output = context.FindTensor("out_0");
  Expect(output != nullptr, "expected SiLU output tensor");
  Expect(output->float_data.size() == 4, "expected SiLU output size 4");
  auto silu = [](float v) { return v * (1.0f / (1.0f + std::exp(-v))); };
  for (std::size_t i = 0; i < output->float_data.size(); ++i) {
    Expect(std::fabs(output->float_data[i] - silu(x.float_data[i])) < 1e-5f, "unexpected SiLU output value");
  }
}

void TestConvSiLUExecutionProducesExpectedOutput() {
  auto graph = MakeGraphWithOps({"ConvSiLU"});
  graph.nodes[0].inputs = {"x", "w", "b"};

  miniort::Tensor x;
  x.name = "x";
  x.dtype = "float32";
  x.shape = {1, 1, 2, 2};
  x.float_data = {-1.f, 0.f,
                  1.f, 2.f};

  miniort::Tensor w;
  w.name = "w";
  w.dtype = "float32";
  w.shape = {1, 1, 1, 1};
  w.float_data = {1.f};

  miniort::Tensor b;
  b.name = "b";
  b.dtype = "float32";
  b.shape = {1};
  b.float_data = {1.f};

  Session session = MakeCpuSession(std::move(graph), SessionOptions{});
  miniort::ExecutionContext context;
  std::unordered_map<std::string, miniort::Tensor> feeds;
  feeds.emplace(x.name, x);
  feeds.emplace(w.name, w);
  feeds.emplace(b.name, b);

  const auto summary = session.Run(feeds, context, nullptr);
  Expect(summary.executed_nodes == 1, "expected ConvSiLU graph to execute one node");
  const auto* output = context.FindTensor("out_0");
  Expect(output != nullptr, "expected ConvSiLU output tensor");
  Expect(output->shape == std::vector<std::int64_t>({1, 1, 2, 2}), "expected ConvSiLU output shape [1,1,2,2]");
  auto silu = [](float v) { return v * (1.0f / (1.0f + std::exp(-v))); };
  for (std::size_t i = 0; i < output->float_data.size(); ++i) {
    Expect(std::fabs(output->float_data[i] - silu(x.float_data[i] + 1.f)) < 1e-5f,
           "unexpected ConvSiLU output value");
  }
}

void TestTanhExecutionProducesExpectedOutput() {
  auto graph = MakeGraphWithOps({"Tanh"});
  graph.nodes[0].inputs = {"x"};

  miniort::Tensor x;
  x.name = "x";
  x.dtype = "float32";
  x.shape = {3};
  x.float_data = {-1.f, 0.f, 1.f};

  Session session = MakeCpuSession(std::move(graph), SessionOptions{});
  miniort::ExecutionContext context;
  std::unordered_map<std::string, miniort::Tensor> feeds;
  feeds.emplace(x.name, x);

  const auto summary = session.Run(feeds, context, nullptr);
  Expect(summary.executed_nodes == 1, "expected Tanh graph to execute one node");
  const auto* output = context.FindTensor("out_0");
  Expect(output != nullptr, "expected Tanh output tensor");
  for (std::size_t i = 0; i < x.float_data.size(); ++i) {
    Expect(std::fabs(output->float_data[i] - std::tanh(x.float_data[i])) < 1e-5f,
           "unexpected Tanh output value");
  }
}

void TestSqueezeExecutionProducesExpectedOutput() {
  auto graph = MakeGraphWithOps({"Squeeze"});
  graph.nodes[0].inputs = {"x", "axes"};

  miniort::Tensor x;
  x.name = "x";
  x.dtype = "float32";
  x.shape = {1, 2, 1, 3};
  x.float_data = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};

  miniort::Tensor axes;
  axes.name = "axes";
  axes.dtype = "int64";
  axes.shape = {2};
  axes.int64_data = {0, 2};

  Session session = MakeCpuSession(std::move(graph), SessionOptions{});
  miniort::ExecutionContext context;
  std::unordered_map<std::string, miniort::Tensor> feeds;
  feeds.emplace(x.name, x);
  feeds.emplace(axes.name, axes);

  const auto summary = session.Run(feeds, context, nullptr);
  Expect(summary.executed_nodes == 1, "expected Squeeze graph to execute one node");
  const auto* output = context.FindTensor("out_0");
  Expect(output != nullptr, "expected Squeeze output tensor");
  Expect(output->shape == std::vector<std::int64_t>({2, 3}), "expected squeezed shape [2,3]");
  Expect(output->float_data == x.float_data, "expected Squeeze to preserve element order");
}

void TestConcatExecutionProducesExpectedOutput() {
  auto graph = MakeGraphWithOps({"Concat"});
  graph.nodes[0].inputs = {"a", "b"};
  graph.nodes[0].attributes["axis"].kind = miniort::AttributeValue::Kind::kInt;
  graph.nodes[0].attributes["axis"].int_value = 1;

  miniort::Tensor a;
  a.name = "a";
  a.dtype = "float32";
  a.shape = {2, 2};
  a.float_data = {1.f, 2.f,
                  3.f, 4.f};

  miniort::Tensor b;
  b.name = "b";
  b.dtype = "float32";
  b.shape = {2, 1};
  b.float_data = {5.f,
                  6.f};

  Session session = MakeCpuSession(std::move(graph), SessionOptions{});
  miniort::ExecutionContext context;
  std::unordered_map<std::string, miniort::Tensor> feeds;
  feeds.emplace(a.name, a);
  feeds.emplace(b.name, b);

  const auto summary = session.Run(feeds, context, nullptr);
  Expect(summary.executed_nodes == 1, "expected Concat graph to execute one node");
  const auto* output = context.FindTensor("out_0");
  Expect(output != nullptr, "expected Concat output tensor");
  Expect(output->shape == std::vector<std::int64_t>({2, 3}), "expected Concat output shape [2,3]");
  const std::vector<float> expected = {1.f, 2.f, 5.f, 3.f, 4.f, 6.f};
  Expect(output->float_data == expected, "unexpected Concat output");
}

void TestTransposeExecutionProducesExpectedOutput() {
  auto graph = MakeGraphWithOps({"Transpose"});
  graph.nodes[0].inputs = {"x"};
  graph.nodes[0].attributes["perm"].kind = miniort::AttributeValue::Kind::kInts;
  graph.nodes[0].attributes["perm"].ints = {1, 0};

  miniort::Tensor x;
  x.name = "x";
  x.dtype = "float32";
  x.shape = {2, 3};
  x.float_data = {1.f, 2.f, 3.f,
                  4.f, 5.f, 6.f};

  Session session = MakeCpuSession(std::move(graph), SessionOptions{});
  miniort::ExecutionContext context;
  std::unordered_map<std::string, miniort::Tensor> feeds;
  feeds.emplace(x.name, x);

  const auto summary = session.Run(feeds, context, nullptr);
  Expect(summary.executed_nodes == 1, "expected Transpose graph to execute one node");
  const auto* output = context.FindTensor("out_0");
  Expect(output != nullptr, "expected Transpose output tensor");
  Expect(output->shape == std::vector<std::int64_t>({3, 2}), "expected Transpose output shape [3,2]");
  const std::vector<float> expected = {1.f, 4.f, 2.f, 5.f, 3.f, 6.f};
  Expect(output->float_data == expected, "unexpected Transpose output");
}

void TestSessionFoldsInitializerTransposeToConstant() {
  Graph graph;
  graph.name = "fold_transpose_graph";

  miniort::Value weight;
  weight.name = "weight";
  weight.info.dtype = "float32";
  weight.info.shape = {"2", "3"};
  weight.info.is_initializer = true;
  miniort::TensorData weight_data;
  weight_data.dtype = "float32";
  weight_data.shape = {2, 3};
  weight_data.float_data = {1.f, 2.f, 3.f,
                            4.f, 5.f, 6.f};
  weight.data = weight_data;
  graph.initializers.emplace(weight.name, weight);

  Node transpose;
  transpose.name = "transpose";
  transpose.op_type = "Transpose";
  transpose.inputs = {"weight"};
  transpose.outputs = {"out_0"};
  miniort::AttributeValue perm;
  perm.kind = miniort::AttributeValue::Kind::kInts;
  perm.ints = {1, 0};
  transpose.attributes.emplace("perm", perm);

  graph.node_name_to_index[transpose.name] = 0;
  graph.topological_order = {0};
  graph.op_type_histogram["Transpose"] = 1;
  graph.nodes.push_back(std::move(transpose));

  Session session = MakeCpuSession(std::move(graph), SessionOptions{});
  const auto& optimized_node = session.graph().nodes[0];
  Expect(optimized_node.op_type == "Constant", "expected initializer Transpose to fold into Constant");
  Expect(optimized_node.inputs.empty(), "expected folded Constant to have no inputs");
  Expect(optimized_node.attributes.contains("value"), "expected folded Constant to contain value tensor");

  miniort::ExecutionContext context;
  const auto summary = session.Run({}, context, nullptr);
  Expect(summary.executed_nodes == 1, "expected folded graph to execute one node");
  const auto* output = context.FindTensor("out_0");
  Expect(output != nullptr, "expected folded transpose output tensor");
  Expect(output->shape == std::vector<std::int64_t>({3, 2}), "expected folded transpose output shape [3,2]");
  const std::vector<float> expected = {1.f, 4.f, 2.f, 5.f, 3.f, 6.f};
  Expect(output->float_data == expected, "unexpected folded transpose output");
}

void TestSoftmaxExecutionProducesExpectedOutput() {
  auto graph = MakeGraphWithOps({"Softmax"});
  graph.nodes[0].inputs = {"x"};
  graph.nodes[0].attributes["axis"].kind = miniort::AttributeValue::Kind::kInt;
  graph.nodes[0].attributes["axis"].int_value = 1;

  miniort::Tensor x;
  x.name = "x";
  x.dtype = "float32";
  x.shape = {2, 3};
  x.float_data = {1.f, 2.f, 3.f,
                  4.f, 5.f, 6.f};

  Session session = MakeCpuSession(std::move(graph), SessionOptions{});
  miniort::ExecutionContext context;
  std::unordered_map<std::string, miniort::Tensor> feeds;
  feeds.emplace(x.name, x);

  const auto summary = session.Run(feeds, context, nullptr);
  Expect(summary.executed_nodes == 1, "expected Softmax graph to execute one node");
  const auto* output = context.FindTensor("out_0");
  Expect(output != nullptr, "expected Softmax output tensor");
  Expect(output->shape == std::vector<std::int64_t>({2, 3}), "expected Softmax output shape [2,3]");
  const std::vector<float> expected = {
      0.09003057f, 0.24472848f, 0.66524094f,
      0.09003057f, 0.24472848f, 0.66524094f,
  };
  for (std::size_t i = 0; i < expected.size(); ++i) {
    Expect(std::fabs(output->float_data[i] - expected[i]) < 1e-5f, "unexpected Softmax output value");
  }
}

void TestSplitExecutionProducesExpectedOutput() {
  Graph graph;
  graph.name = "split_graph";

  Node data;
  data.name = "data_const";
  data.op_type = "Constant";
  data.outputs = {"data"};
  miniort::TensorData data_value;
  data_value.dtype = "float32";
  data_value.shape = {2, 4};
  data_value.float_data = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f};
  data.attributes["value"].tensor = data_value;

  Node split_sizes;
  split_sizes.name = "split_sizes_const";
  split_sizes.op_type = "Constant";
  split_sizes.outputs = {"split_sizes"};
  miniort::TensorData split_sizes_value;
  split_sizes_value.dtype = "int64";
  split_sizes_value.shape = {2};
  split_sizes_value.int64_data = {1, 3};
  split_sizes.attributes["value"].tensor = split_sizes_value;

  Node split;
  split.name = "split";
  split.op_type = "Split";
  split.inputs = {"data", "split_sizes"};
  split.outputs = {"left", "right"};
  split.attributes["axis"].kind = miniort::AttributeValue::Kind::kInt;
  split.attributes["axis"].int_value = 1;

  graph.node_name_to_index[data.name] = 0;
  graph.node_name_to_index[split_sizes.name] = 1;
  graph.node_name_to_index[split.name] = 2;
  graph.topological_order = {0, 1, 2};
  graph.nodes.push_back(std::move(data));
  graph.nodes.push_back(std::move(split_sizes));
  graph.nodes.push_back(std::move(split));

  Session session = MakeCpuSession(std::move(graph), SessionOptions{});
  miniort::ExecutionContext context;
  const auto summary = session.Run({}, context, nullptr);

  Expect(summary.executed_nodes == 3, "expected split graph to execute three nodes");

  const auto* left = context.FindTensor("left");
  Expect(left != nullptr, "expected left split output tensor");
  Expect(left->shape == std::vector<std::int64_t>({2, 1}), "expected left split shape [2,1]");
  Expect(left->float_data == std::vector<float>({1.f, 5.f}), "unexpected left split output");

  const auto* right = context.FindTensor("right");
  Expect(right != nullptr, "expected right split output tensor");
  Expect(right->shape == std::vector<std::int64_t>({2, 3}), "expected right split shape [2,3]");
  Expect(right->float_data == std::vector<float>({2.f, 3.f, 4.f, 6.f, 7.f, 8.f}),
         "unexpected right split output");
}

void TestLayerNormalizationExecutionProducesExpectedOutput() {
  auto graph = MakeGraphWithOps({"LayerNormalization"});
  graph.nodes[0].inputs = {"x", "scale", "bias"};
  graph.nodes[0].attributes["axis"].kind = miniort::AttributeValue::Kind::kInt;
  graph.nodes[0].attributes["axis"].int_value = -1;
  graph.nodes[0].attributes["epsilon"].kind = miniort::AttributeValue::Kind::kFloat;
  graph.nodes[0].attributes["epsilon"].float_value = 1e-5f;

  miniort::Tensor x;
  x.name = "x";
  x.dtype = "float32";
  x.shape = {2, 2};
  x.float_data = {1.f, 3.f, 2.f, 4.f};

  miniort::Tensor scale;
  scale.name = "scale";
  scale.dtype = "float32";
  scale.shape = {2};
  scale.float_data = {1.f, 1.f};

  miniort::Tensor bias;
  bias.name = "bias";
  bias.dtype = "float32";
  bias.shape = {2};
  bias.float_data = {0.f, 0.f};

  Session session = MakeCpuSession(std::move(graph), SessionOptions{});
  miniort::ExecutionContext context;
  std::unordered_map<std::string, miniort::Tensor> feeds;
  feeds.emplace(x.name, x);
  feeds.emplace(scale.name, scale);
  feeds.emplace(bias.name, bias);

  const auto summary = session.Run(feeds, context, nullptr);
  Expect(summary.executed_nodes == 1, "expected LayerNormalization graph to execute one node");
  const auto* output = context.FindTensor("out_0");
  Expect(output != nullptr, "expected LayerNormalization output tensor");
  const float inv_stddev = 1.0f / std::sqrt(1.0f + 1e-5f);
  const std::vector<float> expected = {-inv_stddev, inv_stddev, -inv_stddev, inv_stddev};
  for (std::size_t i = 0; i < expected.size(); ++i) {
    Expect(std::fabs(output->float_data[i] - expected[i]) < 1e-4f, "unexpected LayerNormalization output value");
  }
}

void TestWhereExecutionProducesExpectedOutput() {
  auto graph = MakeGraphWithOps({"Where"});
  graph.nodes[0].inputs = {"cond", "x", "y"};

  miniort::Tensor cond;
  cond.name = "cond";
  cond.dtype = "int64";
  cond.shape = {2, 2};
  cond.int64_data = {1, 0, 0, 1};

  miniort::Tensor x;
  x.name = "x";
  x.dtype = "float32";
  x.shape = {2, 2};
  x.float_data = {1.f, 2.f, 3.f, 4.f};

  miniort::Tensor y;
  y.name = "y";
  y.dtype = "float32";
  y.shape = {2, 2};
  y.float_data = {5.f, 6.f, 7.f, 8.f};

  Session session = MakeCpuSession(std::move(graph), SessionOptions{});
  miniort::ExecutionContext context;
  std::unordered_map<std::string, miniort::Tensor> feeds;
  feeds.emplace(cond.name, cond);
  feeds.emplace(x.name, x);
  feeds.emplace(y.name, y);

  const auto summary = session.Run(feeds, context, nullptr);
  Expect(summary.executed_nodes == 1, "expected Where graph to execute one node");
  const auto* output = context.FindTensor("out_0");
  Expect(output != nullptr, "expected Where output tensor");
  const std::vector<float> expected = {1.f, 6.f, 7.f, 4.f};
  for (std::size_t i = 0; i < expected.size(); ++i) {
    Expect(std::fabs(output->float_data[i] - expected[i]) < 1e-5f, "unexpected Where output value");
  }
}

void TestCastToBoolExecutionProducesExpectedOutput() {
  auto graph = MakeGraphWithOps({"Cast"});
  graph.nodes[0].inputs = {"x"};
  graph.nodes[0].attributes["to"].kind = miniort::AttributeValue::Kind::kInt;
  graph.nodes[0].attributes["to"].int_value = 9;

  miniort::Tensor x;
  x.name = "x";
  x.dtype = "float32";
  x.shape = {5};
  x.float_data = {-2.f, 0.f, 3.f, 0.f, -0.5f};

  Session session = MakeCpuSession(std::move(graph), SessionOptions{});
  miniort::ExecutionContext context;
  std::unordered_map<std::string, miniort::Tensor> feeds;
  feeds.emplace(x.name, x);

  const auto summary = session.Run(feeds, context, nullptr);
  Expect(summary.executed_nodes == 1, "expected Cast graph to execute one node");
  const auto* output = context.FindTensor("out_0");
  Expect(output != nullptr, "expected Cast output tensor");
  Expect(output->int64_data == std::vector<std::int64_t>({1, 0, 1, 0, 1}), "unexpected Cast(bool) output");
}

void TestQwenGapOpsExecutionProducesExpectedOutput() {
  auto graph = MakeGraphWithOps({"Neg", "Sqrt", "Equal", "Less", "ReduceMean", "Trilu"});
  graph.nodes[0].inputs = {"neg_x"};
  graph.nodes[1].inputs = {"sqrt_x"};
  graph.nodes[2].inputs = {"equal_lhs", "equal_rhs"};
  graph.nodes[3].inputs = {"less_lhs", "less_rhs"};
  graph.nodes[4].inputs = {"reduce_x", "reduce_axes"};
  graph.nodes[4].attributes["keepdims"].kind = miniort::AttributeValue::Kind::kInt;
  graph.nodes[4].attributes["keepdims"].int_value = 1;
  graph.nodes[5].inputs = {"trilu_x", "trilu_k"};
  graph.nodes[5].attributes["upper"].kind = miniort::AttributeValue::Kind::kInt;
  graph.nodes[5].attributes["upper"].int_value = 1;

  miniort::Tensor neg_x;
  neg_x.name = "neg_x";
  neg_x.dtype = "float32";
  neg_x.shape = {3};
  neg_x.float_data = {1.f, -2.f, 3.f};

  miniort::Tensor sqrt_x;
  sqrt_x.name = "sqrt_x";
  sqrt_x.dtype = "float32";
  sqrt_x.shape = {3};
  sqrt_x.float_data = {1.f, 4.f, 9.f};

  miniort::Tensor equal_lhs;
  equal_lhs.name = "equal_lhs";
  equal_lhs.dtype = "int64";
  equal_lhs.shape = {2, 2};
  equal_lhs.int64_data = {1, 2, 3, 4};

  miniort::Tensor equal_rhs;
  equal_rhs.name = "equal_rhs";
  equal_rhs.dtype = "int64";
  equal_rhs.shape = {2, 2};
  equal_rhs.int64_data = {1, 0, 3, 5};

  miniort::Tensor less_lhs;
  less_lhs.name = "less_lhs";
  less_lhs.dtype = "float32";
  less_lhs.shape = {2, 2};
  less_lhs.float_data = {1.f, 5.f, 3.f, 7.f};

  miniort::Tensor less_rhs;
  less_rhs.name = "less_rhs";
  less_rhs.dtype = "float32";
  less_rhs.shape = {2, 2};
  less_rhs.float_data = {2.f, 4.f, 3.f, 8.f};

  miniort::Tensor reduce_x;
  reduce_x.name = "reduce_x";
  reduce_x.dtype = "float32";
  reduce_x.shape = {2, 3};
  reduce_x.float_data = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};

  miniort::Tensor reduce_axes;
  reduce_axes.name = "reduce_axes";
  reduce_axes.dtype = "int64";
  reduce_axes.shape = {1};
  reduce_axes.int64_data = {1};

  miniort::Tensor trilu_x;
  trilu_x.name = "trilu_x";
  trilu_x.dtype = "float32";
  trilu_x.shape = {3, 3};
  trilu_x.float_data = {
      1.f, 2.f, 3.f,
      4.f, 5.f, 6.f,
      7.f, 8.f, 9.f,
  };

  miniort::Tensor trilu_k;
  trilu_k.name = "trilu_k";
  trilu_k.dtype = "int64";
  trilu_k.shape = {};
  trilu_k.int64_data = {0};

  Session session = MakeCpuSession(std::move(graph), SessionOptions{});
  miniort::ExecutionContext context;
  std::unordered_map<std::string, miniort::Tensor> feeds;
  feeds.emplace(neg_x.name, neg_x);
  feeds.emplace(sqrt_x.name, sqrt_x);
  feeds.emplace(equal_lhs.name, equal_lhs);
  feeds.emplace(equal_rhs.name, equal_rhs);
  feeds.emplace(less_lhs.name, less_lhs);
  feeds.emplace(less_rhs.name, less_rhs);
  feeds.emplace(reduce_x.name, reduce_x);
  feeds.emplace(reduce_axes.name, reduce_axes);
  feeds.emplace(trilu_x.name, trilu_x);
  feeds.emplace(trilu_k.name, trilu_k);

  const auto summary = session.Run(feeds, context, nullptr);
  Expect(summary.executed_nodes == 6, "expected qwen-gap graph to execute six nodes");

  const auto* neg_out = context.FindTensor("out_0");
  Expect(neg_out != nullptr, "expected Neg output");
  Expect(neg_out->float_data == std::vector<float>({-1.f, 2.f, -3.f}), "unexpected Neg output");

  const auto* sqrt_out = context.FindTensor("out_1");
  Expect(sqrt_out != nullptr, "expected Sqrt output");
  Expect(sqrt_out->float_data == std::vector<float>({1.f, 2.f, 3.f}), "unexpected Sqrt output");

  const auto* equal_out = context.FindTensor("out_2");
  Expect(equal_out != nullptr, "expected Equal output");
  Expect(equal_out->int64_data == std::vector<std::int64_t>({1, 0, 1, 0}), "unexpected Equal output");

  const auto* less_out = context.FindTensor("out_3");
  Expect(less_out != nullptr, "expected Less output");
  Expect(less_out->int64_data == std::vector<std::int64_t>({1, 0, 0, 1}), "unexpected Less output");

  const auto* reduce_out = context.FindTensor("out_4");
  Expect(reduce_out != nullptr, "expected ReduceMean output");
  Expect(reduce_out->shape == std::vector<std::int64_t>({2, 1}), "unexpected ReduceMean output shape");
  const std::vector<float> expected_reduce = {2.f, 5.f};
  for (std::size_t i = 0; i < expected_reduce.size(); ++i) {
    Expect(std::fabs(reduce_out->float_data[i] - expected_reduce[i]) < 1e-6f, "unexpected ReduceMean output value");
  }

  const auto* trilu_out = context.FindTensor("out_5");
  Expect(trilu_out != nullptr, "expected Trilu output");
  const std::vector<float> expected_trilu = {
      1.f, 2.f, 3.f,
      0.f, 5.f, 6.f,
      0.f, 0.f, 9.f,
  };
  Expect(trilu_out->float_data == expected_trilu, "unexpected Trilu output");
}

void TestMatMulExecutionSupportsBatchedInputs() {
  auto graph = MakeGraphWithOps({"MatMul"});
  graph.nodes[0].inputs = {"a", "b"};

  miniort::Tensor lhs;
  lhs.name = "a";
  lhs.dtype = "float32";
  lhs.shape = {1, 2, 2, 3};
  lhs.float_data = {
      1.f, 2.f, 3.f,
      4.f, 5.f, 6.f,
      7.f, 8.f, 9.f,
      10.f, 11.f, 12.f,
  };

  miniort::Tensor rhs;
  rhs.name = "b";
  rhs.dtype = "float32";
  rhs.shape = {1, 2, 3, 2};
  rhs.float_data = {
      1.f, 2.f,
      3.f, 4.f,
      5.f, 6.f,
      7.f, 8.f,
      9.f, 10.f,
      11.f, 12.f,
  };

  Session session = MakeCpuSession(std::move(graph), SessionOptions{});
  miniort::ExecutionContext context;
  std::unordered_map<std::string, miniort::Tensor> feeds;
  feeds.emplace(lhs.name, lhs);
  feeds.emplace(rhs.name, rhs);

  const auto summary = session.Run(feeds, context, nullptr);
  Expect(summary.executed_nodes == 1, "expected batched MatMul graph to execute one node");
  const auto* output = context.FindTensor("out_0");
  Expect(output != nullptr, "expected batched MatMul output tensor");
  Expect(output->shape == std::vector<std::int64_t>({1, 2, 2, 2}), "expected MatMul output shape [1,2,2,2]");
  const std::vector<float> expected = {
      22.f, 28.f,
      49.f, 64.f,
      220.f, 244.f,
      301.f, 334.f,
  };
  for (std::size_t i = 0; i < expected.size(); ++i) {
    Expect(std::fabs(output->float_data[i] - expected[i]) < 1e-5f, "unexpected batched MatMul output value");
  }
}

void TestGatherExecutionSupportsEmbeddingLookup() {
  auto graph = MakeGraphWithOps({"Gather"});
  graph.nodes[0].inputs = {"table", "indices"};

  miniort::Tensor table;
  table.name = "table";
  table.dtype = "float32";
  table.shape = {4, 3};
  table.float_data = {
      1.f, 2.f, 3.f,
      4.f, 5.f, 6.f,
      7.f, 8.f, 9.f,
      10.f, 11.f, 12.f,
  };

  miniort::Tensor indices;
  indices.name = "indices";
  indices.dtype = "int64";
  indices.shape = {2, 2};
  indices.int64_data = {2, 0, 3, 1};

  Session session = MakeCpuSession(std::move(graph), SessionOptions{});
  miniort::ExecutionContext context;
  std::unordered_map<std::string, miniort::Tensor> feeds;
  feeds.emplace(table.name, table);
  feeds.emplace(indices.name, indices);

  const auto summary = session.Run(feeds, context, nullptr);
  Expect(summary.executed_nodes == 1, "expected Gather graph to execute one node");
  const auto* output = context.FindTensor("out_0");
  Expect(output != nullptr, "expected Gather output tensor");
  Expect(output->shape == std::vector<std::int64_t>({2, 2, 3}), "expected Gather output shape [2,2,3]");
  const std::vector<float> expected = {
      7.f, 8.f, 9.f,
      1.f, 2.f, 3.f,
      10.f, 11.f, 12.f,
      4.f, 5.f, 6.f,
  };
  Expect(output->float_data == expected, "unexpected Gather embedding output");
}

void TestGatherExecutionSupportsNonZeroAxis() {
  auto graph = MakeGraphWithOps({"Gather"});
  graph.nodes[0].inputs = {"x", "indices"};
  graph.nodes[0].attributes["axis"].kind = miniort::AttributeValue::Kind::kInt;
  graph.nodes[0].attributes["axis"].int_value = 1;

  miniort::Tensor x;
  x.name = "x";
  x.dtype = "int64";
  x.shape = {2, 3};
  x.int64_data = {10, 11, 12, 20, 21, 22};

  miniort::Tensor indices;
  indices.name = "indices";
  indices.dtype = "int64";
  indices.shape = {2};
  indices.int64_data = {2, 0};

  Session session = MakeCpuSession(std::move(graph), SessionOptions{});
  miniort::ExecutionContext context;
  std::unordered_map<std::string, miniort::Tensor> feeds;
  feeds.emplace(x.name, x);
  feeds.emplace(indices.name, indices);

  const auto summary = session.Run(feeds, context, nullptr);
  Expect(summary.executed_nodes == 1, "expected Gather(axis=1) graph to execute one node");
  const auto* output = context.FindTensor("out_0");
  Expect(output != nullptr, "expected Gather(axis=1) output tensor");
  Expect(output->shape == std::vector<std::int64_t>({2, 2}), "expected Gather(axis=1) output shape [2,2]");
  const std::vector<std::int64_t> expected = {12, 10, 22, 20};
  Expect(output->int64_data == expected, "unexpected Gather(axis=1) output");
}

#if defined(__APPLE__)
void TestAppleDefaultProvidersPreferAccelerateForSupportedOps() {
  auto graph = MakeGraphWithOps({"Sigmoid", "SiLU", "Conv", "ConvSiLU", "Add", "Mul", "Sub", "Div", "MatMul", "Gemm"});

  Session session(std::move(graph), SessionOptions{});
  Expect(session.graph().nodes.size() == 10, "expected ten-node graph");
  for (const auto& node : session.graph().nodes) {
    Expect(node.execution_provider == "Accelerate",
           "expected supported elementwise op to prefer Accelerate on Apple");
  }
  Expect(session.assignment_summary().provider_node_counts.contains("Accelerate"),
         "expected Accelerate provider count");
  Expect(session.assignment_summary().provider_node_counts.at("Accelerate") == 10,
         "expected Accelerate to own ten nodes");
}

void TestAppleAccelerateSupportsQwenGapOps() {
  auto graph = MakeGraphWithOps({"Neg", "Sqrt", "Equal", "Less", "ReduceMean", "Trilu", "Cast"});
  graph.nodes[0].inputs = {"neg_x"};
  graph.nodes[1].inputs = {"sqrt_x"};
  graph.nodes[2].inputs = {"equal_lhs", "equal_rhs"};
  graph.nodes[3].inputs = {"less_lhs", "less_rhs"};
  graph.nodes[4].inputs = {"reduce_x", "reduce_axes"};
  graph.nodes[5].inputs = {"trilu_x", "trilu_k"};
  graph.nodes[6].inputs = {"cast_x"};
  graph.nodes[6].attributes["to"].kind = miniort::AttributeValue::Kind::kInt;
  graph.nodes[6].attributes["to"].int_value = 9;

  Session session(std::move(graph), SessionOptions{});
  for (const auto& node : session.graph().nodes) {
    Expect(node.execution_provider == "Accelerate", "expected Qwen gap ops to prefer Accelerate on Apple");
  }

  miniort::Tensor neg_x;
  neg_x.name = "neg_x";
  neg_x.dtype = "float32";
  neg_x.shape = {2};
  neg_x.float_data = {1.f, -3.f};

  miniort::Tensor sqrt_x;
  sqrt_x.name = "sqrt_x";
  sqrt_x.dtype = "float32";
  sqrt_x.shape = {2};
  sqrt_x.float_data = {1.f, 9.f};

  miniort::Tensor equal_lhs;
  equal_lhs.name = "equal_lhs";
  equal_lhs.dtype = "int64";
  equal_lhs.shape = {2};
  equal_lhs.int64_data = {4, 5};

  miniort::Tensor equal_rhs;
  equal_rhs.name = "equal_rhs";
  equal_rhs.dtype = "int64";
  equal_rhs.shape = {2};
  equal_rhs.int64_data = {4, 7};

  miniort::Tensor less_lhs;
  less_lhs.name = "less_lhs";
  less_lhs.dtype = "float32";
  less_lhs.shape = {2};
  less_lhs.float_data = {1.f, 3.f};

  miniort::Tensor less_rhs;
  less_rhs.name = "less_rhs";
  less_rhs.dtype = "float32";
  less_rhs.shape = {2};
  less_rhs.float_data = {2.f, 3.f};

  miniort::Tensor reduce_x;
  reduce_x.name = "reduce_x";
  reduce_x.dtype = "float32";
  reduce_x.shape = {1, 2};
  reduce_x.float_data = {2.f, 4.f};

  miniort::Tensor reduce_axes;
  reduce_axes.name = "reduce_axes";
  reduce_axes.dtype = "int64";
  reduce_axes.shape = {1};
  reduce_axes.int64_data = {1};

  miniort::Tensor trilu_x;
  trilu_x.name = "trilu_x";
  trilu_x.dtype = "float32";
  trilu_x.shape = {2, 2};
  trilu_x.float_data = {1.f, 2.f, 3.f, 4.f};

  miniort::Tensor trilu_k;
  trilu_k.name = "trilu_k";
  trilu_k.dtype = "int64";
  trilu_k.shape = {};
  trilu_k.int64_data = {0};

  miniort::Tensor cast_x;
  cast_x.name = "cast_x";
  cast_x.dtype = "float32";
  cast_x.shape = {4};
  cast_x.float_data = {-1.f, 0.f, 2.f, 0.f};

  miniort::ExecutionContext context;
  std::unordered_map<std::string, miniort::Tensor> feeds;
  feeds.emplace(neg_x.name, neg_x);
  feeds.emplace(sqrt_x.name, sqrt_x);
  feeds.emplace(equal_lhs.name, equal_lhs);
  feeds.emplace(equal_rhs.name, equal_rhs);
  feeds.emplace(less_lhs.name, less_lhs);
  feeds.emplace(less_rhs.name, less_rhs);
  feeds.emplace(reduce_x.name, reduce_x);
  feeds.emplace(reduce_axes.name, reduce_axes);
  feeds.emplace(trilu_x.name, trilu_x);
  feeds.emplace(trilu_k.name, trilu_k);
  feeds.emplace(cast_x.name, cast_x);

  const auto summary = session.Run(feeds, context, nullptr);
  Expect(summary.executed_nodes == 7, "expected Accelerate Qwen gap graph to execute seven nodes");
  const auto* cast_out = context.FindTensor("out_6");
  Expect(cast_out != nullptr, "expected Cast(bool) output tensor");
  Expect(cast_out->int64_data == std::vector<std::int64_t>({1, 0, 1, 0}),
         "unexpected Accelerate Cast(bool) output");
}

void TestAppleAccelerateSupportsGpt2HotPathOps() {
  auto graph = MakeGraphWithOps({"Add", "Tanh", "Softmax", "Where", "Gather", "Pow", "LayerNormalization", "MatMul"});
  graph.nodes[0].inputs = {"lhs", "rhs"};
  graph.nodes[1].inputs = {"tanh_x"};
  graph.nodes[2].inputs = {"softmax_x"};
  graph.nodes[3].inputs = {"cond", "where_x", "where_y"};
  graph.nodes[4].inputs = {"table", "indices"};
  graph.nodes[5].inputs = {"pow_x", "pow_y"};
  graph.nodes[6].inputs = {"ln_x", "ln_scale", "ln_bias"};
  graph.nodes[7].inputs = {"a", "b"};

  miniort::Tensor lhs;
  lhs.name = "lhs";
  lhs.dtype = "float32";
  lhs.shape = {2, 3};
  lhs.float_data = {1.f, 2.f, 3.f,
                    4.f, 5.f, 6.f};

  miniort::Tensor rhs;
  rhs.name = "rhs";
  rhs.dtype = "float32";
  rhs.shape = {3};
  rhs.float_data = {10.f, 20.f, 30.f};

  miniort::Tensor tanh_x;
  tanh_x.name = "tanh_x";
  tanh_x.dtype = "float32";
  tanh_x.shape = {3};
  tanh_x.float_data = {-1.f, 0.f, 1.f};

  miniort::Tensor softmax_x;
  softmax_x.name = "softmax_x";
  softmax_x.dtype = "float32";
  softmax_x.shape = {2, 2};
  softmax_x.float_data = {1.f, 2.f, 3.f, 4.f};

  miniort::Tensor cond;
  cond.name = "cond";
  cond.dtype = "int64";
  cond.shape = {2, 2};
  cond.int64_data = {1, 0, 0, 1};

  miniort::Tensor where_x;
  where_x.name = "where_x";
  where_x.dtype = "float32";
  where_x.shape = {2, 2};
  where_x.float_data = {1.f, 2.f, 3.f, 4.f};

  miniort::Tensor where_y;
  where_y.name = "where_y";
  where_y.dtype = "float32";
  where_y.shape = {2, 2};
  where_y.float_data = {5.f, 6.f, 7.f, 8.f};

  miniort::Tensor table;
  table.name = "table";
  table.dtype = "float32";
  table.shape = {4, 3};
  table.float_data = {
      1.f, 2.f, 3.f,
      4.f, 5.f, 6.f,
      7.f, 8.f, 9.f,
      10.f, 11.f, 12.f,
  };

  miniort::Tensor indices;
  indices.name = "indices";
  indices.dtype = "int64";
  indices.shape = {2};
  indices.int64_data = {2, 0};

  miniort::Tensor pow_x;
  pow_x.name = "pow_x";
  pow_x.dtype = "float32";
  pow_x.shape = {3};
  pow_x.float_data = {2.f, 3.f, 4.f};

  miniort::Tensor pow_y;
  pow_y.name = "pow_y";
  pow_y.dtype = "float32";
  pow_y.shape = {3};
  pow_y.float_data = {3.f, 2.f, 1.f};

  miniort::Tensor ln_x;
  ln_x.name = "ln_x";
  ln_x.dtype = "float32";
  ln_x.shape = {2, 2};
  ln_x.float_data = {1.f, 3.f, 2.f, 4.f};

  miniort::Tensor ln_scale;
  ln_scale.name = "ln_scale";
  ln_scale.dtype = "float32";
  ln_scale.shape = {2};
  ln_scale.float_data = {1.f, 1.f};

  miniort::Tensor ln_bias;
  ln_bias.name = "ln_bias";
  ln_bias.dtype = "float32";
  ln_bias.shape = {2};
  ln_bias.float_data = {0.f, 0.f};

  miniort::Tensor x;
  x.name = "x";
  x.dtype = "float32";
  x.shape = {2, 2};
  x.float_data = {1.f, 3.f,
                  2.f, 4.f};

  miniort::Tensor scale;
  scale.name = "scale";
  scale.dtype = "float32";
  scale.shape = {2};
  scale.float_data = {1.f, 1.f};

  miniort::Tensor bias;
  bias.name = "bias";
  bias.dtype = "float32";
  bias.shape = {2};
  bias.float_data = {0.f, 0.f};

  miniort::Tensor a;
  a.name = "a";
  a.dtype = "float32";
  a.shape = {1, 2, 2, 3};
  a.float_data = {
      1.f, 2.f, 3.f,
      4.f, 5.f, 6.f,
      7.f, 8.f, 9.f,
      10.f, 11.f, 12.f,
  };

  miniort::Tensor b;
  b.name = "b";
  b.dtype = "float32";
  b.shape = {1, 2, 3, 2};
  b.float_data = {
      1.f, 2.f,
      3.f, 4.f,
      5.f, 6.f,
      7.f, 8.f,
      9.f, 10.f,
      11.f, 12.f,
  };

  Session session(std::move(graph), SessionOptions{});
  for (const auto& node : session.graph().nodes) {
    Expect(node.execution_provider == "Accelerate", "expected GPT-2 hot path ops to prefer Accelerate");
  }

  miniort::ExecutionContext context;
  std::unordered_map<std::string, miniort::Tensor> feeds;
  feeds.emplace(lhs.name, lhs);
  feeds.emplace(rhs.name, rhs);
  feeds.emplace(tanh_x.name, tanh_x);
  feeds.emplace(softmax_x.name, softmax_x);
  feeds.emplace(cond.name, cond);
  feeds.emplace(where_x.name, where_x);
  feeds.emplace(where_y.name, where_y);
  feeds.emplace(table.name, table);
  feeds.emplace(indices.name, indices);
  feeds.emplace(pow_x.name, pow_x);
  feeds.emplace(pow_y.name, pow_y);
  feeds.emplace(ln_x.name, ln_x);
  feeds.emplace(ln_scale.name, ln_scale);
  feeds.emplace(ln_bias.name, ln_bias);
  feeds.emplace(a.name, a);
  feeds.emplace(b.name, b);

  const auto summary = session.Run(feeds, context, nullptr);
  Expect(summary.executed_nodes == 8, "expected GPT-2 hot path graph to execute eight nodes");

  const auto* add_output = context.FindTensor("out_0");
  Expect(add_output != nullptr, "expected broadcast Add output tensor");
  const std::vector<float> expected_add = {11.f, 22.f, 33.f, 14.f, 25.f, 36.f};
  Expect(add_output->float_data == expected_add, "unexpected broadcast Add output");

  const auto* tanh_output = context.FindTensor("out_1");
  Expect(tanh_output != nullptr, "expected Tanh output tensor");
  for (std::size_t i = 0; i < tanh_x.float_data.size(); ++i) {
    Expect(std::fabs(tanh_output->float_data[i] - std::tanh(tanh_x.float_data[i])) < 1e-5f,
           "unexpected Accelerate Tanh output");
  }

  const auto* softmax_output = context.FindTensor("out_2");
  Expect(softmax_output != nullptr, "expected Softmax output tensor");
  const float e1 = std::exp(1.f);
  const float e2 = std::exp(2.f);
  const float e3 = std::exp(3.f);
  const float e4 = std::exp(4.f);
  const std::vector<float> expected_softmax = {
      e1 / (e1 + e2), e2 / (e1 + e2),
      e3 / (e3 + e4), e4 / (e3 + e4),
  };
  for (std::size_t i = 0; i < expected_softmax.size(); ++i) {
    Expect(std::fabs(softmax_output->float_data[i] - expected_softmax[i]) < 1e-5f,
           "unexpected Accelerate Softmax output");
  }

  const auto* where_output = context.FindTensor("out_3");
  Expect(where_output != nullptr, "expected Where output tensor");
  const std::vector<float> expected_where = {1.f, 6.f, 7.f, 4.f};
  Expect(where_output->float_data == expected_where, "unexpected Accelerate Where output");

  const auto* gather_output = context.FindTensor("out_4");
  Expect(gather_output != nullptr, "expected Gather output tensor");
  const std::vector<float> expected_gather = {
      7.f, 8.f, 9.f,
      1.f, 2.f, 3.f,
  };
  Expect(gather_output->float_data == expected_gather, "unexpected Accelerate Gather output");

  const auto* pow_output = context.FindTensor("out_5");
  Expect(pow_output != nullptr, "expected Pow output tensor");
  const std::vector<float> expected_pow = {8.f, 9.f, 4.f};
  for (std::size_t i = 0; i < expected_pow.size(); ++i) {
    Expect(std::fabs(pow_output->float_data[i] - expected_pow[i]) < 1e-5f,
           "unexpected Accelerate Pow output");
  }

  const auto* ln_output = context.FindTensor("out_6");
  Expect(ln_output != nullptr, "expected LayerNormalization output tensor");
  const float inv_stddev = 1.0f / std::sqrt(1.0f + 1e-5f);
  const std::vector<float> expected_ln = {-inv_stddev, inv_stddev, -inv_stddev, inv_stddev};
  for (std::size_t i = 0; i < expected_ln.size(); ++i) {
    Expect(std::fabs(ln_output->float_data[i] - expected_ln[i]) < 1e-4f,
           "unexpected Accelerate LayerNormalization output");
  }

  const auto* mm_output = context.FindTensor("out_7");
  Expect(mm_output != nullptr, "expected batched MatMul output tensor");
  Expect(mm_output->shape == std::vector<std::int64_t>({1, 2, 2, 2}), "expected batched MatMul output shape");
  const std::vector<float> expected_mm = {
      22.f, 28.f,
      49.f, 64.f,
      220.f, 244.f,
      301.f, 334.f,
  };
  for (std::size_t i = 0; i < expected_mm.size(); ++i) {
    Expect(std::fabs(mm_output->float_data[i] - expected_mm[i]) < 1e-5f,
           "unexpected Accelerate batched MatMul output");
  }
}
#endif

}  // namespace

int main() {
  try {
  TestAssignmentSummaryMarksSupportedAndUnsupportedOps();
  TestSessionRejectsUnassignedNodesWhenConfigured();
  TestRunInjectsAllocatorIntoExecutionContext();
  TestIdentityExecutionPassesThroughTensorData();
  TestGptCacheBindingMatchesExpectedSchema();
  TestGptCacheBindingRejectsMalformedSchemas();
  TestGptCacheBindingSupportsGenericTensorNames();
  TestEmptyInt64ConstantSurvivesAndFeedsConstantOfShape();
  TestMatMulExecutionProducesExpectedOutput();
#if defined(MINIORT_BUILD_CUDA_EP)
    TestCudaProviderPrefersMatMulAndFallsBackForUnsupportedOps();
    TestCudaMatMulExecutionProducesExpectedOutput();
    TestCudaGemmExecutionProducesExpectedOutput();
    TestCudaElementwiseExecutionProducesExpectedOutput();
    TestCudaConvExecutionProducesExpectedOutput();
    TestCudaMaxPoolExecutionProducesExpectedOutput();
#endif
    TestConvExecutionProducesExpectedOutput();
    TestGemmExecutionProducesExpectedOutput();
    TestSiLUExecutionProducesExpectedOutput();
    TestConvSiLUExecutionProducesExpectedOutput();
    TestTanhExecutionProducesExpectedOutput();
    TestSqueezeExecutionProducesExpectedOutput();
    TestConcatExecutionProducesExpectedOutput();
    TestTransposeExecutionProducesExpectedOutput();
    TestSessionFoldsInitializerTransposeToConstant();
    TestSoftmaxExecutionProducesExpectedOutput();
    TestSplitExecutionProducesExpectedOutput();
    TestLayerNormalizationExecutionProducesExpectedOutput();
    TestWhereExecutionProducesExpectedOutput();
    TestCastToBoolExecutionProducesExpectedOutput();
    TestQwenGapOpsExecutionProducesExpectedOutput();
    TestMatMulExecutionSupportsBatchedInputs();
    TestGatherExecutionSupportsEmbeddingLookup();
    TestGatherExecutionSupportsNonZeroAxis();
#if defined(__APPLE__)
    TestAppleDefaultProvidersPreferAccelerateForSupportedOps();
    TestAppleAccelerateSupportsQwenGapOps();
    TestAppleAccelerateSupportsGpt2HotPathOps();
#endif
    std::cout << "runtime_tests: ok\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "runtime_tests: fail: " << ex.what() << "\n";
    return 1;
  }
}
