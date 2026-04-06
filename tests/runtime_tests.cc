#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "miniort/model/graph.h"
#include "miniort/runtime/execution_context.h"
#include "miniort/runtime/tensor.h"
#include "miniort/runtime/session.h"

namespace {

using miniort::Graph;
using miniort::Node;
using miniort::ProviderAssignmentPolicy;
using miniort::Session;
using miniort::SessionAssignmentSummary;
using miniort::SessionOptions;

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

  Session session(std::move(graph), SessionOptions{});
  miniort::ExecutionContext context;

  Expect(!context.HasAllocator(), "expected context to start without allocator");
  const auto summary = session.Run({}, context, nullptr);
  Expect(summary.executed_nodes == 0, "expected empty graph to execute zero nodes");
  Expect(context.HasAllocator(), "expected session to inject allocator during run");
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

  Session session(std::move(graph), SessionOptions{});
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

  Session session(std::move(graph), SessionOptions{});
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

  Session session(std::move(graph), SessionOptions{});
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

  Session session(std::move(graph), SessionOptions{});
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

  Session session(std::move(graph), SessionOptions{});
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
#endif

}  // namespace

int main() {
  try {
    TestAssignmentSummaryMarksSupportedAndUnsupportedOps();
    TestSessionRejectsUnassignedNodesWhenConfigured();
    TestRunInjectsAllocatorIntoExecutionContext();
    TestMatMulExecutionProducesExpectedOutput();
    TestConvExecutionProducesExpectedOutput();
    TestGemmExecutionProducesExpectedOutput();
    TestSiLUExecutionProducesExpectedOutput();
    TestConvSiLUExecutionProducesExpectedOutput();
#if defined(__APPLE__)
    TestAppleDefaultProvidersPreferAccelerateForSupportedOps();
#endif
    std::cout << "runtime_tests: ok\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "runtime_tests: fail: " << ex.what() << "\n";
    return 1;
  }
}
