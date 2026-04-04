#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "miniort/loader/onnx_loader.h"
#include "miniort/runtime/execution_context.h"
#include "miniort/runtime/memory_profile.h"
#include "miniort/runtime/profiling.h"
#include "miniort/runtime/session.h"
#include "miniort/tools/image_loader.h"

namespace {

struct Options {
  std::string model_path;
  std::string image_path;
};

Options ParseArgs(int argc, char* argv[]) {
  if (argc < 2) {
    throw std::runtime_error("usage: miniort_memory_trace <model.onnx> [--image path]");
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

  return options;
}

const char* MemoryClassLabel(const miniort::TensorMemoryProfile& profile) {
  if (profile.is_initializer) {
    return "initializer";
  }
  if (profile.is_input) {
    return "input";
  }
  if (profile.is_output) {
    return "output";
  }
  return "intermediate";
}

std::size_t SafeTopoIndex(std::size_t index) {
  return index == std::numeric_limits<std::size_t>::max() ? 0 : index;
}

bool IsAliveAtTopoIndex(const miniort::TensorMemoryProfile& profile, std::size_t topo_index) {
  if (profile.bytes == 0) {
    return false;
  }
  if (profile.is_initializer || profile.is_input || profile.is_output) {
    return true;
  }
  if (profile.producer_topo_index == std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  const auto last_use = SafeTopoIndex(profile.last_use_topo_index);
  return profile.producer_topo_index <= topo_index && topo_index < last_use;
}

std::string JoinNames(const std::vector<std::string>& names, std::size_t limit) {
  std::ostringstream oss;
  std::size_t count = 0;
  for (const auto& name : names) {
    if (count != 0) {
      oss << ", ";
    }
    oss << name;
    ++count;
    if (count >= limit) {
      if (names.size() > limit) {
        oss << ", ...";
      }
      break;
    }
  }
  return oss.str();
}

void PrintMemoryPlan(const miniort::MemoryProfile& plan, std::ostream& os, std::size_t limit) {
  os << "memory plan\n";
  os << "  tensors=" << plan.tensors.size() << "\n";
  os << "  initializer_count=" << plan.initializer_count << "\n";
  os << "  initializer_bytes=" << miniort::FormatBytes(plan.initializer_bytes) << "\n";
  os << "  estimated_peak_bytes=" << miniort::FormatBytes(plan.estimated_peak_bytes) << "\n";
  os << "  estimated_peak_live_tensors=" << plan.estimated_peak_live_tensors << "\n";
  os << "  tensor lifetimes:\n";
  for (std::size_t i = 0; i < plan.tensors.size() && i < limit; ++i) {
    const auto& tensor = plan.tensors[i];
    os << "    - " << tensor.name
       << " kind=" << MemoryClassLabel(tensor)
       << " bytes=" << miniort::FormatBytes(tensor.bytes)
       << " producer=" << (tensor.producer_topo_index == std::numeric_limits<std::size_t>::max()
                               ? std::string("-")
                               : std::to_string(tensor.producer_topo_index))
       << " first_use=" << (tensor.first_use_topo_index == std::numeric_limits<std::size_t>::max()
                                ? std::string("-")
                                : std::to_string(tensor.first_use_topo_index))
       << " last_use=" << (tensor.last_use_topo_index == std::numeric_limits<std::size_t>::max()
                               ? std::string("-")
                               : std::to_string(tensor.last_use_topo_index))
       << "\n";
  }
  if (plan.tensors.size() > limit) {
    os << "    - ...\n";
  }
}

struct RuntimeInitializerStats {
  std::size_t count{0};
  std::size_t bytes{0};
};

RuntimeInitializerStats CollectRuntimeInitializerStats(const miniort::ExecutionContext& context) {
  RuntimeInitializerStats stats;
  for (const auto& [name, tensor] : context.tensors()) {
    (void)name;
    if (!tensor.is_initializer) {
      continue;
    }
    ++stats.count;
    stats.bytes += miniort::EstimateTensorBytes(tensor);
  }
  return stats;
}

void PrintNodeMemorySnapshot(const miniort::MemoryProfile& plan, const miniort::ExecutionContext& context,
                             std::size_t topo_index,
                             const miniort::Node& node, std::ostream* trace, std::size_t live_limit,
                             std::size_t& peak_live_bytes, std::size_t& peak_live_tensors,
                             std::size_t& peak_topo_index) {
  std::size_t live_bytes = 0;
  std::size_t live_tensors = 0;
  std::vector<std::string> live_names;
  std::vector<std::string> reusable_now;

  for (const auto& [name, tensor] : context.tensors()) {
    const auto it = plan.tensor_to_index.find(name);
    if (it == plan.tensor_to_index.end()) {
      continue;
    }
    const auto& profile = plan.tensors[it->second];
    if (!IsAliveAtTopoIndex(profile, topo_index)) {
      if (profile.producer_topo_index != std::numeric_limits<std::size_t>::max() &&
          profile.last_use_topo_index == topo_index && !profile.is_initializer && !profile.is_input &&
          !profile.is_output) {
        reusable_now.push_back(name);
      }
      continue;
    }

    const auto bytes = miniort::EstimateTensorBytes(tensor);
    if (bytes == 0) {
      continue;
    }
    live_bytes += bytes;
    ++live_tensors;
    live_names.push_back(name);
  }

  if (live_bytes > peak_live_bytes) {
    peak_live_bytes = live_bytes;
    peak_live_tensors = live_tensors;
    peak_topo_index = topo_index;
  }

  if (trace != nullptr) {
    *trace << "  [mem] node[" << topo_index << "] " << node.name << " op=" << node.op_type
           << " live=" << live_tensors << " tensors"
           << " bytes=" << miniort::FormatBytes(live_bytes);
    if (!reusable_now.empty()) {
      *trace << " reusable_now=" << reusable_now.size();
    }
    *trace << "\n";

    if (!reusable_now.empty()) {
      *trace << "        reusable: " << JoinNames(reusable_now, live_limit) << "\n";
    }

    if (live_limit != 0 && !live_names.empty()) {
      *trace << "        live: " << JoinNames(live_names, live_limit) << "\n";
    }
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    const auto options = ParseArgs(argc, argv);
    constexpr std::size_t kLiveLimit = 8;
    constexpr std::size_t kContextDumpLimit = 12;
    constexpr std::size_t kMaxNodes = 20;
    std::ostream* trace = &std::cout;

    auto graph = miniort::LoadOnnxGraph(options.model_path, trace);
    if (graph.inputs.empty() || graph.outputs.empty()) {
      throw std::runtime_error("graph must have at least one input and one output");
    }

    const auto memory_plan = miniort::BuildMemoryProfile(graph);
    PrintMemoryPlan(memory_plan, std::cout, 24);

    std::unordered_map<std::string, miniort::Tensor> feeds;
    if (!options.image_path.empty()) {
      const auto& input = graph.inputs.front();
      feeds.emplace(input.name,
                    miniort::LoadImageAsNchwTensor(std::filesystem::path(options.image_path), input.name, input.info,
                                                   trace));
    }

    std::size_t peak_live_bytes = 0;
    std::size_t peak_live_tensors = 0;
    std::size_t peak_topo_index = 0;

    miniort::Session session(std::move(graph),
                             {.allow_missing_kernels = true,
                              .auto_bind_placeholder_inputs = true,
                              .evict_dead_tensors = true,
                              .max_nodes = kMaxNodes,
                              .after_node =
                                  [&](std::size_t topo_index, const miniort::Node& node,
                                      const miniort::ExecutionContext& context, std::ostream* hook_trace) {
                                    PrintNodeMemorySnapshot(memory_plan, context, topo_index, node,
                                                            hook_trace, kLiveLimit, peak_live_bytes,
                                                            peak_live_tensors, peak_topo_index);
                                  }});

    miniort::ExecutionContext context;
    const auto summary = session.Run(feeds, context, trace);
    const auto runtime_initializer_stats = CollectRuntimeInitializerStats(context);

    std::cout << "\nfinal_context\n";
    context.Dump(std::cout, kContextDumpLimit);
    std::cout << "\nmemory_summary\n";
    std::cout << "  initializer_count=" << memory_plan.initializer_count << "\n";
    std::cout << "  initializer_bytes=" << miniort::FormatBytes(memory_plan.initializer_bytes) << "\n";
    std::cout << "  runtime_initializer_count=" << runtime_initializer_stats.count << "\n";
    std::cout << "  runtime_initializer_bytes=" << miniort::FormatBytes(runtime_initializer_stats.bytes) << "\n";
    std::cout << "  approx_total_bytes_at_peak="
              << miniort::FormatBytes(memory_plan.initializer_bytes + peak_live_bytes) << "\n";
    std::cout << "  estimated_peak_bytes=" << miniort::FormatBytes(memory_plan.estimated_peak_bytes) << "\n";
    std::cout << "  observed_peak_bytes=" << miniort::FormatBytes(peak_live_bytes)
              << " at node_index=" << peak_topo_index << "\n";
    std::cout << "  observed_peak_live_tensors=" << peak_live_tensors << "\n";
    std::cout << "  summary executed=" << summary.executed_nodes
              << " skipped=" << summary.skipped_nodes
              << " materialized_outputs=" << summary.materialized_outputs
              << " released_tensors=" << summary.released_tensors << "\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return EXIT_FAILURE;
  }
}
