#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "miniort/loader/onnx_loader.h"
#include "miniort/runtime/session.h"
#include "miniort/tools/phase_output.h"

namespace {

struct Options {
  std::string model_path;
};

Options ParseArgs(int argc, char* argv[]) {
  if (argc < 2) {
    throw std::runtime_error("usage: miniort_provider_segments <model.onnx>");
  }
  return Options{.model_path = argv[1]};
}

std::string FormatOpCounts(const std::unordered_map<std::string, std::size_t>& counts) {
  std::vector<std::pair<std::string, std::size_t>> entries(counts.begin(), counts.end());
  std::sort(entries.begin(), entries.end(),
            [](const auto& lhs, const auto& rhs) {
              if (lhs.second != rhs.second) {
                return lhs.second > rhs.second;
              }
              return lhs.first < rhs.first;
            });

  std::string result;
  for (std::size_t i = 0; i < entries.size(); ++i) {
    if (i != 0) {
      result += ", ";
    }
    result += entries[i].first + ":" + std::to_string(entries[i].second);
  }
  return result;
}

void PrintProviderSegmentSummary(const std::vector<miniort::ProviderSegment>& segments) {
  std::unordered_map<std::string, std::size_t> segment_counts_by_provider;
  std::unordered_map<std::string, std::size_t> node_counts_by_provider;
  std::unordered_map<std::string, std::size_t> max_segment_nodes_by_provider;

  for (const auto& segment : segments) {
    ++segment_counts_by_provider[segment.provider];
    node_counts_by_provider[segment.provider] += segment.node_count;
    auto& max_nodes = max_segment_nodes_by_provider[segment.provider];
    max_nodes = std::max(max_nodes, segment.node_count);
  }

  std::vector<std::string> providers;
  providers.reserve(segment_counts_by_provider.size());
  for (const auto& [provider, count] : segment_counts_by_provider) {
    (void)count;
    providers.push_back(provider);
  }
  std::sort(providers.begin(), providers.end());

  std::cout << "provider_segment_summary\n";
  std::cout << "  total_segments=" << segments.size() << "\n";
  for (const auto& provider : providers) {
    std::cout << "  - " << provider
              << ": segments=" << segment_counts_by_provider[provider]
              << " nodes=" << node_counts_by_provider[provider]
              << " max_segment_nodes=" << max_segment_nodes_by_provider[provider] << "\n";
  }
}

void PrintProviderSegments(const std::vector<miniort::ProviderSegment>& segments) {
  std::cout << "provider_segments\n";
  for (const auto& segment : segments) {
    std::cout << "  - segment[" << segment.index << "]"
              << " provider=" << segment.provider
              << " topo=[" << segment.start_topo << "," << segment.end_topo << "]"
              << " nodes=" << segment.node_count
              << " boundary_inputs=" << segment.boundary_inputs.size()
              << " boundary_outputs=" << segment.boundary_outputs.size()
              << " ops={" << FormatOpCounts(segment.op_counts) << "}\n";
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    const auto options = ParseArgs(argc, argv);
    miniort::PrintPhaseBanner(std::cout, "provider-segments", "Inspect Provider Segments",
                              "按拓扑顺序查看连续同 provider 节点段。");
    miniort::PrintPhaseStep(std::cout, 1, 3, "Load ONNX Graph", options.model_path);
    auto graph = miniort::LoadOnnxGraph(options.model_path);

    miniort::PrintPhaseStep(std::cout, 2, 3, "Assign Execution Providers",
                            "构造 Session 后读取每个节点的 provider。");
    const miniort::Session session(std::move(graph));

    miniort::PrintPhaseStep(std::cout, 3, 3, "Build Provider Segments",
                            "统计连续同 provider 段和边界 tensor 数。");
    const auto& segments = session.provider_segments();
    PrintProviderSegmentSummary(segments);
    PrintProviderSegments(segments);
    miniort::PrintPhaseResult(std::cout, "provider segment inspection complete",
                              "这些段是后续 subgraph/device-residency 优化的候选边界。");
    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return EXIT_FAILURE;
  }
}
