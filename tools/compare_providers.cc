#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "miniort/loader/onnx_loader.h"
#include "miniort/optimizer/graph_optimizer.h"
#include "miniort/runtime/cpu_execution_provider.h"
#include "miniort/runtime/execution_context.h"
#include "miniort/runtime/session.h"
#include "miniort/tools/image_loader.h"
#include "miniort/tools/phase_output.h"

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
  std::string model_path;
  std::string image_path;
  std::size_t repeat{1};
  std::size_t warmup{0};
  bool allow_missing{false};
  bool evict_dead_tensors{false};
  bool planned_memory_reuse{false};
  bool graph_opt{false};
  bool profile_op_types{false};
};

struct LatencyStats {
  double mean_ms{0.0};
  double min_ms{0.0};
  double p50_ms{0.0};
  double p95_ms{0.0};
  double max_ms{0.0};
};

struct OpProfileEntry {
  double total_ms{0.0};
  std::size_t count{0};
};

using OpProfileMap = std::unordered_map<std::string, OpProfileEntry>;

Options ParseArgs(int argc, char* argv[]) {
  if (argc < 4) {
    throw std::runtime_error(
        "usage: miniort_compare_providers <model.onnx> --image path [--repeat N] [--warmup N] [--allow-missing] "
        "[--evict-dead-tensors] [--planned-memory-reuse] [--graph-opt] [--profile-op-types]");
  }

  Options options;
  options.model_path = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--image" && i + 1 < argc) {
      options.image_path = argv[++i];
      continue;
    }
    if (arg == "--repeat" && i + 1 < argc) {
      options.repeat = static_cast<std::size_t>(std::stoull(argv[++i]));
      continue;
    }
    if (arg == "--warmup" && i + 1 < argc) {
      options.warmup = static_cast<std::size_t>(std::stoull(argv[++i]));
      continue;
    }
    if (arg == "--allow-missing") {
      options.allow_missing = true;
      continue;
    }
    if (arg == "--evict-dead-tensors") {
      options.evict_dead_tensors = true;
      continue;
    }
    if (arg == "--planned-memory-reuse") {
      options.planned_memory_reuse = true;
      options.evict_dead_tensors = true;
      continue;
    }
    if (arg == "--graph-opt") {
      options.graph_opt = true;
      continue;
    }
    if (arg == "--profile-op-types") {
      options.profile_op_types = true;
      continue;
    }
    throw std::runtime_error("unknown argument: " + arg);
  }
  if (options.image_path.empty()) {
    throw std::runtime_error("--image is required");
  }
  if (options.repeat == 0) {
    throw std::runtime_error("--repeat must be greater than 0");
  }
  return options;
}

void AccumulateProfile(const miniort::RunSummary& summary, OpProfileMap& profile) {
  for (const auto& [op_type, duration_ms] : summary.op_kernel_time_ms) {
    auto& entry = profile[op_type];
    entry.total_ms += duration_ms;
    if (const auto count_it = summary.op_executed_node_counts.find(op_type); count_it != summary.op_executed_node_counts.end()) {
      entry.count += count_it->second;
    }
  }
}

double RunOnce(miniort::Session& session, const std::unordered_map<std::string, miniort::Tensor>& feeds,
               OpProfileMap* profile) {
  miniort::ExecutionContext context;
  const auto start = Clock::now();
  const auto summary = session.Run(feeds, context, nullptr);
  const auto end = Clock::now();
  if (summary.executed_nodes == 0) {
    throw std::runtime_error("session executed zero nodes");
  }
  if (profile != nullptr) {
    AccumulateProfile(summary, *profile);
  }
  return std::chrono::duration<double, std::milli>(end - start).count();
}

double Percentile(const std::vector<double>& sorted_samples, double percentile) {
  if (sorted_samples.empty()) {
    throw std::runtime_error("cannot compute percentile for empty samples");
  }
  const auto rank = percentile / 100.0 * static_cast<double>(sorted_samples.size() - 1);
  const auto lower = static_cast<std::size_t>(rank);
  const auto upper = std::min(lower + 1, sorted_samples.size() - 1);
  const auto fraction = rank - static_cast<double>(lower);
  return sorted_samples[lower] * (1.0 - fraction) + sorted_samples[upper] * fraction;
}

LatencyStats SummarizeSamples(std::vector<double> samples) {
  if (samples.empty()) {
    throw std::runtime_error("no latency samples were collected");
  }

  double total_ms = 0.0;
  for (const auto sample : samples) {
    total_ms += sample;
  }

  std::sort(samples.begin(), samples.end());
  LatencyStats stats;
  stats.mean_ms = total_ms / static_cast<double>(samples.size());
  stats.min_ms = samples.front();
  stats.p50_ms = Percentile(samples, 50.0);
  stats.p95_ms = Percentile(samples, 95.0);
  stats.max_ms = samples.back();
  return stats;
}

LatencyStats RunBenchmark(miniort::Session& session, const std::unordered_map<std::string, miniort::Tensor>& feeds,
                          std::size_t warmup, std::size_t repeat, OpProfileMap* profile) {
  for (std::size_t i = 0; i < warmup; ++i) {
    (void)RunOnce(session, feeds, nullptr);
  }

  std::vector<double> samples;
  samples.reserve(repeat);
  for (std::size_t i = 0; i < repeat; ++i) {
    samples.push_back(RunOnce(session, feeds, profile));
  }
  return SummarizeSamples(std::move(samples));
}

void PrintOpProfile(const std::string& label, const OpProfileMap& profile, double measured_total_ms) {
  if (profile.empty()) {
    return;
  }
  std::vector<std::pair<std::string, OpProfileEntry>> entries;
  entries.reserve(profile.size());
  for (const auto& [op_type, entry] : profile) {
    entries.emplace_back(op_type, entry);
  }
  std::sort(entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
    if (lhs.second.total_ms == rhs.second.total_ms) {
      return lhs.first < rhs.first;
    }
    return lhs.second.total_ms > rhs.second.total_ms;
  });

  std::cout << label << "_op_profile\n";
  for (const auto& [op_type, entry] : entries) {
    const double avg_node_ms = entry.count == 0 ? 0.0 : entry.total_ms / static_cast<double>(entry.count);
    const double pct = measured_total_ms == 0.0 ? 0.0 : entry.total_ms / measured_total_ms * 100.0;
    std::cout << "  - op=" << op_type
              << " total_ms=" << entry.total_ms
              << " count=" << entry.count
              << " avg_node_ms=" << avg_node_ms
              << " pct_of_measured=" << pct << "\n";
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    const auto options = ParseArgs(argc, argv);
    miniort::PrintPhaseBanner(std::cout, "phase5", "Compare Execution Providers",
                              "看默认 provider 路径和纯 CPU 路径的差异。");
    miniort::PrintPhaseStep(std::cout, 1, 4, "Load ONNX Graph", options.model_path);
    auto graph = miniort::LoadOnnxGraph(options.model_path, nullptr);
    if (options.graph_opt) {
      graph = miniort::OptimizeGraph(std::move(graph),
                                     {.enable_constant_folding = true,
                                      .enable_dead_node_cleanup = true,
                                      .enable_shape_simplification = true},
                                     nullptr,
                                     nullptr);
    }
    if (graph.inputs.empty()) {
      throw std::runtime_error("graph has no inputs");
    }

    const auto& input = graph.inputs.front();
    std::unordered_map<std::string, miniort::Tensor> feeds;
    miniort::PrintPhaseStep(std::cout, 2, 4, "Prepare Runtime Input", options.image_path);
    feeds.emplace(input.name,
                  miniort::LoadImageAsNchwTensor(std::filesystem::path(options.image_path), input.name, input.info,
                                                 nullptr));

    miniort::SessionOptions session_options;
    session_options.auto_bind_placeholder_inputs = true;
    session_options.allow_missing_kernels = options.allow_missing;
    session_options.allow_unassigned_nodes = options.allow_missing;
    session_options.evict_dead_tensors = options.evict_dead_tensors;
    session_options.planned_memory_reuse = options.planned_memory_reuse;

    miniort::PrintPhaseStep(std::cout, 3, 4, "Create Sessions",
                            "分别构造默认 provider 路径和 CPU-only 路径。");
    miniort::Session mixed_session(graph, session_options);
    miniort::Session cpu_only_session(
        graph, std::vector<std::shared_ptr<const miniort::ExecutionProvider>>{
                   std::make_shared<miniort::CpuExecutionProvider>()},
        session_options);

    miniort::PrintPhaseStep(std::cout, 4, 4, "Run And Compare",
                            "关注 mean/p50/p95、delta_ms 和 speedup_pct。");
    OpProfileMap mixed_profile;
    OpProfileMap cpu_profile;
    const auto mixed_stats = RunBenchmark(mixed_session, feeds, options.warmup, options.repeat,
                                          options.profile_op_types ? &mixed_profile : nullptr);
    const auto cpu_stats = RunBenchmark(cpu_only_session, feeds, options.warmup, options.repeat,
                                        options.profile_op_types ? &cpu_profile : nullptr);
    const auto delta_ms = cpu_stats.mean_ms - mixed_stats.mean_ms;
    const auto speedup_pct = delta_ms / cpu_stats.mean_ms * 100.0;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "provider_compare\n";
    std::cout << "  warmup=" << options.warmup << "\n";
    std::cout << "  repeat=" << options.repeat << "\n";
    std::cout << "  allow_missing=" << (options.allow_missing ? "true" : "false") << "\n";
    std::cout << "  evict_dead_tensors=" << (options.evict_dead_tensors ? "true" : "false") << "\n";
    std::cout << "  planned_memory_reuse=" << (options.planned_memory_reuse ? "true" : "false") << "\n";
    std::cout << "  graph_opt=" << (options.graph_opt ? "true" : "false") << "\n";
    std::cout << "  profile_op_types=" << (options.profile_op_types ? "true" : "false") << "\n";
    std::cout << "  mixed_ms=" << mixed_stats.mean_ms << "\n";
    std::cout << "  cpu_only_ms=" << cpu_stats.mean_ms << "\n";
    std::cout << "  delta_ms=" << delta_ms << "\n";
    std::cout << "  speedup_pct=" << speedup_pct << "\n";
    std::cout << "  mixed_latency_ms mean=" << mixed_stats.mean_ms
              << " min=" << mixed_stats.min_ms
              << " p50=" << mixed_stats.p50_ms
              << " p95=" << mixed_stats.p95_ms
              << " max=" << mixed_stats.max_ms << "\n";
    std::cout << "  cpu_only_latency_ms mean=" << cpu_stats.mean_ms
              << " min=" << cpu_stats.min_ms
              << " p50=" << cpu_stats.p50_ms
              << " p95=" << cpu_stats.p95_ms
              << " max=" << cpu_stats.max_ms << "\n";
    if (options.profile_op_types) {
      PrintOpProfile("mixed", mixed_profile, mixed_stats.mean_ms * static_cast<double>(options.repeat));
      PrintOpProfile("cpu_only", cpu_profile, cpu_stats.mean_ms * static_cast<double>(options.repeat));
    }
    std::cout << "metric_guide\n";
    std::cout << "  mixed_ms: default provider path average latency; CUDA/Accelerate may be used when enabled\n";
    std::cout << "  cpu_only_ms: CPU-only provider average latency for the same model and input\n";
    std::cout << "  delta_ms: cpu_only_ms - mixed_ms; positive means the default provider path is faster\n";
    std::cout << "  speedup_pct: delta_ms / cpu_only_ms * 100; larger positive values mean more speedup\n";
    std::cout << "  p50/p95: median and tail latency across measured repeats, after warmup runs\n";
    miniort::PrintPhaseResult(std::cout, "phase5 complete", "你现在看到的是 provider 对比视角。");
    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return EXIT_FAILURE;
  }
}
