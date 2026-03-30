#pragma once

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iosfwd>
#include <ostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace miniort {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
using TimingMap = std::unordered_map<std::string, double>;

inline double DurationMs(TimePoint start, TimePoint end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

inline void AddTiming(TimingMap& timings, std::string label, double duration_ms) {
  timings[std::move(label)] += duration_ms;
}

class ScopedTimer {
 public:
  ScopedTimer(std::string label, std::ostream* trace = nullptr, double* sink = nullptr)
      : label_(std::move(label)), trace_(trace), sink_(sink), start_(Clock::now()) {}

  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;

  ~ScopedTimer() {
    const auto duration_ms = DurationMs(start_, Clock::now());
    if (sink_ != nullptr) {
      *sink_ += duration_ms;
    }
    if (trace_ != nullptr) {
      *trace_ << "  [time] " << label_ << ": " << std::fixed << std::setprecision(3) << duration_ms << " ms\n";
    }
  }

 private:
  std::string label_;
  std::ostream* trace_{nullptr};
  double* sink_{nullptr};
  TimePoint start_;
};

inline void PrintTimingSummary(const TimingMap& timings, std::ostream& os, std::string_view header) {
  if (timings.empty()) {
    return;
  }

  std::vector<std::pair<std::string, double>> entries;
  entries.reserve(timings.size());
  for (const auto& [label, duration_ms] : timings) {
    entries.emplace_back(label, duration_ms);
  }
  std::sort(entries.begin(), entries.end(),
            [](const auto& lhs, const auto& rhs) {
              if (lhs.second == rhs.second) {
                return lhs.first < rhs.first;
              }
              return lhs.second > rhs.second;
            });

  os << header << "\n";
  for (const auto& [label, duration_ms] : entries) {
    os << "  - " << label << ": " << std::fixed << std::setprecision(3) << duration_ms << " ms\n";
  }
}

}  // namespace miniort

