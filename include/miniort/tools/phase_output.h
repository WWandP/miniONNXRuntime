#pragma once

#include <ostream>
#include <string_view>

namespace miniort {

inline void PrintPhaseBanner(std::ostream& os, std::string_view phase_id, std::string_view title,
                             std::string_view goal) {
  os << "\n============================================================\n";
  os << "[" << phase_id << "] " << title << "\n";
  os << "goal: " << goal << "\n";
  os << "============================================================\n";
}

inline void PrintPhaseStep(std::ostream& os, int step_index, int step_total, std::string_view title,
                           std::string_view detail = {}) {
  os << "\n[" << step_index << "/" << step_total << "] " << title << "\n";
  if (!detail.empty()) {
    os << "  " << detail << "\n";
  }
}

inline void PrintPhaseResult(std::ostream& os, std::string_view label, std::string_view detail = {}) {
  os << "\n[result] " << label << "\n";
  if (!detail.empty()) {
    os << "  " << detail << "\n";
  }
}

}  // namespace miniort
