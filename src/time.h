#pragma once

#include <chrono>

inline std::chrono::time_point<std::chrono::high_resolution_clock> GetTimeNow() { return std::chrono::high_resolution_clock::now(); }

inline double GetMilliseconds(std::chrono::time_point<std::chrono::high_resolution_clock> start,
                              std::chrono::time_point<std::chrono::high_resolution_clock> end) {
  return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);
}
