#pragma once

#include <iostream>

#ifdef DEBUG
#define CheckCudaError(instruction) \
  { AssertNoCudaError((instruction), __FILE__, __LINE__); }
#else
#define CheckCudaError(instruction) instruction
#endif

inline void AssertNoCudaError(cudaError_t error_code, const char* file, int line) {
  if (error_code != cudaSuccess) {
    std::cout << "Error: " << cudaGetErrorString(error_code) << " " << file << " " << line << "\n";
  }
}
