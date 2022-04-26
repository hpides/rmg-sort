#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

#include <iomanip>
#include <iostream>

typedef unsigned long long int uint64_cu;

template <typename T>
using PinnedHostVector = thrust::host_vector<T, thrust::system::cuda::experimental::pinned_allocator<T>>;

template <typename T>
void printHostVector(T* begin, size_t n) {
  for (size_t i = 0; i < n; i++) {
    std::cout << *(begin + i) << ",";
  }
  std::cout << std::endl;
}

template <typename T>
void printDeviceVector(const thrust::device_vector<T>& device_vector) {
  thrust::copy(device_vector.begin(), device_vector.end(), std::ostream_iterator<int>(std::cout, ","));
  std::cout << std::endl;
}

template <typename T>
void printHistogram(const thrust::device_vector<T>& device_vector, bool reduced = true) {
  for (size_t i = 0; i < device_vector.size(); i++) {
    if (reduced && device_vector[i] == 0) continue;
    std::cout << "[" << std::setfill('0') << std::setw(3) << i << "]: ";
    std::cout << std::setfill('0') << std::setw(10) << device_vector[i] << ",\t\t";
    if ((i + 1) % 4 == 0 || reduced) std::cout << "\n";
  }
  std::cout << std::endl;
}

template <typename T>
void printHistogram(const thrust::device_vector<T>& device_vector, bool reduced, size_t max) {
  for (size_t i = 0; i < max; i++) {
    if (reduced && device_vector[i] == 0) continue;
    std::cout << "[" << std::setfill('0') << std::setw(3) << i << "]: ";
    std::cout << std::setfill('0') << std::setw(10) << device_vector[i] << ",\t\t";
    if ((i + 1) % 4 == 0 || reduced) std::cout << "\n";
  }
  std::cout << std::endl;
}

class CudaEventTimer {
 public:
  CudaEventTimer() {}

  ~CudaEventTimer() { Destroy(); }

  void Create() {
    CheckCudaError(cudaEventCreate(&start_event));
    CheckCudaError(cudaEventCreate(&stop_event));
  }

  void Destroy() {
    CheckCudaError(cudaEventDestroy(start_event));
    CheckCudaError(cudaEventDestroy(stop_event));
  }

  void StartTimer(cudaStream_t& stream) { CheckCudaError(cudaEventRecord(start_event, stream)); }

  void StopTimer(cudaStream_t& stream) { CheckCudaError(cudaEventRecord(stop_event, stream)); }

  float SynchronizeTimer() {
    CheckCudaError(cudaEventSynchronize(stop_event));
    float milliseconds = 0.0;
    CheckCudaError(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
    return milliseconds;
  }

  cudaEvent_t start_event;
  cudaEvent_t stop_event;
};

size_t GetNumThreadBlocks(size_t num_keys, size_t keys_per_thread, size_t num_threads) {
  size_t num_key_groups = num_keys / keys_per_thread;
  if (num_keys % keys_per_thread != 0) num_key_groups++;

  size_t num_thread_blocks = 1;
  num_thread_blocks = num_key_groups / num_threads;
  if (num_key_groups % num_threads != 0) num_thread_blocks++;

  return num_thread_blocks;
}