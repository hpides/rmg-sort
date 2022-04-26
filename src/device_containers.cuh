#pragma once

#include <map>
#include <vector>

#include "buckets.h"
#include "constants.h"
#include "cuda_utils.cuh"
#include "device_allocator.cuh"
#include "device_histograms.cuh"
#include "double_device_buffer.cuh"

template <typename T>
class DeviceContainers {
 public:
  DeviceContainers(size_t chunk_size, const std::vector<int>& gpus, size_t num_blocks)
      : _double_device_buffers(gpus.size()),
        _device_allocators(gpus.size()),
        _second_device_allocators(gpus.size()),
        _histogram_buffers(gpus.size()),
        _histogram_maps(gpus.size()),
        _next_histogram_index(gpus.size(), 0),
        _chunk_size(chunk_size),
        _gpus(gpus) {
    _epsilon = 0.005 * chunk_size;
    _small_bucket_threshold = 0.01 * chunk_size;
    _local_sort_memory_overhead = radixmgpusort::CUB_SORT_TEMP_MEMORY + (chunk_size * sizeof(T) / 20);

    const size_t num_alloc_bytes = ((2 * _epsilon) + chunk_size) * sizeof(T) + _local_sort_memory_overhead;
    const size_t adjusted_chunk_size = num_alloc_bytes / sizeof(T);

    const size_t num_partition_passes = sizeof(T);
    _max_histograms_per_gpu = (gpus.size() - 1) * (num_partition_passes - 1) + 1;

    for (int i = 0; i < gpus.size(); i++) {
      _gpu_index[gpus[i]] = i;
    }

#pragma omp parallel for
    for (size_t g = 0; g < gpus.size(); g++) {
      CheckCudaError(cudaSetDevice(gpus[g]));

      _double_device_buffers[g].Allocate(adjusted_chunk_size, gpus[g]);

      _device_allocators[g].Malloc(reinterpret_cast<uint8_t*>(thrust::raw_pointer_cast(_double_device_buffers[g].GetTemp()->data())),
                                   reinterpret_cast<uint8_t*>(thrust::raw_pointer_cast(_double_device_buffers[g].GetCurrent()->data())),
                                   num_alloc_bytes);

      _second_device_allocators[g].Malloc(_prefix_sum_memory_overhead);

      _histogram_buffers[g].reserve(_max_histograms_per_gpu);

      for (size_t i = 0; i < _max_histograms_per_gpu; i++) {
        _histogram_buffers[g].emplace_back(gpus[g], gpus.size(), num_blocks);
      }
    }
  }

  ~DeviceContainers() {
    _double_device_buffers.clear();
    _double_device_buffers.shrink_to_fit();

#pragma omp parallel for
    for (size_t g = 0; g < _gpus.size(); g++) {
      _histogram_buffers[g].clear();
      _histogram_buffers[g].shrink_to_fit();
    }
  }

  thrust::device_vector<T>* GetKeys(int gpu) { return _double_device_buffers[_gpu_index[gpu]].GetCurrent(); }

  thrust::device_vector<T>* GetTemp(int gpu) { return _double_device_buffers[_gpu_index[gpu]].GetTemp(); }

  DoubleDeviceBuffer<T>* GetDoubleDeviceBuffer(int gpu) { return &_double_device_buffers[_gpu_index[gpu]]; }

  void FlipBuffers(int gpu) {
    _double_device_buffers[_gpu_index[gpu]].FlipBuffers();
    _device_allocators[_gpu_index[gpu]].Flip();
  }

  DeviceAllocator* GetDeviceAllocator(int gpu) { return &_device_allocators[_gpu_index[gpu]]; }

  DeviceAllocator* GetSecondaryDeviceAllocator(int gpu) { return &_second_device_allocators[_gpu_index[gpu]]; }

  DeviceHistograms* GetHistograms(int gpu, const BucketId& bucket_id) {
    if (_histogram_maps[_gpu_index[gpu]].count(bucket_id) > 0) {
      return &_histogram_buffers[_gpu_index[gpu]][_histogram_maps[_gpu_index[gpu]][bucket_id]];
    }

    return nullptr;
  }

  void AssignNewHistogramBuffer(int gpu, const BucketId& bucket_id) {
    size_t& next_index = _next_histogram_index[_gpu_index[gpu]];

    if (next_index < _max_histograms_per_gpu) {
      _histogram_maps[_gpu_index[gpu]].insert({bucket_id, next_index});
      next_index++;
    }
  }

  inline int GetGpuIndex(int gpu_id) { return _gpu_index[gpu_id]; }

  inline size_t GetEpsilon() const { return _epsilon; }

  inline size_t GetSmallBucketThreshold() const { return _small_bucket_threshold; }

  size_t GetLocalSortMemoryOverhead() { return 2 * _local_sort_memory_overhead; }

  size_t GetMemoryOverhead(int gpu) {
    size_t memory_in_bytes = 0;

    memory_in_bytes += _prefix_sum_memory_overhead;
    memory_in_bytes += _max_histograms_per_gpu * _histogram_buffers[_gpu_index[gpu]][0].GetMemoryInBytes();
    memory_in_bytes += 2 * _local_sort_memory_overhead;
    memory_in_bytes += 2 * _epsilon * 2 * sizeof(T);

    return memory_in_bytes;
  }

  size_t GetMemoryInBytes(int gpu) {
    size_t memory_in_bytes = GetMemoryOverhead(gpu);

    memory_in_bytes += 2 * _chunk_size * sizeof(T);

    return memory_in_bytes;
  }

 private:
  std::vector<DoubleDeviceBuffer<T>> _double_device_buffers;
  std::vector<DeviceAllocator> _device_allocators;
  std::vector<DeviceAllocator> _second_device_allocators;

  std::vector<int> _gpus;
  std::map<int, int> _gpu_index;

  std::vector<std::vector<DeviceHistograms>> _histogram_buffers;
  std::vector<std::map<BucketId, size_t, compareBucketIds>> _histogram_maps;
  std::vector<size_t> _next_histogram_index;
  size_t _max_histograms_per_gpu;

  size_t _chunk_size;
  size_t _epsilon;
  size_t _small_bucket_threshold;
  size_t _local_sort_memory_overhead;
  const size_t _prefix_sum_memory_overhead = 10000000;
};