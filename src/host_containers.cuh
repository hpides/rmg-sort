#pragma once

#include <map>
#include <vector>

#include "buckets.h"
#include "constants.h"
#include "cuda_utils.cuh"
#include "host_histograms.cuh"

template <typename T>
class HostContainers {
 public:
  HostContainers(const std::vector<int>& gpus)
      : _histogram_buffers(gpus.size()), _histogram_maps(gpus.size()), _next_histogram_index(gpus.size(), 0), _gpus(gpus) {
    const size_t num_partition_passes = sizeof(T);
    _max_histograms_per_gpu = (gpus.size() - 1) * (num_partition_passes - 1) + 1;

    for (int i = 0; i < gpus.size(); i++) {
      _gpu_index[gpus[i]] = i;
    }

#pragma omp parallel for
    for (size_t g = 0; g < gpus.size(); g++) {
      _histogram_buffers[g].reserve(_max_histograms_per_gpu);

      for (size_t i = 0; i < _max_histograms_per_gpu; i++) {
        _histogram_buffers[g].emplace_back(gpus[g], gpus.size());
      }
    }
  }

  ~HostContainers() {
#pragma omp parallel for
    for (size_t g = 0; g < _gpus.size(); g++) {
      _histogram_buffers[g].clear();
      _histogram_buffers[g].shrink_to_fit();
    }
  }

  HostHistograms* GetHistograms(int gpu, const BucketId& bucket_id) {
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

 private:
  std::vector<int> _gpus;
  std::map<int, int> _gpu_index;

  std::vector<std::vector<HostHistograms>> _histogram_buffers;
  std::vector<std::map<BucketId, size_t, compareBucketIds>> _histogram_maps;
  std::vector<size_t> _next_histogram_index;
  size_t _max_histograms_per_gpu;
};