#pragma once

#include "constants.h"
#include "cuda_utils.cuh"

class HostHistograms {
 public:
  HostHistograms(int gpu, size_t num_gpus) : _gpu(gpu) {
    _global_histogram.reserve(radixmgpusort::NUM_BUCKETS);
    _global_histogram.resize(radixmgpusort::NUM_BUCKETS);

    _global_prefix_sums.reserve(radixmgpusort::NUM_BUCKETS);
    _global_prefix_sums.resize(radixmgpusort::NUM_BUCKETS);

    _mgpu_striped_histogram.reserve((radixmgpusort::NUM_BUCKETS * num_gpus) + 1);
    _mgpu_striped_histogram.resize((radixmgpusort::NUM_BUCKETS * num_gpus) + 1);

    _bucket_to_gpu_map.reserve(radixmgpusort::NUM_BUCKETS * num_gpus);
    _bucket_to_gpu_map.resize(radixmgpusort::NUM_BUCKETS * num_gpus, -1);
  }

  ~HostHistograms() {
    _global_histogram.clear();
    _global_histogram.shrink_to_fit();

    _global_prefix_sums.clear();
    _global_prefix_sums.shrink_to_fit();

    _mgpu_striped_histogram.clear();
    _mgpu_striped_histogram.shrink_to_fit();

    _bucket_to_gpu_map.clear();
    _bucket_to_gpu_map.shrink_to_fit();
  }

  PinnedHostVector<uint64_t>* GetGlobalHistogram() { return &_global_histogram; }

  PinnedHostVector<uint64_t>* GetGlobalPrefixSums() { return &_global_prefix_sums; }

  PinnedHostVector<uint64_t>* GetMgpuStripedHistogram() { return &_mgpu_striped_histogram; }

  PinnedHostVector<int>* GetBucketToGpuMap() { return &_bucket_to_gpu_map; }

  int GetGpu() const { return _gpu; }

 private:
  PinnedHostVector<uint64_t> _global_histogram;
  PinnedHostVector<uint64_t> _global_prefix_sums;

  PinnedHostVector<uint64_t> _mgpu_striped_histogram;
  PinnedHostVector<int> _bucket_to_gpu_map;

  int _gpu;
};