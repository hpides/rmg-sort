#pragma once

#include "constants.h"
#include "cuda_utils.cuh"

class DeviceHistograms {
 public:
  DeviceHistograms(int gpu, size_t num_gpus, size_t num_blocks) : _gpu(gpu) {
    CheckCudaError(cudaSetDevice(gpu));

    CheckCudaError(cudaMallocHost(reinterpret_cast<void**>(&_non_empty_count), sizeof(size_t), cudaHostAllocMapped));
    *_non_empty_count = 0;
    _memory_in_bytes = 0;

    _global_histogram.reserve(radixmgpusort::NUM_BUCKETS);
    _global_histogram.resize(radixmgpusort::NUM_BUCKETS, 0);

    _memory_in_bytes += radixmgpusort::NUM_BUCKETS * sizeof(uint64_t);

    _global_prefix_sums.reserve(radixmgpusort::NUM_BUCKETS);
    _global_prefix_sums.resize(radixmgpusort::NUM_BUCKETS, 0);

    _memory_in_bytes += radixmgpusort::NUM_BUCKETS * sizeof(uint64_t);

    _global_scatter_offsets.reserve(radixmgpusort::NUM_BUCKETS);
    _global_scatter_offsets.resize(radixmgpusort::NUM_BUCKETS, 0);

    _memory_in_bytes += radixmgpusort::NUM_BUCKETS * sizeof(uint64_t);

    _block_local_histograms.reserve(radixmgpusort::NUM_BUCKETS * num_blocks);
    _block_local_histograms.resize(radixmgpusort::NUM_BUCKETS * num_blocks, 0);

    _memory_in_bytes += radixmgpusort::NUM_BUCKETS * num_blocks * sizeof(uint32_t);

    _mgpu_histograms.reserve(radixmgpusort::NUM_BUCKETS * num_gpus);
    _mgpu_histograms.resize(radixmgpusort::NUM_BUCKETS * num_gpus, 0);

    _memory_in_bytes += radixmgpusort::NUM_BUCKETS * num_gpus * sizeof(uint64_t);

    _mgpu_striped_histogram.reserve((radixmgpusort::NUM_BUCKETS * num_gpus) + 1);
    _mgpu_striped_histogram.resize((radixmgpusort::NUM_BUCKETS * num_gpus) + 1, 0);

    _memory_in_bytes += ((radixmgpusort::NUM_BUCKETS * num_gpus) + 1) * sizeof(uint64_t);

    _bucket_to_gpu_map.reserve(radixmgpusort::NUM_BUCKETS * num_gpus);
    _bucket_to_gpu_map.resize(radixmgpusort::NUM_BUCKETS * num_gpus, -1);

    _memory_in_bytes += radixmgpusort::NUM_BUCKETS * num_gpus * sizeof(int);
  }

  ~DeviceHistograms() {
    CheckCudaError(cudaSetDevice(_gpu));

    _global_histogram.clear();
    _global_histogram.shrink_to_fit();

    _global_prefix_sums.clear();
    _global_prefix_sums.shrink_to_fit();

    _global_scatter_offsets.clear();
    _global_scatter_offsets.shrink_to_fit();

    _block_local_histograms.clear();
    _block_local_histograms.shrink_to_fit();

    _mgpu_histograms.clear();
    _mgpu_histograms.shrink_to_fit();

    _mgpu_striped_histogram.clear();
    _mgpu_striped_histogram.shrink_to_fit();

    _bucket_to_gpu_map.clear();
    _bucket_to_gpu_map.shrink_to_fit();
  }

  thrust::device_vector<uint64_t>* GetGlobalHistogram() { return &_global_histogram; }

  thrust::device_vector<uint64_t>* GetGlobalPrefixSums() { return &_global_prefix_sums; }

  thrust::device_vector<uint64_t>* GetGlobalScatterOffsets() { return &_global_scatter_offsets; }

  thrust::device_vector<uint32_t>* GetBlockLocalHistograms() { return &_block_local_histograms; }

  thrust::device_vector<uint64_t>* GetMgpuHistograms() { return &_mgpu_histograms; }

  thrust::device_vector<uint64_t>* GetMgpuStripedHistogram() { return &_mgpu_striped_histogram; }

  thrust::device_vector<int>* GetBucketToGpuMap() { return &_bucket_to_gpu_map; }

  size_t GetMemoryInBytes() { return _memory_in_bytes; }

  size_t* GetNonEmptyCount() const { return _non_empty_count; }

 private:
  thrust::device_vector<uint64_t> _global_histogram;
  thrust::device_vector<uint64_t> _global_prefix_sums;
  thrust::device_vector<uint64_t> _global_scatter_offsets;
  thrust::device_vector<uint32_t> _block_local_histograms;

  thrust::device_vector<uint64_t> _mgpu_histograms;
  thrust::device_vector<uint64_t> _mgpu_striped_histogram;
  thrust::device_vector<int> _bucket_to_gpu_map;

  size_t* _non_empty_count;
  size_t _memory_in_bytes;
  int _gpu;
};