#pragma once

#include <assert.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <bitset>
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>

#include "buckets.h"
#include "constants.h"
#include "cuda_error.cuh"
#include "cuda_utils.cuh"
#include "device_containers.cuh"
#include "host_containers.cuh"
#include "radix_kernels.cuh"
#include "time.h"

template <typename T>
void ScatterKeysBenchmark(PinnedHostVector<T>* keys, std::vector<int> gpus, OutputMode output_mode) {
  const size_t num_keys = keys->size();
  const size_t keys_per_thread = sizeof(T) == 4 ? 12 : 6;  // number of keys per thread
  const size_t num_block_histograms_to_aggregate = 256;    // number of block-local histograms one thread aggregates into the global one
  const size_t shared_memory_size = keys_per_thread * radixmgpusort::NUM_THREADS_PER_BLOCK * sizeof(T);

  for (int g = 0; g < gpus.size(); g++) {
    CheckCudaError(cudaSetDevice(gpus[g]));
    CheckCudaError(cudaFuncSetCacheConfig(&ScatterKeys<uint32_t>, cudaFuncCachePreferShared));
    CheckCudaError(cudaFuncSetCacheConfig(&ScatterKeys<uint64_t>, cudaFuncCachePreferShared));
    CheckCudaError(cudaFuncSetCacheConfig(&ScatterKeys<float>, cudaFuncCachePreferShared));
    CheckCudaError(cudaFuncSetCacheConfig(&ScatterKeys<double>, cudaFuncCachePreferShared));

    CheckCudaError(cudaFuncSetAttribute(&ScatterKeys<uint32_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));
    CheckCudaError(cudaFuncSetAttribute(&ScatterKeys<uint64_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));
    CheckCudaError(cudaFuncSetAttribute(&ScatterKeys<float>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));
    CheckCudaError(cudaFuncSetAttribute(&ScatterKeys<double>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));
  }

  size_t num_fillers = (num_keys % gpus.size() != 0) ? (gpus.size() - num_keys % gpus.size()) : 0;
  size_t chunk_size = (num_keys + num_fillers) / gpus.size();  // number of keys per GPU chunk

  while (chunk_size < num_fillers) {
    gpus.resize(gpus.size() / 2);
    num_fillers = (num_keys % gpus.size() != 0) ? (gpus.size() - num_keys % gpus.size()) : 0;
    chunk_size = (num_keys + num_fillers) / gpus.size();
  }

  const size_t num_gpus = gpus.size();
  size_t num_blocks = GetNumThreadBlocks(chunk_size, keys_per_thread, radixmgpusort::NUM_THREADS_PER_BLOCK);

  std::cout << "chunk_size : " << chunk_size << std::endl;
  std::cout << "num_fillers: " << num_fillers << std::endl;
  std::cout << "num_blocks: " << num_blocks << std::endl;

  DeviceContainers<T>* device_containers = new DeviceContainers<T>(chunk_size, gpus, num_blocks);
  HostContainers<T>* host_containers = new HostContainers<T>(gpus);

  for (int g = 0; g < num_gpus; g++) {
    device_containers->AssignNewHistogramBuffer(gpus[g], BucketId());
    host_containers->AssignNewHistogramBuffer(gpus[g], BucketId());
  }

  std::cout << std::fixed << std::setprecision(4) << std::endl;
  std::cout << "Memory consumption:" << std::endl;
  std::cout << "=================================================" << std::endl;
  std::cout << "Input array size: " << (chunk_size * sizeof(T)) / 1000000.0 << " MB" << std::endl;
  std::cout << "Total allocated device memory: " << device_containers->GetMemoryInBytes(gpus[0]) / 1000000.0 << " MB" << std::endl;
  std::cout << "Memory overhead (over 2n): " << device_containers->GetMemoryOverhead(gpus[0]) / 1000000.0 << " MB" << std::endl;
  std::cout << "=================================================" << std::endl;
  std::cout << std::endl;

  std::vector<std::array<cudaStream_t, 2>> streams(num_gpus);

#pragma omp parallel for
  for (size_t g = 0; g < num_gpus; g++) {
    CheckCudaError(cudaSetDevice(gpus[g]));
    CheckCudaError(cudaStreamCreateWithFlags(&streams[g][0], cudaStreamNonBlocking));
    CheckCudaError(cudaStreamCreateWithFlags(&streams[g][1], cudaStreamNonBlocking));
  }

  std::vector<double> histogram_time(num_gpus, 0.0);
  std::vector<double> key_scatter_time(num_gpus, 0.0);

  auto start_sort = GetTimeNow();

#pragma omp parallel for
  for (size_t g = 0; g < num_gpus; g++) {
    CheckCudaError(cudaSetDevice(gpus[g]));

    size_t g_chunk_size = chunk_size - (g == num_gpus - 1 ? num_fillers : 0);

    CheckCudaError(cudaMemcpyAsync(thrust::raw_pointer_cast(device_containers->GetKeys(gpus[g])->data()),
                                   thrust::raw_pointer_cast(keys->data() + (chunk_size * g)), sizeof(T) * g_chunk_size,
                                   cudaMemcpyHostToDevice, streams[g][0]));

    CheckCudaError(cudaStreamSynchronize(streams[g][0]));

    auto start = GetTimeNow();

    ComputeHistogram<T><<<num_blocks, radixmgpusort::NUM_THREADS_PER_BLOCK, 0, streams[g][0]>>>(
        thrust::raw_pointer_cast(device_containers->GetKeys(gpus[g])->data()),
        thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], BucketId())->GetBlockLocalHistograms()->data()), g_chunk_size,
        keys_per_thread, sizeof(T) * radixmgpusort::NUM_RADIX_BITS);

    AggregateHistogram<T><<<(num_blocks / num_block_histograms_to_aggregate) + 1, radixmgpusort::NUM_THREADS_PER_BLOCK, 0, streams[g][0]>>>(
        reinterpret_cast<uint64_cu*>(
            thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], BucketId())->GetGlobalHistogram()->data())),
        thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], BucketId())->GetBlockLocalHistograms()->data()), num_blocks,
        num_block_histograms_to_aggregate);

    CheckCudaError(cudaStreamSynchronize(streams[g][0]));
    auto end = GetTimeNow();

    histogram_time[g] = GetMilliseconds(start, end);

    if (output_mode == OutputMode::DEBUG) {
      size_t global_checksum = 0;
      size_t block_local_checksum = 0;

      if (chunk_size < 100000) {
        global_checksum = std::accumulate(device_containers->GetHistograms(gpus[g], BucketId())->GetGlobalHistogram()->begin(),
                                          device_containers->GetHistograms(gpus[g], BucketId())->GetGlobalHistogram()->end(), 0llu);

        block_local_checksum =
            std::accumulate(device_containers->GetHistograms(gpus[g], BucketId())->GetBlockLocalHistograms()->begin(),
                            device_containers->GetHistograms(gpus[g], BucketId())->GetBlockLocalHistograms()->end(), 0llu);
      }

      std::cout << "On GPU " << gpus[g] << std::endl;
      std::cout << "Histogram checksum: " << global_checksum << std::endl;
      std::cout << "Block-local checksum: " << block_local_checksum << std::endl;
      std::cout << "Global histogram:" << std::endl;
      printHistogram<uint64_t>(*device_containers->GetHistograms(gpus[g], BucketId())->GetGlobalHistogram());

      if (chunk_size < 100000) {
        assert(global_checksum == g_chunk_size);
        assert(global_checksum == block_local_checksum);
      }
    }

    thrust::exclusive_scan(thrust::cuda::par(*device_containers->GetSecondaryDeviceAllocator(gpus[g])).on(streams[g][0]),
                           device_containers->GetHistograms(gpus[g], BucketId())->GetGlobalHistogram()->begin(),
                           device_containers->GetHistograms(gpus[g], BucketId())->GetGlobalHistogram()->end(),
                           device_containers->GetHistograms(gpus[g], BucketId())->GetGlobalPrefixSums()->begin());

    CheckHistogramSkewness<<<1, 1, 0, streams[g][0]>>>(
        thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], BucketId())->GetGlobalHistogram()->data()),
        device_containers->GetHistograms(gpus[g], BucketId())->GetNonEmptyCount());
    CheckCudaError(cudaStreamSynchronize(streams[g][0]));

    if (output_mode == OutputMode::DEBUG) {
      std::cout << "On GPU " << gpus[g] << std::endl;
      std::cout << "non empty count: " << *device_containers->GetHistograms(gpus[g], BucketId())->GetNonEmptyCount() << std::endl;
      std::cout << "global prefix sum: " << std::endl;
      printHistogram<uint64_t>(*device_containers->GetHistograms(gpus[g], BucketId())->GetGlobalPrefixSums(), false);
      if (num_keys < radixmgpusort::MAX_KEYS_DEBUG_PRINT) {
        std::cout << "block local histogram: " << std::endl;
        printHistogram<uint32_t>(*device_containers->GetHistograms(gpus[g], BucketId())->GetBlockLocalHistograms(), false,
                                 num_blocks * radixmgpusort::NUM_BUCKETS);
      }
    }

    start = GetTimeNow();
    if (*device_containers->GetHistograms(gpus[g], BucketId())->GetNonEmptyCount() > 1) {
      ScatterKeys<T><<<num_blocks, radixmgpusort::NUM_THREADS_PER_BLOCK, shared_memory_size, streams[g][0]>>>(
          thrust::raw_pointer_cast(device_containers->GetKeys(gpus[g])->data()),
          thrust::raw_pointer_cast(device_containers->GetTemp(gpus[g])->data()),
          thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], BucketId())->GetGlobalPrefixSums()->data()),
          thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], BucketId())->GetBlockLocalHistograms()->data()),
          reinterpret_cast<uint64_cu*>(
              thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], BucketId())->GetGlobalScatterOffsets()->data())),
          g_chunk_size, keys_per_thread, sizeof(T) * radixmgpusort::NUM_RADIX_BITS);
    }

    CheckCudaError(cudaStreamSynchronize(streams[g][0]));
    end = GetTimeNow();

    key_scatter_time[g] = GetMilliseconds(start, end);

    if (*device_containers->GetHistograms(gpus[g], BucketId())->GetNonEmptyCount() > 1) {
      device_containers->FlipBuffers(gpus[g]);
    }

    if (output_mode == OutputMode::DEBUG) {
      thrust::device_vector<uint64_t> global_histogram_check(radixmgpusort::NUM_BUCKETS, 0);

      ComputeHistogram<T><<<num_blocks, radixmgpusort::NUM_THREADS_PER_BLOCK, 0, streams[g][0]>>>(
          thrust::raw_pointer_cast(device_containers->GetKeys(gpus[g])->data()),
          thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], BucketId())->GetBlockLocalHistograms()->data()), g_chunk_size,
          keys_per_thread, sizeof(T) * radixmgpusort::NUM_RADIX_BITS);

      AggregateHistogram<T>
          <<<(num_blocks / num_block_histograms_to_aggregate) + 1, radixmgpusort::NUM_THREADS_PER_BLOCK, 0, streams[g][0]>>>(
              reinterpret_cast<uint64_cu*>(thrust::raw_pointer_cast(global_histogram_check.data())),
              thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], BucketId())->GetBlockLocalHistograms()->data()),
              num_blocks, num_block_histograms_to_aggregate);
      CheckCudaError(cudaStreamSynchronize(streams[g][0]));

      for (size_t i = 0; i < radixmgpusort::NUM_BUCKETS; i++) {
        if ((*device_containers->GetHistograms(gpus[g], BucketId())->GetGlobalHistogram())[i] != global_histogram_check[i]) {
          std::cout << "Error: Key scatter wrote back false keys!" << std::endl;
          std::cout << "Error: Key scatter wrote back false keys on GPU " << gpus[g] << "!" << std::endl;
          break;
        }
      }
    }
  }

  auto end_sort = GetTimeNow();

  std::cout << "Time duration breakdown:" << std::endl;
  std::cout << "=================================================" << std::endl;
  for (size_t g = 0; g < num_gpus; g++) {
    std::cout << "> On GPU " << gpus[g] << std::endl;

    std::cout << "Histogram time: \t";
    std::cout << histogram_time[g] << "ms";
    std::cout << std::endl << "Key scatter time: \t";
    std::cout << key_scatter_time[g] << "ms";
    std::cout << std::endl;
    std::cout << "- - - - - - - - - - - - - - - - - - - - - - - - -" << std::endl;
  }

  std::cout << "Total duration: " << GetMilliseconds(start_sort, end_sort) << "ms" << std::endl;

  delete device_containers;
  delete host_containers;

#pragma omp parallel for
  for (size_t g = 0; g < num_gpus; g++) {
    for (size_t i = 0; i < 2; i++) {
      CheckCudaError(cudaStreamDestroy(streams[g][i]));
    }
  }
}
