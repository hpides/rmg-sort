#pragma once

#include <assert.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <bitset>
#include <cub/cub.cuh>
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
size_t DetectSpanningBuckets(DeviceContainers<T>* device_containers, HostContainers<T>* host_containers,
                             std::vector<std::vector<std::pair<int, BucketId>>>& spanning_buckets,
                             std::map<BucketId, std::vector<int>, compareBucketIds>& spanning_bucket_to_gpus_map, std::vector<int>& gpus,
                             size_t iteration, OutputMode output_mode) {
  size_t num_spanning_buckets = 0;
  size_t num_gpus = gpus.size();

  for (size_t g = 0; g < num_gpus; g++) {
    for (size_t s = 0; s < spanning_buckets[iteration - 1].size(); s++) {
      if (spanning_buckets[iteration - 1][s].first == gpus[g]) {
        for (size_t i = 0; i < radixmgpusort::NUM_BUCKETS; i++) {
          HostHistograms* this_host_histograms = host_containers->GetHistograms(gpus[g], spanning_buckets[iteration - 1][s].second);

          if (output_mode == OutputMode::DEBUG) {
            std::cout << "Bucket " << i << " of iteration " << iteration - 1 << " belongs to GPU(s): ";
            for (int x = 0; x < num_gpus; x++) {
              std::cout << (*this_host_histograms->GetBucketToGpuMap())[(i * num_gpus) + x] << ",";
            }
            std::cout << std::endl;
          }

          if ((*this_host_histograms->GetBucketToGpuMap())[(i * num_gpus) + 1] >= 0 &&
              (*this_host_histograms->GetGlobalHistogram())[i] > 0) {
            BucketId new_spanning_bucket = BucketId(iteration, i, &spanning_buckets[iteration - 1][s].second);

            spanning_buckets[iteration].push_back({gpus[g], new_spanning_bucket});

            if (spanning_bucket_to_gpus_map.count(new_spanning_bucket) > 0) {
              spanning_bucket_to_gpus_map[new_spanning_bucket].push_back(gpus[g]);
            } else {
              spanning_bucket_to_gpus_map.insert({new_spanning_bucket, {gpus[g]}});
              num_spanning_buckets++;
            }

            device_containers->AssignNewHistogramBuffer(gpus[g], new_spanning_bucket);
            host_containers->AssignNewHistogramBuffer(gpus[g], new_spanning_bucket);

            if (output_mode == OutputMode::DEBUG) {
              std::cout << "GPU " << gpus[g] << ": Spanning bucket i " << i << " on GPUs ";
              int x = 0;
              while ((*this_host_histograms->GetBucketToGpuMap())[(i * num_gpus) + x] >= 0 && x < num_gpus) {
                std::cout << (*this_host_histograms->GetBucketToGpuMap())[(i * num_gpus) + x] << ",";
                x++;
              }
              std::cout << " with " << (*this_host_histograms->GetGlobalHistogram())[i] << " keys, from "
                        << (*this_host_histograms->GetGlobalPrefixSums())[i] << " to "
                        << (*this_host_histograms->GetGlobalPrefixSums())[i] + (*this_host_histograms->GetGlobalHistogram())[i]
                        << std::endl;
            }
          }
        }
      }
    }
  }

  return num_spanning_buckets;
}

template <typename T>
void RadixMultiGpuSort(PinnedHostVector<T>* keys, std::vector<int> gpus, OutputMode output_mode) {
  const size_t max_partition_passes = sizeof(T);
  const size_t num_keys = keys->size();
  const size_t keys_per_thread = sizeof(T) == 4 ? 12 : 6;  // number of keys per thread
  const size_t num_block_histograms_to_aggregate = 256;    // number of block-local histograms one thread aggregates into the global one
  const size_t shared_memory_size = keys_per_thread * radixmgpusort::NUM_THREADS_PER_BLOCK * sizeof(T);

  // Shared memory buffer size for key scatter kernel:
  // 32-bit types: 12 keys * 1024 threads per block * 4 bytes = 49.152 KB
  // 64-bit types:  6 keys * 1024 threads per block * 8 bytes = 49.152 KB

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
  size_t num_partition_passes_needed = max_partition_passes;
  size_t num_thread_blocks = GetNumThreadBlocks(chunk_size, keys_per_thread, radixmgpusort::NUM_THREADS_PER_BLOCK);

  if (output_mode != OutputMode::CSV) {
    std::cout << "chunk_size : " << chunk_size << std::endl;
    std::cout << "num_fillers: " << num_fillers << std::endl;
    std::cout << "num_thread_blocks: " << num_thread_blocks << std::endl;
  } else {
    std::cout << chunk_size << "," << num_thread_blocks << ",";
  }

  std::vector<std::array<cudaStream_t, radixmgpusort::MAX_NUM_CONCURRENT_KERNELS>> streams(num_gpus);
  std::vector<std::array<cudaEvent_t, radixmgpusort::MAX_NUM_BUCKETS_FOR_REDUCED_SORTING>> events(num_gpus);
  std::vector<CudaEventTimer> cuda_timers(num_gpus);

#pragma omp parallel for
  for (size_t g = 0; g < num_gpus; g++) {
    CheckCudaError(cudaSetDevice(gpus[g]));

    cuda_timers[g].Create();

    for (size_t i = 0; i < radixmgpusort::MAX_NUM_CONCURRENT_KERNELS; i++) {
      CheckCudaError(cudaStreamCreateWithFlags(&streams[g][i], cudaStreamNonBlocking));
    }

    for (size_t i = 0; i < radixmgpusort::MAX_NUM_BUCKETS_FOR_REDUCED_SORTING; i++) {
      CheckCudaError(cudaEventCreate(&events[g][i]));
    }
  }

  // INFO: prefix sum and histogram exchange times are negligible
  std::vector<double> htod_copy_time(num_gpus, 0);
  std::vector<double> sort_time(num_gpus, 0);
  std::vector<double> dtoh_time(num_gpus, 0);

  std::vector<std::vector<double>> histogram_time(max_partition_passes, std::vector<double>(num_gpus, 0.0));
  std::vector<std::vector<double>> key_scatter_time(max_partition_passes, std::vector<double>(num_gpus, 0.0));

  double detect_spanning_buckets_time = 0.0;
  double key_swap_time = 0.0;
  double prepare_key_swap_time = 0.0;
  double total_sort_duration = 0.0;

  if (num_gpus == 1) {
    num_partition_passes_needed = 0;
    CheckCudaError(cudaSetDevice(gpus[0]));

    DoubleDeviceBuffer<T> double_device_buffer;
    DeviceAllocator device_allocator;

    const size_t num_alloc_bytes = chunk_size * sizeof(T) + radixmgpusort::CUB_SORT_TEMP_MEMORY;
    const size_t adjusted_chunk_size = num_alloc_bytes / sizeof(T);

    double_device_buffer.Allocate(adjusted_chunk_size, 0);

    device_allocator.Malloc(reinterpret_cast<uint8_t*>(thrust::raw_pointer_cast(double_device_buffer.GetTemp()->data())),
                            reinterpret_cast<uint8_t*>(thrust::raw_pointer_cast(double_device_buffer.GetCurrent()->data())),
                            num_alloc_bytes);

    if (output_mode == OutputMode::CSV) {
      std::cout << num_alloc_bytes / 1000000.0 << "," << radixmgpusort::CUB_SORT_TEMP_MEMORY / 1000000.0 << ","
                << radixmgpusort::CUB_SORT_TEMP_MEMORY / 1000000.0 << ",";
    }

    auto start_sort = GetTimeNow();
    auto start = GetTimeNow();

    CheckCudaError(cudaMemcpyAsync(thrust::raw_pointer_cast(double_device_buffer.GetCurrent()->data()),
                                   thrust::raw_pointer_cast(keys->data()), sizeof(T) * chunk_size, cudaMemcpyHostToDevice, streams[0][0]));

    CheckCudaError(cudaStreamSynchronize(streams[0][0]));
    auto end = GetTimeNow();

    htod_copy_time[0] = GetMilliseconds(start, end);
    start = end;

    thrust::sort(thrust::cuda::par(device_allocator).on(streams[0][0]), double_device_buffer.GetCurrent()->begin(),
                 double_device_buffer.GetCurrent()->begin() + chunk_size);

    end = GetTimeNow();
    sort_time[0] = GetMilliseconds(start, end);
    start = end;

    CheckCudaError(cudaMemcpyAsync(thrust::raw_pointer_cast(keys->data()),
                                   thrust::raw_pointer_cast(double_device_buffer.GetCurrent()->data()), sizeof(T) * chunk_size,
                                   cudaMemcpyDeviceToHost, streams[0][0]));

    CheckCudaError(cudaStreamSynchronize(streams[0][0]));
    end = GetTimeNow();
    dtoh_time[0] = GetMilliseconds(start, end);

    total_sort_duration = GetMilliseconds(start_sort, end);

  } else {
    DeviceContainers<T>* device_containers = new DeviceContainers<T>(chunk_size, gpus, num_thread_blocks);
    HostContainers<T>* host_containers = new HostContainers<T>(gpus);

    std::vector<uint64_t> gpu_global_offsets(num_gpus + 1, 0);

    // [iteration][gpu_id]: (bucket ids,...)
    std::vector<std::vector<std::pair<int, BucketId>>> spanning_buckets;
    spanning_buckets.reserve(max_partition_passes);
    spanning_buckets.resize(max_partition_passes);

    // [BucketId] --> <spanning_bucket_offset, (last pass spanning bucket fractions,...)>
    std::map<BucketId, std::pair<size_t, std::vector<LPSpanningBucketFraction>>, compareBucketIds> last_pass_spanning_buckets;

    // [BucketId] --> (gpu ids,...)
    std::map<BucketId, std::vector<int>, compareBucketIds> spanning_bucket_to_gpus_map;
    spanning_bucket_to_gpus_map.insert({BucketId(), {}});

    std::vector<std::vector<ReducedSortingBucket<T>>> reduced_sorting_buckets(num_gpus);
    std::vector<std::vector<std::future<void>>> sync_sort_and_copy_back_futures(num_gpus);

    size_t num_spanning_buckets = 1;

    for (int g = 0; g < num_gpus; g++) {
      spanning_buckets[0].push_back({gpus[g], BucketId()});
      spanning_bucket_to_gpus_map[BucketId()].push_back(gpus[g]);

      device_containers->AssignNewHistogramBuffer(gpus[g], BucketId());
      host_containers->AssignNewHistogramBuffer(gpus[g], BucketId());

      reduced_sorting_buckets[g].reserve(num_gpus * radixmgpusort::MAX_NUM_BUCKETS_FOR_REDUCED_SORTING);
      sync_sort_and_copy_back_futures[g].reserve(radixmgpusort::MAX_NUM_BUCKETS_FOR_REDUCED_SORTING);
    }

    std::cout << std::fixed << std::setprecision(4);

    if (output_mode != OutputMode::CSV) {
      std::cout << std::endl;
      std::cout << "Memory consumption:" << std::endl;
      std::cout << "=================================================" << std::endl;
      std::cout << "Input array size: " << (chunk_size * sizeof(T)) / 1000000.0 << " MB" << std::endl;
      std::cout << "Total allocated device memory: " << device_containers->GetMemoryInBytes(gpus[0]) / 1000000.0 << " MB" << std::endl;
      std::cout << "Memory overhead (over 2n): " << device_containers->GetMemoryOverhead(gpus[0]) / 1000000.0 << " MB" << std::endl;
      std::cout << "=================================================" << std::endl;
    } else {
      std::cout << device_containers->GetMemoryInBytes(gpus[0]) / 1000000.0 << ",";
      std::cout << device_containers->GetMemoryOverhead(gpus[0]) / 1000000.0 << ",";
      std::cout << device_containers->GetLocalSortMemoryOverhead() / 1000000.0 << ",";
    }

    auto start_sort = GetTimeNow();

    for (size_t iteration = 0; iteration < sizeof(T); iteration++) {
      if (output_mode == OutputMode::DEBUG) {
        std::cout << std::endl << "=== Iteration " << iteration << " ===" << std::endl;
      }

      if (iteration > 0) {
        num_spanning_buckets = 0;

        auto start = GetTimeNow();
        num_spanning_buckets = DetectSpanningBuckets<T>(device_containers, host_containers, spanning_buckets, spanning_bucket_to_gpus_map,
                                                        gpus, iteration, output_mode);
        auto end = GetTimeNow();
        detect_spanning_buckets_time += GetMilliseconds(start, end);

        if (output_mode == OutputMode::DEBUG) {
          std::cout << "num_spanning_buckets: " << num_spanning_buckets << std::endl;
        }
      }

      if (num_spanning_buckets == 0) {
        num_partition_passes_needed = iteration;
        break;
      }

#pragma omp parallel for
      for (size_t g = 0; g < num_gpus; g++) {
        CheckCudaError(cudaSetDevice(gpus[g]));

        size_t g_chunk_size = chunk_size - (g == num_gpus - 1 ? num_fillers : 0);

        if (iteration == 0) {
          auto start = GetTimeNow();
          CheckCudaError(cudaMemcpyAsync(thrust::raw_pointer_cast(device_containers->GetKeys(gpus[g])->data()),
                                         thrust::raw_pointer_cast(keys->data() + (chunk_size * g)), sizeof(T) * g_chunk_size,
                                         cudaMemcpyHostToDevice, streams[g][0]));

          CheckCudaError(cudaStreamSynchronize(streams[g][0]));
          auto end = GetTimeNow();
          htod_copy_time[g] = GetMilliseconds(start, end);
        }

        auto start = GetTimeNow();
        if (iteration == 0) {
          ComputeHistogram<T><<<num_thread_blocks, radixmgpusort::NUM_THREADS_PER_BLOCK, 0, streams[g][0]>>>(
              thrust::raw_pointer_cast(device_containers->GetKeys(gpus[g])->data()),
              thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], BucketId())->GetBlockLocalHistograms()->data()),
              g_chunk_size, keys_per_thread, (sizeof(T) - iteration) * radixmgpusort::NUM_RADIX_BITS);

          AggregateHistogram<T>
              <<<(num_thread_blocks / num_block_histograms_to_aggregate) + 1, radixmgpusort::NUM_THREADS_PER_BLOCK, 0, streams[g][0]>>>(
                  reinterpret_cast<uint64_cu*>(
                      thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], BucketId())->GetGlobalHistogram()->data())),
                  thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], BucketId())->GetBlockLocalHistograms()->data()),
                  num_thread_blocks, num_block_histograms_to_aggregate);
        } else {
          for (size_t s = 0; s < spanning_buckets[iteration].size(); s++) {
            if (spanning_buckets[iteration][s].first == gpus[g]) {
              BucketId& current_bucket = spanning_buckets[iteration][s].second;
              BucketId* predecessor = current_bucket.predecessor;

              size_t bucket_nr = current_bucket.bucket_number;
              size_t bucket_size = (*host_containers->GetHistograms(gpus[g], *predecessor)->GetGlobalHistogram())[bucket_nr];
              size_t local_num_thread_blocks = GetNumThreadBlocks(bucket_size, keys_per_thread, radixmgpusort::NUM_THREADS_PER_BLOCK);

              if (output_mode == OutputMode::DEBUG) {
                std::cout << "GPU " << gpus[g] << ": Compute Histogram on spanning bucket " << bucket_nr << " (" << bucket_size << " keys)"
                          << std::endl;
                std::cout << "Starting from GetKeys()->data() offset "
                          << (*host_containers->GetHistograms(gpus[g], *predecessor)->GetGlobalPrefixSums())[bucket_nr] << std::endl;
                std::cout << "Use " << local_num_thread_blocks << " blocks for CUDA kernel" << std::endl;
              }

              ComputeHistogram<T><<<local_num_thread_blocks, radixmgpusort::NUM_THREADS_PER_BLOCK, 0, streams[g][0]>>>(
                  thrust::raw_pointer_cast(device_containers->GetKeys(gpus[g])->data() +
                                           (*host_containers->GetHistograms(gpus[g], *predecessor)->GetGlobalPrefixSums())[bucket_nr]),
                  thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], current_bucket)->GetBlockLocalHistograms()->data()),
                  bucket_size, keys_per_thread, (sizeof(T) - iteration) * radixmgpusort::NUM_RADIX_BITS);

              AggregateHistogram<T><<<(local_num_thread_blocks / num_block_histograms_to_aggregate) + 1,
                                      radixmgpusort::NUM_THREADS_PER_BLOCK, 0, streams[g][0]>>>(
                  reinterpret_cast<uint64_cu*>(
                      thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], current_bucket)->GetGlobalHistogram()->data())),
                  thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], current_bucket)->GetBlockLocalHistograms()->data()),
                  local_num_thread_blocks, num_block_histograms_to_aggregate);
            }
          }
        }
        CheckCudaError(cudaStreamSynchronize(streams[g][0]));
        auto end = GetTimeNow();

        histogram_time[iteration][g] = GetMilliseconds(start, end);

        if (output_mode == OutputMode::DEBUG) {
          for (size_t s = 0; s < spanning_buckets[iteration].size(); s++) {
            if (spanning_buckets[iteration][s].first == gpus[g]) {
              BucketId& b = spanning_buckets[iteration][s].second;

              size_t global_checksum = 0;
              size_t block_local_checksum = 0;

              if (chunk_size < 100000) {
                global_checksum = std::accumulate(device_containers->GetHistograms(gpus[g], b)->GetGlobalHistogram()->begin(),
                                                  device_containers->GetHistograms(gpus[g], b)->GetGlobalHistogram()->end(), 0llu);

                block_local_checksum =
                    std::accumulate(device_containers->GetHistograms(gpus[g], b)->GetBlockLocalHistograms()->begin(),
                                    device_containers->GetHistograms(gpus[g], b)->GetBlockLocalHistograms()->end(), 0llu);
              }

              std::cout << "On GPU " << gpus[g] << std::endl;
              std::cout << "Histogram checksum: " << global_checksum << std::endl;
              std::cout << "Block-local checksum: " << block_local_checksum << std::endl;
              std::cout << "Global histogram:" << std::endl;
              printHistogram<uint64_t>(*device_containers->GetHistograms(gpus[g], b)->GetGlobalHistogram());

              if (iteration == 0 && chunk_size < 100000) {
                assert(global_checksum == g_chunk_size);
                assert(global_checksum == block_local_checksum);
              }
            }
          }
        }

        // Global histogram exchange between GPUs
        if (iteration == 0) {
          for (size_t dest_gpu = 0; dest_gpu < num_gpus; dest_gpu++) {
            CheckCudaError(cudaMemcpyAsync(
                thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[dest_gpu], BucketId())->GetMgpuHistograms()->data() +
                                         (g * radixmgpusort::NUM_BUCKETS)),
                thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], BucketId())->GetGlobalHistogram()->data()),
                sizeof(uint64_t) * radixmgpusort::NUM_BUCKETS, cudaMemcpyDeviceToDevice, streams[g][1]));
          }
        } else {
          for (size_t s = 0; s < spanning_buckets[iteration].size(); s++) {
            if (spanning_buckets[iteration][s].first == gpus[g]) {
              BucketId& spanning_bucket = spanning_buckets[iteration][s].second;

              for (auto dest_gpu : spanning_bucket_to_gpus_map[spanning_bucket]) {
                if (output_mode == OutputMode::DEBUG) {
                  std::cout << "Spanning bucket " << spanning_bucket.bucket_number << " of iteration " << spanning_bucket.partition_pass
                            << " on GPU " << gpus[g];
                  std::cout << " sends its histogram to GPU " << dest_gpu << std::endl;
                }
                CheckCudaError(cudaMemcpyAsync(
                    thrust::raw_pointer_cast(device_containers->GetHistograms(dest_gpu, spanning_bucket)->GetMgpuHistograms()->data() +
                                             (g * radixmgpusort::NUM_BUCKETS)),
                    thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], spanning_bucket)->GetGlobalHistogram()->data()),
                    sizeof(uint64_t) * radixmgpusort::NUM_BUCKETS, cudaMemcpyDeviceToDevice, streams[g][1]));
              }
            }
          }
        }

        for (size_t s = 0; s < spanning_buckets[iteration].size(); s++) {
          if (spanning_buckets[iteration][s].first == gpus[g]) {
            BucketId& spanning_bucket = spanning_buckets[iteration][s].second;
            uint64_t pre_offset = 0;
            if (iteration > 0) {
              BucketId* predecessor = spanning_bucket.predecessor;
              size_t bucket_nr = spanning_bucket.bucket_number;
              pre_offset = (*host_containers->GetHistograms(gpus[g], *predecessor)->GetGlobalPrefixSums())[bucket_nr];
            }

            thrust::exclusive_scan(thrust::cuda::par(*device_containers->GetSecondaryDeviceAllocator(gpus[g])).on(streams[g][0]),
                                   device_containers->GetHistograms(gpus[g], spanning_bucket)->GetGlobalHistogram()->begin(),
                                   device_containers->GetHistograms(gpus[g], spanning_bucket)->GetGlobalHistogram()->end(),
                                   device_containers->GetHistograms(gpus[g], spanning_bucket)->GetGlobalPrefixSums()->begin(), pre_offset);

            CheckHistogramSkewness<<<1, 1, 0, streams[g][0]>>>(
                thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], spanning_bucket)->GetGlobalHistogram()->data()),
                device_containers->GetHistograms(gpus[g], spanning_bucket)->GetNonEmptyCount());
            CheckCudaError(cudaStreamSynchronize(streams[g][0]));

            if (output_mode == OutputMode::DEBUG) {
              std::cout << "On GPU " << gpus[g] << std::endl;
              std::cout << "non empty count: " << *device_containers->GetHistograms(gpus[g], spanning_bucket)->GetNonEmptyCount()
                        << std::endl;
              std::cout << "global prefix sum: " << std::endl;
              printHistogram<uint64_t>(*device_containers->GetHistograms(gpus[g], spanning_bucket)->GetGlobalPrefixSums(), false);
              if (num_keys < radixmgpusort::MAX_KEYS_DEBUG_PRINT) {
                std::cout << "block local histogram: " << std::endl;
                printHistogram<uint32_t>(*device_containers->GetHistograms(gpus[g], spanning_bucket)->GetBlockLocalHistograms(), false,
                                         num_thread_blocks * radixmgpusort::NUM_BUCKETS);
              }
            }
          }
        }

        cuda_timers[g].StartTimer(streams[g][0]);
        if (iteration == 0) {
          if (*device_containers->GetHistograms(gpus[g], BucketId())->GetNonEmptyCount() > 1) {
            ScatterKeys<T><<<num_thread_blocks, radixmgpusort::NUM_THREADS_PER_BLOCK,
                             keys_per_thread * radixmgpusort::NUM_THREADS_PER_BLOCK * sizeof(T), streams[g][0]>>>(
                thrust::raw_pointer_cast(device_containers->GetKeys(gpus[g])->data()),
                thrust::raw_pointer_cast(device_containers->GetTemp(gpus[g])->data()),
                thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], BucketId())->GetGlobalPrefixSums()->data()),
                thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], BucketId())->GetBlockLocalHistograms()->data()),
                reinterpret_cast<uint64_cu*>(
                    thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], BucketId())->GetGlobalScatterOffsets()->data())),
                g_chunk_size, keys_per_thread, (sizeof(T) - iteration) * radixmgpusort::NUM_RADIX_BITS);
          }
        } else {
          size_t spanning_bucket_index = 0;
          std::vector<std::pair<size_t, size_t>> key_scatter_offsets(num_gpus - 1, {0, 0});

          for (size_t s = 0; s < spanning_buckets[iteration].size(); s++) {
            if (spanning_buckets[iteration][s].first == gpus[g]) {
              BucketId& current_bucket = spanning_buckets[iteration][s].second;
              BucketId* predecessor = current_bucket.predecessor;

              if (*device_containers->GetHistograms(gpus[g], current_bucket)->GetNonEmptyCount() > 1) {
                size_t bucket_nr = current_bucket.bucket_number;
                size_t bucket_size = (*host_containers->GetHistograms(gpus[g], *predecessor)->GetGlobalHistogram())[bucket_nr];
                size_t local_num_thread_blocks = GetNumThreadBlocks(bucket_size, keys_per_thread, radixmgpusort::NUM_THREADS_PER_BLOCK);

                uint64_t key_scatter_start_offset =
                    (*host_containers->GetHistograms(gpus[g], *predecessor)->GetGlobalPrefixSums())[bucket_nr];

                key_scatter_offsets[spanning_bucket_index] = {key_scatter_start_offset, key_scatter_start_offset + bucket_size};
                spanning_bucket_index++;

                if (output_mode == OutputMode::DEBUG) {
                  std::cout << "Scatter keys on GPU " << gpus[g] << " and iteration " << iteration << " for bucket " << bucket_nr
                            << ", from " << key_scatter_start_offset << " to " << key_scatter_start_offset + bucket_size << std::endl;
                  std::cout << "Use " << local_num_thread_blocks << " blocks for CUDA kernel for a bucket size of " << bucket_size
                            << std::endl;
                }

                ScatterKeys<T><<<local_num_thread_blocks, radixmgpusort::NUM_THREADS_PER_BLOCK,
                                 keys_per_thread * radixmgpusort::NUM_THREADS_PER_BLOCK * sizeof(T), streams[g][0]>>>(
                    thrust::raw_pointer_cast(device_containers->GetKeys(gpus[g])->data()),
                    thrust::raw_pointer_cast(device_containers->GetTemp(gpus[g])->data()),
                    thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], current_bucket)->GetGlobalPrefixSums()->data()),
                    thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], current_bucket)->GetBlockLocalHistograms()->data()),
                    reinterpret_cast<uint64_cu*>(thrust::raw_pointer_cast(
                        device_containers->GetHistograms(gpus[g], current_bucket)->GetGlobalScatterOffsets()->data())),
                    bucket_size, keys_per_thread, (sizeof(T) - iteration) * radixmgpusort::NUM_RADIX_BITS);
              }
            }
          }

          if (spanning_bucket_index > 0) {
            if (key_scatter_offsets[0].first > 0) {
              CheckCudaError(cudaMemcpyAsync(thrust::raw_pointer_cast(device_containers->GetTemp(gpus[g])->data()),
                                             thrust::raw_pointer_cast(device_containers->GetKeys(gpus[g])->data()),
                                             sizeof(T) * key_scatter_offsets[0].first, cudaMemcpyDeviceToDevice, streams[g][1]));
            }

            for (size_t i = 1; i < spanning_bucket_index; i++) {
              CheckCudaError(cudaMemcpyAsync(
                  thrust::raw_pointer_cast(device_containers->GetTemp(gpus[g])->data() + key_scatter_offsets[i - 1].second),
                  thrust::raw_pointer_cast(device_containers->GetKeys(gpus[g])->data() + key_scatter_offsets[i - 1].second),
                  sizeof(T) * (key_scatter_offsets[i].first - key_scatter_offsets[i - 1].second), cudaMemcpyDeviceToDevice, streams[g][1]));
            }

            size_t max_second = key_scatter_offsets[spanning_bucket_index - 1].second;
            if (g_chunk_size > max_second) {
              CheckCudaError(cudaMemcpyAsync(thrust::raw_pointer_cast(device_containers->GetTemp(gpus[g])->data() + max_second),
                                             thrust::raw_pointer_cast(device_containers->GetKeys(gpus[g])->data() + max_second),
                                             sizeof(T) * (g_chunk_size - max_second), cudaMemcpyDeviceToDevice, streams[g][1]));
            }
          }
        }
        cuda_timers[g].StopTimer(streams[g][0]);
      }

#pragma omp parallel for
      for (size_t g = 0; g < num_gpus; g++) {
        CheckCudaError(cudaSetDevice(gpus[g]));

        CheckCudaError(cudaStreamSynchronize(streams[g][1]));
      }

#pragma omp parallel for
      for (size_t g = 0; g < num_gpus; g++) {
        CheckCudaError(cudaSetDevice(gpus[g]));

        size_t g_chunk_size = chunk_size - (g == num_gpus - 1 ? num_fillers : 0);
        bool contains_spanning_bucket = false;
        bool skipped_key_scatter = true;

        if (iteration == 0) {
          CreateMgpuStripedHistogram<<<num_gpus, radixmgpusort::NUM_BUCKETS, 0, streams[g][1]>>>(
              thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], BucketId())->GetMgpuHistograms()->data()),
              thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], BucketId())->GetMgpuStripedHistogram()->data()), num_gpus);
        } else {
          for (size_t s = 0; s < spanning_buckets[iteration].size(); s++) {
            if (spanning_buckets[iteration][s].first == gpus[g]) {
              contains_spanning_bucket = true;
              CreateMgpuStripedHistogram<<<num_gpus, radixmgpusort::NUM_BUCKETS, 0, streams[g][1]>>>(
                  thrust::raw_pointer_cast(
                      device_containers->GetHistograms(gpus[g], spanning_buckets[iteration][s].second)->GetMgpuHistograms()->data()),
                  thrust::raw_pointer_cast(
                      device_containers->GetHistograms(gpus[g], spanning_buckets[iteration][s].second)->GetMgpuStripedHistogram()->data()),
                  num_gpus);
            }
          }
        }

        if (output_mode == OutputMode::DEBUG && iteration == 0 && chunk_size < radixmgpusort::MAX_KEYS_DEBUG_PRINT) {
          size_t mgpu_histograms_checksum =
              std::accumulate(device_containers->GetHistograms(gpus[g], BucketId())->GetMgpuHistograms()->begin(),
                              device_containers->GetHistograms(gpus[g], BucketId())->GetMgpuHistograms()->end(), 0llu);

          std::cout << "MGPU histogram checksum: " << mgpu_histograms_checksum << std::endl;
          assert(mgpu_histograms_checksum == num_keys);

          size_t mgpu_striped_histogram_checksum =
              std::accumulate(device_containers->GetHistograms(gpus[g], BucketId())->GetMgpuStripedHistogram()->begin(),
                              device_containers->GetHistograms(gpus[g], BucketId())->GetMgpuStripedHistogram()->end(), 0llu);

          assert(mgpu_striped_histogram_checksum == num_keys);
        }

        for (size_t s = 0; s < spanning_buckets[iteration].size(); s++) {
          if (spanning_buckets[iteration][s].first == gpus[g]) {
            BucketId& spanning_bucket = spanning_buckets[iteration][s].second;
            uint64_t pre_offset = 0;
            if (iteration > 0) {
              BucketId* predecessor = spanning_bucket.predecessor;
              size_t bucket_nr = spanning_bucket.bucket_number;
              pre_offset = (*host_containers->GetHistograms(gpus[g], *predecessor)->GetMgpuStripedHistogram())[bucket_nr * num_gpus];
            }

            if (*device_containers->GetHistograms(gpus[g], spanning_bucket)->GetNonEmptyCount() > 1) {
              skipped_key_scatter = false;
            }

            thrust::exclusive_scan(thrust::cuda::par(*device_containers->GetSecondaryDeviceAllocator(gpus[g])).on(streams[g][1]),
                                   device_containers->GetHistograms(gpus[g], spanning_bucket)->GetMgpuStripedHistogram()->begin(),
                                   device_containers->GetHistograms(gpus[g], spanning_bucket)->GetMgpuStripedHistogram()->end(),
                                   device_containers->GetHistograms(gpus[g], spanning_bucket)->GetMgpuStripedHistogram()->begin(),
                                   pre_offset);

            DetermineBucketToGpuMapping<<<1, radixmgpusort::NUM_BUCKETS, 0, streams[g][1]>>>(
                thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], spanning_bucket)->GetMgpuStripedHistogram()->data()),
                thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], spanning_bucket)->GetBucketToGpuMap()->data()),
                chunk_size, num_fillers, num_gpus, device_containers->GetEpsilon());
          }
        }

        CheckCudaError(cudaStreamSynchronize(streams[g][0]));
        key_scatter_time[iteration][g] = cuda_timers[g].SynchronizeTimer();
        CheckCudaError(cudaStreamSynchronize(streams[g][1]));

        if (!skipped_key_scatter && (iteration == 0 || contains_spanning_bucket)) {
          device_containers->FlipBuffers(gpus[g]);
        }

        for (size_t s = 0; s < spanning_buckets[iteration].size(); s++) {
          if (spanning_buckets[iteration][s].first == gpus[g]) {
            BucketId& spanning_bucket = spanning_buckets[iteration][s].second;

            CheckCudaError(cudaMemcpyAsync(
                thrust::raw_pointer_cast(host_containers->GetHistograms(gpus[g], spanning_bucket)->GetBucketToGpuMap()->data()),
                thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], spanning_bucket)->GetBucketToGpuMap()->data()),
                sizeof(int) * radixmgpusort::NUM_BUCKETS * num_gpus, cudaMemcpyDeviceToHost, streams[g][0]));

            CheckCudaError(cudaMemcpyAsync(
                thrust::raw_pointer_cast(host_containers->GetHistograms(gpus[g], spanning_bucket)->GetMgpuStripedHistogram()->data()),
                thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], spanning_bucket)->GetMgpuStripedHistogram()->data()),
                sizeof(uint64_t) * ((radixmgpusort::NUM_BUCKETS * num_gpus) + 1), cudaMemcpyDeviceToHost, streams[g][0]));

            CheckCudaError(cudaMemcpyAsync(
                thrust::raw_pointer_cast(host_containers->GetHistograms(gpus[g], spanning_bucket)->GetGlobalHistogram()->data()),
                thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], spanning_bucket)->GetGlobalHistogram()->data()),
                sizeof(uint64_t) * radixmgpusort::NUM_BUCKETS, cudaMemcpyDeviceToHost, streams[g][0]));

            CheckCudaError(cudaMemcpyAsync(
                thrust::raw_pointer_cast(host_containers->GetHistograms(gpus[g], spanning_bucket)->GetGlobalPrefixSums()->data()),
                thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], spanning_bucket)->GetGlobalPrefixSums()->data()),
                sizeof(uint64_t) * radixmgpusort::NUM_BUCKETS, cudaMemcpyDeviceToHost, streams[g][0]));
          }
        }

        CheckCudaError(cudaStreamSynchronize(streams[g][0]));

        // CHECK IF CORRECT: Keys at GPU g are the same before and after the key scatter step
        if (output_mode == OutputMode::DEBUG) {
          if (iteration == 0) {
            thrust::device_vector<uint64_t> global_histogram_check(radixmgpusort::NUM_BUCKETS, 0);

            ComputeHistogram<T><<<num_thread_blocks, radixmgpusort::NUM_THREADS_PER_BLOCK, 0, streams[g][0]>>>(
                thrust::raw_pointer_cast(device_containers->GetKeys(gpus[g])->data()),
                thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], BucketId())->GetBlockLocalHistograms()->data()),
                g_chunk_size, keys_per_thread, (sizeof(T) - iteration) * radixmgpusort::NUM_RADIX_BITS);

            AggregateHistogram<T>
                <<<(num_thread_blocks / num_block_histograms_to_aggregate) + 1, radixmgpusort::NUM_THREADS_PER_BLOCK, 0, streams[g][0]>>>(
                    reinterpret_cast<uint64_cu*>(thrust::raw_pointer_cast(global_histogram_check.data())),
                    thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], BucketId())->GetBlockLocalHistograms()->data()),
                    num_thread_blocks, num_block_histograms_to_aggregate);
            CheckCudaError(cudaStreamSynchronize(streams[g][0]));

            for (size_t i = 0; i < radixmgpusort::NUM_BUCKETS; i++) {
              if ((*device_containers->GetHistograms(gpus[g], BucketId())->GetGlobalHistogram())[i] != global_histogram_check[i]) {
                std::cout << "Error: Key scatter wrote back false keys!" << std::endl;
                std::cout << "Error: Key scatter of iteration 0 wrote back false keys on GPU " << gpus[g] << "!" << std::endl;
                break;
              }
            }
          } else {
            thrust::device_vector<uint64_t> global_histogram_check(radixmgpusort::NUM_BUCKETS, 0);

            for (size_t s = 0; s < spanning_buckets[iteration].size(); s++) {
              BucketId& current_bucket = spanning_buckets[iteration][s].second;
              if (spanning_buckets[iteration][s].first == gpus[g]) {
                BucketId* predecessor = current_bucket.predecessor;

                size_t bucket_nr = current_bucket.bucket_number;
                size_t bucket_size = (*host_containers->GetHistograms(gpus[g], *predecessor)->GetGlobalHistogram())[bucket_nr];
                size_t local_num_thread_blocks = GetNumThreadBlocks(bucket_size, keys_per_thread, radixmgpusort::NUM_THREADS_PER_BLOCK);

                ComputeHistogram<T><<<local_num_thread_blocks, radixmgpusort::NUM_THREADS_PER_BLOCK, 0, streams[g][0]>>>(
                    thrust::raw_pointer_cast(device_containers->GetKeys(gpus[g])->data() +
                                             (*host_containers->GetHistograms(gpus[g], *predecessor)->GetGlobalPrefixSums())[bucket_nr]),
                    thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], current_bucket)->GetBlockLocalHistograms()->data()),
                    bucket_size, keys_per_thread, (sizeof(T) - iteration) * radixmgpusort::NUM_RADIX_BITS);

                AggregateHistogram<T><<<(local_num_thread_blocks / num_block_histograms_to_aggregate) + 1,
                                        radixmgpusort::NUM_THREADS_PER_BLOCK, 0, streams[g][0]>>>(
                    reinterpret_cast<uint64_cu*>(thrust::raw_pointer_cast(global_histogram_check.data())),
                    thrust::raw_pointer_cast(device_containers->GetHistograms(gpus[g], current_bucket)->GetBlockLocalHistograms()->data()),
                    local_num_thread_blocks, num_block_histograms_to_aggregate);
                CheckCudaError(cudaStreamSynchronize(streams[g][0]));

                for (size_t i = 0; i < radixmgpusort::NUM_BUCKETS; i++) {
                  if ((*device_containers->GetHistograms(gpus[g], current_bucket)->GetGlobalHistogram())[i] != global_histogram_check[i]) {
                    std::cout << "Error: Key scatter of spanning bucket " << bucket_nr << " of iteration " << iteration
                              << " wrote back false keys on GPU " << gpus[g] << "!" << std::endl;
                    break;
                  }
                }
              }
            }
          }
        }
      }
    }

    auto start = GetTimeNow();

    for (size_t iteration = 0; iteration < max_partition_passes; iteration++) {
      for (size_t s = 0; s < spanning_buckets[iteration].size(); s++) {
        int spanning_bucket_gpu = spanning_buckets[iteration][s].first;
        BucketId& spanning_bucket = spanning_buckets[iteration][s].second;

        HostHistograms* host_histograms = host_containers->GetHistograms(spanning_bucket_gpu, spanning_bucket);
        for (size_t i = 0; i < radixmgpusort::NUM_BUCKETS; i++) {
          PinnedHostVector<int>* current_bucket_to_gpu_map = host_histograms->GetBucketToGpuMap();

          if ((*current_bucket_to_gpu_map)[(i * num_gpus) + 1] == -1) {
            int dest_gpu = (*current_bucket_to_gpu_map)[i * num_gpus];
            if (dest_gpu >= 0) {
              uint64_t offset = (*host_histograms->GetMgpuStripedHistogram())[(i + 1) * num_gpus];

              if (offset > gpu_global_offsets[dest_gpu + 1]) {
                gpu_global_offsets[dest_gpu + 1] = offset;
              }
            }
          } else if (num_partition_passes_needed == max_partition_passes && iteration == max_partition_passes - 1) {
            // Handle spanning buckets from the last possible partitioning pass (on the last 8 bits),
            // i.e. buckets that are filled with the same key. Distribute them across the determined GPUs
            // so that an equal load balance is achieved, i.e fill up the GPUs until the chunk size is met.

            int source_gpu = device_containers->GetGpuIndex(spanning_bucket_gpu);

            size_t bucket_starting_offset = (*host_histograms->GetMgpuStripedHistogram())[i * num_gpus];
            size_t bucket_ending_offset = (*host_histograms->GetMgpuStripedHistogram())[(i + 1) * num_gpus];
            size_t source_gpu_bucket_size = (*host_histograms->GetGlobalHistogram())[i];

            if (source_gpu_bucket_size > 0) {
              BucketId last_pass_bucket = BucketId(max_partition_passes - 1, i, &spanning_buckets[iteration][s].second);

              if (last_pass_spanning_buckets.count(last_pass_bucket) == 0) {
                last_pass_spanning_buckets[last_pass_bucket] = {0, {}};
                last_pass_spanning_buckets[last_pass_bucket].second.reserve(num_gpus);
              }

              int j = 0;
              size_t source_offset = 0;
              while ((*current_bucket_to_gpu_map)[(i * num_gpus) + j] >= 0 && j < num_gpus && source_gpu_bucket_size > 0) {
                int dest_gpu = (*current_bucket_to_gpu_map)[(i * num_gpus) + j];

                LPSpanningBucketFraction lp_fraction;
                lp_fraction.source_gpu = source_gpu;
                lp_fraction.dest_gpu = dest_gpu;

                size_t current_offset = bucket_starting_offset + last_pass_spanning_buckets[last_pass_bucket].first;

                if (current_offset + source_gpu_bucket_size <= (dest_gpu + 1) * chunk_size) {
                  lp_fraction.fraction_size = source_gpu_bucket_size;
                  lp_fraction.source_offset = source_offset;
                  lp_fraction.dest_offset = last_pass_spanning_buckets[last_pass_bucket].first;

                  source_offset += source_gpu_bucket_size;
                  last_pass_spanning_buckets[last_pass_bucket].first += source_gpu_bucket_size;
                  last_pass_spanning_buckets[last_pass_bucket].second.push_back(lp_fraction);

                  if (gpu_global_offsets[dest_gpu + 1] < current_offset + source_gpu_bucket_size) {
                    gpu_global_offsets[dest_gpu + 1] = current_offset + source_gpu_bucket_size;
                  }

                  source_gpu_bucket_size = 0;

                } else {
                  if ((dest_gpu + 1) * chunk_size > current_offset) {
                    size_t num_keys_to_fill_chunk = (dest_gpu + 1) * chunk_size - current_offset;

                    lp_fraction.fraction_size = num_keys_to_fill_chunk;
                    lp_fraction.source_offset = source_offset;
                    lp_fraction.dest_offset = last_pass_spanning_buckets[last_pass_bucket].first;

                    source_offset += num_keys_to_fill_chunk;
                    last_pass_spanning_buckets[last_pass_bucket].first += num_keys_to_fill_chunk;
                    last_pass_spanning_buckets[last_pass_bucket].second.push_back(lp_fraction);
                    source_gpu_bucket_size -= num_keys_to_fill_chunk;

                    gpu_global_offsets[dest_gpu + 1] = (dest_gpu + 1) * chunk_size;
                  }
                }
                j++;
              }
            }
          }
        }
      }
    }

    auto end = GetTimeNow();

    prepare_key_swap_time = GetMilliseconds(start, end);

    if (output_mode == OutputMode::DEBUG && num_keys < radixmgpusort::MAX_KEYS_DEBUG_PRINT) {
      for (size_t g = 0; g < num_gpus; g++) {
        std::cout << "CHUNK ON GPU " << gpus[g] << ":" << std::endl;
        for (size_t i = 0; i < chunk_size; i++) {
          std::cout << (*device_containers->GetKeys(gpus[g]))[i] << ",";
        }
        std::cout << std::endl;
      }
    }

    if (output_mode == OutputMode::DEBUG) {
      for (size_t g = 0; g < gpu_global_offsets.size(); g++) {
        std::cout << "gpu_global_offsets[" << g << "]: " << gpu_global_offsets[g] << std::endl;
      }
      std::cout << std::endl;
    }

    start = GetTimeNow();

#pragma omp parallel for
    for (size_t g = 0; g < num_gpus; g++) {
      CheckCudaError(cudaSetDevice(gpus[g]));

      for (size_t iteration = 0; iteration < max_partition_passes; iteration++) {
        for (size_t s = 0; s < spanning_buckets[iteration].size(); s++) {
          int spanning_bucket_gpu = spanning_buckets[iteration][s].first;
          BucketId& spanning_bucket = spanning_buckets[iteration][s].second;

          if (spanning_bucket_gpu == gpus[g]) {
            for (size_t i = 0; i < radixmgpusort::NUM_BUCKETS; i++) {
              HostHistograms* host_histograms = host_containers->GetHistograms(spanning_bucket_gpu, spanning_bucket);
              PinnedHostVector<int>* current_bucket_to_gpu_map = host_histograms->GetBucketToGpuMap();

              if ((*host_histograms->GetGlobalHistogram())[i] > 0) {
                if ((*current_bucket_to_gpu_map)[(i * num_gpus) + 1] == -1) {
                  int dest_gpu = (*current_bucket_to_gpu_map)[i * num_gpus];
                  if (dest_gpu >= 0) {
                    if (output_mode == OutputMode::DEBUG) {
                      std::cout << std::endl
                                << "Bucket i = " << i << " on GPU " << gpus[g] << " of iteration " << iteration << " = "
                                << spanning_bucket.partition_pass << " belongs to GPU " << dest_gpu << "!" << std::endl;
                      std::cout << "Copy keys FROM GPU " << gpus[g] << " at " << (*host_histograms->GetGlobalPrefixSums())[i] << std::endl;
                      std::cout << "Copy keys TO GPU " << gpus[dest_gpu] << " at "
                                << (*host_histograms->GetMgpuStripedHistogram())[(i * num_gpus) + g] - gpu_global_offsets[dest_gpu]
                                << std::endl;
                      std::cout << "mgpu_striped_histogram[(i * num_gpus) + g]: "
                                << (*host_histograms->GetMgpuStripedHistogram())[(i * num_gpus) + g] << std::endl;
                      std::cout << "global_offsets[dest_gpu]: " << gpu_global_offsets[dest_gpu] << std::endl;
                      std::cout << "Copy " << (*host_histograms->GetGlobalHistogram())[i] << " keys..." << std::endl << std::endl;
                    }

                    CheckCudaError(
                        cudaMemcpyAsync(thrust::raw_pointer_cast(device_containers->GetTemp(gpus[dest_gpu])->data() +
                                                                 (*host_histograms->GetMgpuStripedHistogram())[(i * num_gpus) + g] -
                                                                 gpu_global_offsets[dest_gpu]),
                                        thrust::raw_pointer_cast(device_containers->GetKeys(gpus[g])->data() +
                                                                 (*host_histograms->GetGlobalPrefixSums())[i]),
                                        sizeof(T) * (*host_histograms->GetGlobalHistogram())[i], cudaMemcpyDeviceToDevice, streams[g][0]));
                  }
                }
              }
            }
          }
        }
      }
    }

    for (auto const& [bucket_id, lp_fraction_pair] : last_pass_spanning_buckets) {
      for (auto const& lp_fraction : lp_fraction_pair.second) {
        size_t i = bucket_id.bucket_number;
        int source_gpu = lp_fraction.source_gpu;
        int dest_gpu = lp_fraction.dest_gpu;

        HostHistograms* host_histograms = host_containers->GetHistograms(gpus[source_gpu], *bucket_id.predecessor);

        if (output_mode == OutputMode::DEBUG) {
          std::cout << std::endl
                    << ">>> Bucket i = " << i << " on GPU " << gpus[source_gpu] << " of iteration " << bucket_id.partition_pass;
          std::cout << "belongs to GPU " << gpus[dest_gpu] << " !" << std::endl;
          std::cout << "Copy keys FROM GPU " << gpus[source_gpu] << " at "
                    << (*host_histograms->GetGlobalPrefixSums())[i] + lp_fraction.source_offset << std::endl;
          std::cout << "Copy keys TO GPU " << gpus[dest_gpu] << " at "
                    << (*host_histograms->GetMgpuStripedHistogram())[i * num_gpus] + lp_fraction.dest_offset - gpu_global_offsets[dest_gpu];
          std::cout << std::endl
                    << "mgpu_striped_histogram[i * num_gpus] + lp_fraction.dest_offset: "
                    << (*host_histograms->GetMgpuStripedHistogram())[i * num_gpus] + lp_fraction.dest_offset << std::endl;
          std::cout << "global_offsets[dest_gpu]: " << gpu_global_offsets[dest_gpu] << std::endl;
          std::cout << "Copy " << lp_fraction.fraction_size << " keys..." << std::endl << std::endl;
        }

        CheckCudaError(cudaMemcpyAsync(thrust::raw_pointer_cast(device_containers->GetTemp(gpus[dest_gpu])->data() +
                                                                (*host_histograms->GetMgpuStripedHistogram())[i * num_gpus] +
                                                                lp_fraction.dest_offset - gpu_global_offsets[dest_gpu]),
                                       thrust::raw_pointer_cast(device_containers->GetKeys(gpus[source_gpu])->data() +
                                                                (*host_histograms->GetGlobalPrefixSums())[i] + lp_fraction.source_offset),
                                       sizeof(T) * lp_fraction.fraction_size, cudaMemcpyDeviceToDevice, streams[source_gpu][0]));
      }
    }

    for (size_t g = 0; g < num_gpus; g++) {
      CheckCudaError(cudaSetDevice(gpus[g]));
      CheckCudaError(cudaStreamSynchronize(streams[g][0]));
    }

    end = GetTimeNow();

    key_swap_time = GetMilliseconds(start, end);

#pragma omp parallel for
    for (size_t g = 0; g < num_gpus; g++) {
      CheckCudaError(cudaSetDevice(gpus[g]));

      if (gpu_global_offsets[g + 1] > 0 || g == 0) {
        size_t gpu_local_chunk_size = gpu_global_offsets[g + 1] - gpu_global_offsets[g];
        size_t balanced_chunk_size = chunk_size - (g == num_gpus - 1 ? num_fillers : 0);

        device_containers->FlipBuffers(gpus[g]);

        auto start = GetTimeNow();

        size_t num_buckets_to_sort = 0;
        ReducedSortingBucket<T>* prev_bucket = nullptr;

        for (auto it = spanning_bucket_to_gpus_map.begin(); it != spanning_bucket_to_gpus_map.end(); it++) {
          const BucketId& spanning_bucket = it->first;
          int spanning_bucket_gpu = it->second[0];

          for (size_t i = 0; i < radixmgpusort::NUM_BUCKETS; i++) {
            HostHistograms* host_histograms = host_containers->GetHistograms(spanning_bucket_gpu, spanning_bucket);
            PinnedHostVector<int>* current_bucket_to_gpu_map = host_histograms->GetBucketToGpuMap();

            int dest_gpu = (*current_bucket_to_gpu_map)[i * num_gpus];
            if (dest_gpu == g && (*current_bucket_to_gpu_map)[(i * num_gpus) + 1] == -1) {
              size_t bucket_end = (*host_histograms->GetMgpuStripedHistogram())[(i + 1) * num_gpus];
              size_t bucket_start = (*host_histograms->GetMgpuStripedHistogram())[i * num_gpus];
              size_t bucket_size = bucket_end - bucket_start;
              if (bucket_size > 1 && spanning_bucket.partition_pass < max_partition_passes - 1) {
                bucket_end -= gpu_global_offsets[g];
                bucket_start -= gpu_global_offsets[g];

                // Group together neighbouring buckets that are small. Reduces the number of CUB radix sort kernel launches.
                if (prev_bucket != nullptr && prev_bucket->partition_pass == spanning_bucket.partition_pass &&
                    prev_bucket->bucket_size + bucket_size < device_containers->GetSmallBucketThreshold() &&
                    prev_bucket->bucket_start + prev_bucket->bucket_size == bucket_start) {
                  prev_bucket->bucket_size += bucket_size;

                  // Find the most significant bit position where the buckets differ as the new best-case end bit.
                  uint8_t xor_bucket = static_cast<uint8_t>(i) ^ static_cast<uint8_t>(prev_bucket->bucket_number);
                  uint32_t msb_dif_position = 0;
                  while (xor_bucket) {
                    xor_bucket >>= 1;
                    msb_dif_position++;
                  }

                  if (msb_dif_position > prev_bucket->msb_dif_position) {
                    prev_bucket->msb_dif_position = msb_dif_position;
                  }

                  continue;
                }

                ReducedSortingBucket<T> b;
                b.bucket_size = bucket_size;
                b.bucket_start = bucket_start;
                b.cub_double_buffer =
                    cub::DoubleBuffer(thrust::raw_pointer_cast(device_containers->GetKeys(gpus[g])->data() + bucket_start),
                                      thrust::raw_pointer_cast(device_containers->GetTemp(gpus[g])->data() + bucket_start));

                b.msb_dif_position = 0;
                b.partition_pass = spanning_bucket.partition_pass;
                b.bucket_number = i;

                reduced_sorting_buckets[g].emplace_back(b);
                prev_bucket = &reduced_sorting_buckets[g][num_buckets_to_sort];
                num_buckets_to_sort++;
              }
            }
          }
        }

        std::sort(reduced_sorting_buckets[g].begin(), reduced_sorting_buckets[g].end(), CompareReducedSortingBuckets<T>());

        if (output_mode == OutputMode::DEBUG) {
          std::cout << "num_buckets_to_sort: " << num_buckets_to_sort << std::endl;
          for (size_t s = 0; s < num_buckets_to_sort; s++) {
            ReducedSortingBucket<T>& b = reduced_sorting_buckets[g][s];
            uint32_t end_bit =
                std::min(sizeof(T) * 8, (sizeof(T) - b.partition_pass - 1) * radixmgpusort::NUM_RADIX_BITS + 1 + b.msb_dif_position);
            std::cout << "Reduced sorting bucket has " << b.bucket_size << " keys with ent_bit " << end_bit << std::endl;
          }
        }

        if (num_buckets_to_sort <= radixmgpusort::MAX_NUM_BUCKETS_FOR_REDUCED_SORTING) {
          device_containers->GetDeviceAllocator(gpus[g])->SetOffset(sizeof(T) * gpu_local_chunk_size);

          if (num_buckets_to_sort >= radixmgpusort::MIN_NUM_BUCKETS_FOR_SORT_COPY_OVERLAP) {
            size_t sorted_keys_offset = 0;
            size_t transferred_keys = 0;

            for (size_t s = 0; s < num_buckets_to_sort; s++) {
              ReducedSortingBucket<T>& b = reduced_sorting_buckets[g][s];

              uint32_t end_bit =
                  std::min(sizeof(T) * 8, (sizeof(T) - b.partition_pass - 1) * radixmgpusort::NUM_RADIX_BITS + 1 + b.msb_dif_position);

              if (output_mode == OutputMode::DEBUG) {
                std::cout << "Sort bucket #" << b.bucket_number << " of iteration " << b.partition_pass << " on GPU " << gpus[g];
                std::cout << " from offset " << b.bucket_start << " - " << b.bucket_size << " keys - ";
                std::cout << " on bit range [0," << end_bit << ")" << std::endl;
              }

              size_t num_temporary_bytes = 0;
              cub::DeviceRadixSort::SortKeys(NULL, num_temporary_bytes, b.cub_double_buffer, b.bucket_size, 0, end_bit,
                                             streams[g][s % radixmgpusort::MAX_NUM_CONCURRENT_KERNELS]);

              uint8_t* temporary_storage = device_containers->GetDeviceAllocator(gpus[g])->allocate(num_temporary_bytes);

              cub::DeviceRadixSort::SortKeys((void*)temporary_storage, num_temporary_bytes, b.cub_double_buffer, b.bucket_size, 0, end_bit,
                                             streams[g][s % radixmgpusort::MAX_NUM_CONCURRENT_KERNELS]);

              CheckCudaError(cudaEventRecord(events[g][s], streams[g][s % radixmgpusort::MAX_NUM_CONCURRENT_KERNELS]));

              CheckCudaError(cudaStreamWaitEvent(streams[g][s % radixmgpusort::MAX_NUM_CONCURRENT_KERNELS], events[g][s]));
              if (b.cub_double_buffer.Current() == thrust::raw_pointer_cast(device_containers->GetTemp(gpus[g])->data() + b.bucket_start)) {
                CheckCudaError(cudaMemcpyAsync(b.cub_double_buffer.Alternate(), b.cub_double_buffer.Current(), sizeof(T) * b.bucket_size,
                                               cudaMemcpyDeviceToDevice, streams[g][s % radixmgpusort::MAX_NUM_CONCURRENT_KERNELS]));
              }

              sorted_keys_offset = b.bucket_start + b.bucket_size;

              size_t keys_to_transfer = sorted_keys_offset - transferred_keys;

              CheckCudaError(cudaMemcpyAsync(thrust::raw_pointer_cast(keys->data() + gpu_global_offsets[g] + transferred_keys),
                                             thrust::raw_pointer_cast(device_containers->GetKeys(gpus[g])->data() + transferred_keys),
                                             sizeof(T) * keys_to_transfer, cudaMemcpyDeviceToHost,
                                             streams[g][s % radixmgpusort::MAX_NUM_CONCURRENT_KERNELS]));

              transferred_keys += keys_to_transfer;
            }

            if (gpu_local_chunk_size > transferred_keys) {
              CheckCudaError(cudaMemcpyAsync(thrust::raw_pointer_cast(keys->data() + gpu_global_offsets[g] + transferred_keys),
                                             thrust::raw_pointer_cast(device_containers->GetKeys(gpus[g])->data() + transferred_keys),
                                             sizeof(T) * (gpu_local_chunk_size - transferred_keys), cudaMemcpyDeviceToHost, streams[g][0]));
            }

            for (size_t s = 1; s < std::min(num_buckets_to_sort, radixmgpusort::MAX_NUM_CONCURRENT_KERNELS); s++) {
              CheckCudaError(cudaStreamSynchronize(streams[g][s]));
            }
            CheckCudaError(cudaStreamSynchronize(streams[g][0]));

            auto end = GetTimeNow();
            sort_time[g] = GetMilliseconds(start, end);

          } else {
            for (size_t s = 0; s < num_buckets_to_sort; s++) {
              ReducedSortingBucket<T>& b = reduced_sorting_buckets[g][s];

              uint32_t end_bit =
                  std::min(sizeof(T) * 8, (sizeof(T) - b.partition_pass - 1) * radixmgpusort::NUM_RADIX_BITS + 1 + b.msb_dif_position);

              if (output_mode == OutputMode::DEBUG) {
                std::cout << "Sort bucket #" << b.bucket_number << " of iteration " << b.partition_pass << " on GPU " << gpus[g];
                std::cout << " from offset " << b.bucket_start << " - " << b.bucket_size << " keys - ";
                std::cout << " on bit range [0," << end_bit << ")" << std::endl;
              }

              size_t num_temporary_bytes = 0;
              cub::DeviceRadixSort::SortKeys(NULL, num_temporary_bytes, b.cub_double_buffer, b.bucket_size, 0, end_bit,
                                             streams[g][s % radixmgpusort::MAX_NUM_CONCURRENT_KERNELS]);

              uint8_t* temporary_storage = device_containers->GetDeviceAllocator(gpus[g])->allocate(num_temporary_bytes);

              cub::DeviceRadixSort::SortKeys((void*)temporary_storage, num_temporary_bytes, b.cub_double_buffer, b.bucket_size, 0, end_bit,
                                             streams[g][s % radixmgpusort::MAX_NUM_CONCURRENT_KERNELS]);
            }

            for (size_t s = 0; s < std::min(num_buckets_to_sort, radixmgpusort::MAX_NUM_CONCURRENT_KERNELS); s++) {
              CheckCudaError(cudaStreamSynchronize(streams[g][s]));
            }

            for (size_t i = 0; i < num_buckets_to_sort; i++) {
              ReducedSortingBucket<T>& b = reduced_sorting_buckets[g][i];
              if (b.cub_double_buffer.Current() == thrust::raw_pointer_cast(device_containers->GetTemp(gpus[g])->data() + b.bucket_start)) {
                CheckCudaError(cudaMemcpyAsync(b.cub_double_buffer.Alternate(), b.cub_double_buffer.Current(), sizeof(T) * b.bucket_size,
                                               cudaMemcpyDeviceToDevice, streams[g][0]));
              }
            }

            CheckCudaError(cudaStreamSynchronize(streams[g][0]));

            auto end = GetTimeNow();
            sort_time[g] = GetMilliseconds(start, end);
          }

        } else {
          thrust::sort(thrust::cuda::par(*device_containers->GetDeviceAllocator(gpus[g])).on(streams[g][0]),
                       device_containers->GetKeys(gpus[g])->begin(), device_containers->GetKeys(gpus[g])->begin() + gpu_local_chunk_size);

          CheckCudaError(cudaStreamSynchronize(streams[g][0]));

          auto end = GetTimeNow();
          sort_time[g] = GetMilliseconds(start, end);
        }

        if (num_buckets_to_sort > radixmgpusort::MAX_NUM_BUCKETS_FOR_REDUCED_SORTING ||
            num_buckets_to_sort < radixmgpusort::MIN_NUM_BUCKETS_FOR_SORT_COPY_OVERLAP) {
          start = GetTimeNow();

          if (output_mode == OutputMode::DEBUG) {
            std::cout << "Copy " << gpu_local_chunk_size << " sorted keys back to the CPU at " << gpu_global_offsets[g] << " to "
                      << gpu_global_offsets[g + 1] << std::endl;
          }

          if (gpu_global_offsets[g] < balanced_chunk_size * g) {
            size_t skip_keys = balanced_chunk_size * g - gpu_global_offsets[g];

            CheckCudaError(cudaMemcpyAsync(thrust::raw_pointer_cast(keys->data() + balanced_chunk_size * g),
                                           thrust::raw_pointer_cast(device_containers->GetKeys(gpus[g])->data() + skip_keys),
                                           sizeof(T) * (gpu_local_chunk_size - skip_keys), cudaMemcpyDeviceToHost, streams[g][0]));

            CheckCudaError(cudaMemcpyAsync(thrust::raw_pointer_cast(keys->data() + gpu_global_offsets[g]),
                                           thrust::raw_pointer_cast(device_containers->GetKeys(gpus[g])->data()), sizeof(T) * skip_keys,
                                           cudaMemcpyDeviceToHost, streams[g][0]));
          } else {
            CheckCudaError(cudaMemcpyAsync(thrust::raw_pointer_cast(keys->data() + gpu_global_offsets[g]),
                                           thrust::raw_pointer_cast(device_containers->GetKeys(gpus[g])->data()),
                                           sizeof(T) * gpu_local_chunk_size, cudaMemcpyDeviceToHost, streams[g][0]));
          }
          CheckCudaError(cudaStreamSynchronize(streams[g][0]));

          end = GetTimeNow();
          dtoh_time[g] = GetMilliseconds(start, end);
        }
      }
    }

    auto end_sort = GetTimeNow();
    total_sort_duration = GetMilliseconds(start_sort, end_sort);

    delete device_containers;
    delete host_containers;
  }

  if (output_mode != OutputMode::CSV) {
    std::cout << std::endl << "Sort duration breakdown:" << std::endl;
    std::cout << "=================================================" << std::endl;

    if (num_gpus > 1) {
      std::cout << "Number of partitioning passes needed: " << num_partition_passes_needed << std::endl;
    }

    for (size_t g = 0; g < num_gpus; g++) {
      std::cout << "> On GPU " << gpus[g] << std::endl;
      std::cout << "Copy HtoD time: " << htod_copy_time[g] << "ms" << std::endl;

      if (num_gpus > 1) {
        std::cout << "Histogram time: \t";
        for (size_t iteration = 0; iteration < num_partition_passes_needed; iteration++) {
          std::cout << "\t(" << iteration + 1 << "): " << histogram_time[iteration][g] << "ms";
        }
        std::cout << std::endl << "Key scatter time: \t";
        for (size_t iteration = 0; iteration < num_partition_passes_needed; iteration++) {
          std::cout << "\t(" << iteration + 1 << "): " << key_scatter_time[iteration][g] << "ms";
        }
        std::cout << std::endl;

        std::cout << "P2P key swap time: " << key_swap_time << "ms" << std::endl;
      }
      std::cout << "Sort time: " << sort_time[g] << "ms" << std::endl;
      std::cout << "Copy DtoH time: " << dtoh_time[g] << "ms" << std::endl << std::endl;
    }

    if (num_gpus > 1) {
      std::cout << "Determine spanning buckets time: " << detect_spanning_buckets_time << "ms" << std::endl;
      std::cout << "Prepare key swap time: " << prepare_key_swap_time << "ms" << std::endl;
    }

    std::cout << "=================================================" << std::endl;
    std::cout << "Total sort duration: " << total_sort_duration << "ms" << std::endl;

  } else {
    std::cout << num_partition_passes_needed << ",";
    std::cout << "\"";

    for (size_t g = 0; g < num_gpus; g++) {
      std::cout << htod_copy_time[g] << ",";

      if (num_gpus > 1) {
        std::cout << "{(";
        for (size_t iteration = 0; iteration < num_partition_passes_needed; iteration++) {
          std::cout << histogram_time[iteration][g] << ((iteration == num_partition_passes_needed - 1) ? "" : ",");
        }
        std::cout << "),(";
        for (size_t iteration = 0; iteration < num_partition_passes_needed; iteration++) {
          std::cout << key_scatter_time[iteration][g] << ((iteration == num_partition_passes_needed - 1) ? "" : ",");
        }
        std::cout << ")},";
      }

      std::cout << key_swap_time << ",";
      std::cout << sort_time[g] << ",";
      std::cout << dtoh_time[g] << ((g == num_gpus - 1) ? "" : ",");
    }
    std::cout << "\",";
    std::cout << total_sort_duration << std::endl;
  }

#pragma omp parallel for
  for (size_t g = 0; g < num_gpus; g++) {
    for (size_t i = 0; i < 2; i++) {
      CheckCudaError(cudaStreamDestroy(streams[g][i]));
    }
  }
}
