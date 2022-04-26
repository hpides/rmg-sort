#pragma once

#include "cuda.h"
#include "cuda_error.cuh"
#include "cuda_runtime.h"
#include "cuda_utils.cuh"

__device__ __forceinline__ int GetAbsolute(int v) { return v < 0 ? -v : v; }

template <std::uint8_t num_bytes>
using uint_type = typename std::conditional<num_bytes == 4, uint32_t, uint64_t>::type;

template <typename T>
__device__ __forceinline__ void GetRadixBucket(T* key_value, uint_type<sizeof(T)> radix_mask, size_t radix_msb,
                                               uint_type<sizeof(T)>* bucket) {
  *bucket = radix_mask & *key_value;
  *bucket = *bucket >> (radix_msb - radixmgpusort::NUM_RADIX_BITS);
}

template <>
__device__ __forceinline__ void GetRadixBucket<float>(float* key_value, uint32_t radix_mask, size_t radix_msb, uint32_t* bucket) {
  uint32_t key = *(reinterpret_cast<uint32_t*>(key_value));
  *bucket = radix_mask & key;
  *bucket = *bucket >> (radix_msb - radixmgpusort::NUM_RADIX_BITS);
}

template <>
__device__ __forceinline__ void GetRadixBucket<double>(double* key_value, uint64_t radix_mask, size_t radix_msb, uint64_t* bucket) {
  uint64_t key = *(reinterpret_cast<uint64_t*>(key_value));
  *bucket = radix_mask & key;
  *bucket = *bucket >> (radix_msb - radixmgpusort::NUM_RADIX_BITS);
}

template <typename T>
__global__ void ComputeHistogram(T* input, uint32_t* block_local_histograms, size_t n, size_t k, size_t radix_msb) {
  const size_t index = (k * (size_t)blockDim.x * (size_t)blockIdx.x) + threadIdx.x;

  // initialize thread block-local histogram buffer in shared memory
  __shared__ __align__(sizeof(uint32_t)) uint32_t smem_histogram[radixmgpusort::NUM_BUCKETS];

  uint_type<sizeof(T)> radix_mask = (1 << radixmgpusort::NUM_RADIX_BITS) - 1;
  radix_mask = radix_mask << (radix_msb - radixmgpusort::NUM_RADIX_BITS);
  uint_type<sizeof(T)> start_bucket = 0;
  uint_type<sizeof(T)> bucket = 0;

  if (threadIdx.x < radixmgpusort::NUM_BUCKETS) {
    smem_histogram[threadIdx.x] = 0;
  }

  __syncthreads();

  if (index < n) {
    int count = 1;
    GetRadixBucket<T>(&input[index], radix_mask, radix_msb, &start_bucket);

    for (size_t i = 1; i < k && index + (i * blockDim.x) < n; i++) {
      GetRadixBucket<T>(&input[index + (i * blockDim.x)], radix_mask, radix_msb, &bucket);

      if (bucket == start_bucket) {
        count++;
      } else {
        atomicAdd(&smem_histogram[bucket], 1);
      }
    }
    atomicAdd(&smem_histogram[start_bucket], count);
  }

  __syncthreads();

  if (threadIdx.x < radixmgpusort::WARP_SIZE) {
    for (size_t bucket = threadIdx.x; bucket < radixmgpusort::NUM_BUCKETS; bucket += radixmgpusort::WARP_SIZE) {
      block_local_histograms[radixmgpusort::NUM_BUCKETS * blockIdx.x + bucket] = smem_histogram[bucket];
    }
  }
}

template <typename T>
__global__ void AggregateHistogram(uint64_cu* global_histogram, uint32_t* block_local_histograms, size_t total_num_blocks,
                                   size_t blocks_per_thread) {
  // write histogram information back to global memory
  // blocks_per_thread defines how many blocks one thread will aggregate
  if (threadIdx.x < radixmgpusort::NUM_BUCKETS) {
    size_t offset = blockIdx.x * radixmgpusort::NUM_BUCKETS * blocks_per_thread;
    uint64_cu count = 0;
    for (size_t i = 0; i < blocks_per_thread && blockIdx.x * blocks_per_thread + i < total_num_blocks; i++) {
      count += block_local_histograms[offset + (radixmgpusort::NUM_BUCKETS * i) + threadIdx.x];
    }
    atomicAdd(&global_histogram[threadIdx.x], (uint64_cu)count);
  }
}

__global__ void CheckHistogramSkewness(uint64_t* histogram, size_t* non_empty_count) {
  *non_empty_count = 0;
  for (size_t i = 0; i < radixmgpusort::NUM_BUCKETS; i++) {
    *non_empty_count += histogram[i] > 0;
  }
}

template <typename T>
__global__ void ScatterKeys(T* input, T* output, uint64_t* global_prefix_sums, uint32_t* block_local_histograms, uint64_cu* global_offsets,
                            size_t n, size_t k, size_t radix_msb) {
  const size_t index = ((size_t)blockDim.x * (size_t)blockIdx.x + threadIdx.x) * k;

  // stores the pre-scattered keys of this thread block in shared memory
  extern __shared__ __align__(sizeof(uint64_t)) uint8_t smem_buffer[];
  T* key_buffer = reinterpret_cast<T*>(smem_buffer);

  __shared__ int thread_bucket_map[radixmgpusort::NUM_THREADS_PER_BLOCK];

  __shared__ uint32_t local_histogram[radixmgpusort::NUM_BUCKETS];  // load the block-local histogram into shared memory
  __shared__ uint32_t local_offsets[radixmgpusort::NUM_BUCKETS];  // stores the block-local offsets for the key scatter within shared memory
  __shared__ uint64_t global_offset_per_bucket[radixmgpusort::NUM_BUCKETS];  // stores the global memory write offsets per bucket

  const uint64_t global_input_start = global_prefix_sums[0];

  if (threadIdx.x < radixmgpusort::NUM_BUCKETS) {
    local_histogram[threadIdx.x] = block_local_histograms[radixmgpusort::NUM_BUCKETS * blockIdx.x + threadIdx.x];
    global_offset_per_bucket[threadIdx.x] =
        global_prefix_sums[threadIdx.x] + atomicAdd(&global_offsets[threadIdx.x], (uint64_cu)local_histogram[threadIdx.x]);
  }

  const uint32_t buckets_to_handle_per_warp =
      radixmgpusort::NUM_BUCKETS / (radixmgpusort::NUM_THREADS_PER_BLOCK / radixmgpusort::WARP_SIZE);

  __syncthreads();

  if (threadIdx.x == 0) {
    uint64_t prefix_sum = 0;
    for (size_t i = 0; i < radixmgpusort::NUM_BUCKETS; i++) {
      local_offsets[i] = prefix_sum;
      prefix_sum += local_histogram[i];
    }

    int bucket = 0;
    for (size_t i = 0; i < radixmgpusort::NUM_THREADS_PER_BLOCK; i++) {
      if (i % radixmgpusort::WARP_SIZE == 0) {
        thread_bucket_map[i] = bucket;
        bucket += buckets_to_handle_per_warp;
      }
    }
  }

  __syncthreads();

  uint_type<sizeof(T)> radix_mask = (1 << radixmgpusort::NUM_RADIX_BITS) - 1;
  radix_mask = radix_mask << (radix_msb - radixmgpusort::NUM_RADIX_BITS);

  uint_type<sizeof(T)> bucket = 0;
  T key_value = 0;

  for (size_t i = 0; i < k && index + i < n; i++) {
    key_value = input[global_input_start + index + i];
    GetRadixBucket<T>(&key_value, radix_mask, radix_msb, &bucket);

    uint64_t offset = atomicAdd(&local_offsets[bucket], 1);
    key_buffer[offset] = key_value;
  }

  __syncthreads();

  // scatter keys back to global memory, writes are coalesced because of the pre-scatter in shared memory
  for (int b = 0; b < buckets_to_handle_per_warp; b++) {
    uint32_t bucket = thread_bucket_map[threadIdx.x - (threadIdx.x % radixmgpusort::WARP_SIZE)] + b;
    const uint32_t local_offset = local_offsets[bucket] - local_histogram[bucket];

    for (size_t i = threadIdx.x % radixmgpusort::WARP_SIZE; i < local_histogram[bucket]; i += radixmgpusort::WARP_SIZE) {
      output[global_offset_per_bucket[bucket] + i] = key_buffer[local_offset + i];
    }
  }
}

__global__ void CreateMgpuStripedHistogram(uint64_t* mgpu_histograms, uint64_t* mgpu_striped_histogram, size_t num_gpus) {
  if (blockDim.x != radixmgpusort::NUM_BUCKETS) {
    printf("ERROR: CreateStripedGpuHistogram requires exactly radixmgpusort::NUM_BUCKETS threads per block!\n");
  }

  if (gridDim.x != num_gpus) {
    printf("ERROR: CreateStripedGpuHistogram requires exactly num_gpus thread blocks!\n");
  }

  __shared__ uint64_t smem_mgpu_histogram[radixmgpusort::NUM_BUCKETS];

  if (threadIdx.x < radixmgpusort::NUM_BUCKETS) {
    smem_mgpu_histogram[threadIdx.x] = (uint64_t)mgpu_histograms[blockIdx.x * radixmgpusort::NUM_BUCKETS + threadIdx.x];
  }

  __syncthreads();

  if (threadIdx.x < radixmgpusort::NUM_BUCKETS) {
    mgpu_striped_histogram[threadIdx.x * num_gpus + blockIdx.x] = smem_mgpu_histogram[threadIdx.x];
  }
}

__device__ void GetStartGpuDistance(uint64_t* splitters, uint64_t value, int* distance, size_t num_gpus) {
  *distance = 0;
  for (size_t i = 0; i < num_gpus; i++) {
    *distance = i;
    if (value < splitters[i]) break;
  }
}

__device__ void GetEndGpuDistance(uint64_t* splitters, uint64_t value, int* distance, size_t num_gpus) {
  *distance = 0;
  for (size_t i = 0; i < num_gpus; i++) {
    *distance = i;
    if (value <= splitters[i]) break;
  }
}

__global__ void DetermineBucketToGpuMapping(uint64_t* mgpu_striped_prefix_sums, int* bucket_to_gpu_map, size_t chunk_size,
                                            size_t num_fillers, size_t num_gpus, size_t epsilon) {
  if (blockDim.x != radixmgpusort::NUM_BUCKETS) {
    printf("ERROR: DetermineBucketToGpuMapping requires exactly radixmgpusort::NUM_BUCKETS threads per block!\n");
  }

  if (gridDim.x != 1) {
    printf("ERROR: DetermineBucketToGpuMapping requires exactly 1 thread block!\n");
  }

  __shared__ uint64_t smem_mgpu_striped_prefix_sums[radixmgpusort::NUM_BUCKETS + 1];
  __shared__ uint64_t splitters[radixmgpusort::MAX_NUM_GPUS];

  if (threadIdx.x == 0) {
    smem_mgpu_striped_prefix_sums[radixmgpusort::NUM_BUCKETS] = mgpu_striped_prefix_sums[radixmgpusort::NUM_BUCKETS * num_gpus];

    for (size_t g = 0; g < num_gpus; g++) {
      splitters[g] = (g + 1) * chunk_size;
    }
  }

  if (threadIdx.x < radixmgpusort::NUM_BUCKETS) {
    smem_mgpu_striped_prefix_sums[threadIdx.x] = mgpu_striped_prefix_sums[threadIdx.x * num_gpus];
  }

  __syncthreads();

  if (threadIdx.x < radixmgpusort::NUM_BUCKETS) {
    size_t bucket_size = smem_mgpu_striped_prefix_sums[threadIdx.x + 1] - smem_mgpu_striped_prefix_sums[threadIdx.x];

    if (bucket_size > 0) {
      int start_gpu = 0;
      GetStartGpuDistance(&splitters[0], smem_mgpu_striped_prefix_sums[threadIdx.x], &start_gpu, num_gpus);

      int end_gpu = 0;
      GetEndGpuDistance(&splitters[0], smem_mgpu_striped_prefix_sums[threadIdx.x + 1], &end_gpu, num_gpus);

      if (end_gpu - start_gpu == 1) {
        int start_overflow = splitters[start_gpu] - smem_mgpu_striped_prefix_sums[threadIdx.x];
        int end_overflow = GetAbsolute(splitters[start_gpu] - smem_mgpu_striped_prefix_sums[threadIdx.x + 1]);

        if (start_overflow > end_overflow) {
          if (end_overflow <= epsilon) {
            end_gpu--;
          }
        } else {
          if (start_overflow <= epsilon) {
            start_gpu++;
          }
        }
      } else if (end_gpu - start_gpu >= 2) {
        int start_overflow = splitters[start_gpu] - smem_mgpu_striped_prefix_sums[threadIdx.x];
        int end_overflow = GetAbsolute(splitters[end_gpu - 1] - smem_mgpu_striped_prefix_sums[threadIdx.x + 1]);

        if (start_overflow <= epsilon) {
          start_gpu++;
        }

        if (end_overflow <= epsilon) {
          end_gpu--;
        }
      }

      for (size_t g = start_gpu; g <= end_gpu; g++) {
        bucket_to_gpu_map[(threadIdx.x * num_gpus) + g - start_gpu] = g;
      }
    }
  }
}