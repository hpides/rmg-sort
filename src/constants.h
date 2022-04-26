#pragma once

#include <omp.h>

namespace radixmgpusort {

const size_t NUM_BUCKETS = 256;
const size_t NUM_RADIX_BITS = 8;

const size_t MAX_NUM_GPUS = 64;
const size_t WARP_SIZE = 32;                // number of threads per warp
const size_t NUM_THREADS_PER_BLOCK = 1024;  // number of threads per block
const size_t MAX_NUM_CONCURRENT_KERNELS = 2;
const size_t MAX_NUM_BUCKETS_FOR_REDUCED_SORTING = 128;
const size_t MIN_NUM_BUCKETS_FOR_SORT_COPY_OVERLAP = 4;

const size_t MAX_KEYS_DEBUG_PRINT = 1000;
const size_t BYTE_PADDING = 128;
const size_t CUB_SORT_TEMP_MEMORY = 128000000;
const size_t NUM_CPU_THREADS = omp_get_num_procs();

}  // namespace radixmgpusort
