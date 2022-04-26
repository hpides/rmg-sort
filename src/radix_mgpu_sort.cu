#include <omp.h>

#include <algorithm>
#include <bitset>
#include <numeric>
#include <parallel/algorithm>
#include <string>

#include "arguments.h"
#include "constants.h"
#include "cuda_error.cuh"
#include "cuda_utils.cuh"
#include "data_generator.h"
#include "parser.cuh"
#include "radix_mgpu_sort.cuh"

template <typename T>
void BenchmarkSort(const Arguments& arguments) {
  bool success = true;

  CheckCudaError(cudaSetDevice(0));

  int num_devices;
  CheckCudaError(cudaGetDeviceCount(&num_devices));

  int device = 0;
  cudaDeviceProp prop;
  CheckCudaError(cudaGetDeviceProperties(&prop, device));

  if (arguments.output_mode != OutputMode::CSV) {
    printf("Total CUDA-enabled device count: %i\n", num_devices);
    printf(" Device name: %s\n", prop.name);
    printf(" Memory clock rate (KHz): %d\n", prop.memoryClockRate);
    printf(" Memory bus width (bits): %d\n", prop.memoryBusWidth);
    printf(" Peak memory bandwidth (GB/s): %f\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);

    if (prop.deviceOverlap) {
      std::cout << " Concurrent copy/compute (asynchronous) engines: " << prop.asyncEngineCount << std::endl;
      std::cout << " Concurrent kernel executions: " << prop.concurrentKernels << std::endl << std::endl;
    }

    arguments.PrintDefault();
  } else {
    std::cout << "\"" << prop.name << "\"," << num_devices << ",";
    arguments.PrintCSV();
  }

  PinnedHostVector<T> keys(arguments.num_keys);
  std::vector<T> keys2(arguments.num_keys);

  omp_set_num_threads(radixmgpusort::NUM_CPU_THREADS);
  omp_set_schedule(omp_sched_static, 1);
  omp_set_nested(1);

  DataGenerator data_generator(radixmgpusort::NUM_CPU_THREADS);
  if (arguments.distribution_type == "skewed") {
    data_generator.ComputeDistribution<T>(&keys[0], arguments.num_keys, arguments.distribution_type, arguments.bit_entropy);
  } else if (arguments.distribution_type == "zipf") {
    data_generator.ComputeDistribution<T>(&keys[0], arguments.num_keys, arguments.distribution_type, arguments.zipf_exponent);
  } else if (arguments.distribution_type == "custom") {
    data_generator.ComputeDistribution<T>(&keys[0], arguments.num_keys, arguments.distribution_type, arguments.file_name);
  } else {
    data_generator.ComputeDistribution<T>(&keys[0], arguments.num_keys, arguments.distribution_type);
  }

  thrust::copy(keys.begin(), keys.end(), keys2.begin());

  if (arguments.output_mode == OutputMode::DEBUG && arguments.num_keys < radixmgpusort::MAX_KEYS_DEBUG_PRINT) {
    for (size_t i = 0; i < arguments.num_keys; i++) {
      std::bitset<sizeof(T) * 8> x(keys[i]);
      std::cout << keys[i] << " (" << x << ")\n";
    }
  }

  omp_set_num_threads(arguments.gpus.size());

#pragma omp parallel for
  for (size_t i = 0; i < arguments.gpus.size(); ++i) {
    CheckCudaError(cudaSetDevice(arguments.gpus[i]));

    for (size_t j = 0; j < arguments.gpus.size(); ++j) {
      if (i != j) {
        CheckCudaError(cudaDeviceEnablePeerAccess(arguments.gpus[j], 0));
      }
    }

    CheckCudaError(cudaDeviceSynchronize());
  }

  RadixMultiGpuSort<T>(&keys, arguments.gpus, arguments.output_mode);

  omp_set_num_threads(radixmgpusort::NUM_CPU_THREADS);
  __gnu_parallel::sort(keys2.begin(), keys2.end());

  size_t count = 0;
  if (arguments.output_mode == OutputMode::DEBUG && arguments.num_keys < radixmgpusort::MAX_KEYS_DEBUG_PRINT) {
    for (size_t i = 0; i < arguments.num_keys; i++) {
      std::bitset<sizeof(T) * 8> x(keys[i]);

      if (keys[i] != keys2[i]) {
        count++;
        std::cout << "\033[1;31m" << keys[i] << " (" << x << ") != " << keys2[i] << " WRONG at " << i << " \033[0m\n";
        success = false;
      } else {
        if (arguments.output_mode == OutputMode::DEBUG && arguments.num_keys < radixmgpusort::MAX_KEYS_DEBUG_PRINT) {
          std::cout << keys[i] << " (" << x << ") == " << keys2[i] << "\n";
        }
      }
    }

  } else {
#pragma omp parallel for num_threads(radixmgpusort::NUM_CPU_THREADS)
    for (size_t i = 0; i < arguments.num_keys; i++) {
      if (keys[i] != keys2[i]) {
        success = false;
        count++;
      }
    }
  }

  if (!success) {
    std::cerr << "Error: Invalid sort order: " << count << " wrong positions." << std::endl;
  }
}

int main(int argc, char* argv[]) {
  Arguments arguments;
  Parser parser("radix-mgpu-sort");

  if (!parser.Parse(argc, argv, &arguments)) {
    return -1;
  }

  if (arguments.data_type == "uint32") {
    BenchmarkSort<uint32_t>(arguments);
  } else if (arguments.data_type == "uint64") {
    BenchmarkSort<uint64_t>(arguments);
  } else if (arguments.data_type == "float32") {
    BenchmarkSort<float>(arguments);
  } else if (arguments.data_type == "float64") {
    BenchmarkSort<double>(arguments);
  }

  CheckCudaError(cudaDeviceReset());

  return 0;
}
