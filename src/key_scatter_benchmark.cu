#include <inttypes.h>
#include <omp.h>
#include <stdio.h>

#include <algorithm>
#include <bitset>
#include <numeric>
#include <parallel/algorithm>
#include <string>

#include "arguments.h"
#include "cuda_error.cuh"
#include "cuda_utils.cuh"
#include "data_generator.h"
#include "key_scatter_benchmark.cuh"
#include "parser.cuh"

template <typename T>
void KeyScatterBenchmark(const Arguments& arguments) {
  PinnedHostVector<T> keys(arguments.num_keys);
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

  if (arguments.output_mode == OutputMode::DEBUG && arguments.num_keys < radixmgpusort::MAX_KEYS_DEBUG_PRINT) {
    for (size_t i = 0; i < arguments.num_keys; i++) {
      std::bitset<sizeof(T) * 8> x(keys[i]);
      std::cout << keys[i] << " (" << x << ")\n";
    }
  }

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

  omp_set_schedule(omp_sched_static, 1);
  omp_set_nested(1);
  omp_set_num_threads(arguments.gpus.size());

  ScatterKeysBenchmark<T>(&keys, arguments.gpus, arguments.output_mode);
}

int main(int argc, char* argv[]) {
  Arguments arguments;
  Parser parser("key-scatter-benchmark");

  if (!parser.Parse(argc, argv, &arguments)) {
    return -1;
  }

  arguments.PrintDefault();

  if (arguments.data_type == "uint32") {
    KeyScatterBenchmark<uint32_t>(arguments);
  } else if (arguments.data_type == "uint64") {
    KeyScatterBenchmark<uint64_t>(arguments);
  } else if (arguments.data_type == "float32") {
    KeyScatterBenchmark<float>(arguments);
  } else if (arguments.data_type == "float64") {
    KeyScatterBenchmark<double>(arguments);
  }

  CheckCudaError(cudaDeviceReset());

  return 0;
}
