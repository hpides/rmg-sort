#include <omp.h>

#include <algorithm>
#include <numeric>
#include <parallel/algorithm>
#include <string>

#include "arguments.h"
#include "constants.h"
#include "cuda_error.cuh"
#include "data_generator.h"
#include "paradis.h"
#include "parser.cuh"
#include "radix_mgpu_sort.cuh"
#include "time.h"

template <typename T>
void BenchmarkSort(const std::string& algorithm, const Arguments& arguments, SortingAlgorithmType sorting_algorithm_type) {
  CheckCudaError(cudaSetDevice(0));

  int num_devices;
  CheckCudaError(cudaGetDeviceCount(&num_devices));

  int device = 0;
  cudaDeviceProp prop;
  CheckCudaError(cudaGetDeviceProperties(&prop, device));

  if (arguments.output_mode != OutputMode::CSV) {
    std::cout << "sorting algorithm: " << algorithm << std::endl;
    arguments.PrintDefault();
  } else {
    std::cout << "\"" << prop.name << "\"," << num_devices << ",\"" << algorithm << "\",";
    arguments.PrintCSV();
  }

  PinnedHostVector<T> keys(arguments.num_keys);
  std::vector<T> keys2(arguments.num_keys);
  double total_sort_duration = 0.0;

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
  __gnu_parallel::sort(keys2.begin(), keys2.end());

  if (algorithm == "radix-mgpu-sort" || algorithm == "thrust-sort") {
    omp_set_num_threads(arguments.gpus.size());
    omp_set_schedule(omp_sched_static, 1);
    omp_set_nested(1);

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

  } else if (algorithm == "gnu-parallel-sort") {
    omp_set_num_threads(arguments.num_cpu_threads);
    auto start = GetTimeNow();

    __gnu_parallel::sort(keys.begin(), keys.end());

    auto end = GetTimeNow();
    total_sort_duration = GetMilliseconds(start, end);
  } else if (algorithm == "paradis-sort") {
    auto start = GetTimeNow();

    paradis::sort<T>(keys.data(), keys.data() + arguments.num_keys, arguments.num_cpu_threads);

    auto end = GetTimeNow();
    total_sort_duration = GetMilliseconds(start, end);
  }

  if (sorting_algorithm_type == SortingAlgorithmType::CPU) {
    if (arguments.output_mode != OutputMode::CSV) {
      std::cout << "Total sort duration: " << total_sort_duration << std::endl;
    } else {
      std::cout << "0,0,0,0,0,0,\"\",";
      std::cout << total_sort_duration << std::endl;
    }
  }

  bool success = std::is_sorted(keys.begin(), keys.end()) && (keys == keys2);

  if (!success) {
    std::cerr << "Error: Invalid sort order." << std::endl;
  }
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "-- Usage: ./sort-benchmark <algorithm> (...) \n";
    std::cerr << "-- Algorithm options are: {radix-mgpu-sort, thrust-sort, gnu-parallel-sort, paradis-sort}.\n";
    std::cerr << "Error: Specify the sorting algorithm to benchmark.\n";
    return -1;
  }

  std::string algorithm = "";
  if (argc >= 2) {
    algorithm = argv[1];
  }

  Arguments arguments;
  SortingAlgorithmType sorting_algorithm_type;

  if (algorithm == "radix-mgpu-sort") {
    sorting_algorithm_type = SortingAlgorithmType::MULTI_GPU;
    Parser parser(algorithm, sorting_algorithm_type);

    char** args = &argv[1];
    if (!parser.Parse(argc - 1, args, &arguments)) {
      return -1;
    }
  } else if (algorithm == "thrust-sort") {
    sorting_algorithm_type = SortingAlgorithmType::SINGLE_GPU;
    Parser parser(algorithm, sorting_algorithm_type);

    char** args = &argv[1];
    if (!parser.Parse(argc - 1, args, &arguments)) {
      return -1;
    }

  } else if (algorithm == "gnu-parallel-sort") {
    sorting_algorithm_type = SortingAlgorithmType::CPU;
    Parser parser(algorithm, sorting_algorithm_type);

    char** args = &argv[1];
    if (!parser.Parse(argc - 1, args, &arguments)) {
      return -1;
    }
  } else if (algorithm == "paradis-sort") {
    sorting_algorithm_type = SortingAlgorithmType::CPU;
    Parser parser(algorithm, sorting_algorithm_type);

    char** args = &argv[1];
    if (!parser.Parse(argc - 1, args, &arguments)) {
      return -1;
    }
  } else {
    std::cerr << "-- Usage: ./sort-benchmark <algorithm> (...) \n";
    std::cerr << "-- Algorithm options are: {radix-mgpu-sort, thrust-sort, gnu-parallel-sort, paradis-sort}.\n";
    std::cerr << "Error: Invalid sorting algorithm specified.\n";
    return -1;
  }

  if (arguments.data_type == "uint32") {
    BenchmarkSort<uint32_t>(algorithm, arguments, sorting_algorithm_type);
  } else if (arguments.data_type == "uint64") {
    BenchmarkSort<uint64_t>(algorithm, arguments, sorting_algorithm_type);
  } else if (arguments.data_type == "float32") {
    BenchmarkSort<float>(algorithm, arguments, sorting_algorithm_type);
  } else if (arguments.data_type == "float64") {
    BenchmarkSort<double>(algorithm, arguments, sorting_algorithm_type);
  }

  CheckCudaError(cudaDeviceReset());

  return 0;
}
