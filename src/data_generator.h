#pragma once

#include <assert.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <parallel/algorithm>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "zipf_distribution.h"

class DataGenerator {
 public:
  explicit DataGenerator(size_t num_threads, uint32_t distribution_seed = 2147483647)
      : _num_threads(num_threads), _distribution_seed(distribution_seed) {}

  template <typename T>
  void ComputeDistribution(T* begin, size_t num_keys, const std::string& distribution_type) {
    if (distribution_type == "uniform") {
      ComputeUniformDistribution<T>(begin, num_keys);
    } else if (distribution_type == "normal") {
      ComputeNormalDistribution<T>(begin, num_keys);
    } else if (distribution_type == "zero") {
      ComputeZeroDistribution<T>(begin, num_keys);
    } else if (distribution_type == "sorted") {
      ComputeSortedDistribution<T>(begin, num_keys);
    } else if (distribution_type == "reverse-sorted") {
      ComputeReverseSortedDistribution<T>(begin, num_keys);
    } else if (distribution_type == "nearly-sorted") {
      ComputeNearlySortedDistribution<T>(begin, num_keys);
    }
  }

  template <typename T>
  void ComputeDistribution(T* begin, size_t num_keys, const std::string& distribution_type, size_t bit_entropy) {
    if (distribution_type == "skewed") {
      ComputeSkewedDistribution<T>(begin, num_keys, bit_entropy);
    } else {
      throw std::runtime_error("Skewed data distribution requires bit entropy parameter.");
    }
  }

  template <typename T>
  void ComputeDistribution(T* begin, size_t num_keys, const std::string& distribution_type, double exponent) {
    if (distribution_type == "zipf") {
      ComputeZipfDistribution(begin, num_keys, exponent);
    } else {
      throw std::runtime_error("Zipf data distribution requires exponent parameter.");
    }
  }

  template <typename T>
  void ComputeDistribution(T* begin, size_t num_keys, const std::string& distribution_type, const std::string& file_name) {
    if (distribution_type == "custom") {
      LoadCustomDistributionFromFile(begin, num_keys, file_name);
    } else {
      throw std::runtime_error("Custom data distribution requires file name parameter.");
    }
  }

 private:
  template <typename T>
  void ComputeUniformDistribution(T* begin, size_t num_keys) {
#pragma omp parallel num_threads(_num_threads)
    {
      std::mt19937 random_generator = SeedRandomGenerator(_distribution_seed + static_cast<size_t>(omp_get_thread_num()));
      std::uniform_real_distribution<double> uniform_dist(0, std::numeric_limits<T>::max());

#pragma omp for schedule(static)
      for (size_t i = 0; i < num_keys; ++i) {
        *(begin + i) = static_cast<T>(uniform_dist(random_generator));
      }
    }
  }

  template <typename T>
  void ComputeNormalDistribution(T* begin, size_t num_keys) {
    const double mean = std::numeric_limits<T>::max() / 2.0;
    const double stddev = mean / 3.0;

#pragma omp parallel num_threads(_num_threads)
    {
      std::mt19937 random_generator = SeedRandomGenerator(_distribution_seed + static_cast<size_t>(omp_get_thread_num()));
      std::normal_distribution<double> normal_dist(mean, stddev);

#pragma omp for schedule(static)
      for (size_t i = 0; i < num_keys; ++i) {
        *(begin + i) = static_cast<T>(std::fabs(normal_dist(random_generator)));
      }
    }
  }

  template <typename T>
  void ComputeZeroDistribution(T* begin, size_t num_keys) {
#pragma omp parallel for num_threads(_num_threads) schedule(static)
    for (size_t i = 0; i < num_keys; ++i) {
      *(begin + i) = 0;
    }
  }

  template <typename T>
  void ComputeSortedDistribution(T* begin, size_t num_keys) {
    ComputeUniformDistribution<T>(begin, num_keys);

    __gnu_parallel::sort(begin, begin + num_keys);
  }

  template <typename T>
  void ComputeReverseSortedDistribution(T* begin, size_t num_keys) {
    ComputeUniformDistribution<T>(begin, num_keys);

    __gnu_parallel::sort(begin, begin + num_keys, std::greater<T>());
  }

  template <typename T>
  void ComputeNearlySortedDistribution(T* begin, size_t num_keys) {
    ComputeSortedDistribution<T>(begin, num_keys);

#pragma omp parallel num_threads(_num_threads)
    {
      std::mt19937 random_generator = SeedRandomGenerator(_distribution_seed + static_cast<size_t>(omp_get_thread_num()));

#pragma omp for schedule(static)
      for (size_t i = 0; i < num_keys - 1; ++i) {
        const double mean = 0.0;

        double stddev = *(begin + i + 1) - *(begin + i);

        if (stddev < std::numeric_limits<double>::max() / 2.0) {
          stddev *= 2.0;
        }

        std::normal_distribution<double> normal_dist(mean, stddev);

        T diff = static_cast<T>(std::fabs(normal_dist(random_generator)));

        if (*(begin + i) > std::numeric_limits<T>::max() - diff) {
          *(begin + i) = std::numeric_limits<T>::max();
        } else {
          *(begin + i) += diff;
        }
      }
    }
  }

  template <typename T>
  void ComputeSkewedDistribution(T* begin, size_t num_keys, size_t bit_entropy) {
    assert(bit_entropy <= sizeof(T) * 8);

#pragma omp parallel num_threads(_num_threads)
    {
      T max = (1 << bit_entropy) - 1;

      if (bit_entropy == sizeof(T) * 8) {
        max = std::numeric_limits<T>::max();
      } else if (bit_entropy == 1) {
        max = 2;
      }

      std::mt19937 random_generator = SeedRandomGenerator(_distribution_seed + static_cast<size_t>(omp_get_thread_num()));
      std::uniform_real_distribution<double> uniform_dist(0, max);

#pragma omp for schedule(static)
      for (size_t i = 0; i < num_keys; ++i) {
        *(begin + i) = static_cast<T>(uniform_dist(random_generator));
      }
    }
  }

  template <typename T>
  void ComputeZipfDistribution(T* begin, size_t num_keys, double exponent) {
    assert(exponent >= 0.0);

#pragma omp parallel num_threads(_num_threads)
    {
      std::mt19937 random_generator = SeedRandomGenerator(_distribution_seed + static_cast<size_t>(omp_get_thread_num()));
      ZipfDistribution<T> zipf_dist(std::numeric_limits<T>::max(), exponent);

#pragma omp for schedule(static)
      for (size_t i = 0; i < num_keys; ++i) {
        *(begin + i) = zipf_dist(random_generator);
      }
    }
  }

  template <typename T>
  void LoadCustomDistributionFromFile(T* begin, size_t num_keys, const std::string& file_name) {
    size_t key_count = 0;
    std::string line;
    std::ifstream input_file(file_name);

    if (input_file.is_open()) {
      while (std::getline(input_file, line, ',')) {
        T value;

        if (std::is_integral<T>::value) {
          value = static_cast<T>(std::stoull(line));
        } else if (std::is_floating_point<T>::value) {
          value = static_cast<T>(std::stod(line));
        }

        *(begin + key_count) = value;
        key_count++;

        if (key_count >= num_keys) break;
      }

      input_file.close();
    }
  }

  std::mt19937 SeedRandomGenerator(uint32_t distribution_seed) {
    const size_t seeds_bytes = sizeof(std::mt19937::result_type) * std::mt19937::state_size;
    const size_t seeds_length = seeds_bytes / sizeof(std::seed_seq::result_type);

    std::vector<std::seed_seq::result_type> seeds(seeds_length);
    std::generate(seeds.begin(), seeds.end(), [&]() {
      distribution_seed = (distribution_seed << 1) | (distribution_seed >> (-1 & 31));
      return distribution_seed;
    });
    std::seed_seq seed_sequence(seeds.begin(), seeds.end());

    return std::mt19937{seed_sequence};
  }

  const size_t _distribution_seed;
  size_t _num_threads;
};