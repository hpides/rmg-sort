#pragma once

#include <algorithm>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "cuda_error.cuh"

class ArgumentParser {
 public:
  ArgumentParser() {
    int cuda_device_count;
    CheckCudaError(cudaGetDeviceCount(&cuda_device_count));

    valid_gpus.resize(cuda_device_count);
    std::iota(valid_gpus.begin(), valid_gpus.end(), 0);

    valid_distribution_types.push_back("uniform");
    valid_distribution_types.push_back("normal");
    valid_distribution_types.push_back("zero");
    valid_distribution_types.push_back("sorted");
    valid_distribution_types.push_back("reverse-sorted");
    valid_distribution_types.push_back("nearly-sorted");
    valid_distribution_types.push_back("skewed");
    valid_distribution_types.push_back("zipf");
    valid_distribution_types.push_back("custom");

    valid_data_types.push_back("uint32");
    valid_data_types.push_back("uint64");
    valid_data_types.push_back("float32");
    valid_data_types.push_back("float64");

    valid_output_modes.push_back(OutputMode::DEFAULT);
    valid_output_modes.push_back(OutputMode::DEBUG);
    valid_output_modes.push_back(OutputMode::CSV);
  }

  bool ParseGpus(const std::string& gpus_to_parse, std::vector<int>* gpus) const {
    std::string trimmed_gpus = gpus_to_parse;
    trimmed_gpus.erase(std::remove_if(trimmed_gpus.begin(), trimmed_gpus.end(), [](unsigned char c) { return std::isspace(c); }),
                       trimmed_gpus.end());

    if (trimmed_gpus.empty()) {
      return false;
    }

    if (!std::all_of(trimmed_gpus.begin(), trimmed_gpus.end(), [](unsigned char c) { return std::isdigit(c) || c == ','; })) {
      return false;
    }

    if (trimmed_gpus.find(",,") != std::string::npos || trimmed_gpus.front() == ',' || trimmed_gpus.back() == ',') {
      return false;
    }

    gpus->clear();
    std::string trimmed_gpu;
    std::stringstream trimmed_stream(trimmed_gpus);
    while (std::getline(trimmed_stream, trimmed_gpu, ',')) {
      gpus->emplace_back(std::stoi(trimmed_gpu));
    }

    return IsValidGpuSet(*gpus);
  }

  bool IsValidNumCpuThreads(int num_cpu_threads) { return num_cpu_threads >= 1 && num_cpu_threads <= radixmgpusort::NUM_CPU_THREADS; }

  bool IsValidGpuSet(const std::vector<int>& gpus) const {
    if (gpus.empty()) {
      return false;
    }

    if (std::set<int>(gpus.begin(), gpus.end()).size() != gpus.size()) {
      return false;
    }

    for (const auto& gpu : gpus) {
      if (std::find(valid_gpus.begin(), valid_gpus.end(), gpu) == valid_gpus.end()) {
        return false;
      }
    }

    return true;
  }

  bool IsValidDistributionType(const std::string& distribution_tpye) {
    if (std::find(valid_distribution_types.begin(), valid_distribution_types.end(), distribution_tpye) == valid_distribution_types.end()) {
      return false;
    } else {
      return true;
    }
  }

  bool IsValidDataType(const std::string& data_type) {
    if (std::find(valid_data_types.begin(), valid_data_types.end(), data_type) == valid_data_types.end()) {
      return false;
    } else {
      return true;
    }
  }

  bool ParseOutputMode(const std::string& output_mode_to_parse, OutputMode& output_mode) {
    std::string output_mode_string = output_mode_to_parse;
    std::transform(output_mode_string.begin(), output_mode_string.end(), output_mode_string.begin(), ::tolower);

    if (std::find(valid_output_modes.begin(), valid_output_modes.end(), output_mode) != valid_output_modes.end()) {
      for (size_t i = 0; i < valid_output_modes.size(); i++) {
        if (output_mode_string == Arguments::GetOutputModeName(valid_output_modes[i])) {
          output_mode = valid_output_modes[i];
          return true;
        }
      }
    }

    return false;
  }

  bool IsValidBitEntropy(size_t key_bit_width, size_t bit_entropy) { return bit_entropy <= key_bit_width; }

  bool IsValidZipfExponent(double exponent) { return exponent >= 0.0; }

  std::vector<int> GetValidGpus() const { return valid_gpus; }
  std::vector<std::string> GetValidDistributionsTypes() const { return valid_distribution_types; }
  std::vector<std::string> GetValidDataTypes() const { return valid_data_types; }
  std::vector<OutputMode> GetValidOutputModes() const { return valid_output_modes; }

 private:
  std::vector<int> valid_gpus;
  std::vector<std::string> valid_distribution_types;
  std::vector<std::string> valid_data_types;
  std::vector<OutputMode> valid_output_modes;
};
