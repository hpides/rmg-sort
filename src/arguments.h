#pragma once

#include <iostream>
#include <string>

#include "constants.h"

enum class OutputMode { DEFAULT = 0, DEBUG = 1, CSV = 2 };

struct Arguments {
  size_t num_keys;
  std::vector<int> gpus;
  int num_cpu_threads = radixmgpusort::NUM_CPU_THREADS;
  std::string data_type = "uint32";
  std::string distribution_type = "uniform";
  size_t bit_entropy = 0;
  double zipf_exponent = 0.0;
  std::string file_name = "../data/custom.csv";
  OutputMode output_mode = OutputMode::DEFAULT;

  static std::string GetOutputModeName(const OutputMode& output_mode) {
    if (output_mode == OutputMode::DEFAULT) {
      return "default";
    } else if (output_mode == OutputMode::DEBUG) {
      return "debug";
    } else if (output_mode == OutputMode::CSV) {
      return "csv";
    }

    return "";
  }

  void PrintDefault() const {
    std::cout << "execution mode: " << GetOutputModeName(output_mode) << std::endl;
    std::cout << "num_keys: " << num_keys << std::endl;
    std::cout << "gpus: ";
    for (int g = 0; g < gpus.size(); g++) {
      std::cout << gpus[g] << (g == gpus.size() - 1 ? "\n" : ",");
    }
    std::cout << "num_cpu_threads: " << num_cpu_threads << std::endl;
    std::cout << "data_type: " << data_type << std::endl;
    std::cout << "distribution_type: " << distribution_type << std::endl;
    if (distribution_type == "skewed") {
      std::cout << "bit_entropy: " << bit_entropy << std::endl;
    } else if (distribution_type == "zipf") {
      std::cout << "zipf_exponent: " << zipf_exponent << std::endl;
    } else if (distribution_type == "custom") {
      std::cout << "file_name: " << file_name << std::endl;
    }
  }

  void PrintCSV() const {
    std::cout << num_keys << ",\"";
    for (int g = 0; g < gpus.size(); g++) {
      std::cout << gpus[g] << (g == gpus.size() - 1 ? "" : ",");
    }
    if (gpus.size() == 0) {
      std::cout << "-";
    }
    std::cout << "\"," << num_cpu_threads << ",";
    std::cout << "\"" << data_type << "\",\"" << distribution_type << "\",";
    if (distribution_type == "skewed") {
      std::cout << bit_entropy << ",";
    } else if (distribution_type == "zipf") {
      std::cout << zipf_exponent << ",";
    } else if (distribution_type == "custom") {
      std::cout << file_name << ",";
    } else {
      std::cout << "0,";
    }
  }
};