#pragma once

#include <algorithm>
#include <bitset>
#include <numeric>
#include <sstream>
#include <string>

#include "argument_parser.cuh"
#include "cuda_error.cuh"

enum class SortingAlgorithmType { MULTI_GPU = 0, SINGLE_GPU = 1, CPU = 2 };

class Parser {
 public:
  Parser(const std::string& program_name, SortingAlgorithmType sorting_algorithm_type = SortingAlgorithmType::MULTI_GPU)
      : _sorting_algorithm_type(sorting_algorithm_type) {
    std::stringstream usage;
    ArgumentParser argument_parser;

    usage << "-- Usage: ./" << program_name << " <num_keys> ";
    std::string gpu_option = "";
    if (_sorting_algorithm_type == SortingAlgorithmType::MULTI_GPU) {
      gpu_option = "<gpus> ";
      usage << gpu_option;
    } else if (_sorting_algorithm_type == SortingAlgorithmType::SINGLE_GPU) {
      gpu_option = "<gpu-id> ";
      usage << gpu_option;
    } else if (_sorting_algorithm_type == SortingAlgorithmType::CPU) {
      usage << "<num_cpu_threads> ";
    }

    usage << "<data_type=uint32> <distribution_type=uniform> "
             "<distribution_parameter=0> <output_mode=default>\n";

    usage << "-- Example: ./" << program_name << " 2000000000 ";
    if (_sorting_algorithm_type == SortingAlgorithmType::MULTI_GPU) {
      usage << "0,1,2,3 ";
    } else if (_sorting_algorithm_type == SortingAlgorithmType::SINGLE_GPU) {
      usage << "0 ";
    } else if (_sorting_algorithm_type == SortingAlgorithmType::CPU) {
      usage << "128 ";
    }
    usage << "uint32 uniform\n";
    usage << "-- Argument options: \n";

    if (_sorting_algorithm_type == SortingAlgorithmType::MULTI_GPU || _sorting_algorithm_type == SortingAlgorithmType::SINGLE_GPU) {
      usage << "\t " << gpu_option << "= {";
      auto valid_gpus = argument_parser.GetValidGpus();
      for (size_t i = 0; i < valid_gpus.size(); i++) {
        usage << valid_gpus[i] << ((i == valid_gpus.size() - 1) ? "}\n" : ",");
      }
    } else if (_sorting_algorithm_type == SortingAlgorithmType::CPU) {
      usage << "\t <num_cpu_threads> = {1.." << radixmgpusort::NUM_CPU_THREADS << "}\n";
    }

    usage << "\t <data_type> = {";
    auto valid_data_types = argument_parser.GetValidDataTypes();
    for (size_t i = 0; i < valid_data_types.size(); i++) {
      usage << valid_data_types[i] << ((i == valid_data_types.size() - 1) ? "}\n" : ", ");
    }

    usage << "\t <distribution_type> = {";
    auto valid_distribution_types = argument_parser.GetValidDistributionsTypes();
    for (size_t i = 0; i < valid_distribution_types.size(); i++) {
      usage << valid_distribution_types[i] << ((i == valid_distribution_types.size() - 1) ? "}\n" : ", ");
    }

    usage << "\t <distribution_parameter> = {bit_entropy=0, zipf_exponent=0.0, file_name=../data/custom.csv} depending on "
             "<distribution_type>\n";
    usage
        << "\t\t Specify the bit entropy for <distribution_type> = skewed... <bit_entropy> = {0..32} or {0..64} depending on <data_type>\n";
    usage << "\t\t Specify the zipf exponent for <distribution_type> = zipf... <zipf_exponent> = {0.0..5.0}\n";
    usage << "\t\t Specify the file name for <distribution_type> = custom... <file_name> = file name and path to csv file containing input "
             "keys\n";

    usage << "\t <output_mode> = {";
    auto valid_output_modes = argument_parser.GetValidOutputModes();
    for (size_t i = 0; i < valid_output_modes.size(); i++) {
      usage << Arguments::GetOutputModeName(valid_output_modes[i]) << ((i == valid_output_modes.size() - 1) ? "}\n" : ", ");
    }

    _usage_info = usage.str();
  }

  bool Parse(int argc, char* argv[], Arguments* arguments) {
    if (argc < 3 || argc > 7) {
      std::cerr << _usage_info << std::endl;
      if (argc < 2) {
        std::cerr << "Error: Please specify the number of keys n" << std::endl;
      }

      if (argc < 3) {
        if (_sorting_algorithm_type == SortingAlgorithmType::MULTI_GPU) {
          std::cerr << "Error: Please specify the CUDA devices option, <gpus> = i,...,j" << std::endl;
        } else if (_sorting_algorithm_type == SortingAlgorithmType::SINGLE_GPU) {
          std::cerr << "Error: Please specify the GPU id." << std::endl;
        } else if (_sorting_algorithm_type == SortingAlgorithmType::CPU) {
          std::cerr << "Error: Please specify the number of CPU threads." << std::endl;
        }
      }

      return false;
    }

    ArgumentParser argument_parser;

    if (!IsNumericInteger(std::string(argv[1]))) {
      std::cerr << _usage_info << std::endl;
      return false;
    }

    arguments->num_keys = std::stoull(argv[1]);

    if (_sorting_algorithm_type == SortingAlgorithmType::MULTI_GPU || _sorting_algorithm_type == SortingAlgorithmType::SINGLE_GPU) {
      const std::string gpus_to_parse = argv[2];

      if (!argument_parser.ParseGpus(gpus_to_parse, &arguments->gpus)) {
        std::cerr << _usage_info << std::endl;
        std::cerr << "Error: Invalid CUDA device configuration. Available CUDA devices are: ";
        auto valid_gpus = argument_parser.GetValidGpus();
        for (size_t i = 0; i < valid_gpus.size(); i++) {
          std::cerr << valid_gpus[i] << ((i == valid_gpus.size() - 1) ? "\n" : ",");
        }
        return false;
      }

      if (_sorting_algorithm_type == SortingAlgorithmType::SINGLE_GPU) {
        if (arguments->gpus.size() != 1) {
          std::cerr << _usage_info << std::endl;
          std::cerr << "Error: Please specify exactly one GPU." << std::endl;
          return false;
        }
      }
    }

    if (_sorting_algorithm_type == SortingAlgorithmType::CPU) {
      if (!IsNumericInteger(std::string(argv[2]))) {
        std::cerr << _usage_info << std::endl;
        return false;
      }

      int num_cpu_threads = std::stoi(argv[2]);
      if (!argument_parser.IsValidNumCpuThreads(num_cpu_threads)) {
        std::cerr << _usage_info << std::endl;
        std::cerr << "Error: Invalid number of CPU threads. Available options are {1.." << radixmgpusort::NUM_CPU_THREADS << "}\n";
        return false;
      } else {
        arguments->num_cpu_threads = num_cpu_threads;
      }
    }

    if (argc >= 4) {
      arguments->data_type = argv[3];

      if (!argument_parser.IsValidDataType(arguments->data_type)) {
        std::cerr << _usage_info << std::endl;
        std::cerr << "Error: Invalid data type. Supported data types are: " << std::endl;
        auto valid_data_types = argument_parser.GetValidDataTypes();
        for (size_t i = 0; i < valid_data_types.size(); i++) {
          std::cerr << valid_data_types[i] << ((i == valid_data_types.size() - 1) ? "\n" : ", ");
        }
        return false;
      }
    }

    if (argc >= 5) {
      arguments->distribution_type = argv[4];

      if (!argument_parser.IsValidDistributionType(arguments->distribution_type)) {
        std::cerr << _usage_info << std::endl;
        std::cerr << "Error: Invalid distribution type. Supported data distributions are: " << std::endl;
        auto valid_distribution_types = argument_parser.GetValidDistributionsTypes();
        for (size_t i = 0; i < valid_distribution_types.size(); i++) {
          std::cerr << valid_distribution_types[i] << ((i == valid_distribution_types.size() - 1) ? "\n" : ", ");
        }
        return false;
      }
    }

    if (argc >= 6) {
      if (arguments->distribution_type == "skewed") {
        if (!IsNumericInteger(std::string(argv[5]))) {
          std::cerr << _usage_info << std::endl;
          return false;
        }

        arguments->bit_entropy = std::stoull(argv[5]);
        size_t num_bits = std::stoull(arguments->data_type.substr(arguments->data_type.size() - 2, std::string::npos));

        if (!argument_parser.IsValidBitEntropy(num_bits, arguments->bit_entropy)) {
          std::cerr << _usage_info << std::endl;
          std::cerr << "Error: Invalid bit entropy for " << arguments->data_type << ". Bit entropy must be within [0," << num_bits << "]."
                    << std::endl;
          return false;
        }
      } else if (arguments->distribution_type == "zipf") {
        if (!IsNumericFloat(std::string(argv[5]))) {
          std::cerr << _usage_info << std::endl;
          return false;
        }

        arguments->zipf_exponent = std::stod(argv[5]);

        if (!argument_parser.IsValidZipfExponent(arguments->zipf_exponent)) {
          std::cerr << _usage_info << std::endl;
          std::cerr << "Error: Invalid zipf exponent for the zipf data distribution. Zipf exponent must be >= 0.0." << std::endl;
          return false;
        }
      } else if (arguments->distribution_type == "custom") {
        arguments->file_name = argv[5];
      }
    }

    if (argc >= 7) {
      std::string output_mode_string = argv[6];
      if (!argument_parser.ParseOutputMode(output_mode_string, arguments->output_mode)) {
        std::cerr << _usage_info << std::endl;
        std::cerr << "Error: Invalid output mode. Supported output modes are: " << std::endl;
        auto valid_output_modes = argument_parser.GetValidOutputModes();
        for (size_t i = 0; i < valid_output_modes.size(); i++) {
          std::cerr << Arguments::GetOutputModeName(valid_output_modes[i]) << ((i == valid_output_modes.size() - 1) ? "\n" : ", ");
        }
        return false;
      }
    }

    if (arguments->distribution_type == "custom") {
      std::ifstream input_file(arguments->file_name);

      if (!input_file.is_open()) {
        std::cerr << _usage_info << std::endl;
        std::cerr << "Error: Invalid file name for the custom data distribution. Cannot open file." << std::endl;
        return false;
      }
    }

    return true;
  }

  void PrintUsageInfo() const { std::cerr << _usage_info << std::endl; }

  bool IsNumericInteger(const std::string& string_to_parse) {
    if (!std::all_of(string_to_parse.begin(), string_to_parse.end(), [](unsigned char c) { return std::isdigit(c); })) {
      return false;
    }

    return true;
  }

  bool IsNumericFloat(const std::string& string_to_parse) {
    std::string str_to_parse = string_to_parse;

    auto it = std::find(str_to_parse.begin(), str_to_parse.end(), '.');
    if (it != str_to_parse.end()) {
      str_to_parse.erase(it);
    }

    if (!std::all_of(str_to_parse.begin(), str_to_parse.end(), [](unsigned char c) { return std::isdigit(c); })) {
      return false;
    }

    return true;
  }

 private:
  std::string _usage_info;
  SortingAlgorithmType _sorting_algorithm_type;
};