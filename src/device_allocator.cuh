#pragma once

#include <chrono>
#include <map>
#include <new>
#include <thread>

#include "constants.h"
#include "cuda_error.cuh"

struct DeviceAllocator {
  using value_type = uint8_t;

  void SetOffset(size_t num_bytes) {
    size_t aligned_num_bytes = num_bytes % radixmgpusort::BYTE_PADDING == 0
                                   ? num_bytes
                                   : num_bytes + (radixmgpusort::BYTE_PADDING - num_bytes % radixmgpusort::BYTE_PADDING);

    next_pointer = begin_pointers[begin_pointer_flag] + aligned_num_bytes;
  }

  void Malloc(size_t num_bytes) {
    size_t aligned_num_bytes = num_bytes % radixmgpusort::BYTE_PADDING == 0
                                   ? num_bytes
                                   : num_bytes + (radixmgpusort::BYTE_PADDING - num_bytes % radixmgpusort::BYTE_PADDING);

    CheckCudaError(cudaMalloc(reinterpret_cast<void**>(&begin_pointers[0]), sizeof(value_type) * aligned_num_bytes));

    next_pointer = begin_pointers[0];

    capacity = sizeof(value_type) * aligned_num_bytes;
  }

  void Malloc(value_type* primary_pointer, value_type* secondary_pointer, size_t device_vector_size) {
    begin_pointers[0] = primary_pointer;
    begin_pointers[1] = secondary_pointer;

    capacity = device_vector_size * sizeof(value_type);

    next_pointer = begin_pointers[begin_pointer_flag];
  }

  void Flip() {
    if (!pointer_to_num_bytes.empty() || begin_pointers[1] == nullptr) {
      throw std::logic_error("Bad Flip() on DeviceAllocator");
    }

    begin_pointer_flag = !begin_pointer_flag;

    next_pointer = begin_pointers[begin_pointer_flag];
  }

  void Free() {
    if (begin_pointers[1] == nullptr) {
      CheckCudaError(cudaFree(begin_pointers[0]));
    }
    begin_pointers[0] = nullptr;
  }

  value_type* allocate(size_t num_bytes) {
    if (next_pointer + (sizeof(value_type) * num_bytes) > begin_pointers[begin_pointer_flag] + capacity) {
      std::cout << "Error: Tried to allocate " << num_bytes << " bytes when " << next_pointer - begin_pointers[begin_pointer_flag]
                << " bytes are already allocated and only " << capacity << " bytes are available." << std::endl;
      throw std::bad_alloc();
    }

    value_type* current = next_pointer;

    size_t aligned_num_bytes = num_bytes % radixmgpusort::BYTE_PADDING == 0
                                   ? num_bytes
                                   : num_bytes + (radixmgpusort::BYTE_PADDING - num_bytes % radixmgpusort::BYTE_PADDING);
    next_pointer += aligned_num_bytes;

    pointer_to_num_bytes.emplace(current, aligned_num_bytes);

    return current;
  }

  void deallocate(value_type* current, size_t num_bytes) {
    if (pointer_to_num_bytes.count(current) == 1) {
      next_pointer -= pointer_to_num_bytes[current];

      pointer_to_num_bytes.erase(current);
    }
  }

  std::map<value_type*, size_t> pointer_to_num_bytes;

  std::array<value_type*, 2> begin_pointers = {nullptr, nullptr};
  value_type* next_pointer = nullptr;

  bool begin_pointer_flag = false;
  size_t capacity = 0;
};
