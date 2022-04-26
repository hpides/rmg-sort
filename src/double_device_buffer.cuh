#pragma once

#include <thrust/device_vector.h>

#include <vector>

template <typename T>
class DoubleDeviceBuffer {
 public:
  DoubleDeviceBuffer() {}

  ~DoubleDeviceBuffer() {
    CheckCudaError(cudaSetDevice(_gpu));

    for (size_t i = 0; i < _buffers.size(); i++) {
      _buffers[i].clear();
      _buffers[i].shrink_to_fit();
    }
  }

  void Allocate(size_t chunk_size, int gpu) {
    CheckCudaError(cudaSetDevice(gpu));

    _buffers.reserve(2);
    _buffers.emplace_back(chunk_size);
    _buffers.emplace_back(chunk_size);

    _gpu = gpu;
  }

  thrust::device_vector<T>* GetCurrent() { return &_buffers[_flag]; }

  thrust::device_vector<T>* GetTemp() { return &_buffers[!_flag]; }

  void FlipBuffers() { _flag = !_flag; }

 private:
  std::vector<thrust::device_vector<T>> _buffers;
  bool _flag = false;
  int _gpu;
};