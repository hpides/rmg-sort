#include <cub/cub.cuh>

#include "cuda_error.cuh"
#include "cuda_utils.cuh"
#include "data_generator.h"
#include "device_allocator.cuh"
#include "time.h"

int main(int argc, char* argv[]) {
  if (argc != 5) {
    std::cerr << "Usage: ./cub-radix-sort-benchmark <num_keys> <gpu_id> <bit_entropy> <set_end_bit>" << std::endl;
    return -1;
  }

  size_t num_keys = std::stoull(argv[1]);
  int gpu = std::stoi(argv[2]);
  size_t bit_entropy = std::stoi(argv[3]);
  bool set_end_bit = std::stoi(argv[4]);

  size_t end_bit = sizeof(uint32_t) * 8;
  if (set_end_bit) {
    end_bit = std::min(sizeof(uint32_t) * 8, bit_entropy + 1);
  }

  cudaStream_t stream;
  CheckCudaError(cudaSetDevice(gpu));
  CheckCudaError(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  PinnedHostVector<uint32_t> keys(num_keys);

  DataGenerator data_generator(radixmgpusort::NUM_CPU_THREADS);
  data_generator.ComputeDistribution<uint32_t>(&keys[0], num_keys, "skewed", bit_entropy);

  DeviceAllocator device_allocator;
  device_allocator.Malloc(num_keys * sizeof(uint32_t) + radixmgpusort::CUB_SORT_TEMP_MEMORY);

  thrust::device_vector<uint32_t> device_vector(num_keys);

  uint32_t* alt_key_buffer = reinterpret_cast<uint32_t*>(device_allocator.allocate(num_keys * sizeof(uint32_t)));

  cub::DoubleBuffer<uint32_t> keys_double_buffer(thrust::raw_pointer_cast(device_vector.data()), alt_key_buffer);

  thrust::copy(keys.begin(), keys.end(), device_vector.begin());

  CheckCudaError(cudaDeviceSynchronize());

  auto start = GetTimeNow();

  size_t num_temporary_bytes = 0;
  cub::DeviceRadixSort::SortKeys(NULL, num_temporary_bytes, keys_double_buffer, num_keys, 0, end_bit, stream);

  uint8_t* temporary_storage = device_allocator.allocate(num_temporary_bytes);

  cub::DeviceRadixSort::SortKeys((void*)temporary_storage, num_temporary_bytes, keys_double_buffer, num_keys, 0, end_bit, stream);

  device_allocator.deallocate(temporary_storage, num_temporary_bytes);

  CheckCudaError(cudaStreamSynchronize(stream));

  auto end = GetTimeNow();

  double sort_time = GetMilliseconds(start, end);

  std::cout << "Sort duration: " << sort_time << "ms" << std::endl;

  if (keys_double_buffer.Current() == thrust::raw_pointer_cast(device_vector.data())) {
    thrust::copy(device_vector.begin(), device_vector.end(), keys.begin());
  } else if (keys_double_buffer.Current() == alt_key_buffer) {
    CheckCudaError(cudaMemcpyAsync(thrust::raw_pointer_cast(keys.data()), alt_key_buffer, sizeof(uint32_t) * num_keys,
                                   cudaMemcpyDeviceToHost, stream));
    CheckCudaError(cudaStreamSynchronize(stream));
  } else {
    std::cerr << "Error: SortKeys failed." << std::endl;
  }

  return 0;
}