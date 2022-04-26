cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(radix-partitioned-multi-gpu-sort LANGUAGES CUDA CXX)

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -fopenmp")

# thrust submodule
add_subdirectory(third_party/thrust)
set(THRUST_REQUIRED_SYSTEMS CUDA OMP)
thrust_create_target(ThrustCUDA HOST OMP DEVICE CUDA)

set(PARADIS_INCLUDE_DIR "third_party/simple_paradis/src/")

# radix-mgpu-sort
add_executable(radix-mgpu-sort "src/radix_mgpu_sort.cu")
target_link_libraries(radix-mgpu-sort PUBLIC Thrust::Thrust ThrustCUDA)
set_property(TARGET radix-mgpu-sort PROPERTY CUDA_ARCHITECTURES 80 72 70)

# sort-benchmark
add_executable(sort-benchmark "src/sort_benchmark.cu")
target_link_libraries(sort-benchmark PUBLIC Thrust::Thrust ThrustCUDA)
set_property(TARGET sort-benchmark PROPERTY CUDA_ARCHITECTURES 80 72 70)
target_include_directories(sort-benchmark SYSTEM PRIVATE ${PARADIS_INCLUDE_DIR})

# key-scatter-benchmark
add_executable(key-scatter-benchmark "src/key_scatter_benchmark.cu")
target_link_libraries(key-scatter-benchmark PUBLIC Thrust::Thrust ThrustCUDA)
set_property(TARGET key-scatter-benchmark PROPERTY CUDA_ARCHITECTURES 80 72 70)

# cub-radix-sort-benchmark
add_executable(cub-radix-sort-benchmark "src/cub_radix_sort_benchmark.cu")
target_link_libraries(cub-radix-sort-benchmark PUBLIC Thrust::Thrust ThrustCUDA)
set_property(TARGET cub-radix-sort-benchmark PROPERTY CUDA_ARCHITECTURES 80 72 70)