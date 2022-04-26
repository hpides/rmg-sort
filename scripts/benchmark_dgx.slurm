#!/bin/bash
#SBATCH --mail-user=ivan.ilic@student.hpi.de
#SBATCH --mail-type=BEGIN,END
#SBATCH -A rabl
#SBATCH --partition sorcery
#SBATCH --nodelist dgxa100-01
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=8
#SBATCH --mem=100G
#SBATCH --time="3:0:0"

./build_dgx.sh
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2
nsys profile -o DGXA100_2Bkeys_2GPUs_uniform.qdrep numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2,4,6
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2,4,6
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2,4,6
nsys profile -o DGXA100_2Bkeys_4GPUs_uniform.qdrep numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2,4,6
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,1,2,3,4,5,6,7
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,1,2,3,4,5,6,7
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,1,2,3,4,5,6,7
nsys profile -o DGXA100_2Bkeys_8GPUs_uniform.qdrep numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,1,2,3,4,5,6,7
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2 uint32 skewed 32
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2 uint32 skewed 32
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2 uint32 skewed 31
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2 uint32 skewed 31
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2 uint32 skewed 30
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2 uint32 skewed 30
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2 uint32 skewed 29
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2 uint32 skewed 29
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2 uint32 skewed 28
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2 uint32 skewed 28
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2 uint32 skewed 27
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2 uint32 skewed 27
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2 uint32 skewed 26
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2 uint32 skewed 26
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2 uint32 skewed 25
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2 uint32 skewed 25
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2 uint32 skewed 24
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2 uint32 skewed 24
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2,4,6 uint32 skewed 32
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2,4,6 uint32 skewed 32
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2,4,6 uint32 skewed 31
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2,4,6 uint32 skewed 31
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2,4,6 uint32 skewed 30
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2,4,6 uint32 skewed 30
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2,4,6 uint32 skewed 29
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2,4,6 uint32 skewed 29
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2,4,6 uint32 skewed 28
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2,4,6 uint32 skewed 28
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2,4,6 uint32 skewed 27
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2,4,6 uint32 skewed 27
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2,4,6 uint32 skewed 26
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2,4,6 uint32 skewed 26
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2,4,6 uint32 skewed 25
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2,4,6 uint32 skewed 25
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2,4,6 uint32 skewed 24
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,2,4,6 uint32 skewed 24
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,1,2,3,4,5,6,7 uint32 skewed 32
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,1,2,3,4,5,6,7 uint32 skewed 32
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,1,2,3,4,5,6,7 uint32 skewed 31
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,1,2,3,4,5,6,7 uint32 skewed 31
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,1,2,3,4,5,6,7 uint32 skewed 30
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,1,2,3,4,5,6,7 uint32 skewed 30
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,1,2,3,4,5,6,7 uint32 skewed 29
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,1,2,3,4,5,6,7 uint32 skewed 29
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,1,2,3,4,5,6,7 uint32 skewed 28
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,1,2,3,4,5,6,7 uint32 skewed 28
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,1,2,3,4,5,6,7 uint32 skewed 27
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,1,2,3,4,5,6,7 uint32 skewed 27
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,1,2,3,4,5,6,7 uint32 skewed 26
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,1,2,3,4,5,6,7 uint32 skewed 26
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,1,2,3,4,5,6,7 uint32 skewed 25
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,1,2,3,4,5,6,7 uint32 skewed 25
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,1,2,3,4,5,6,7 uint32 skewed 24
numactl -N 0 -m 0 ./build_dgx/radix-mgpu-sort 2000000000 0,1,2,3,4,5,6,7 uint32 skewed 24
