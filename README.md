# RMG Sort: Radix-Partitioning-Based Multi-GPU Sorting

In recent years, graphics processing units (GPUs) emerged as database accelerators due to their massive parallelism and high-bandwidth memory. Sorting is a core database operation with many applications, such as output ordering, index creation, grouping, and sort-merge joins. Many single-GPU sorting algorithms have been shown to outperform highly parallel CPU algorithms. Today’s systems include multiple GPUs with direct high-bandwidth peer-to-peer (P2P) interconnects. However, previous multi-GPU sorting algorithms do not efficiently harness the P2P transfer capability of modern interconnects, such as NVLink and NVSwitch. In this paper, we propose RMG sort, a novel radix partitioning-based multi-GPU sorting algorithm. We present a most-significant-bit
partitioning strategy that efficiently utilizes high-speed P2P interconnects while reducing inter-GPU communication. Independent of the number of GPUs, we exchange radix partitions between the GPUs in one all-to-all P2P key swap and achieve nearly-perfect load balancing. We evaluate RMG sort on two modern multi-GPU systems. Our experiments show that RMG sort scales well with the input size and the number of GPUs, outperforming a parallel CPU-based sort by up to 20×. Compared to two state-of-the-art, merge-based, multi-GPU sorting algorithms, we achieve speedups of up to 1.3× and 1.8× across both systems. Excluding the CPU-GPU data transfer times and on eight GPUs, RMG sort outperforms the two merge-based multi-GPU sorting algorithms up to 2.7× and 9.2×

# Run everything in one script

To run one script that builds the project, runs evaluation experiments, and generates PDF plots for the results, there are two options:

**Option A)** Clone this repository or obtain the sources via .zip download. Then, from within the rmg-sort root directory, execute:
```
./run_all.sh
```

**Option B)** Just call a script that will automatically install the required pip-dependencies, clone this repository for you, and call the `run_all.sh` script. For this, you only need this one single script file on your filesystem:
```
./install_and_run_all.slurm
```
Since this is a slurm file, it is schedulable via slurm. Use `sbatch install_and_run_all.slurm` on your slurm summon server.


# Step by Step Guide

## How to initialize the project
```
git submodule update --init --recursive
```

## How to build and run
```
./build.sh
numactl -N 0 -m 0 ./build/radix-mgpu-sort 2000000000 0,1,2,3
numactl -N 0 -m 0 ./build/radix-mgpu-sort 30000 0,1,2,3 uint32 uniform 0 DEBUG
```

## How to run evaluation experiments from a live ssh-session on a multi-GPU server:
```
python3 scripts/run_experiments.py build
```
This creates an `experiments` folder and places the benchmark results into a subfolder, named after the current date/time (e.g., `2022_02_22_23_59_59`).


## How to schedule to run evaluation experiments via slurm:
```
sbatch scripts/run_experiments.slurm
```
Change the slurm options in ```run_experiments.slurm``` as needed, e.g. the server name and the requested resources.


## How to generate plots for the experiment results

Plot evaluation results of folder ```experiments/2022_02_22_23_59_59```, creating `.pdf` plots in that folder:
```
python3 scripts/plot_experiments.py 2022_02_22_23_59_59
```

## How to run automatic tests
```
python3 -u scripts/run_tests.py | tee test_results.txt
```

Run tests for all data distribution types with ```<g>``` GPUs:
```
python3 -u scripts/run_tests.py <g> build default | tee default_test_results.txt
python3 -u scripts/run_tests.py <g> build skew | tee skew_test_results.txt
python3 -u scripts/run_tests.py <g> build custom | tee custom_test_results.txt
python3 -u scripts/run_tests.py <g> build other | tee other_test_results.txt```
