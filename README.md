# radix-mgpu-sort
RMG Sort: Radix-Partitioning-Based Multi-GPU Sorting

Sorting is a fundamental database operation with use cases in index creation, duplicate removal, grouping, and sort-merge joins. In recent years, Graphics Processing Units (GPUs) emerged as database accelerators due to their massive parallelism and high-bandwidth memory. Many single-GPU sorting algorithms have been proposed and shown to outperform highly parallel CPU algorithms. Today's accelerator platforms include multiple GPUs with direct high-bandwidth Peer-to-Peer (P2P) interconnects. However, the few published multi-GPU sorting algorithms do not efficiently harness the all-to-all P2P transfer capability of modern interconnects, such as NVLink and NVSwitch. All previous multi-GPU sorting algorithms are sort-merge approaches. Their merging workload grows for increasing numbers of GPUs.

In this thesis, we propose a novel radix-partitioning-based multi-GPU sorting algorithm (RMG sort). We present a most-significant-bit (MSB) radix partitioning strategy that efficiently utilizes high-speed P2P interconnects while reducing the inter-GPU communication compared to prior merge-based algorithms. Independent of the number of GPUs, we exchange radix partitions between the GPUs in one all-to-all P2P key swap. We analyze the performance of RMG sort on two modern multi-GPU systems with different interconnect topologies. Our evaluation shows that RMG sort scales well with the number of GPUs. We measure it to outperform highly parallel CPU sorting algorithms up to $20\times$. Compared to two state-of-the-art merge-based multi-GPU sorting algorithms, we achieve speedups of up to $1.26\times$ and $1.8\times$ across both systems.

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

## How to run evaluation experiments
```
python3 scripts/run_experiments.py build
```

## How to generate plots for the experiment results

Plot evaluation results of folder ```experiments/<results>```:
```
python3 scripts/plot_experiments.py <results>
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
