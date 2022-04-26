# radix-mgpu-sort
RMG Sort: Radix-Partitioning-Based Multi-GPU Sorting

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
