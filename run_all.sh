#!/bin/bash
# Run this script from within the rmg-sort directory.
git submodule update --init --recursive
./build.sh
python3 scripts/run_experiments.py build results
python3 scripts/plot_experiments.py results
