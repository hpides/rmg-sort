#!/bin/bash

# ./build.sh
python3 -u scripts/run_tests.py 4 build default | tee default_test_results_ac922.txt
python3 -u scripts/run_tests.py 4 build skew | tee skew_test_results_ac922.txt
python3 -u scripts/run_tests.py 4 build custom | tee custom_test_results_ac922.txt
python3 -u scripts/run_tests.py 4 build other | tee other_test_results_ac922.txt
