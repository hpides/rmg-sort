#!/bin/bash
#SBATCH --mail-user=ivan.ilic@student.hpi.de
#SBATCH --mail-type=BEGIN,END
#SBATCH -A rabl
#SBATCH --partition sorcery
#SBATCH --nodelist dgxa100-01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus=8
#SBATCH --mem=100G
#SBATCH --time="1-12"

./build_dgx.sh
python3 -u scripts/run_tests.py 8 build_dgx default | tee default_test_results_dgx.txt
python3 -u scripts/run_tests.py 8 build_dgx skew | tee skew_test_results_dgx.txt
python3 -u scripts/run_tests.py 8 build_dgx custom | tee custom_test_results_dgx.txt
python3 -u scripts/run_tests.py 8 build_dgx other | tee other_test_results_dgx.txt
