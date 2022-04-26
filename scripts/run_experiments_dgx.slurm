#!/bin/bash
#SBATCH --mail-user=ivan.ilic@student.hpi.de
#SBATCH --mail-type=BEGIN,END
#SBATCH -A rabl
#SBATCH --partition sorcery
#SBATCH --nodelist dgxa100-01
#SBATCH --nodes=1
#SBATCH --cpus-per-task=256
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=8
#SBATCH --mem=900G
#SBATCH --time="1-12"

./build_dgx.sh
python3 scripts/run_experiments.py build_dgx
