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

# To be scheduled via slurm. Call this script from within the directory that you want to install rmg-sort into.
pip install matplotlib
pip install pandas
git clone https://github.com/hpides/rmg-sort.git
cd rmg-sort
./run_all.sh
