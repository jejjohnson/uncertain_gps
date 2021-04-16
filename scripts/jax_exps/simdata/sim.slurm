#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --exclude=nodo17
#SBATCH --workdir=/home/emmanuel/projects/uncertain_gps/
#SBATCH --job-name=egp_sim
#SBATCH --output=/home/emmanuel/logs/egp_sim_%A.out


module load Anaconda3
source activate jax_py38


srun --ntasks 1 python -u scripts/jax_exps/simdata/sim.py --pred_model exact
