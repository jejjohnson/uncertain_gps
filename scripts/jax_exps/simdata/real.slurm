#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --exclude=nodo17
#SBATCH --workdir=/home/emmanuel/projects/uncertain_gps/
#SBATCH --job-name=egp_real
#SBATCH --output=/home/emmanuel/logs/egp_real_%A.out


module load Anaconda3
source activate jax_py38


srun --ntasks 1 python -u scripts/jax_exps/simdata/real.py --pred_model exact
srun --ntasks 1 python -u scripts/jax_exps/simdata/real.py --pred_model mc
srun --ntasks 1 python -u scripts/jax_exps/simdata/real.py --pred_model sigma
srun --ntasks 1 python -u scripts/jax_exps/simdata/real.py --pred_model gh
srun --ntasks 1 python -u scripts/jax_exps/simdata/real.py --pred_model taylor_o1
srun --ntasks 1 python -u scripts/jax_exps/simdata/real.py --pred_model taylor_o2
srun --ntasks 1 python -u scripts/jax_exps/simdata/real.py --pred_model mm_mc
srun --ntasks 1 python -u scripts/jax_exps/simdata/real.py --pred_model mm_sigma
srun --ntasks 1 python -u scripts/jax_exps/simdata/real.py --pred_model mm_gh