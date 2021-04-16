#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --exclude=nodo17
#SBATCH --workdir=/home/emmanuel/projects/uncertain_gps/
#SBATCH --job-name=egp_real_fapar
#SBATCH --output=/home/emmanuel/logs/egp_real_fapar_%A.out


module load Anaconda3
source activate jax_py38


srun --ntasks 1 python -u scripts/jax_exps/simdata/real.py --pred_model exact --y_var fapar
srun --ntasks 1 python -u scripts/jax_exps/simdata/real.py --pred_model mc --y_var fapar
srun --ntasks 1 python -u scripts/jax_exps/simdata/real.py --pred_model sigma --y_var fapar
srun --ntasks 1 python -u scripts/jax_exps/simdata/real.py --pred_model gh --y_var fapar
srun --ntasks 1 python -u scripts/jax_exps/simdata/real.py --pred_model taylor_o1 --y_var fapar
srun --ntasks 1 python -u scripts/jax_exps/simdata/real.py --pred_model taylor_o2 --y_var fapar
srun --ntasks 1 python -u scripts/jax_exps/simdata/real.py --pred_model mm_mc --y_var fapar
srun --ntasks 1 python -u scripts/jax_exps/simdata/real.py --pred_model mm_sigma --y_var fapar
srun --ntasks 1 python -u scripts/jax_exps/simdata/real.py --pred_model mm_gh --y_var fapar