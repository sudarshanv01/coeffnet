#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --time=5:00:00
#SBATCH --account=silimorphous
#SBATCH --job-name=job
#SBATCH --output=frequency_flattening-%j.out
#SBATCH --error=frequency_flattening-%j.error
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source /home/svijay/bin/ENVIRONMENT_CONDA
source /home/svijay/bin/ENVIRONMENT_ML
conda activate molml

python train_hamiltonian_model.py --use_wandb