#!/bin/bash -l

#SBATCH --job-name=job
#SBATCH --partition=es1
#SBATCH --qos=condo_mp_es1
#SBATCH --account=lr_mp
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --constraint=es1_v100
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1

module load cuda/10.2
conda activate molml

# --- Test ---
# python train_inner_interpolate_model.py --use_wandb --reprocess_dataset --num_updates 5
python train_hamiltonian_model.py --use_wandb

# --- Raytune ---
# python raytune_model.py --model diffclassifier 
# python raytune_model.py --model interpolate_diff
# python raytune_model.py --model interpolate

# --- Best Config --- 
# python train_diffclassifier_model.py --use_best_config --use_wandb
# python train_interpolate_diff_model.py --use_best_config --use_wandb
# python train_interpolate_model.py --use_best_config --use_wandb
