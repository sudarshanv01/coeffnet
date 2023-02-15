#!/bin/bash -l

#SBATCH --job-name=job
#SBATCH --time=5:00:00  
#SBATCH --constraint=gpu
#SBATCH --qos=regular         # use the `regular` queue
#SBATCH --account=jcesr_g    # don't forget the `_g`; you may want to use `jcesr_g`
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

module load python
module load cudatoolkit/11.5
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
