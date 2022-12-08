#!/bin/bash -l

#SBATCH --job-name=job
#SBATCH --time=05:00:00  
#SBATCH --constraint=gpu
#SBATCH --qos=regular         # use the `regular` queue
#SBATCH --account=jcesr_g    # don't forget the `_g`; you may want to use `jcesr_g`
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

module load python
module load cudatoolkit/11.5
conda activate molml

# python train_interpolate_model.py 
# python raytune_model.py --wandb_project interpolate_diff_raytune --model interpolate_diff
python train_interpolate_diff_model.py --use_best_config --use_wandb --reprocess_dataset