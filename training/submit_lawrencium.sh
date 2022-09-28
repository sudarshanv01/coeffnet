#!/bin/bash -l

#SBATCH --job-name=job
#SBATCH --partition=es1
#SBATCH --qos=condo_mp_es1
#SBATCH --account=lr_mp
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2 
#SBATCH --constraint=es1_v100
#SBATCH --error=slurm-%j.err

module load cuda/10.2
conda activate molml

python model.py 
