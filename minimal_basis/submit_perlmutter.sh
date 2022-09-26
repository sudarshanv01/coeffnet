#!/bin/bash -l

#SBATCH --job-name=job
#SBATCH --time=0-4:00:00  
#SBATCH --ntasks=1      
#SBATCH --cpus-per-task=2     # 2 cpus for the job
#SBATCH --gpus=1              # 1 gpu for the job 
#SBATCH --constraint=gpu
#SBATCH --qos=regular         # use the `regular` queue
#SBATCH --account=jcesr_g    # don't forget the `_g`; you may want to use `jcesr_g`

module load pytorch/1.11.0
module load python
conda activate molml

python model.py 
