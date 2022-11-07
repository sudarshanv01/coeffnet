#!/bin/bash -l

#SBATCH --job-name=job
#SBATCH --time=00:10:00  
#SBATCH --ntasks=1      
#SBATCH --cpus-per-task=2     # 2 cpus for the job
#SBATCH --gpus=2              # 1 gpu for the job 
#SBATCH --constraint=gpu
#SBATCH --qos=debug         # use the `regular` queue
#SBATCH --account=jcesr_g    # don't forget the `_g`; you may want to use `jcesr_g`
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

module load python
module load cudatoolkit/11.5
conda activate molml

# python train_model.py --model hamiltonian
python train_model.py --model charge --output_dir charge_checkpt --debug
