#!/bin/bash -l

#SBATCH --job-name=job
#SBATCH --time=02:00:00  
#SBATCH --constraint=gpu
#SBATCH --qos=regular         # use the `regular` queue
#SBATCH --account=jcesr_g    # don't forget the `_g`; you may want to use `jcesr_g`
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

module load python
module load cudatoolkit/11.5
conda activate molml

python train_model.py --model charge
# python train_model.py --model hamiltonian
# python train_model.py --model equi_hamiltonian
