#!/bin/bash
#SBATCH -A m1727
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1

module load conda
conda activate /global/common/software/m1727/vpadma/analysis_env

srun python3 ../../paltas/paltas/Analysis/train_model.py ../py_files/$1 --tensorboard_dir $PSCRATCH/full/$2 --h5
