#!/bin/bash
#SBATCH -A m1727
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=01:30:00
#SBATCH --nodes=1

module load python

srun python ../../paltas/paltas/generate.py ../py_files/$1 $PSCRATCH/generated_images/$2 --n $3 --tf_record --h5