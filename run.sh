#!/bin/sh
#SBATCH --time=5:00:00           # Run time in hh:mm:ss
#SBATCH --mem=4096               # Maximum memory required (in megabytes)
#SBATCH --job-name=cse479_job
#SBATCH --partition=cseos2g
#SBATCH --gres=gpu:1

module load "tensorflow/py37/1.14"

python -u demo.py