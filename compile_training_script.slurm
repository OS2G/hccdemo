#!/bin/bash
#SBATCH --job-name=Demo
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint='gpu_32gb'
#SBATCH --output=/work/cseos2g/papapalpi/HCC-Tutorial/Output-%x-%j.out

# SLURM Inherits your environment. cd $SLURM_SUBMIT_DIR not needed
pwd; hostname; date

srun bash launch_script.bash
