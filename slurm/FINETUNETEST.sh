#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --qos=cs
#SBATCH --mem-per-cpu 10000MB
#SBATCH --ntasks 56
#SBATCH --nodes=2
#SBATCH --output="FINETUNETESTyaml.slurm"
#SBATCH --time 24:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#%Module

#module load cuda/12.4.1-pw6cogp

module purge
export PATH="/grphome/grp_handwriting/qwen/qwenenv:$PATH"
eval "$(conda shell.bash hook)"
conda activate /grphome/grp_handwriting/qwen/qwenenv

python -u /grphome/grp_handwriting/qwen/finetuneTest.py