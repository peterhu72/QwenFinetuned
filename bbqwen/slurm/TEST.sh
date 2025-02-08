#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --qos=cs
#SBATCH --mem-per-cpu 10000MB
#SBATCH --ntasks 56
#SBATCH --nodes=1
#SBATCH --output="TESTyaml.slurm"
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

python -u /grphome/grp_handwriting/qwen/bbqwen/finetune_bb.py