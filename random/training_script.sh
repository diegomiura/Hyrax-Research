#!/bin/bash
#SBATCH --job-name="a.out_symmetric"
#SBATCH --output="a.out.%j.%N.out"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=50G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=16   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bemi-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH -t 01:00:00
#SBATCH -e slurm-%j.err

echo "Loading modules"

# ——— load your environment ———
# (Adjust to how you activate conda / module load on Delta)
module load anaconda3_cpu
source activate hyrax      # or `conda activate hyrax`

echo "Done loading modules"

echo "Training"

hyrax train --runtime-config="/work/hdd/bemi/dmiura/800batch/my_config.toml"

echo "Done training"
