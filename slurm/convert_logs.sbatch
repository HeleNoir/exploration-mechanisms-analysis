#!/usr/bin/env bash

#SBATCH --time=96:00:00
#SBATCH --partition=cpu
#SBATCH --nodelist=oc226
#SBATCH --cpus-per-task=20
#SBATCH --mem=25G
#SBATCH --output=/data/oc226/%u/output/output-%j.txt

algorithm=$1
dimension=$2

srun nix develop --impure --command python convert_log_files.py -a $algorithm -d $dimension
