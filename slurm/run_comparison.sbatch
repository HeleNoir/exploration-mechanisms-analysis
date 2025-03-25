#!/usr/bin/env bash

#SBATCH --time=96:00:00
#SBATCH --partition=cpu
#SBATCH --nodelist=oc226
#SBATCH --cpus-per-task=20
#SBATCH --mem=50G
#SBATCH --output=/data/oc226/%u/output/output-%j.txt

dimension=$1

srun nix develop --impure --command python comparison.py -d $dimension
