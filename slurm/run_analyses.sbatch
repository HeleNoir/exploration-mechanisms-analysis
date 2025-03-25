#!/usr/bin/env bash

#SBATCH --time=96:00:00
#SBATCH --partition=cpu
#SBATCH --nodelist=oc226
#SBATCH --cpus-per-task=20
#SBATCH --mem=50G
#SBATCH --output=/data/oc226/%u/output/output-%j.txt

dataset=$1
algorithm=$2

srun nix develop --impure --command python analysis.py -d $dataset -a $algorithm
