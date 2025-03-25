#!/usr/bin/env bash

dim=$1

sbatch slurm/run_analyses.sbatch "PSO" $dim
sbatch slurm/run_analyses.sbatch "PSO_RR" $dim
sbatch slurm/run_analyses.sbatch "SHADE" $dim
sbatch slurm/run_analyses.sbatch "PSO_GPGM" $dim
sbatch slurm/run_analyses.sbatch "PSO_NPGM" $dim
sbatch slurm/run_analyses.sbatch "PSO_PDM" $dim
sbatch slurm/run_analyses.sbatch "PSO_SRM" $dim