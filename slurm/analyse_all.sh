#!/usr/bin/env bash

export dim=$1

sbatch slurm/run_analyses.sbatch "PSO"
sbatch slurm/run_analyses.sbatch "PSO_RR"
sbatch slurm/run_analyses.sbatch "SHADE"
sbatch slurm/run_analyses.sbatch "PSO_GPGM"
sbatch slurm/run_analyses.sbatch "PSO_NPGM"
sbatch slurm/run_analyses.sbatch "PSO_PDM"
sbatch slurm/run_analyses.sbatch "PSO_SRM"