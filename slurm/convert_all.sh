#!/usr/bin/env bash

export dim=$1

sbatch slurm/convert_logs.sbatch "PSO"
sbatch slurm/convert_logs.sbatch "PSO_RR"
sbatch slurm/convert_logs.sbatch "SHADE"
sbatch slurm/convert_logs.sbatch "PSO_GPGM"
sbatch slurm/convert_logs.sbatch "PSO_NPGM"
sbatch slurm/convert_logs.sbatch "PSO_PDM"
sbatch slurm/convert_logs.sbatch "PSO_SRM"