#!/usr/bin/env bash

dim=$1

sbatch slurm/convert_logs.sbatch "PSO" $dim
sbatch slurm/convert_logs.sbatch "PSO_RR" $dim
sbatch slurm/convert_logs.sbatch "SHADE" $dim
sbatch slurm/convert_logs.sbatch "PSO_GPGM" $dim
sbatch slurm/convert_logs.sbatch "PSO_NPGM" $dim
sbatch slurm/convert_logs.sbatch "PSO_PDM" $dim
sbatch slurm/convert_logs.sbatch "PSO_SRM" $dim