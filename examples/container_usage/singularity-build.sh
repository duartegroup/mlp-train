#!/bin/bash
#SBATCH --job-name=singularity-build
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --partition=short

SINGULARITY_CACHEDIR=$TMPDIR SINGULARITY_TMPDIR=$TMPDIR singularity build $SCRATCH/mlp-train.sif docker://ghcr.io/jackleland/mlp-train:latest

rsync -a $SCRATCH/ $DATA/singularity/