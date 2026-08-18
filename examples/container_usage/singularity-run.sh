#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH --account=dtce-schmidt
#SBATCH --time=02:00:00
#SBATCH --mem-per-gpu=32G
#SBATCH --job-name=mlp-train-gputest
#SBATCH --output=logs/mlp-train-%A.out
#SBATCH --error=logs/mlp-train-%A.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jack.leland@dtc.ox.ac.uk

module load OpenMPI

nvidia-smi
which singularity 
singularity --version

mpirun singularity exec --nv -B $DATA/mlp-train:/data --pwd /data $DATA/singularity/mlp-train.sif /usr/bin/micromamba run -n mlptrain-mace python /data/water_mace.py