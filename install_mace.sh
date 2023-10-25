#!/usr/bin/env bash
# Exit on error
set -euo pipefail

CONDA_ENV_NAME="mlptrain-mace"

echo "* Looking for mamba or conda executable *"
if which mamba; then
    CONDA_EXE=mamba
elif which micromamba; then
    CONDA_EXE=micromamba
elif which conda; then
    CONDA_EXE=conda
else
  echo "ERROR: conda executable not found!"
  exit 1
fi

echo "* Installing everything to a new conda environment called: ${CONDA_ENV_NAME} *"
$CONDA_EXE env create -n ${CONDA_ENV_NAME} --file environment_mace.yml

echo "* Installing mlptrain in editable mode *"
$CONDA_EXE run -n ${CONDA_ENV_NAME} pip install -e .

echo "* DONE! *"
