#!/usr/bin/env bash
# Exit on error
set -euo pipefail

CONDA_ENV_NAME="mlptrain-mace"

echo "* Looking for mamba or conda executable *"
if which mamba; then
  CONDAEXE=mamba
elif which micromamba; then
    CONDAEXE=micromamba
elif which conda; then
  CONDAEXE=conda
else
  echo "ERROR: conda executable not found!"
  exit 1
fi

echo "* Installing everything to a new conda environment called: ${CONDA_ENV_NAME} *"
$CONDAEXE env create -n ${CONDA_ENV_NAME} --file environment_mace.yml

#echo "* Installing MACE and its dependencies (PyTorch, e3nn, ...) *"
#$CONDAEXE run -n ${CONDA_ENV_NAME} pip install -r requirements_mace.txt

echo "* Installing mlptrain in editable mode *"
$CONDAEXE run -n ${CONDA_ENV_NAME} pip install -e .

echo "* DONE! *"
