#!/usr/bin/env bash
# Exit on error
set -euo pipefail

CONDA_ENV_NAME="mlptrain-mace"

echo "* Looking for mamba or conda executable *"
if which mamba; then
  CONDAEXE=mamba
elif which conda; then
  CONDAEXE=conda
else
  echo "conda executable not found!"
  exit 1
fi

echo "* Installing everything to a new conda environment called: ${CONDA_ENV_NAME} *"

echo "* Installing mlp-train dependencies via conda *"
$CONDAEXE create -n ${CONDA_ENV_NAME} -f environment.yml -c conda-forge --yes

echo "* Installing MACE and its dependencies (PyTorch, e3nn, ...) *"
$CONDAEXE run -n ${CONDA_ENV_NAME} pip install -r requirements_mace.txt

echo "* Installing mlptrain in editable mode *"
$CONDAEXE run -n ${CONDA_ENV_NAME} pip install -e .

echo "* DONE! *"
