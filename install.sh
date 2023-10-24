#!/usr/bin/env bash

set -o errexit

echo "* Looking for mamba or conda executable *"
if which mamba; then
  CONDAEXE=mamba
elif which conda; then
  CONDAEXE=conda
else
  echo "* ERROR conda executable not found! *"
  exit 1
fi

# Install the conda dependencies
$CONDAEXE install -c conda-forge --file requirements.txt --yes

echo "* Installing GAP requirements *"
pip install quippy-ase

echo "* Installing ASE from master branch *"
pip install git+https://gitlab.com/ase/ase@f2615a6e9a

# Finally install the mlptrain Python package in editable mode
pip install -e .
