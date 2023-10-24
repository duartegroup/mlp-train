#!/usr/bin/env bash

set -o errexit

if [[ -n `which mamba` ]]; then
  CONDAEXE=mamba
elif [[ -n `which conda` ]]; then
  CONDAEXE=conda
else
  echo "conda not found!"
  exit 1
fi

# Install the conda dependencies
$CONDAEXE install -c conda-forge --file requirements.txt

# Install ASE from master branch in a subshell
(
  cd scripts || exit
  source install_ase.sh
)

# Finally install the mlptrain Python package in editable mode
pip install -e .
