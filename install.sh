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

# Install xtb if not available
echo "Looking for xtb executable in PATH"
if ! which xtb; then
  echo "* Installing xTB via conda *"
  $CONDAEXE install -c conda-forge xtb
fi

# Finally install the mlptrain Python package in editable mode
pip install -e .
