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

# Install ASE from master branch in a subshell
(
  cd scripts || exit
  source install_ase.sh
)

# Install xtb if not available
echo "* Looking for xtb executable in PATH *"
if ! which xtb; then
  echo "* Installing xTB from conda-forge *"
  $CONDAEXE install -c conda-forge --yes xtb
fi

echo "* Installing GAP requirements *"
pip install quippy-ase

# Finally install the mlptrain Python package in editable mode
pip install -e .
