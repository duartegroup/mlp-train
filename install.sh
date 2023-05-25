#!/usr/bin/env bash

set -o errexit

# Install the conda dependencies
conda install -c conda-forge --file requirements.txt

# Ensure the ASE install is from https://github.com/rosswhitfield/ase
conda uninstall ase --yes || true

# Install ASE in a subshell
(
  cd scripts || exit
  source install_ase.sh
)

# Finally install the Python package
pip install -e .
