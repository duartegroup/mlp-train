#!/bin/bash

# NOTE: You need to install Julia >=1.9 before running this script!

# Exit on error
set -euo pipefail

CONDA_ENV_NAME="mlptrain-ace"
PYTHON_VERSION="3.9"

if [[ -n `which mamba` ]]; then
  CONDAEXE=mamba
elif [[ -n `which conda` ]]; then
  CONDAEXE=conda
else
  echo "conda not found!"
  exit 1
fi

echo "Installing everything to a new conda environment called: $CONDA_ENV_NAME"
$CONDAEXE create --name ${CONDA_ENV_NAME} -c conda-forge python=${PYTHON_VERSION} pip --file ../requirements.txt --yes
# NOTE: `conda activate` does not work in scripts, we use `conda run` below.
# https://stackoverflow.com/a/72395091

# ----------------------------------------------------
echo "Adding required registries and packages to Julia"

echo "using Pkg
Pkg.Registry.add(\"General\")
Pkg.Registry.add(RegistrySpec(url=\"https://github.com/ACEsuit/ACEregistry\"))
Pkg.add(PackageSpec(name=\"ACEpotentials\", version=\"0.6.3\"))
Pkg.add(\"ASE\")" > add_julia_pkgs.jl
julia add_julia_pkgs.jl

echo "Setting up Python-Julia integration"
conda run -n ${CONDA_ENV_NAME} python -m pip install julia
conda run -n ${CONDA_ENV_NAME} python -c "import julia; julia.install()"

# ----------------------------------------------------
echo "Pointing PyCall to the version of Python in the new env"

echo "ENV[\"PYTHON\"] = \"$(eval "which python")\"
using Pkg
Pkg.build(\"PyCall\")" > pycall.jl
julia pycall.jl
rm pycall.jl

echo "Installing pyjulip"
conda run -n ${CONDA_ENV_NAME} python -m pip install pyjulip@git+https://github.com/casv2/pyjulip.git@8316043f66

# Useful env variables:
# JULIA_NUM_THREADS

echo "Installing mlptrain package in editable mode" 
conda run -n ${CONDA_ENV_NAME} python -m pip install -e ../

echo "Updating ASE"
conda run -n ${CONDA_ENV_NAME} python -m pip install git+https://github.com/rosswhitfield/ase@f2615a6e9a

echo "DONE!"
