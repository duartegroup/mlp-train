#!/usr/bin/env bash
# Exit on error
set -euo pipefail

# NOTE: You need to install Julia >=1.6 before running this script!

CONDA_ENV_NAME="mlptrain-ace"

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

echo "* Looking for Julia executable *"
if ! which julia; then
  echo "* ERROR: julia not found! *"
  echo "* Install Julia >= 1.6 first and add it to your PATH *"
  exit 1
fi

echo "Installing everything to a new conda environment called: $CONDA_ENV_NAME"
$CONDAEXE env create -n ${CONDA_ENV_NAME} --file environment_ace.yml
# NOTE: `conda activate` does not work in scripts, we use `conda run` below.
# https://stackoverflow.com/a/72395091

echo "* Adding required registries and packages to Julia *"
echo "using Pkg
Pkg.Registry.add(\"General\")
Pkg.Registry.add(RegistrySpec(url=\"https://github.com/JuliaMolSim/MolSim.git\"))
Pkg.add(PackageSpec(name=\"JuLIP\", version=\"0.10.1\"))
Pkg.add(PackageSpec(name=\"ACE\", version=\"0.8.4\"))
Pkg.add(PackageSpec(name=\"IPFitting\", version=\"0.5.0\"))
Pkg.add(\"IJulia\")
Pkg.add(\"ASE\")" > add_julia_pkgs.jl
julia add_julia_pkgs.jl

echo "* Setting up Python-Julia integration *"
conda run -n ${CONDA_ENV_NAME} python -c "import julia; julia.install()"

echo "* Pointing PyCall to the version of Python in the new env *"

echo "ENV[\"PYTHON\"] = \"$(eval "which python")\"
using Pkg
Pkg.build(\"PyCall\")" > pycall.jl
julia pycall.jl

echo "* Installing mlptrain package in editable mode *" 
conda run -n ${CONDA_ENV_NAME} pip install -e .

rm -f add_julia_pkgs.jl pycall.jl
echo "* DONE! *"
