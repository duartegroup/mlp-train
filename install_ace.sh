#!/usr/bin/env bash
# Exit on error
set -euo pipefail

# Install mlptrain together with dependencies for ACE
# NOTE: You need to install Julia >=1.6 before running this script!

export CONDA_ENV_NAME=mlptrain-ace
export CONDA_ENV_FILE=environment_ace.yml

echo "* Looking for Julia executable *"
if ! which julia; then
  echo "* ERROR: julia not found! *"
  echo "* Install Julia >= 1.10 first and add it to your PATH *"
  exit 1
fi

source create_conda_environment.sh

echo "* Adding required registries and packages to Julia *"
echo "using Pkg
Pkg.Registry.add(\"General\")
Pkg.Registry.add(RegistrySpec(url=\"https://github.com/ACEsuit/ACEregistry\"))
Pkg.add(PackageSpec(name=\"ACEpotentials\",version=\"0.6.9\"))
Pkg.add(\"JuLIP\")
Pkg.add(\"PrettyTables\")
Pkg.add(\"ExtXYZ\")
Pkg.add(\"ASE\")" > add_julia_pkgs.jl

julia add_julia_pkgs.jl

# NOTE: `conda activate` does not work in scripts, need to use `conda run`, see:
# https://stackoverflow.com/a/72395091
echo "* Setting up Python-Julia integration *"
$CONDA_EXE run -n ${CONDA_ENV_NAME} python -c "import julia; julia.install()"

rm -f add_julia_pkgs.jl
echo "* DONE! *"
