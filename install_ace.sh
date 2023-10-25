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
  echo "* Install Julia >= 1.6 first and add it to your PATH *"
  exit 1
fi

source create_conda_environment.sh
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

rm -f add_julia_pkgs.jl pycall.jl
echo "* DONE! *"
