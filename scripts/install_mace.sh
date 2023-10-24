# Make sure to be in the scripts directory before running this script

set -euo pipefail

PYTHON_VERSION="3.9"

if [[ -n `which mamba` ]]; then
  CONDAEXE=mamba
elif [[ -n `which conda` ]]; then
  CONDAEXE=conda
else
  echo "conda not found!"
  exit 1
fi

echo "* Installing everything to a new conda environment called: mace *"

echo "* Installing mlp-train dependencies via conda *"
$CONDAEXE create --name mace python=${PYTHON_VERSION} --file ../requirements.txt -c conda-forge --yes

# Install ASE from master branch
./install_ase.sh

echo "* Installing MACE and its dependencies (PyTorch, e3nn, ...) *"
$CONDAEXE run -n mace pip install -r ../requirements_mace.txt

echo "* Installing mlptrain in editable mode *"
$CONDAEXE run -n mace pip install -e ../

echo "* DONE! *"
