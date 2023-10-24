# Make sure to be in the scripts directory before running this script

set -euo pipefail

PYTHON_VERSION="3.9"
CONDA_ENV_NAME="mlptrain-mace"

echo "* Looking for mamba or conda executable *"
if which mamba; then
  CONDAEXE=mamba
elif which conda; then
  CONDAEXE=conda
else
  echo "conda executable not found!"
  exit 1
fi

echo "* Installing everything to a new conda environment called: ${CONDA_ENV_NAME}*"

echo "* Installing mlp-train dependencies via conda *"
$CONDAEXE create --name ${CONDA_ENV_NAME} python=${PYTHON_VERSION} pip --file ../requirements.txt -c conda-forge --yes

# Install ASE from master branch
echo "* Installing ASE from master branch *"
$CONDAEXE run -n ${CONDA_ENV_NAME} pip install git+https://github.com/rosswhitfield/ase@f2615a6e9a

echo "* Installing MACE and its dependencies (PyTorch, e3nn, ...) *"
$CONDAEXE run -n ${CONDA_ENV_NAME} pip install -r ../requirements_mace.txt

echo "* Installing mlptrain in editable mode *"
$CONDAEXE run -n ${CONDA_ENV_NAME} pip install -e ../

echo "* DONE! *"
