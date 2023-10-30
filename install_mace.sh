#!/usr/bin/env bash
set -euo pipefail

# Install mlptrain together with dependencies for MACE
export CONDA_ENV_NAME=mlptrain-mace
export CONDA_ENV_FILE=environment_mace.yml

source create_conda_environment.sh
echo "* Installing OpenMM-ML *"
# NOTE: The upstream PR to openmm-ml has not been merged yet: https://github.com/openmm/openmm-ml/pull/61)
pip install git+https://github.com/sef43/openmm-ml@mace

echo "* DONE! *"
