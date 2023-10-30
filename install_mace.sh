#!/usr/bin/env bash
set -euo pipefail

# Install mlptrain together with dependencies for MACE
export CONDA_ENV_NAME=mlptrain-mace
export CONDA_ENV_FILE=environment_mace.yml

source create_conda_environment.sh
echo "* DONE! *"
