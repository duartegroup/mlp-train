#!/usr/bin/env bash
set -euo pipefail

# Install mlptrain together with dependencies for GAP
export CONDA_ENV_NAME=mlptrain-gap
export CONDA_ENV_FILE=environment.yml

source create_conda_environment.sh
echo "* DONE! *"
