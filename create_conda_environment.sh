#!/bin/bash
# NOTE: This script should not be called on its own,
# but should be sourced from other install scripts such as install_ace.sh
set -euo pipefail

if [[ -z ${CONDA_ENV_NAME} || -z ${CONDA_ENV_FILE} ]];then
  echo "ERROR: Please pass in conda environment name as the first parameter"
  echo "ERROR: Please pass in conda environment file as the second parameter"
  exit 1
fi

if [[ ! -f ${CONDA_ENV_FILE} ]];then
    echo "ERROR: File ${CONDA_ENV_FILE} does not exist"
    exit 1
fi

echo "* Looking for mamba or conda executable *"
if which mamba; then
    export CONDA_EXE=mamba
elif which micromamba; then
    export CONDA_EXE=micromamba
elif which conda; then
    export CONDA_EXE=conda
else
    echo "* ERROR: conda executable not found! *"
    exit 1
fi

if [[ ${CONDA_DEFAULT_ENV-} != "gha-test-env"  ]];then
    echo "Installing everything to a new conda environment called: $CONDA_ENV_NAME"
    $CONDA_EXE env create -n "${CONDA_ENV_NAME}" --file ${CONDA_ENV_FILE}
else
    CONDA_ENV_NAME="gha-test-env"
    # On GitHub the environment is auto-created by setup-micromamba action
    echo "* Skipping conda install, we're on Github, it's already there! *"
fi

echo "* Installing mlptrain package in editable mode *"

if [[ ${CONDA_EXE} = "mamba" ]];then
    # For some reason `mamba run` does not seem to work, use conda instead.
    CONDA_EXE=conda
fi
${CONDA_EXE} run -n ${CONDA_ENV_NAME} python3 -m pip install -e .
