#!/usr/bin/env bash

set -o errexit

echo "* Looking for mamba or conda executable *"
if which mamba; then
    CONDAEXE=mamba
elif which micromamba; then
    CONDAEXE=micromamba
elif which conda; then
    CONDAEXE=conda
else
    echo "* ERROR conda executable not found! *"
    exit 1
fi

CONDA_ENV_NAME="mlptrain"

if [[ $CONDA_DEFAULT_ENV != "gha-test-env"  ]];then
    echo "Installing everything to a new conda environment called: $CONDA_ENV_NAME"
    $CONDAEXE env create -n ${CONDA_ENV_NAME} --file environment.yml
else
    CONDA_ENV_NAME="gha-test-env"
    # On GitHub the environment is auto-created by setup-micromamba action
    echo "* Skipping conda install, we're on Github, it's already there! *"
fi

echo "* Installing mlptrain package in editable mode *"
$CONDAEXE run -n ${CONDA_ENV_NAME} pip install -e .
echo "* DONE! *"
