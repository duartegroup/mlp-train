#!/usr/bin/env bash

set -o errexit

echo "* Looking for mamba or conda executable *"
if which mamba; then
    CONDAEXE=mamba
elif which conda; then
    CONDAEXE=conda
else
    echo "* ERROR conda executable not found! *"
    exit 1
fi

if [[ $CONDA_DEFAULT_ENV != "test-env"  ]];then
    echo "* Installing base dependencies via conda *"
    $CONDAEXE install -c conda-forge --file environment.yml --yes
else
    # On GitHub the environment is auto-created by setup-micromamba action
    echo "* Skipping conda install, we're on Github! *"
fi

echo "* Installing GAP requirements *"
pip install -r requirements_gap.txt

echo "* Installing mlptrain package in editable mode *"
pip install -e .
