[![Test with pytest](https://github.com/duartegroup/mlp-train/actions/workflows/pytest.yml/badge.svg?event=push)](https://github.com/duartegroup/mlp-train/actions/workflows/pytest.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/License-MIT%202.0-blue.svg)](https://opensource.org/licenses/mit)
[![GitHub issues](https://img.shields.io/github/issues/duartegroup/mlp-train.svg)](https://github.com/duartegroup/mlp-train/issues)

# mlp-train
General machine learning potentials (MLP) training for molecular systems in gas phase and solution

Available models:
- GAP
- ACE
- MACE


## Install

Each model is installed into individual conda environment:

```
# Install GAP
./install_gap.sh

# Install ACE
./install_ace.sh

# Install MACE
./install_mace.sh 
```

### Notes

- Units are: distance (Å), energy (eV), force (eV Å$`^{-1}`$), time (fs)

## Using with OpenMM

The OpenMM backend only works with MACE at the moment. The necessary dependencies are installed automatically via conda:

```console
./install_mace.sh
```

You should now be able to run `water_openmm.py` in `./examples` or run the jupyter notebook on Google Colab [`water_openmm_colab.ipynb`](./examples/water_openmm_colab.ipynb).

You can use OpenMM during active learning by passing the keyword argument `md_program="OpenMM"` to the `al_train` method.
You can run MD with OpenMM using `mlptrain.md_openmm.run_mlp_md_openmm()`

## For developers

We are happy to accept pull requests from users. Please first fork mlp-train repository. We use `pre-commit`, `Ruff` and `pytest` to check the code. Your PR needs to pass through these checks before is accepted. `Pre-commit` is installed as one the dependecies. To use it in your repository, run the following command in the mlp-train folder:

```
pre-commit install 
```

`Pre-commit` will then run automatically at each commit and will take care of installation and running of `Ruff`.

## Citations

If _mlptrain_ is used in a publication please consider citing the [paper](https://doi.org/10.1039/D2CP02978B):

```
@article{MLPTraining2022,
  doi = {10.1039/D2CP02978B},
  url = {https://doi.org/10.1039/D2CP02978B},
  year = {2022},
  publisher = {The Royal Society of Chemistry},
  author = {Young, Tom and Johnston-Wood, Tristan and Zhang, Hanwen and Duarte, Fernanda},
  title = {Reaction dynamics of Diels-Alder reactions from machine learned potentials},
  journal = {Phys. Chem. Chem. Phys.}
}
```

## Contact

For bugs or implementation requests, please use [GitHub Issues](https://github.com/duartegroup/mlp-train/issues)

