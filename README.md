# mlp-train
General machine learnt potential (MLP) training for molecular systems

***
### Install

```
./install.sh
```

Machine learning potentials can be installed directly from [scripts/](scripts).

### Notes

- Units are: distance (Å), energy (eV), force (eV Å-1), time (fs)
- Training using molecular mechanics (MM) is not supported as we've not found it to be efficient


## Using with OpenMM

The OpenMM backend only works with MACE at the moment

First install openmm-torch from conda-forge:

```
conda install -c conda-forge openmm-torch
```

Then install MACE:
```
pip install git+https://github.com/ACEsuit/mace.git
```

Then install this specific fork of OpenMM-ML:
```
pip install git+https://github.com/sef43/openmm-ml@mace
```

Then install this fork of MLP-Train
```
git clone https://github.com/sef43/mlptrain
cd mlptrain
pip install -e .
```

xtb can be installed from conda-forge
```
conda install -c conda-forge xtb
```


Now run `water_openmm.py` in ./examples

## Citation

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
