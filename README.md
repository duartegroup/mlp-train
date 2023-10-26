[![Test with pytest](https://github.com/duartegroup/mlp-train/actions/workflows/pytest.yml/badge.svg?event=push)](https://github.com/duartegroup/mlp-train/actions/workflows/pytest.yml)

# mlp-train
General machine learning potentials (MLP) training for molecular systems in gas phase and solution

Available models:
- GAP
- ACE
- MACE


***
### Install

Each model is installed into individual conda environment:

```
# Install GAP
./install_gap.sh

# Install ACE
install_ace.sh

#Install MACE
install_mace.sh 
```


### Notes

- Units are: distance (Å), energy (eV), force (eV Å$^{-1}$), time (fs)

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
