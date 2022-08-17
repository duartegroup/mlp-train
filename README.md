# mlp-train
General machine learnt potential (MLP) training for molecular systems

***
### Install


```
conda install --file requirements.txt
pip install -e .
```

ACE can be installed directly from `scripts/` with `source install_ace.sh`.

### Notes

- Units are: distance (Å), energy (eV), force (eV Å-1), time (fs)
- Training using molecular mechanics (MM) is not supported as we've not found it to be efficient

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
