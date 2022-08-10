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

If _mlptrain_ is used in a publication please consider citing the [paper](https://doi.org/10.26434/chemrxiv-2022-59qc9):

```
@article{MLPTraining2022,
  doi = {10.26434/chemrxiv-2022-59qc9},
  url = {https://doi.org/10.26434/chemrxiv-2022-59qc9},
  year = {2022},
  publisher = {ChemRxiv},
  author = {Tom Young and Tristan Johnston-Wood and Hanwen Zhang and Fernanda Duarte},
  title = {Reaction dynamics of Diels-Alder reactions from machine learned potentials},
  journal = {ChemRxiv}
}
```