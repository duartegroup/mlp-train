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
