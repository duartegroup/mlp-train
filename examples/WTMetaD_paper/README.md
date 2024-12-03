#  Active learning meets metadynamics: Automated workflow for reactive machine learning potentials

This folder contains codes and initial files for various computational tasks discussed in the paper "Active learning meets metadynamics: Automated workflow for reactive machine learning potentials" ([https://chemrxiv.org/engage/chemrxiv/article-details/671fe54b98c8527d9ea7647a]).

The manuscript demonstrates three reactions: the SN2 reaction in implicit solvent (referred to as r1), rearrangement reaction in the gas phase (r2), and glycosylation reaction in explicit solvent (r3). Python scripts for each reaction can be found in the corresponding folders.

For r1 and r2, active learning (AL) was utilised with both WTMetaD-IB and downhill sampling methods to train the potential. Input geometries (as *.xyz files) and training script are located within folders named after the training methods, such as al_downhill.

In the case of r2, free energy calculations were performed using different enhanced sampling methods. All scripts can be found in the free_energy folder within the r2 directory. Within the free_energy folder, each enhanced sampling method is organised in separate folders containing Python scripts and necessary configuration files.