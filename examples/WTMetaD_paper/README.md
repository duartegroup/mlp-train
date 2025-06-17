#  Active learning meets metadynamics: Automated workflow for reactive machine learning potentials

This folder contains codes and initial files for various computational tasks discussed in the paper "Active learning meets metadynamics: Automated workflow for reactive machine learning potentials" ([https://chemrxiv.org/engage/chemrxiv/article-details/671fe54b98c8527d9ea7647a]).

The manuscript demonstrates three reactions: the SN2 reaction in implicit solvent (referred to as r1), rearrangement reaction in the gas phase (r2), and glycosylation reaction in explicit solvent (r3). Python scripts for each reaction can be found in the corresponding folders. Access to training and testing data for all three reactions is available on Figshare with DOI: 10.6084/m9.figshare.28631591. For more questions, please contact the corresponding author, Fernanda Duarte.

For r1 and r2, active learning (AL) was utilised with both WTMetaD-IB and downhill sampling methods to train the potential. For r3, MLIP was trained by AL with WTMetaD-IB sampling method. Input geometries (as *.xyz files) and training script are located within folders named after the training methods (if with different samplign method), such as al_downhill.

In the case of r2, free energy calculations were performed using different enhanced sampling methods. All scripts can be found in the free_energy folder within the r2 directory. Within the free_energy folder, each enhanced sampling method is organised in separate folders containing Python scripts and necessary configuration files.
