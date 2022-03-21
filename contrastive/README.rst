SimCLR training
###############

This folder contains scripts to run preprocessing SimCLR training
 and evaluation on cingulate folding structures.

Description of parts
====================

backbones
---------
Definition of neural network backbones.

configs
-------
yaml hydra-like configuration files.

data
----
Definition of datasets and data modules.

evaluation
----------
Scripts to evaluate the models (clustering,...).

models
------
Model definitions.

notebooks
---------
Notebooks.

preprocessing
-------------
Command to get input crops from HCP dataset.

utils
-----
All utility functions that don't enter in other categories.

augmentations.py
----------------
File containing the augmentation classes.

losses.py
---------
File containing loss functions

train.py
--------
Python script to launch the training

synthesize_results.py
---------------------
Python script to synthesize the results.


Tutorial: generate the whole pipeline
=====================================

To generate postprocessing in "test" mode, we run the following command:

python3 postprocessing_results.py test=True checkpoint_path="/host/volatile/jc225751/Runs/23_to_midl2021_software/Output/first-model/logs/default/version_0/checkpoints/epoch\=99-step\=5599.ckpt"


