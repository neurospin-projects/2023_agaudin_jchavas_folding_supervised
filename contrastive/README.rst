SimCLR training
###############

This folder contains scripts to run preprocessing SimCLR training
and evaluation on cingulate folding structures.

Description of parts
====================

backbones
---------
Definition of neural network backbones, currently densenet and convnet (a more simple
CNN).

configs
-------
yaml hydra-like configuration files.

data
----
Definition of datasets and data modules.

evaluation
----------
Scripts to evaluate the models (clustering, classifiers, etc).

models
------
Model definitions, for instance SimCLR, supervised or not.

notebooks
---------
Notebooks. Mainly used for tests or visualisation, but some are doing more relevant
tasks.

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


Tutorial: generate the whole pipeline (deprecated?)
===================================================

To generate postprocessing in "test" mode, we run the following command:

python3 postprocessing_results.py test=True checkpoint_path="/host/volatile/jc225751/Runs/23_to_midl2021_software/Output/first-model/logs/default/version_0/checkpoints/epoch\=99-step\=5599.ckpt"


Tutorial: train a SimCLR model
==============================

You first have to set up the config the right way. For that, you have to modify the configs/config.yaml file and set the fields to the right yaml file. 'mode' has to be 'encoder'.

Then run the command line:
python3 train.py

* For multiple trainings, use the following instead (example with varying temperature):

.. code-block:: shell

    python3 train.py temperature=0.1,0.5 --multirun   (training one after another)

    python3 train.py temperature=0.1,0.5 hydra/launcher=joblib --multirun   (training in parallel)

/!\ When you use multirun, you have to take care about not modifying the config files, as the
config used is the one at the start of the model training => if you change it during a multirun,
all the models trained after will use the modified config.

* The path where the results are stored is written in the configs/hydra/local.yaml. Section run is for normal run and sweep for multirun.


Tutorial: generate and rate embeddings
======================================

To assert the quality of a representation, the current method is to use classifiers.
The python files involved are: 
- evaluation/generate_embeddings.py
- evaluation/pca_embeddings.py
- evaluation/train_multiple_classifiers.py
- evaluation/embeddings_pipeline.py

More information about these programs and the related yaml files is available in the 
**evaluation/README_classifier.rst**.

* /!\ To use most of these programs, you have to set up the **config_no_save.yaml** file instead of config.yaml. (The reason is to avoid to save countless small networks, that  can then be confused with the SimCLR.)

* /!\ To use these programs, you have to have the same network as the one used during training. It means that you have to choose the right backbone in config_no_save.yaml, the same output and latent space sizes in the corresponding yaml file, and that you need to have the same network structure be on the right branch at a compatible commit).