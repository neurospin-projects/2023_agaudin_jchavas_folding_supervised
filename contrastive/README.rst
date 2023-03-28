===============
SimCLR training
===============

This folder contains scripts to run preprocessing SimCLR training
and evaluation on cingulate folding structures.

Description of parts
====================

backbones
---------
Definition of neural network backbones, currently densenet, convnet (a more simple
CNN) and pointnet (a network that takes point clouds as inputs).

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
tasks. A README explains more about each one in the notebooks section.

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

synthesize_results.py
---------------------
Python script to synthesize the results.

train.py
--------
Python script to launch the training



Tutorial: generate the whole pipeline (deprecated?)
===================================================

To generate postprocessing in "test" mode, we run the following command:

.. code-block:: shell

    python3 postprocessing_results.py test=True checkpoint_path="/host/volatile/jc225751/Runs/23_to_midl2021_software/Output/first-model/logs/default/version_0/checkpoints/epoch\=99-step\=5599.ckpt"


Tutorial: train a SimCLR model
==============================

You first have to set up the config the right way. For that, you have to modify the configs/config.yaml file and set the fields to the right yaml file. 'mode' has to be 'encoder'.

Then run the command line:

.. code-block:: shell

    python3 train.py

* For multiple trainings, use the following instead (example with varying temperature):

.. code-block:: shell

    python3 train.py temperature=0.1,0.5 --multirun   (training one after another)

    python3 train.py temperature=0.1,0.5 hydra/launcher=joblib --multirun   (training in parallel)

**/!\\** When you use multirun, you have to take care about not modifying the config files, as the
config used is the one at the start of the model training => if you change it during a multirun,
all the models trained after will use the modified config.

* The path where the results are stored is written in the ``configs/hydra/local.yaml``. Section run is for normal run and sweep for multirun.


Tutorial: generate and rate embeddings
======================================

To assert the quality of a representation, the current method is to use classifiers.

The python files involved are:

* ``evaluation/generate_embeddings.py``
* ``evaluation/pca_embeddings.py``
* ``evaluation/train_multiple_classifiers.py``
* ``evaluation/embeddings_pipeline.py``

More information about these programs and the related yaml files is available in the 
**evaluation/README_classifier.rst**.

**/!\\** To use most of these programs, you have to set up the **config_no_save.yaml** file instead of config.yaml.
(The reason is to avoid to save countless small networks, that can then be confused with the SimCLR.)

**/!\\** To use these programs, you have to have the same network as the one used during training. It means that you 
have to choose the right backbone in ``config_no_save.yaml``, the same output and latent space sizes in the corresponding 
yaml file, and that you need to have the same network structure, i.e. be on the right branch at a compatible commit.


Tutorial: generate a csv database of the models
===============================================

As a lot of models are trained, methods to create a database where their addresses and parameters are stored have been implemented.

The files involved in this process are:

* **utils/models_database.py** contains all the functions needed to preprocess the models, create the database and postprocess it.
* **evaluation/SimCLR_performance_criteria.py** compute the exclusion criteria based on the trivial minimum (all embeddings are collinear) for all the targeted models (same loop set up as ``embeddings_pipeline``).
* **evaluation/generate_bdd.py** actually loop on the targeted folders and create a database containing all the encountered models.


The produced database contains the path to the model, its loss values at the end of the training, its svm' accuracy and auc, and 
some parameters contained in its ``partial_config.yaml``. The config parameters contained in the database are the ones that changed at
least once between models.

You can notice that there is no way right now to add new models to the database. The only way to add new ones is to generate entirely
a new database, which is still not too long since there are not too much models yet.