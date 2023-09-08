#############
config README
#############

This folders contains the configuration yaml files that allows us to pass relevant keywords to hydra.
Here is a README to explain how this is organized, and what the keywords are used for.

In each folder, there are one or more yaml files that contains keywords specific to the topic of the folder.
There is a main file ``config.yaml`` that is used to specify for each folder which yaml file should be used.

The ``config_no_save.yaml`` is used by embeddings pipeline, in order not to save all the classifiers just like 
SimCLR models are. The keywords it contains are actually not relevant, as they are overridden when the 
function is called.

The ``sweep.yaml`` file is related to WandB, and is used to launch grid searches. You have to specify in it 
the variables of the config that you want to change during the search, and how they should be changed.
More about it in the `doc of WandB <https://docs.wandb.ai/guides/sweeps/define-sweep-configuration>`.


Folders
=======

augmentations
-------------
Parameters related to SimCLR augmentations. Each file is one type of augmentation, or a combination of several ones.

backbone
--------
Parameters related to the backbone of SimCLR, i.e. of the encoders from the skeleton to the latent space. Some 
parameters might be specific to a backbone.

classifier
----------
Parameters related to the classifiers used after the training of the SimCLR model. Each file correspond to a type 
of classifier.

dataset
-------
Parameters related to the dataset used for training and evaluation. Each file contains the path to the numpy arrays,
the info csv and a few more info, like the input shape. 

embeddings
----------
Parameters related to the generation of model embeddings, like which model to use or where to store them. This file 
doesn't have to be modified if you use embeddings_pipeline or supervised_pipeline, as everything is handled by the 
functions.

hydra
-----
Mandatory yaml for hydra to function properly. It contains notably the path where the models are saved.

label
-----
Files that contains only the label name. They are separated from the dataset yaml in order to avoid having several 
dataset yaml for the exact same dataset (one for each label).

mode
----
Files that determine the way the model is trained. It is in particular how you can choose if you want the model to 
be trained in a supervised way (classifier, regresser) or in an unsupervised way (decoder, encoder). //!\ The decoder 
mode might not be handled by the current version of the repository.

model
-----
A single file that contains parameters related to the SimCLR model as a whole, such as drop rate, temperature or if 
you want to use pretraining. It also contains parameters about the "converter" (a layer that fuse latent spaces when 
multiple encoder are used).

platform
--------
Parameters related to the environment set for training (choice between cpu and gpu, number of workers, use of BrainVISA).

projection_head
---------------
Parameters related to the projection head, such as its layers' activation and shape.

trainer
-------
Parameters related to the training process, such as learning rate, proportions of train/val, max number of epochs, etc.

wandb
-----
Parameters linked to wandb. It should contain your own account and project names. It also contains parameters about grid 
searches.