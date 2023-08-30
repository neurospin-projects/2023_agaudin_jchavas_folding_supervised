=============
config README
=============

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