Classifier training
###################

The classifier is a model put after the SimCLR model that is
 supposed to classify embeddings as representations of crops
 with or without paracingulate.


There are two main python files used for that:

generate_embeddings
-------------------
generate_embeddings creates the embeddings of crops with a chosen
 SimCLR. It requires you to set in the embeddings.yaml config file 
 the path to the model folder (ending by YYYY-MM-DD/hh-mm-ss) and
 the path where to store the embeddings (created if needed).

train_classifier
----------------
train_classifier creates the classifier, trains it on the chosen
 embeddings, and produce evaluation results (e.g. ROC curve). It 
 is linked to the classifier config, where you have to provide the
 embeddings and labels paths to train the classifier on, embeddings 
 you want to have the result on (if different), and parameters for 
 the classifier and the training.
