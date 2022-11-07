(*) notebooks are the important ones, either by the results they contain or by 
the fact that they implement methods used by other programs.

- anatomist_visualisation_model_selection
Notebook to get the extreme predictions' subject name for a given classifier.

- beta-VAE_analysis (*)
Notebook to preprocess and display classifiers results for the beta-VAEs computed
by Louise.

- classifier_results (*)
Notebook where to plot the curves and display the performance of all models in 
a chosen folder. It is also where to conduct the performance analysis about the
models database.

- consistency_analysis
Notebook where the nn distance is correlated to the auc_std. Should be used for
all analysis meant to know about a model consistency in its embeddings generation.

- generate_pointnet_data
Test notebook to know how to process pointnet data (padding and apply transforms).

- latent_space_visualization (*)
Notebook with methods to plot the embeddings (umap) and do some clustering.

- linear_classifier
Test notebook for the implementation and interpretation of the classifiers. Also
used for plotting ROC curves without mean and median agregated models.

- nearest_neighbours_extended
Notebook to compute the nn distance between latent spaces.

- nearest_neighbours
Test notebook for the nn distance. Contains the old version of the distance.

- pca_analysis
Notebook to generate PCA embeddings and to compare PCA and other dimension 
reduction methods to SimCLR.

- test_switch_to_numpy
Test notebook to transition from pickle datasets to numpy.