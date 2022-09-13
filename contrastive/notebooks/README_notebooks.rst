(*) notebooks are the important ones, either by the results they contain or by 
the fact that they implement methods used by other programs.

- classifier_results
Notebook where to plot the curves and display the performance of all models in 
a define folder. It is also where to conduct the performance analysis about the
models database.

- consistency_analysis
Notebook where the nn distance is correlated to the auc_std. Should be used for
all analysis meant to know about a model consistency in its embeddings generation.

- generate_bdd (*)
Notebook to generate the model database. You have to specify the folders with the
models you want to include, and then the database will be generated at the specified 
address.

- generate_pointnet_data
Test notebook to know how to process pointnet data (padding and apply transforms).

- latent_space_visualization (*)
Notebook with methods to plot the embeddings (umap) and do some clustering.

- linear_classifier
Test notebook for the implementation and interpretation of the classifiers.

- nearest_neighbours_extended
Notebook to compute the nn distance between latent spaces.

- nearest_neighbours
Test notebook for the nn distance. Contains the old version of the distance.

- pca_analysis
Notebook to compare PCA and other dimension reduction methods to SimCLR.

- test_switch_to_numpy
Test notebook to transition from pickle datasets to numpy.

- vanilla_pointnet
Test notebook to implement Pointnet outside of a SimCLR paradigm. Unfinished.