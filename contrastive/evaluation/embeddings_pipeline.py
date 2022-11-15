import os
import yaml
import omegaconf

from contrastive.evaluation.generate_embeddings import compute_embeddings
from contrastive.evaluation.train_multiple_classifiers import train_classifiers

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


# Auxilary function used to process the config linked to the model.
# For instance, change the embeddings save path to being next to the model.
def preprocess_config(sub_dir, dataset, classifier_name='svm', verbose=False):
    if verbose:
        print(os.getcwd())
    cfg = omegaconf.OmegaConf.load(sub_dir+'/.hydra/config.yaml')

    # replace the dataset
    with open(f'./configs/dataset/{dataset}.yaml', 'r') as file:
        dataset_yaml = yaml.load(file, yaml.FullLoader)
    for key in dataset_yaml:
        cfg[key] = dataset_yaml[key]
    
    # get the right classifiers parameters
    with open(f'./configs/classifier/{classifier_name}.yaml', 'r') as file:
        dataset_yaml = yaml.load(file, yaml.FullLoader)
    for key in dataset_yaml:
        cfg[key] = dataset_yaml[key]

    # replace the possibly incorrect config parameters
    cfg.training_labels = cfg['subject_labels_file']
    cfg.model_path = sub_dir
    cfg.embeddings_save_path = sub_dir + f"/{dataset}_embeddings"
    cfg.training_embeddings = sub_dir + f"/{dataset}_embeddings/full_embeddings.csv"

    return cfg


# main function
# creates embeddings and train classifiers for all models contained in the folder
@ignore_warnings(category=ConvergenceWarning)
def embeddings_pipeline(dir_path, dataset='cingulate_ACCpatterns', classifier_name='svm',
                        overwrite=False, verbose=False):
    """
    - dir_path: path where to apply recursively the process.
    - dataset: dataset the embeddings are generated from.
    - classifier_name: parameter to select the desired classifer type (currently neural_network
    or svm).
    - overwrite: redo the process on models where embeddings already exist.
    - verbose: verbose.
    """

    print("/!\ Convergence warnings are disabled")
    # walks recursively through the subfolders
    for name in os.listdir(dir_path):
        sub_dir = dir_path + '/' + name
        # checks if directory
        if os.path.isdir(sub_dir):
            # check if directory associated to a model
            if os.path.exists(sub_dir+'/.hydra/config.yaml'):
                print("Treating", sub_dir)

                # check if embeddings and ROC already computed
                # if already computed and don't want to overwrite, then pass
                # else apply the normal process
                if os.path.exists(sub_dir + f"/{dataset}_embeddings") and (not overwrite):
                    print("Model already treated (existing folder with embeddings). Set \
overwrite to True if you still want to compute them.")

                elif '#' in sub_dir:
                    print("Model with an incompatible structure with the current one. Pass.")

                else:
                    print("Start post processing")
                    # get the config and correct it to suit what is needed for classifiers
                    cfg = preprocess_config(sub_dir, dataset, classifier_name=classifier_name)
                    if verbose:
                        print("CONFIG FILE", type(cfg))
                        print(cfg)
                    # save the modified config next to the real one
                    with open(sub_dir+'/.hydra/config_classifiers.yaml', 'w') as file:
                        yaml.dump(omegaconf.OmegaConf.to_yaml(cfg), file)
                    
                    print(f"\nbefore compute_embeddings: training_labels = {cfg.training_labels}\n")

                    # apply the functions
                    compute_embeddings(cfg)
                    # reload config for train_classifiers to work properly
                    cfg = omegaconf.OmegaConf.load(sub_dir+'/.hydra/config_classifiers.yaml')
                    print(f"\nbefore train_classifiers: training_labels = {cfg.training_labels}\n")
                    train_classifiers(cfg)

            else:
                print(f"{sub_dir} not associated to a model. Continue")
                embeddings_pipeline(sub_dir,  dataset=dataset, classifier_name=classifier_name,
                                    overwrite=overwrite, verbose=verbose)
        else:
            print(f"{sub_dir} is a file. Continue.")


# embeddings_pipeline("/neurospin/dico/agaudin/Runs/04_pointnet/Output",
embeddings_pipeline("/neurospin/dico/data/deep_folding/papers/ipmi2023/models/contrastive/trained_on_HCP_half_2",
dataset='cingulate_ACCpatterns_1', verbose=True, classifier_name='svm', overwrite=False)
#label_names: ["NEOFAC_A", "NEOFAC_O", "NEOFAC_C", "NEOFAC_N", "NEOFAC_E"]
#label_names: ["NEOFAC_C"]