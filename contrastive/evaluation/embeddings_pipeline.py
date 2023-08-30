import os
import yaml
import json
import omegaconf

from generate_embeddings import compute_embeddings
from train_multiple_classifiers import train_classifiers
from utils_pipelines import get_save_folder_name, change_config_datasets, change_config_label

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


# Auxilary function used to process the config linked to the model.
# For instance, change the embeddings save path to being next to the model.
def preprocess_config(sub_dir, datasets, label, folder_name, classifier_name='svm', verbose=False):
    """Loads the associated config of the given model and changes what has to be done,
    mainly the datasets, the classifier type and a few other keywords.
    
    Arguments:
        - sub_dir: str. Path to the directory containing the saved model.
        - datasets: list of str. List of the datasets to be used for the results generation.
        - label: str. Name of the label to be used for evaluation.
        - folder_name: str. Name of the directory where to store both embeddings and aucs.
        - classifier_name: str. Should correspond to a classifier yaml file's name 
        (currently either 'svm' or 'neural_network').
        - verbose: bool. Verbose.
        
    Output:
        - cfg: the config as an omegaconf object."""

    if verbose:
        print(os.getcwd())
    cfg = omegaconf.OmegaConf.load(sub_dir+'/.hydra/config.yaml')

    # replace the datasets
    change_config_datasets(cfg, datasets)
    # replace the label
    change_config_label(cfg, label)

    # get the right classifiers parameters
    with open(f'./configs/classifier/{classifier_name}.yaml', 'r') as file:
        dataset_yaml = yaml.load(file, yaml.FullLoader)
    for key in dataset_yaml:
        cfg[key] = dataset_yaml[key]

    # replace the possibly incorrect config parameters
    cfg.model_path = sub_dir
    cfg.embeddings_save_path = \
        sub_dir + f"/{folder_name}_embeddings"
    cfg.training_embeddings = \
        sub_dir + f"/{folder_name}_embeddings/full_embeddings.csv"
    cfg.apply_transformations = False

    return cfg


# main function
# creates embeddings and train classifiers for all models contained in folder
@ignore_warnings(category=ConvergenceWarning)
def embeddings_pipeline(dir_path, datasets, label, short_name=None, classifier_name='svm',
                        overwrite=False, use_best_model=False, permutations=True, verbose=False):
    """Pipeline to generate automatically the embeddings and compute the 
    associated aucs for all the models contained in a given directory.

    Arguments:
        - dir_path: str. Path where the models are stored and where is applied 
        recursively the process.
        - datasets: list of str. Datasets the embeddings are generated from.
        - label: str. Name of the label to be used for evaluation.
        - short_name: str or None. Name of the directory where to store both embeddings 
        and aucs. If None, use datasets to generate the folder name.
        - classifier_name: str. Parameter to select the desired classifer type
        (currently neural_network or svm).
        - overwrite: bool. Redo the process on models where embeddings already exist.
        - use_best_model: bool. Use the best model saved during to generate embeddings. 
        The 'normal' model is always used, the best is only added.
        - verbose: bool. Verbose.
    """

    print("/!\\ Convergence warnings are disabled")
    # walks recursively through the subfolders
    for name in os.listdir(dir_path):
        sub_dir = dir_path + '/' + name
        # checks if directory
        if os.path.isdir(sub_dir):
            # check if directory associated to a model
            if os.path.exists(sub_dir+'/.hydra/config.yaml'):
                print("\nTreating", sub_dir)

                # check if embeddings and ROC already computed
                # if already computed and don't want to overwrite, then pass
                # else apply the normal process
                folder_name = get_save_folder_name(datasets=datasets, short_name=short_name)
                if (
                    os.path.exists(sub_dir + f"/{folder_name}_embeddings")
                    and (not overwrite)
                ):
                    print("Model already treated "
                          "(existing folder with embeddings). "
                          "Set overwrite to True if you still want "
                          "to compute them.")

                elif '#' in sub_dir:
                    print(
                        "Model with an incompatible structure "
                        "with the current one. Pass.")

                else:
                    print("Start post processing")
                    # get the config and correct it to suit
                    # what is needed for classifiers
                    cfg = preprocess_config(sub_dir, datasets, label, folder_name,
                                            classifier_name=classifier_name)
                    cfg.permutations = permutations
                    if verbose:
                        print("CONFIG FILE", type(cfg))
                        print(json.dumps(omegaconf.OmegaConf.to_container(
                            cfg, resolve=True), indent=4, sort_keys=True))
                    # save the modified config next to the real one
                    with open(sub_dir+'/.hydra/config_classifiers.yaml', 'w') \
                            as file:
                        yaml.dump(omegaconf.OmegaConf.to_yaml(cfg), file)

                    # apply the functions
                    compute_embeddings(cfg)
                    # reload config for train_classifiers to work properly
                    cfg = omegaconf.OmegaConf.load(
                        sub_dir+'/.hydra/config_classifiers.yaml')
                    train_classifiers(cfg)

                    # compute embeddings for the best model if saved
                    if (use_best_model and os.path.exists(sub_dir+'/logs/best_model_weights.pt')):
                        print("COMPUTE AGAIN WITH THE BEST MODEL")
                        # apply the functions
                        cfg = omegaconf.OmegaConf.load(
                            sub_dir+'/.hydra/config_classifiers.yaml')
                        cfg.use_best_model = True
                        compute_embeddings(cfg)
                        # reload config for train_classifiers to work properly
                        cfg = omegaconf.OmegaConf.load(
                            sub_dir+'/.hydra/config_classifiers.yaml')
                        cfg.use_best_model = True
                        cfg.training_embeddings = cfg.embeddings_save_path + \
                            '_best_model/full_embeddings.csv'
                        cfg.embeddings_save_path = \
                            cfg.embeddings_save_path + '_best_model'
                        train_classifiers(cfg)

            else:
                print(f"{sub_dir} not associated to a model. Continue")
                embeddings_pipeline(sub_dir,
                                    datasets=datasets,
                                    label=label,
                                    short_name=short_name,
                                    classifier_name=classifier_name,
                                    overwrite=overwrite,
                                    use_best_model=use_best_model,
                                    permutations=permutations,
                                    verbose=verbose)
        else:
            print(f"{sub_dir} is a file. Continue.")


# STs_HCP_full_top-separated
# STs_dHCP_374_subjects
# STs_utrecht_dhcp_morpho
# STs_utrecht_40wks
embeddings_pipeline("/neurospin/dico/jlaval/Runs/01_deep_supervised/Program/Output/2023-06-15/",
datasets=['STs_dHCP_374_subjects'], label='gravityCenter_x', short_name='STs_dHCP_374_subjects',
classifier_name='svm', overwrite=True, use_best_model=False, permutations=False, verbose=False)
