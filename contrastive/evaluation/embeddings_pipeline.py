import os
import yaml
import json
import omegaconf

from generate_embeddings import compute_embeddings
from train_multiple_classifiers import train_classifiers

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


# Auxilary function used to process the config linked to the model.
# For instance, change the embeddings save path to being next to the model.
def preprocess_config(sub_dir, dataset, classifier_name='svm', verbose=False):
    if verbose:
        print(os.getcwd())
    cfg = omegaconf.OmegaConf.load(sub_dir+'/.hydra/config.yaml')

    # replace the dataset
    # first, remove some keys of the older dataset
    keys_to_remove = ['train_val_csv_file', 'train_csv_file', 'val_csv_file',
                      'test_intra_csv_file', 'test_csv_file']
    for key in keys_to_remove:
        if key in cfg.keys():
            cfg.pop(key)
    # add the ones of the target dataset
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
                        overwrite=False, verbose=False, use_best_model=False):
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
                        print(json.dumps(omegaconf.OmegaConf.to_container(cfg, resolve=True), indent=4, sort_keys=True))
                    # save the modified config next to the real one
                    with open(sub_dir+'/.hydra/config_classifiers.yaml', 'w') as file:
                        yaml.dump(omegaconf.OmegaConf.to_yaml(cfg), file)
                    
                    print(f"\nbefore compute_embeddings: training_labels = {cfg.training_labels}\n")

                    # apply the functions
                    compute_embeddings(cfg)
                    # reload config for train_classifiers to work properly
                    cfg = omegaconf.OmegaConf.load(sub_dir+'/.hydra/config_classifiers.yaml')
                    train_classifiers(cfg)
                    
                    # compute embeddings for the best model if saved
                    # FULL BRICOLAGE
                    if use_best_model and os.path.exists(sub_dir+'/logs/best_model_weights.pt'):
                        # apply the functions
                        cfg = omegaconf.OmegaConf.load(sub_dir+'/.hydra/config_classifiers.yaml')
                        cfg.use_best_model = True
                        compute_embeddings(cfg)
                        # reload config for train_classifiers to work properly
                        cfg = omegaconf.OmegaConf.load(sub_dir+'/.hydra/config_classifiers.yaml')
                        cfg.use_best_model = True
                        cfg.training_embeddings = cfg.embeddings_save_path+'_best_model/full_embeddings.csv'
                        cfg.embeddings_save_path = cfg.embeddings_save_path+'_best_model'
                        train_classifiers(cfg)

            else:
                print(f"{sub_dir} not associated to a model. Continue")
                embeddings_pipeline(sub_dir,  dataset=dataset, classifier_name=classifier_name,
                                    overwrite=overwrite, verbose=verbose)
        else:
            print(f"{sub_dir} is a file. Continue.")

#'cingulate_ACCpatterns_1'
#'cingulate_Utrecht_dHCP'
embeddings_pipeline("/neurospin/dico/jlaval/Runs/01_deep_supervised/Program/Output/2023-05-02/",
dataset='cingulate_ACCpatterns_1', verbose=True, classifier_name='svm', overwrite=False)
