import os
import yaml
import json
import omegaconf

from generate_embeddings import compute_embeddings
from train_multiple_classifiers import train_classifiers

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


def get_save_folder_name(datasets, short_name):
    if short_name is not None:
        folder_name = short_name
    else:
        folder_name = ''
        for dataset in datasets:
            folder_name = folder_name + dataset + '_'
        folder_name = folder_name[:-1]  # remove the last _
    
    return folder_name


# Auxilary function used to process the config linked to the model.
# For instance, change the embeddings save path to being next to the model.
def preprocess_config(sub_dir, datasets, folder_name, classifier_name='svm', verbose=False):
    if verbose:
        print(os.getcwd())
    cfg = omegaconf.OmegaConf.load(sub_dir+'/.hydra/config.yaml')

    # replace the datasets
    # first, remove the keys of the older datasets
    cfg['dataset'] = {}

    # add the ones of the target datasets
    for dataset in datasets:
        with open(f'./configs/dataset/{dataset}.yaml', 'r') as file:
            dataset_yaml = yaml.load(file, yaml.FullLoader)
        cfg.dataset[dataset] = {}
        for key in dataset_yaml:
            cfg.dataset[dataset][key] = dataset_yaml[key]

    # get the right classifiers parameters
    with open(f'./configs/classifier/{classifier_name}.yaml', 'r') as file:
        dataset_yaml = yaml.load(file, yaml.FullLoader)
    for key in dataset_yaml:
        cfg[key] = dataset_yaml[key]

    # replace the possibly incorrect config parameters
    # cfg.training_labels = cfg.data[0]['subject_labels_file']  # pas besoin de cette ligne
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
def embeddings_pipeline(dir_path,
                        datasets='cingulate_ACCpatterns', short_name=None,
                        classifier_name='svm',
                        overwrite=False, verbose=False, use_best_model=False):
    """
    - dir_path: path where to apply recursively the process.
    - dataset: dataset the embeddings are generated from.
    - classifier_name: parameter to select the desired classifer type
    (currently neural_network or svm).
    - overwrite: redo the process on models where embeddings already exist.
    - verbose: verbose.
    """

    print("/!\\ Convergence warnings are disabled")
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
                    cfg = preprocess_config(sub_dir, datasets, folder_name,
                                            classifier_name=classifier_name)
                    if verbose:
                        print("CONFIG FILE", type(cfg))
                        print(json.dumps(omegaconf.OmegaConf.to_container(
                            cfg, resolve=True), indent=4, sort_keys=True))
                    # save the modified config next to the real one
                    with open(sub_dir+'/.hydra/config_classifiers.yaml', 'w') \
                            as file:
                        yaml.dump(omegaconf.OmegaConf.to_yaml(cfg), file)

                    print("\nbefore compute_embeddings: "
                          f"training_labels = {cfg.training_labels}\n")

                    # apply the functions
                    compute_embeddings(cfg)
                    # reload config for train_classifiers to work properly
                    cfg = omegaconf.OmegaConf.load(
                        sub_dir+'/.hydra/config_classifiers.yaml')
                    train_classifiers(cfg)

                    # compute embeddings for the best model if saved
                    # FULL BRICOLAGE
                    if (use_best_model and os.path.exists(sub_dir+'/logs/best_model_weights.pt')):
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
                                    short_name=short_name,
                                    classifier_name=classifier_name,
                                    overwrite=overwrite,
                                    use_best_model=use_best_model,
                                    verbose=verbose)
        else:
            print(f"{sub_dir} is a file. Continue.")


embeddings_pipeline("/neurospin/dico/agaudin/Runs/09_new_repo/Output/2023-05-31/test",
datasets=['cingulate_ACCpatterns'], short_name='cing_ACC',
classifier_name='svm', overwrite=True, use_best_model=False, verbose=False)
