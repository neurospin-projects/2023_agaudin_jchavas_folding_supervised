import os
import yaml
import json
import omegaconf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist

from contrastive.evaluation.generate_embeddings import compute_embeddings


# Auxilary function used to process the config linked to the model.
# For instance, change the embeddings save path to eing next to the model.
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
    cfg.model_path = sub_dir
    cfg.embeddings_save_path = sub_dir + f"/{dataset}_embeddings"
    cfg.training_embeddings = sub_dir + \
        f"/{dataset}_embeddings/full_embeddings.csv"

    # add possibly missing config parameters
    if 'projection_head_hidden_layers' not in cfg.keys():
        cfg.projection_head_hidden_layers = None

    return cfg


def compute_hist_sim_zij(model_path, emb_types=['val'], q=0.1, threshold=0.90,
                         save=False, verbose=False):
    path = model_path + '/cingulate_HCP_embeddings'
    for emb_type in emb_types:
        # load the data
        full_path = path + f'/{emb_type}_embeddings.csv'
        embs = pd.read_csv(full_path, index_col=0)

        # compute the similarities
        sims = cdist(embs, embs, metric='cosine')
        sims = 1 - sims

        # plot them
        x = [sims[i, j] for i in range(embs.shape[0])
             for j in range(embs.shape[0]) if i < j]
        if verbose:
            print("check size (should be the same two numbers):",
                  len(x), np.sum(range(embs.shape[0])))

        if save:
            x = np.array(x)
            np.save(path+f'/{emb_type}_hist_sim_zij.npy', x)

        plt.figure()
        plt.hist(x, bins=np.linspace(-1, 1, 50))
        plt.title(f"{emb_type} embeddings")

        # check if model to exclude based on similarity repartition
        # (using quantile)
        quant = np.quantile(x, q)
        print("10% lowest similarity", quant)
        good_model = True
        if quant >= threshold:
            good_model = False
            print(f"PROBLEMATIC MODEL: {model_path}")

        if save:
            plt.savefig(path+f'/{emb_type}_hist_sim_zij.jpg')
            good_model_dict = {'quantile-percentage': q,
                               'quantile': quant,
                               'threshold': threshold,
                               'exclude': not good_model}
            with open(path+'/good_model.json', 'w') as file:
                json.dump(good_model_dict, file)
        else:
            plt.show()


def control_sim_zij(dir_path, dataset='cingulate_HCP', emb_types=['val'],
                    q=0.1, threshold=0.90, overwrite=False, verbose=False):
    """
    - dir_path: path where to apply recursively the process.
    - dataset: dataset the embeddings are generated from.
    - overwrite: redo the process on models where embeddings already exist.
    - verbose: verbose.
    """
    print("Start")
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
                if (
                    os.path.exists(sub_dir + f"/{dataset}_embeddings")
                    and (not overwrite)
                ):
                    print("Model already treated "
                          "(existing folder with embeddings). "
                          "Set overwrite to True "
                          "if you still want to compute them.")

                else:
                    print("Start post processing")
                    # get the config
                    # and correct it to suit what is needed for classifiers
                    cfg = preprocess_config(sub_dir, dataset)
                    if verbose:
                        print("CONFIG FILE", type(cfg))
                        print(cfg)
                    # save the modified config next to the real one
                    with open(sub_dir+'/.hydra/config_classifiers.yaml', 'w') \
                            as file:
                        yaml.dump(omegaconf.OmegaConf.to_yaml(cfg), file)

                    # apply the functions
                    compute_embeddings(cfg)

                    compute_hist_sim_zij(sub_dir, emb_types=emb_types, q=q,
                                         threshold=threshold,
                                         save=True, verbose=verbose)

            else:
                print(f"{sub_dir} not associated to a model. Continue")
                control_sim_zij(sub_dir, emb_types=emb_types, q=q,
                                threshold=threshold,
                                overwrite=overwrite, verbose=verbose)
        else:
            print(f"{sub_dir} is a file. Continue.")


control_sim_zij("/neurospin/dico/agaudin/Runs/04_pointnet/Output",
                dataset='cingulate_HCP', verbose=False, emb_types=['val'],
                overwrite=False)
