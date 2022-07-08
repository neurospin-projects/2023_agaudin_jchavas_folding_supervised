import os
import yaml
import omegaconf

from contrastive.evaluation.generate_embeddings import compute_embeddings
from contrastive.evaluation.train_multiple_classifiers import train_classifiers



def preprocess_config(sub_dir, dataset, verbose=False):
    if verbose:
        print(os.getcwd())
    cfg = omegaconf.OmegaConf.load(sub_dir+'/.hydra/config.yaml')
    # replace the possibly incorrect config parameters
    cfg.model_path = sub_dir
    cfg.embeddings_save_path = sub_dir + f"/{dataset}_embeddings"
    cfg.training_embeddings = sub_dir + f"/{dataset}_embeddings/full_embeddings.csv"
    cfg.classifier_name = "svm"

    # replace the dataset
    with open(f'./configs/dataset/{dataset}.yaml', 'r') as file:
        dataset_yaml = yaml.load(file, yaml.FullLoader)
    for key in dataset_yaml:
        cfg[key] = dataset_yaml[key]

    return cfg



def embeddings_pipeline(dir_path, dataset='cingulate_ACCpatterns', overwrite=False, verbose=False):
    # walks recursivley through the subfolders
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
                    print("Model already treated (existing folder with embeddings and ROC). Set \
overwrite to True if you still want to compute them.")

                else:
                    print("Start post processing")
                    # get the config and correct it to suit what is needed for classifiers
                    cfg = preprocess_config(sub_dir, dataset)
                    if verbose:
                        print("CONFIG FILE", type(cfg))
                        print(cfg)
                    # save the modified config next to the real one
                    with open(sub_dir+'/.hydra/config_classifiers.yaml', 'w') as file:
                        yaml.dump(omegaconf.OmegaConf.to_yaml(cfg), file)
                    
                    # apply the functions
                    compute_embeddings(cfg)
                    # reload it for train_classifiers to work properly
                    cfg = omegaconf.OmegaConf.load(sub_dir+'/.hydra/config_classifiers.yaml')
                    train_classifiers(cfg)

            else:
                print(f"{sub_dir} not associated to a model. Continue")
                embeddings_pipeline(sub_dir)
        else:
            print(f"{sub_dir} is a file. Continue.")


embeddings_pipeline("/neurospin/dico/agaudin/Runs/03_monkeys/Output/analysis_folders/convnet",
dataset='cingulate_ACCpatterns', overwrite=False)