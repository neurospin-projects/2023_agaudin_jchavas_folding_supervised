# Get the output of a target model for all subsets of a the given dataset.


import omegaconf
import torch
import pandas as pd
import os
import glob
import json
import yaml

from contrastive.utils.config import process_config
from contrastive.data.datamodule import DataModule_Evaluation
from contrastive.evaluation.utils_pipelines import *
from contrastive.models.contrastive_learner_fusion import \
    ContrastiveLearnerFusion

from contrastive.utils.logs import set_root_logger_level, set_file_logger
log = set_file_logger(__file__)


# Auxilary function used to process the config linked to the model.
# For instance, change the embeddings save path to being next to the model.
def preprocess_config(sub_dir, datasets, label):
    """Loads the associated config of the given model and changes what has to be done,
    mainly the datasets and a few other keywords.
    
    Arguments:
        - sub_dir: str. Path to the directory containing the saved model.
        - datasets: list of str. List of the datasets to be used for the results generation.
        
    Output:
        - cfg: the config as an omegaconf object."""

    log.debug(os.getcwd())
    cfg = omegaconf.OmegaConf.load(sub_dir+'/.hydra/config.yaml')

    # replace the datasets in place
    change_config_datasets(cfg, datasets)
    change_config_label(cfg, label)

    # replace the possibly incorrect config parameters
    cfg.with_labels = True
    cfg.apply_augmentations = False

    return cfg


def save_outputs_as_csv(outputs, filenames, labels, csv_path=None, verbose=False):
    columns_names = ['dim'+str(i+1) for i in range(outputs.shape[1])]
    values = pd.DataFrame(outputs.numpy(), columns=columns_names)
    labels = pd.DataFrame(labels, columns=['labels']).astype(int)
    filenames = pd.DataFrame(filenames, columns=['ID'])
    df_outputs = pd.concat([labels, values, filenames], axis=1)
    
    # remove one copy each ID
    df_outputs = df_outputs.groupby('ID').mean()
    df_outputs.labels = df_outputs.labels.astype(int)

    if verbose:
        print("outputs:", df_outputs.iloc[:10, :])
        print("nb of elements:", df_outputs.shape[0])

    # Solves the case in which index type is tensor
    if len(df_outputs.index) > 0:  # avoid cases where empty df
        if type(df_outputs.index[0]) != str:
            index = [idx.item() for idx in df_outputs.index]
            index_name = df_outputs.index.name
            df_outputs.index = index
            df_outputs.index.names = [index_name]

    if csv_path:
        df_outputs.to_csv(csv_path)
    else:
        return df_outputs


def compute_embeddings(config, model_path, folder_name=None, use_best_model=False):
    """Compute the embeddings (= output of the backbone(s)) for a given model. 
    It relies on the hydra config framework, especially the backbone, datasets 
    and model parts.
    
    It saves csv files for each subset of the datasets (train, val, test_intra, 
    test) and one with all subjects."""
    
    config = process_config(config)

    # create new models in mode visualisation
    data_module = DataModule_Evaluation(config)
    data_module.setup(stage='validate')

    # create a new instance of the current model version,
    # then load hydra weights.
    print("No trained_model.pt saved. Create a new instance and load weights.")

    model = ContrastiveLearnerFusion(config, sample_data=data_module)
    # fetch and load weights
    paths = model_path+"/logs/*/version_0/checkpoints"+r'/*.ckpt'
    if use_best_model:
        paths = model_path+"/logs/best_model_weights.pt"
    files = glob.glob(paths)
    print("model_weights:", files[0])
    cpkt_path = files[0]
    checkpoint = torch.load(
        cpkt_path, map_location=torch.device(config.device))
    model.load_state_dict(checkpoint['state_dict'])

    model.to(torch.device(config.device))
    model.eval()

    # create a save path is necessary
    save_path = model_path+f"/{folder_name}_outputs"
    log.debug(f"Save path = {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # get the data and compute auc
        # train and val auc
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # test_intra
    try:
        test_intra_loader = data_module.test_intra_dataloader()
        test_intra_out, test_intra_filenames, test_intra_labels = \
            model.compute_outputs_skeletons(test_intra_loader)
        full_save_path = save_path + '/test_intra_outputs.csv'
        save_outputs_as_csv(test_intra_out, test_intra_filenames,
                            test_intra_labels, full_save_path)
    except:
        log.info("No test intra for this dataset.")
    
    # train-val-test outputs
    loaders = {'train': train_loader,
               'val': val_loader,
               'test': test_loader}
    for subset in loaders.keys():
        loader = loaders[subset]
        outputs, filenames, labels = model.compute_outputs_skeletons(loader)
        full_save_path = save_path + f'/{subset}_outputs.csv'
        save_outputs_as_csv(outputs, filenames, labels, full_save_path)

    # save what are the datasets have been used for the performance computation
    datasets = config.dataset.keys()
    save_used_datasets(save_path, datasets)
    save_used_label(save_path, config)


def get_outputs(model_path, datasets, label, short_name, use_best_model):
    """Get the output of a target model for all subsets of a the given dataset.
    
    Arguments:
        - model_path: str. Path to the model folder.
        - datasets: list of str. Name of the datasets used as inputs. 
        Most likely, they are the same than the one used for training.
        - short_name: str. A shorter name to give to the folder where the otputs are saved.
        - use_best_model: bool. Choose which weights to use to generate the outputs."""
    # get the config
    cfg = preprocess_config(model_path, datasets, label)
    log.debug(f"CONFIG FILE {type(cfg)}")
    log.debug(json.dumps(omegaconf.OmegaConf.to_container(
        cfg, resolve=True), indent=4, sort_keys=True))
    # save the modified config next to the real one
    with open(model_path+'/.hydra/config_evaluation.yaml', 'w') \
            as file:
        yaml.dump(omegaconf.OmegaConf.to_yaml(cfg), file)

    folder_name = get_save_folder_name(datasets, short_name)
    compute_embeddings(cfg, os.path.abspath(model_path),
                       folder_name=folder_name,
                       use_best_model=use_best_model)


get_outputs(model_path='/neurospin/dico/agaudin/Runs/09_new_repo/Output/2023-06-20/15-42-18_0',
            datasets=["cingulate_schiz_strat_bis"],
            label='diagnosis',
            short_name='cing_schiz_strat_bis',
            use_best_model=True)