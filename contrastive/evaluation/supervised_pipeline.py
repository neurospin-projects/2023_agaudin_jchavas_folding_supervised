import os
import glob
import yaml
import json
import omegaconf
import torch
import pickle
import pandas as pd

from contrastive.data.datamodule import DataModule_Evaluation
from contrastive.evaluation.grad_cam import compute_all_grad_cams
from contrastive.models.contrastive_learner_fusion import \
    ContrastiveLearnerFusion
from contrastive.utils.config import process_config
from contrastive.utils.logs import set_root_logger_level, set_file_logger

from utils_pipelines import *

log = set_file_logger(__file__)


def checks_before_compute(sub_dir, datasets, short_name, overwrite=False,
                          use_best_model=True):
    """Checks if a model has to be treated or not"""

    config_path = sub_dir+'/.hydra/config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.BaseLoader)

    # check if the model is a classifier
    if config['mode'] != 'classifier':
        print(f"{sub_dir} is not a classifier. Continue.")
        return False

    # check if test values are already saved
    folder_name = get_save_folder_name(datasets, short_name)
    if os.path.exists(sub_dir + f"/{folder_name}_supervised_results") and (not overwrite):
        print("Model already treated (existing folder with results). Set \
overwrite to True if you still want to evaluate it.")
        return False

    # check if the model is not excluded by hand
    if '#' in sub_dir:
        print("Model with an incompatible structure with the current one. "
              "Pass.")
        return False

    # check if the best_model has been saved
    if use_best_model:
        if not os.path.exists(os.path.abspath(sub_dir)
                              + "/logs/best_model_weights.pt"):
            print("The best model weigths have not been saved for this model. "
                  "Set the keyword use_best_model to False "
                  "if you want to evaluate it with its lasts weights.")
            return False

    return True


# Auxilary function used to process the config linked to the model.
# For instance, change the embeddings save path to being next to the model.
def preprocess_config(sub_dir, datasets, label):
    """Loads the associated config of the given model and changes what has to be done,
    mainly the datasets and a few other keywords.
    
    Arguments:
        - sub_dir: str. Path to the directory containing the saved model.
        - datasets: list of str. List of the datasets to be used for the results generation.
        - label: str. Name of the chosen label yaml file.
        
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


def supervised_auc_eval(config, model_path, folder_name=None, use_best_model=True,
                        save_outputs=False):
    """Actually computes the test, train and val (and test_intra if existing) auc 
    of a target model."""

    config = process_config(config)

    log.debug(config.mode)
    log.debug(config.with_labels)
    log.debug(config.apply_augmentations)

    # create new models in mode visualisation
    set_root_logger_level(0)
    data_module = DataModule_Evaluation(config)
    data_module.setup()
    set_root_logger_level(1)

    # create a new instance of the current model version
    # then load hydra weights.
    log.info("No trained_model.pt saved. "
             "Create a new instance and load weights.")

    model = ContrastiveLearnerFusion(config, sample_data=data_module)
    # fetch and load weights
    paths = model_path+"/logs/*/version_0/checkpoints"+r'/*.ckpt'
    if use_best_model:
        paths = model_path+"/logs/best_model_weights.pt"
    files = glob.glob(paths)
    log.debug("model_weights:", files[0])
    cpkt_path = files[0]
    model.load_pretrained_model(cpkt_path, encoder_only=False)

    model.to(torch.device(config.device))
    model.eval()

    # get the data and compute auc
        # train and val auc
    loaders_dict = {}
    loaders_dict['train'] = data_module.train_dataloader()
    loaders_dict['val'] = data_module.val_dataloader()
    loaders_dict['test'] = data_module.test_dataloader()

    # test_intra
    try:
        test_intra_loader = data_module.test_intra_dataloader()
        loaders_dict['test_intra'] = test_intra_loader
    except:
        log.info("No test intra for this dataset.")
    
    # create a save path is necessary
    save_path = model_path+f"/{folder_name}_supervised_results"
    log.debug(f"Save path = {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # compute aucs
    aucs_dict = {}
    for subset_name, loader in loaders_dict.items():
        aucs_dict[subset_name+"_auc"] = model.compute_output_auc(loader)
    log.info(aucs_dict)

    # save the aucs
    if use_best_model:
        json_path = save_path+'/aucs_best_model.json'
    else:
        json_path = save_path+'/aucs.json'
    with open(json_path, 'w') as file:
        json.dump(aucs_dict, file)

    # compute and save grad cam if required
    if len(config.data) == 1:
        attributions_dict = compute_all_grad_cams(loaders_dict, model,
                                                  with_labels=config.with_labels)
        if use_best_model:
            filename = '/attributions_best_model.pkl'
        else:
            filename = '/attributions.pkl'
        with open(save_path+filename, 'wb') as f:
            pickle.dump(attributions_dict, f)

    # compute and save outputs if required
    if save_outputs:
        log.info("Generate the outputs of the model.")
        # set the save path
        if use_best_model:
            outputs_save_path = os.path.join(save_path, 'best_model_outputs')
        else:
            outputs_save_path = os.path.join(save_path, 'outputs')
        if not os.path.exists(outputs_save_path):
            log.info("Creating the save path for the outputs...")
            os.makedirs(outputs_save_path)
        
        full_csv = pd.DataFrame([])
        # compute the output for each subset
        for subset in loaders_dict.keys():
            loader = loaders_dict[subset]
            outputs, filenames, labels = model.compute_outputs_skeletons(loader)
            full_save_path = outputs_save_path + f'/{subset}_outputs.csv'
            subset_csv = save_outputs_as_csv(outputs, filenames, labels, full_save_path)
            full_csv = pd.concat([full_csv, subset_csv], axis=0)
        full_csv.to_csv(outputs_save_path + '/full_outputs.csv')
        log.info("Outputs saved")

    # save what are the datasets have been used for the performance computation
    datasets = config.dataset.keys()
    save_used_datasets(save_path, datasets)
    save_used_label(save_path, config)


def pipeline(dir_path, datasets, label, short_name=None, overwrite=False, use_best_model=True,
             save_outputs=False):
    """Pipeline to generate automatically the output aucs for supervised classifiers only.

    Arguments:
        - dir_path: str. Path where the models are stored and where is applied 
        recursively the process.
        - datasets: list of str. Datasets the results are generated from. /!\ Only 
        uses the test (and test_intra) subsets.
        - label: str. Label chosen for the auc computation.
        - short_name: str or None. Name of the directory where to store both embeddings 
        and aucs. If None, use datasets to generate the folder name.
        - overwrite: bool. Redo the process on models where embeddings already exist.
        - use_best_model: bool. Use the best model saved during to generate embeddings. 
        The 'normal' model is always used, the best is only added.
        - save_outputs: bool. Save the outputs (i.e. after the projection head) in the
        save folder.
    """
    # walks recursively through the subfolders
    for name in os.listdir(dir_path):
        sub_dir = dir_path + '/' + name

        # checks if directory
        if os.path.isdir(sub_dir):
            # check if directory associated to a model
            if os.path.exists(sub_dir+'/.hydra/config.yaml'):
                print("\n")
                print(f"Treating {sub_dir}")
                # checks to know if the model should be treated
                cont_bool = checks_before_compute(
                    sub_dir, datasets, overwrite=overwrite,
                    short_name=short_name, use_best_model=use_best_model)
                if cont_bool:
                    print("Start post processing")
                    # get the config
                    # and correct it to suit what is needed for classifiers
                    cfg = preprocess_config(sub_dir, datasets, label)
                    # save the modified config next to the real one
                    with open(sub_dir+'/.hydra/config_evaluation.yaml', 'w') \
                            as file:
                        yaml.dump(omegaconf.OmegaConf.to_yaml(cfg), file)

                    folder_name = get_save_folder_name(datasets, short_name)
                    supervised_auc_eval(cfg, os.path.abspath(sub_dir),
                                         folder_name=folder_name, use_best_model=False,
                                         save_outputs=save_outputs)
                    if use_best_model:  # do both
                        log.info("Repeat with the best model")
                        cfg = preprocess_config(sub_dir, datasets, label)
                        supervised_auc_eval(cfg, os.path.abspath(sub_dir),
                                         folder_name=folder_name, use_best_model=True,
                                         save_outputs=save_outputs)

            else:
                print(f"{sub_dir} not associated to a model. Continue")
                pipeline(sub_dir, datasets, label, short_name=short_name,
                         overwrite=overwrite, use_best_model=use_best_model,
                         save_outputs=save_outputs)
        else:
            print(f"{sub_dir} is a file. Continue.")

if __name__ == "__main__":
    pipeline("/neurospin/dico/agaudin/Runs/09_new_repo/Output/grid_searches/step3/occipital/densenet",    
            datasets=["occipital_schiz_R_strat_bis", 'occipital_schiz_L_strat_bis'],
            label='diagnosis', short_name='schiz_diag', overwrite=True, use_best_model=True,
            save_outputs=True)
    
    # regions = os.listdir("/neurospin/dico/agaudin/Runs/09_new_repo/Output/grid_searches/step2")
    # lesconnasses1sur3 = ['occipito_temporal', 'fissure_parieto_occipital', 'inferior_temporal', 'precentral',
    #          'FIP']
    # lesconnasses2sur3 = ['postcentral', 'pericalcarine', 'SFintermediate', 'STs', 'fissure_lateral']
    # lesconnasses3sur3 = ['fissure_collateral',
    #          'SC_sylv', 'SFmedian', 'BROCA', 'lobule_parietal_sup']
    # for region in lesconnasses3sur3:
    #     pipeline(f"/neurospin/dico/agaudin/Runs/09_new_repo/Output/grid_searches/step2/{region}",
    #             datasets=[f"{region}_schiz_R_strat_bis", f'{region}_schiz_L_strat_bis'],
    #             label='diagnosis', short_name='schiz_diag', overwrite=True, use_best_model=True,
    #             save_outputs=True)
