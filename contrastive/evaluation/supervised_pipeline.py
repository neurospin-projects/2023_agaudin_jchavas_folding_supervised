import os
import glob
import yaml
import json
import omegaconf
import torch

from contrastive.data.datamodule import DataModule_Evaluation
from contrastive.models.contrastive_learner_fusion import \
    ContrastiveLearnerFusion
from contrastive.utils.config import process_config
from contrastive.utils.logs import set_root_logger_level, set_file_logger

from utils_pipelines import get_save_folder_name, change_config_datasets, \
    change_config_label, save_used_datasets, save_used_label

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


def supervised_auc_eval(config, model_path, folder_name=None, use_best_model=True):
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
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # test_intra
    try:
        test_intra_loader = data_module.test_intra_dataloader()
        test_intra_auc = model.compute_output_auc(test_intra_loader)
        log.info(f"test_intra_auc = {test_intra_auc}")
    except:
        log.info("No test intra for this dataset.")

    # train-val-test aucs
    train_auc = model.compute_output_auc(train_loader)
    val_auc = model.compute_output_auc(val_loader)
    test_auc = model.compute_output_auc(test_loader)

    # create a save path is necessary
    save_path = model_path+f"/{folder_name}_supervised_results"
    log.debug(f"Save path = {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save the results
    results_dico = {'train_auc': train_auc,
                    'val_auc': val_auc,
                    'test_auc': test_auc}
    log.info(f"aucs = {results_dico}")
    # if test_intra has been computed
    if 'test_intra_auc' in locals():
        results_dico['test_intra_auc'] = test_intra_auc

    if use_best_model:
        json_path = save_path+'/aucs_best_model.json'
    else:
        json_path = save_path+'/aucs.json'
    with open(json_path, 'w') as file:
        json.dump(results_dico, file)

    # save what are the datasets have been used for the performance computation
    datasets = config.dataset.keys()
    save_used_datasets(save_path, datasets)
    save_used_label(save_path, config)


def pipeline(dir_path, datasets, label, short_name=None, overwrite=False, use_best_model=True):
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
                    log.debug(f"CONFIG FILE {type(cfg)}")
                    log.debug(json.dumps(omegaconf.OmegaConf.to_container(
                        cfg, resolve=True), indent=4, sort_keys=True))
                    # save the modified config next to the real one
                    with open(sub_dir+'/.hydra/config_evaluation.yaml', 'w') \
                            as file:
                        yaml.dump(omegaconf.OmegaConf.to_yaml(cfg), file)

                    folder_name = get_save_folder_name(datasets, short_name)
                    supervised_auc_eval(cfg, os.path.abspath(sub_dir),
                                         folder_name=folder_name, use_best_model=False)
                    if use_best_model:  # do both
                        log.info("Repeat with the best model")
                        cfg = preprocess_config(sub_dir, datasets, label)
                        supervised_auc_eval(cfg, os.path.abspath(sub_dir),
                                         folder_name=folder_name, use_best_model=True)

            else:
                print(f"{sub_dir} not associated to a model. Continue")
                pipeline(sub_dir, datasets, label, short_name=short_name,
                         overwrite=overwrite, use_best_model=use_best_model)
        else:
            print(f"{sub_dir} is a file. Continue.")


pipeline("/volatile2/jc225751/Runs/61_classifier_regresser/Program/Output/2023-09-08_finetuning_after_HCP",
         datasets=["cingulate_HCP_stratified_extreme_Flanker_left",
                   "cingulate_HCP_stratified_extreme_Flanker_right"],
         label='Flanker_AgeAdj_class',
         short_name='flanker_class', overwrite=True, use_best_model=True)
