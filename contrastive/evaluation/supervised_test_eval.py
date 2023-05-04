import os
import yaml
import json
import hydra
import omegaconf
import glob
import torch

from sklearn.metrics import roc_auc_score

from contrastive.utils.config import process_config
from contrastive.utils.logs import set_root_logger_level, set_file_logger
from contrastive.models.contrastive_learner_with_labels import ContrastiveLearner_WithLabels
from contrastive.models.contrastive_learner_visualization import ContrastiveLearner_Visualization
from contrastive.data.datamodule import DataModule_Evaluation


log = set_file_logger(__file__)


def checks_before_compute(sub_dir, dataset, overwrite=False):
    config_path = sub_dir+'/.hydra/config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.BaseLoader)

    # check if the model is a classifier
    if config['mode'] != 'classifier':
        print(f"{sub_dir} is not a classifier. Continue.")
        return False
    
    # check if test values are already saved
    if os.path.exists(sub_dir + f"/{dataset}_results") and (not overwrite):
        print("Model already treated (existing folder with results). Set \
overwrite to True if you still want to compute them.")
        return False
        
    # check if the model is not excluded by hand
    if '#' in sub_dir:
        print("Model with an incompatible structure with the current one. Pass.")
        return False
    
    return True


# Auxilary function used to process the config linked to the model.
# For instance, change the embeddings save path to being next to the model.
def preprocess_config(sub_dir, dataset):
    
    log.debug(os.getcwd())
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


    # replace the possibly incorrect config parameters
    cfg.with_labels = True
    cfg.apply_augmentations = False

    return cfg


def compute_test_auc(model, dataloader):
    # get the model's output
    Y_pred, labels, filenames = model.compute_outputs_skeletons(dataloader)
    log.debug(f"prediction shape {Y_pred.shape}")
    log.debug(f"prediction = {Y_pred[:10]}")
    log.debug(f"filenames {filenames[:10]}")
    log.debug(f"labels {labels[:10]}")
    # take only one view (normally both are the same) 
    Y_pred = Y_pred[::2,:]
    filenames = filenames[::2]
    labels = labels[::2]

    # apply softmax (applied in the loss during training)
    Y_pred = torch.nn.functional.softmax(Y_pred, dim=1)

    # compute auc
    test_auc = roc_auc_score(labels, Y_pred[:,1])

    return test_auc


def supervised_test_eval(config, model_path, use_best_model=True):

    config = process_config(config)

    log.debug(config.mode)
    log.debug(config.with_labels)
    log.debug(config.apply_augmentations)

    # create new models in mode visualisation
    set_root_logger_level(0)
    data_module = DataModule_Evaluation(config)
    data_module.setup()
    set_root_logger_level(1)

    # create a new instance of the current model version then load hydra weights.
    log.info("No trained_model.pt saved. Create a new instance and load weights.")

    model = ContrastiveLearner_WithLabels(config, sample_data=data_module)
    # fetch and load weights
    paths = model_path+"/logs/*/version_0/checkpoints"+r'/*.ckpt'
    if use_best_model:
        paths = model_path+"/logs/best_model_weights.pt"
    files = glob.glob(paths)
    log.debug("model_weights:", files[0])
    cpkt_path = files[0]
    model.load_pretrained_model(cpkt_path, encoder_only=False)

    model.eval()

    test_loader = data_module.test_dataloader()
    try:
        test_intra_loader = data_module.test_intra_dataloader()
        test_intra_auc = compute_test_auc(model, test_intra_loader)
    except:
        log.info("No test intra for this dataset.")

    test_auc = compute_test_auc(model, test_loader)
    log.info(f"test_auc = {test_auc}")

    # create a save path is necessary
    save_path = model_path+f"/{config.dataset_name}_results"
    log.debug(f"Save path = {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # save the results
    results_dico = {'test_auc': test_auc}
    # if test_intra has been computed
    if 'test_intra_auc' in locals():
        results_dico['test_intra_auc'] = test_intra_auc

    with open(save_path+'/test_results.json', 'w') as file:
        json.dump(results_dico, file)


def pipeline(dir_path, dataset, overwrite=False, use_best_model=True):
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
                cont_bool = checks_before_compute(sub_dir, dataset, overwrite=overwrite)
                if cont_bool:
                    print("Start post processing")
                    # get the config and correct it to suit what is needed for classifiers
                    cfg = preprocess_config(sub_dir, dataset)
                    log.debug(f"CONFIG FILE {type(cfg)}")
                    log.debug(json.dumps(omegaconf.OmegaConf.to_container(cfg, resolve=True), indent=4, sort_keys=True))
                    # save the modified config next to the real one
                    with open(sub_dir+'/.hydra/config_evaluation.yaml', 'w') as file:
                        yaml.dump(omegaconf.OmegaConf.to_yaml(cfg), file)
                    
                    supervised_test_eval(cfg, os.path.abspath(sub_dir),
                                         use_best_model=use_best_model)


            else:
                print(f"{sub_dir} not associated to a model. Continue")
                pipeline(sub_dir, dataset, overwrite=False, test_intra=False, use_best_model=True)
        else:
            print(f"{sub_dir} is a file. Continue.")


pipeline("/neurospin/dico/agaudin/Runs/09_new_repo/Output/supervised/ACCpatterns/L",
         dataset="cingulate_ACCpatterns_left", overwrite=True, use_best_model=True)