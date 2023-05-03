import hydra
import glob
import torch

from sklearn.metrics import roc_auc_score

from contrastive.utils.config import process_config
from contrastive.utils.logs import set_root_logger_level, set_file_logger
from contrastive.models.contrastive_learner_with_labels import ContrastiveLearner_WithLabels
from contrastive.models.contrastive_learner_visualization import ContrastiveLearner_Visualization
from contrastive.data.datamodule import DataModule_Evaluation


log = set_file_logger(__file__)


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


@hydra.main(config_name='config_no_save', config_path="../configs")
def supervised_test_eval(config):

    config = process_config(config)

    config.use_best_model = True
    config.model_path = "/neurospin/dico/agaudin/Runs/09_new_repo/Output/supervised/ACCpatterns/R/09-14-24_0"
    config.mode = 'classifier'
    config.with_labels = True
    config.drop_rate = 0

    # create new models in mode visualisation
    set_root_logger_level(0)
    data_module = DataModule_Evaluation(config)
    data_module.setup()
    set_root_logger_level(1)

    # create a new instance of the current model version then load hydra weights.
    log.info("No trained_model.pt saved. Create a new instance and load weights.")

    model = ContrastiveLearner_WithLabels(config, sample_data=data_module)
    # fetch and load weights
    paths = config.model_path+"/logs/*/version_0/checkpoints"+r'/*.ckpt'
    if config.use_best_model:
        paths = config.model_path+"/logs/best_model_weights.pt"
    files = glob.glob(paths)
    log.debug("model_weights:", files[0])
    cpkt_path = files[0]
    model.load_pretrained_model(cpkt_path, encoder_only=False)

    model.eval()

    test_loader = data_module.test_dataloader()
    try:
        test_intra_loader = data_module.test_intra_dataloader()
        compute_test_auc(model, test_intra_loader)
    except:
        log.info("No test intra for this dataset.")

    test_auc = compute_test_auc(model, test_loader)
    log.info(f"test_auc = {test_auc}")

    return test_auc

if __name__ == "__main__":
    supervised_test_eval()