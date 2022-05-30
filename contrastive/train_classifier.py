import hydra
import torch
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, roc_auc_score
from contrastive.models.binary_classifier import BinaryClassifier 

from contrastive.utils.config import process_config
from contrastive.utils.logs import set_root_logger_level


@hydra.main(config_name='config', config_path="configs")
def train_classifier(config):
    config = process_config(config)

    set_root_logger_level(config.verbose)

    
    # import the embeddings (supposed to be already computed)
    #dir_path = config.embeddings_path
    dir_path = "/neurospin/dico/agaudin/Runs/02_explicabilite_humains_2022/\
Output/2022-05-18/11-00-10/"  # should be passed as an argument/ in config
    train_embeddings = pd.read_csv(dir_path+'train_embeddings.csv', index_col=0)
    val_embeddings = pd.read_csv(dir_path+'val_embeddings.csv', index_col=0)
    #test_embeddings = pd.read_csv(dir_path+'test_embeddings.csv', index_col=0)
    # regroup them in one dataframe (discuss with Joël)
    embeddings = pd.concat([train_embeddings, val_embeddings],#, test_embeddings],
                           axis=0, ignore_index=True)
    embeddings.sort_values(by='ID', inplace=True)

    # get the labels (0 = no paracingulate, 1 = paracingulate)
    labels = pd.read_csv(config.labels_paracingulate)
    labels.sort_values(by='ID', inplace=True)
    # supposed to contain one column 'ID' and one column 'label' with the actual labels

    # cast the dataset to the torch format
    X =  torch.from_numpy(embeddings.loc[:, embeddings.columns != 'ID'].values).type(torch.FloatTensor)
    Y = torch.from_numpy(labels.label.values).type(torch.FloatTensor)

    # create and train the model
    bin_class = BinaryClassifier(config.num_representation_features, 1, 
                                 activation=config.classifier_activation,
                                 loss=config.classifier_loss)

    print(bin_class)

    train_set = TensorDataset(X, Y)
    train_loader = DataLoader(train_set, batch_size=10)

    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(model=bin_class, train_dataloaders=train_loader)

    # compute predictions
    labels_pred = bin_class.forward(X)
    labels_pred = labels_pred.detach().numpy()

    # plot ROC curve
    curves = roc_curve(Y.detach_().numpy(), labels_pred)
    roc_auc = roc_auc_score(Y.detach_().numpy(), labels_pred)

    plt.plot(curves[0], curves[1], label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0,1],[0,1],color='r')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train_classifier()