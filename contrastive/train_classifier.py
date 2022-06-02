import hydra
import torch
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import json

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from contrastive.models.binary_classifier import BinaryClassifier 

from contrastive.data.utils import read_labels

from contrastive.utils.config import process_config
from contrastive.utils.logs import set_root_logger_level


def load_and_format_embeddings(dir_path, labels_path, config):
    train_embeddings = pd.read_csv(dir_path+'train_embeddings.csv', index_col=0)
    val_embeddings = pd.read_csv(dir_path+'val_embeddings.csv', index_col=0)
    test_embeddings = pd.read_csv(dir_path+'test_embeddings.csv', index_col=0)

    n_train = train_embeddings.shape[0]
    n_val = val_embeddings.shape[0]
    n_test = test_embeddings.shape[0]

    # regroup them in one dataframe (discuss with Joël)
    embeddings = pd.concat([train_embeddings, val_embeddings, test_embeddings],
                           axis=0, ignore_index=False)
    embeddings.sort_values(by='ID', inplace=True)
    print("sorted embeddings:", embeddings.head())

    # get the labels (0 = no paracingulate, 1 = paracingulate)
    # /!\ use read_labels
    labels = read_labels(labels_path, config.subject_column_name, config.label_names)
    labels = pd.DataFrame(labels.values, columns=['ID', 'label'])
    labels.sort_values(by='ID', inplace=True, ignore_index=True)
    print("sorted labels", labels.head())
    # supposed to contain one column 'ID' and one column 'label' with the actual labels
    # /!\ multiple labels is not handled

    # cast the dataset to the torch format
    X =  torch.from_numpy(embeddings.loc[:, embeddings.columns != 'ID'].values).type(torch.FloatTensor)
    Y = torch.from_numpy(labels.label.values.astype('float32')).type(torch.FloatTensor)

    return X, Y, n_train, n_val, n_test


@hydra.main(config_name='config', config_path="configs")
def train_classifier(config):
    config = process_config(config)

    set_root_logger_level(config.verbose)

    # set up load and save paths
    train_embs_path = config.training_embeddings
    train_lab_paths = config.training_labels
    # if not specified, the embeddings the results are created from are the ones used for training

    EoI_path = config.embeddings_of_interest if config.embeddings_of_interest else train_embs_path
    LoI_path = config.labels_of_interest if config.labels_of_interest else train_lab_paths

    # if not specified, the outputs of the classifier will be stored next to the embeddings
    # used to generate them
    results_save_path = config.results_save_path if config.results_save_path else EoI_path
    
    
    # import the embeddings (supposed to be already computed)
    X,Y,n_train,n_val,_ = load_and_format_embeddings(train_embs_path, train_lab_paths, config)


    # create and train the model
    hidden_layers = list(config.classifier_hidden_layers)
    layers_shapes = [config.num_representation_features]+hidden_layers+[1]
    bin_class = BinaryClassifier(layers_shapes,
                                 activation=config.classifier_activation,
                                 loss=config.classifier_loss)

    print("model", bin_class)
    #print("layers", bin_class.layers)
    #print("loss", bin_class.loss)
    #print("activation", bin_class.activation)

    class_train_set = TensorDataset(X, Y)
    train_loader = DataLoader(class_train_set, batch_size=10)

    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(model=bin_class, train_dataloaders=train_loader)


    # load new embeddings and labels if needed
    if (EoI_path == train_embs_path) and (LoI_path == train_lab_paths):
        pass
    else:
        # load embeddings of interest
        X,Y,n_train,n_val,_ = load_and_format_embeddings(train_embs_path, train_lab_paths, config)

    labels_pred = bin_class.forward(X)
    labels_pred = labels_pred.detach().numpy()


    # plot ROC curve
    labels_true = Y.detach_().numpy()
    curves = roc_curve(labels_true, labels_pred)
    roc_auc = roc_auc_score(labels_true, labels_pred)

    plt.plot(curves[0], curves[1], label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0,1],[0,1],color='r')
    plt.legend()
    plt.savefig(results_save_path+"ROC_curve.png")
    plt.show()


    # choose labels predicted with frontier = 0.5
    labels_pred = (labels_pred >= 0.5).astype('int')
     # separate train, val and test predictions and true values (if necessary)
    labels_pred_train = labels_pred[:n_train]
    labels_pred_val = labels_pred[n_train:n_train+n_val]
    labels_pred_test = labels_pred[n_train+n_val:]
    # if not already separated:
    labels_train = labels_true[:n_train]
    labels_val = labels_true[n_train:n_train+n_val]
    labels_test = labels_true[n_train+n_val:]

    # compute accuracies
    accuracy = accuracy_score(labels_true, labels_pred)
    accuracy_train = accuracy_score(labels_train, labels_pred_train)
    accuracy_val = accuracy_score(labels_val, labels_pred_val)
    accuracy_test = accuracy_score(labels_test, labels_pred_test)

    accuracies = {'total': accuracy,
                  'train': accuracy_train,
                  'val': accuracy_val,
                  'test': accuracy_test,
                  'auc': roc_auc
                  }

    print(accuracies)  # find another way to get them than printing
    with open(results_save_path+"accuracies.json", 'w') as file:
        json.dump(accuracies, file)



if __name__ == "__main__":
    train_classifier()