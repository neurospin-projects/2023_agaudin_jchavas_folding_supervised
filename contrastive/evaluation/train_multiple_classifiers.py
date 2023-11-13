import hydra
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import json
import os

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import auc, roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_predict, train_test_split

from pqdm.processes import pqdm
from joblib import cpu_count
from functools import partial

from sklearn.preprocessing import StandardScaler
# from contrastive.models.binary_classifier import BinaryClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from contrastive.data.utils import read_labels

from contrastive.utils.config import process_config
from contrastive.utils.logs import set_root_logger_level, set_file_logger
from contrastive.evaluation.utils_pipelines import save_used_label

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


_parallel = True

log = set_file_logger(__file__)


def define_njobs():
    """Returns number of cpus used by main loop
    """
    nb_cpus = cpu_count()
    return max(nb_cpus - 2, 1)


def load_embeddings(dir_path, labels_path, config, subset='full'):
    """Load the embeddings and the labels.

    Arguments:
        - dir_path: path where the embeddings are stored. Either 
        the folder that contains them or directly the target file.
        - labels_path: the file where the labels are stored.
        - config: the omegaconf object related to the current ANN model.
        - subset: str. Target subset of the data the classifiers will be trained 
        on. Usually either 'train', 'val', 'train_val', 'test' or 'test_intra'.
    """
    # load embeddings
    # if targeting directly the target csv file
    if not os.path.isdir(dir_path):
        embeddings = pd.read_csv(dir_path, index_col=0)
    # if only giving the directory (implies constraints on the file name)
    # take only a specified subset
    elif subset != 'full':
        embeddings = pd.read_csv(
                dir_path+f'/{subset}_embeddings.csv', index_col=0)
    # takes all the subjects
    else:
        if os.path.exists(dir_path+'/full_embeddings.csv'):
            embeddings = pd.read_csv(
                dir_path+'/full_embeddings.csv', index_col=0)
        elif os.path.exists(dir_path+'/pca_embeddings.csv'):
            embeddings = pd.read_csv(
                dir_path+'/pca_embeddings.csv', index_col=0)
        else:
            train_embeddings = pd.read_csv(
                dir_path+'/train_embeddings.csv', index_col=0)
            val_embeddings = pd.read_csv(
                dir_path+'/val_embeddings.csv', index_col=0)
            test_embeddings = pd.read_csv(
                dir_path+'/test_embeddings.csv', index_col=0)
            embs_list = [train_embeddings, val_embeddings,
                         test_embeddings]
            try:
                test_intra_embeddings = pd.read_csv(
                    dir_path+'/test_intra_embeddings.csv', index_col=0)
                embs_list.append(test_intra_embeddings)
            except:
                pass
                
            # regroup them in one dataframe
            embeddings = pd.concat(embs_list, axis=0, ignore_index=False)

    embeddings.sort_index(inplace=True)
    log.debug(f"sorted embeddings: {embeddings.head()}")

    # get the labels (0 = no paracingulate, 1 = paracingulate)
    # and match them to the embeddings
    # /!\ use read_labels
    label_scaling = (None if 'label_scaling' not in config.keys()
                     else config.label_scaling)
    labels = read_labels(labels_path, config.data[0].subject_column_name,
                         config.label_names, label_scaling)
    labels.rename(columns={config.label_names[0]: 'label'}, inplace=True)
    labels = labels[labels.Subject.isin(embeddings.index)]
    labels.sort_values(by='Subject', inplace=True, ignore_index=True)
    log.debug(f"sorted labels: {labels.head()}")

    embeddings = embeddings[embeddings.index.isin(labels.Subject)]
    embeddings.sort_index(inplace=True)
    log.debug(f"sorted embeddings: {embeddings.head()}")

    # /!\ multiple labels is not handled

    return embeddings, labels


def compute_indicators(Y, proba_pred):
    """Compute ROC curve and auc, and accuracy."""
    if type(Y) == torch.tensor:
        labels_true = Y.detach_().numpy()
    else:
        labels_true = Y.values.astype('float64')
    curves = roc_curve(labels_true, proba_pred[:, 1])
    roc_auc = roc_auc_score(labels_true, proba_pred[:, 1])

    # choose labels predicted with frontier = 0.5
    labels_pred = np.argmax(proba_pred, axis=1)
    # compute accuracy
    accuracy = accuracy_score(labels_true, labels_pred)
    return curves, roc_auc, accuracy


def compute_auc(column, label_col=None):
    log.debug("COMPUTE AUC")
    log.debug(label_col.head())
    log.debug(column.head())
    return roc_auc_score(label_col, column)


def get_average_model(labels_df):
    """Get a model with performance that is representative of the group, 
    i.e. the one with the median auc."""
    aucs = labels_df.apply(compute_auc, args=[labels_df.label])
    aucs = aucs[aucs.index != 'label']
    aucs = aucs[aucs == aucs.quantile(interpolation='nearest')]
    return (aucs.index[0])


def post_processing_results(labels, embeddings, Curves, aucs, accuracies,
                            values, columns_names, mode, subset, results_save_path):
    """Get the mean and the median AUC and accuracy, plot the ROC curves and 
    the generated files."""

    labels_true = labels.label.values.astype('float64')

    # compute agregated models
    predicted_labels = labels[columns_names]

    labels['median_pred'] = predicted_labels.median(axis=1)
    labels['mean_pred'] = predicted_labels.mean(axis=1)

    # plot ROC curves
    plt.figure()

    # ROC curves of all models
    for curves in Curves[mode]:
        plt.plot(curves[0], curves[1], color='grey', alpha=0.1)
    plt.plot([0, 1], [0, 1], color='r', linestyle='dashed')

    # get the average model (with AUC as a criteria)
    # /!\ This model is a classifier that exists in the pool
    # /!\ This model != 'mean_pred' or 'median_pred'
    average_model = get_average_model(
        labels[['label'] + columns_names].astype('float64'))
    labels['average_model'] = labels[average_model]
    roc_curve_average = roc_curve(labels_true, labels[average_model].values)
    # ROC curves of "special" models
    roc_curve_median = roc_curve(labels_true, labels.median_pred.values)
    roc_curve_mean = roc_curve(labels_true, labels.mean_pred.values)

    plt.plot(roc_curve_average[0], roc_curve_average[1],
             color='red', alpha=0.5, label='average model')
    plt.plot(roc_curve_median[0], roc_curve_median[1],
             color='blue', label='agregated model (median)')
    plt.plot(roc_curve_mean[0], roc_curve_mean[1],
             color='black', label='agregated model (mean)')
    plt.legend()
    plt.title(f"{subset} ROC curves")
    plt.savefig(results_save_path+f"/{subset}_ROC_curves.png")

    # compute accuracy and area under the curve
    print(f"{subset} cross_val accuracy",
          np.mean(accuracies[mode]),
          np.std(accuracies[mode]))
    print(f"{subset} cross_val AUC", np.mean(aucs[mode]), np.std(aucs[mode]))

    values[f'{subset}_total_accuracy'] = \
        [np.mean(accuracies[mode]), np.std(accuracies[mode])]
    values[f'{subset}_auc'] = [np.mean(aucs[mode]), np.std(aucs[mode])]

    # save predicted labels
    labels.to_csv(results_save_path+f"/{subset}_predicted_probas.csv",
                  index=False)
    # DEBUG embeddings.to_csv(results_save_path+f"/effective_embeddings.csv",
    #                         index=True)


def train_one_classifier(config, inputs, i=0):
    """Trains one classifier, whose type is set in config_no_save.

    Args:
        - config: config file
        - inputs: dictionary containing the input data,
        with X key containing embeddings
        and Y key labels. If a test set is defined,
        it also contains X and Y for the test set.
        - i: seed for the SVM.
        Is automatically changed in each call of train_svm_classifiers.
    """

    X = inputs['X']
    Y = inputs['Y']
    outputs = {}

    # choose the classifier type
    # /!\ The chosen classifier must have a predict_proba method.
    if config.classifier_name == 'svm':
        model = SVC(kernel='linear', probability=True,
                    max_iter=config.class_max_epochs, random_state=i)
    elif config.classifier_name == 'neural_network':
        model = MLPClassifier(hidden_layer_sizes=config.classifier_hidden_layers,
                              activation=config.classifier_activation,
                              batch_size=config.class_batch_size,
                              max_iter=config.class_max_epochs, random_state=i)
    else:
        raise ValueError(f"The chosen classifier ({config.classifier_name}) is not handled by the pipeline. \
Choose a classifier type that exists in configs/classifier.")
    
    # SVC predict_proba
    labels_proba = cross_val_predict(model, X, Y, cv=5, method='predict_proba')
    curves, roc_auc, accuracy = compute_indicators(Y, labels_proba)
    outputs['proba_of_1'] = labels_proba[:, 1]

    # Stores in outputs dict
    outputs['curves'] = curves
    outputs['roc_auc'] = roc_auc
    outputs['accuracy'] = accuracy

    return outputs


@ignore_warnings(category=ConvergenceWarning)
def train_n_repeat_classifiers(config, subset='full'):
    """Sets up the save paths, loads the embeddings and then loops 
    to train the n_repeat (=250) classifiers."""
    ## import the data

    # set up load and save paths
    train_embs_path = config.training_embeddings
    # /!\ in fact all_labels (=train_val and test labels)
    train_lab_paths = config.data[0].subject_labels_file

    # if not specified, the outputs of the classifier will be stored next
    # to the embeddings used to generate them
    results_save_path = (config.results_save_path if config.results_save_path
                         else train_embs_path)

    # remove the filename from the path if it is a file
    if not os.path.isdir(results_save_path):
        results_save_folder, _ = os.path.split(results_save_path)
    else:
        results_save_folder, _ = results_save_path, ''
    # add a subfolder with the evaluated label as name
    results_save_folder = results_save_folder + "/" + config.label_names[0]
    if not os.path.exists(results_save_folder):
        os.makedirs(results_save_folder)

    embeddings, labels = load_embeddings(
        train_embs_path, train_lab_paths, config, subset=subset)
    names_col = 'ID' if 'ID' in embeddings.columns else 'Subject'
    X = embeddings.loc[:, embeddings.columns != names_col]
    Y = labels.label

    # Builds objects where the results are saved
    Curves = {'cross_val': []}
    aucs = {'cross_val': []}
    accuracies = {'cross_val': []}
    proba_matrix = np.zeros((labels.shape[0], config.n_repeat))

    # Configures loops

    repeats = range(config.n_repeat)

    inputs = {}
    inputs['X'] = X
    # rescale embeddings
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    inputs['Y'] = Y

    # Actual loop done config.n_repeat times
    if _parallel:
        print(f"Computation done IN PARALLEL: {config.n_repeat} times")
        print(f"Number of subjects used by the SVM: {len(inputs['X'])}")
        func = partial(train_one_classifier, config, inputs)
        outputs = pqdm(repeats, func, n_jobs=define_njobs())
    else:
        outputs = []
        print("Computation done SERIALLY")
        for i in repeats:
            print("model number", i)
            outputs.append(train_one_classifier(config, inputs, i))

    # Put together the results
    for i, o in enumerate(outputs):
        probas_pred = o['proba_of_1']
        curves = o['curves']
        roc_auc = o['roc_auc']
        accuracy = o['accuracy']
        proba_matrix[:, i] = probas_pred
        Curves['cross_val'].append(curves)
        aucs['cross_val'].append(roc_auc)
        accuracies['cross_val'].append(accuracy)

    # add the predictions to the df where the true values are
    columns_names = ["svm_"+str(i) for i in range(config.n_repeat)]
    probas = pd.DataFrame(
        proba_matrix, columns=columns_names, index=labels.index)
    labels = pd.concat([labels, probas], axis=1)

    # post processing (mainly plotting graphs)
    values = {}
    mode = 'cross_val'
    post_processing_results(labels, embeddings, Curves, aucs,
                            accuracies, values, columns_names,
                            mode, subset, results_save_folder)
    
    # save results
    print(f"results_save_path = {results_save_folder}")
    filename = f"{subset}_values.json"
    with open(os.path.join(results_save_folder, filename), 'w+') as file:
        json.dump(values, file)

    # plt.show()
    plt.close('all')

    save_used_label(os.path.dirname(results_save_folder), config)


#@hydra.main(config_name='config_no_save', config_path="../configs")
def train_classifiers(config, subsets=None):
    """Train classifiers (either SVM or neural networks) to classify target embeddings
    with the given label.
    
    All the relevant information should be passed thanks to the input config.
    
    It saves txt files containg the acuuracies, the aucs and figures of the ROC curves."""

    config = process_config(config)

    set_root_logger_level(config.verbose)

    for subset in subsets:
        print("\n")
        log.info(f"USING SUBSET {subset}")
        # the choice of the classifiers' type is now inside the function 
        train_n_repeat_classifiers(config, subset=subset)


if __name__ == "__main__":
    train_classifiers()
