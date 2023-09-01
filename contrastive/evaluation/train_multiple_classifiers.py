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
from sklearn.model_selection import cross_val_predict, train_test_split, permutation_test_score, StratifiedKFold

from pqdm.processes import pqdm
from joblib import cpu_count
from functools import partial

from sklearn.preprocessing import StandardScaler
from contrastive.models.binary_classifier import BinaryClassifier
from sklearn.svm import LinearSVR, SVC

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


def load_embeddings(dir_path, labels_path, config):
    """Load the embeddings and the labels.
    """
    # load embeddings
    # if targeting directly the target csv file
    if not os.path.isdir(dir_path):
        embeddings = pd.read_csv(dir_path, index_col=0)
    # if only giving the directory (implies constraints on the file name)
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

            # regroup them in one dataframe (discuss with Joël)
            embeddings = pd.concat([train_embeddings,
                                    val_embeddings,
                                    test_embeddings],
                                   axis=0, ignore_index=False)

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


# used for torch neural network (special formatting required)
def load_and_format_embeddings(dir_path, labels_path, config):
    # load embeddings
    embeddings, labels = load_embeddings(dir_path, labels_path, config)
    names_col = 'ID' if 'ID' in embeddings.columns else 'Subject'

    # create train-test datasets
    if config.classifier_test_size:
        embeddings_train, embeddings_test, labels_train, labels_test = \
            train_test_split(embeddings, labels,
                             test_size=config.classifier_test_size,
                             random_state=config.classifier_seed)
    else:  # no train-test sets for the classifier
        embeddings_train = embeddings_test = embeddings
        labels_train = labels_test = labels

    # cast the dataset to the torch format
    X_train = embeddings_train.loc[:, embeddings_train.columns != names_col]
    X_train = torch.from_numpy(X_train.values).type(torch.FloatTensor)

    X_test = embeddings_test.loc[:, embeddings_test.columns != names_col]
    X_test = torch.from_numpy(X_test.values).type(torch.FloatTensor)

    Y_train = torch.from_numpy(labels_train.label.values.astype(
        'float32')).type(torch.FloatTensor)
    Y_test = torch.from_numpy(labels_test.label.values.astype(
        'float32')).type(torch.FloatTensor)

    return X_train, X_test, Y_train, Y_test, labels_train, labels_test


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
                            values, columns_names, mode, results_save_path,
                            pvalues_auc=None, perm_scores_mean=None,
                            perm_scores_std=None, perm_scores_95=None):
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
    plt.title(f"{mode} ROC curves")
    plt.savefig(results_save_path+f"/{mode}_ROC_curves.png")

    # compute accuracy and area under the curve
    print(f"{mode} accuracy",
          np.mean(accuracies[mode]),
          np.std(accuracies[mode]))
    print(f"{mode} AUC", np.mean(aucs[mode]), np.std(aucs[mode]))

    values[f'{mode}_total_accuracy'] = \
        [np.mean(accuracies[mode]), np.std(accuracies[mode])]
    values[f'{mode}_auc'] = [np.mean(aucs[mode]), np.std(aucs[mode])]

    # save predicted labels
    labels.to_csv(results_save_path+f"/{mode}_predicted_probas.csv",
                  index=False)
    # DEBUG embeddings.to_csv(results_save_path+f"/effective_embeddings.csv",
    #                         index=True)

    # add permutation auc / pvalue in values if True in config
    if pvalues_auc!=None:
        # for now added only in cross_val mode
        if mode in pvalues_auc.keys():
            values[f'{mode}_permutations'] = \
                [np.mean(pvalues_auc[mode]), np.mean(perm_scores_mean[mode]), np.mean(perm_scores_std[mode]), np.mean(perm_scores_95[mode])]

def train_nn_classifiers(config):
    """Sets up the save paths, loads the embeddings and then loops 
    to train the 250 neural networks.
    
    OBSOLETE"""
    # set up load and save paths
    train_embs_path = config.training_embeddings
    train_lab_paths = config.training_labels

    # if not specified, the embeddings are the ones used for training

    EoI_path = (config.embeddings_of_interest if config.embeddings_of_interest
                else train_embs_path)
    LoI_path = (config.labels_of_interest if config.labels_of_interest
                else train_lab_paths)

    # if not specified, the outputs of the classifier will be stored next
    # to the embeddings used to generate them
    results_save_path = (config.results_save_path if config.results_save_path
                         else EoI_path)
    if not os.path.isdir(results_save_path):
        results_save_path = os.path.dirname(results_save_path)

    # import the embeddings (supposed to be already computed)
    X_train, X_test, Y_train, Y_test, labels_train, labels_test = \
        load_and_format_embeddings(train_embs_path, train_lab_paths, config)

    # create objects that will be filled during the loop
    train_prediction_matrix = np.zeros(
        (labels_train.shape[0], config.n_repeat))
    test_prediction_matrix = np.zeros((labels_test.shape[0], config.n_repeat))

    Curves = {'train': [],
              'test': []}
    aucs = {'train': [],
            'test': []}
    accuracies = {'train': [],
                  'test': []}

    # loop to train the classifiers
    for i in range(config.n_repeat):
        print("model number", i)

        if i == 0:
            hidden_layers = list(config.classifier_hidden_layers)
            layers_shapes = [np.shape(X_train)[1]]+hidden_layers+[1]

        bin_class = BinaryClassifier(layers_shapes,
                                     activation=config.classifier_activation,
                                     loss=config.classifier_loss)

        if i == 0:
            print("model", bin_class)

        class_train_set = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(
            class_train_set, batch_size=config.class_batch_size)

        trainer = pl.Trainer(max_epochs=config.class_max_epochs,
                             logger=False, enable_checkpointing=False)
        trainer.fit(model=bin_class, train_dataloaders=train_loader)

        # load new embeddings and labels if needed
        if (EoI_path == train_embs_path) and (LoI_path == train_lab_paths):
            pass
        else:
            pass

        # predict labels with the classifier (both for train and test sets)
        labels_pred_train = bin_class.forward(X_train).detach().numpy()
        labels_pred_test = bin_class.forward(X_test).detach().numpy()
        # save the predicted labels
        train_prediction_matrix[:, i] = labels_pred_train.flatten()
        test_prediction_matrix[:, i] = labels_pred_test.flatten()

        # compute indicators for train
        curves, roc_auc, accuracy = compute_indicators(
            Y_train, labels_pred_train)
        Curves['train'].append(curves)
        aucs['train'].append(roc_auc)
        accuracies['train'].append(accuracy)

        # compute indicators for test
        curves, roc_auc, accuracy = compute_indicators(
            Y_test, labels_pred_test)
        Curves['test'].append(curves)
        aucs['test'].append(roc_auc)
        accuracies['test'].append(accuracy)

        # plot the histogram of predicted values
        """with_paracingulate = labels[labels.label == 1]
        without_paracingulate = labels[labels.label == 0]

        print(with_paracingulate.shape[0], "-", without_paracingulate.shape[0])

        x_min = min(0, np.min(labels.predicted))
        x_max = max(1, np.max(labels.predicted))

        plt.figure()
        plt.hist(without_paracingulate.predicted,
                 bins=np.arange(x_min,x_max,0.01), alpha=0.6)
        plt.hist(with_paracingulate.predicted,
                 bins=np.arange(x_min,x_max,0.01), alpha=0.6, color='r')
        plt.legend(['without_paracingulate', "with_paracingulate"])

        ax = plt.gca()
        plt.vlines([0.5], ax.get_ylim()[0], ax.get_ylim()[1], color='black')

        plt.savefig(results_save_path+"/prediction_histogram.png")"""

    # add the predictions to the df where the true values are
    columns_names = ["predicted_"+str(i) for i in range(config.n_repeat)]
    train_preds = pd.DataFrame(
        train_prediction_matrix, columns=columns_names,
        index=labels_train.index)
    labels_train = pd.concat([labels_train, train_preds], axis=1)

    test_preds = pd.DataFrame(test_prediction_matrix,
                              columns=columns_names, index=labels_test.index)
    labels_test = pd.concat([labels_test, test_preds], axis=1)

    # evaluation of the aggregation of the models
    values = {}

    for mode in ['train', 'test']:
        if mode == 'train':
            labels = labels_train
        elif mode == 'test':
            labels = labels_test

        post_processing_results(
            labels, Curves, aucs, accuracies, values,
            columns_names, mode, results_save_path)
        # values is changed in place

    with open(results_save_path+"/values.json", 'w+') as file:
        json.dump(values, file)

    plt.close('all')


def train_one_svm_classifier(config, inputs, i=0):
    """Trains one SVC.

    Args:
        - config: config file
        - inputs: dictionary containing the input data,
        with X key containing embeddings
        and Y key labels. If a test set is defined,
        IT also contains X and Y for the test set.
        - i: seed for the SVM.
        Is automatically changed in each call of train_svm_classifiers.
    """

    X = inputs['X']
    Y = inputs['Y']
    test_embs_path = inputs['test_embs_path']
    if test_embs_path:
        X_test = inputs['X_test']
        Y_test = inputs['Y_test']
    outputs = {}

    # SVC predict_proba
    model = SVC(C=0.01, kernel='linear', probability=True, class_weight='balanced',
                max_iter=config.class_max_epochs, random_state=i)
    cv = StratifiedKFold(5)
    labels_proba = cross_val_predict(model, X, Y, cv=cv, method='predict_proba')
    curves, roc_auc, accuracy = compute_indicators(Y, labels_proba)
    outputs['proba_of_1'] = labels_proba[:, 1]

    if config.permutations:
        _, perm_score, pvalue = permutation_test_score(model, X, Y, scoring='roc_auc', cv=cv, n_permutations=200, random_state=i)
        perm_score_mean, perm_score_std, perm_score_95 = np.mean(perm_score), np.std(perm_score), np.quantile(perm_score, 0.95)

    # SVR
    #model = LinearSVR(C=0.01, max_iter=1000, random_state=i)
    #labels_pred = cross_val_predict(model, X, Y, cv=5)
    #curves, roc_auc, accuracy = compute_indicators(Y, labels_pred)
    # WRONG LABELING HERE, MODIFY CODE TO CHANGE THIS MISLEADING NAME
    #outputs['proba_of_1'] = labels_pred

    # Stores in outputs dict

    outputs['curves'] = curves
    outputs['roc_auc'] = roc_auc
    outputs['accuracy'] = accuracy
    if config.permutations:
        outputs['pvalue_auc'] = pvalue
        outputs['perm_score_mean'] = perm_score_mean
        outputs['perm_score_std'] = perm_score_std
        outputs['perm_score_95'] = perm_score_95

    if test_embs_path:
        labels_pred_test = model.predict(X_test)
        curves, roc_auc, accuracy = compute_indicators(
            Y_test, labels_pred_test)
        outputs['curves_test'] = curves
        outputs['roc_auc_test'] = roc_auc
        outputs['accuracy_test'] = accuracy

    return outputs


@ignore_warnings(category=ConvergenceWarning)
def train_svm_classifiers(config):
    """Sets up the save paths, loads the embeddings and then loops 
    to train the 250 SVMs."""
    # import the data

    # set up load and save paths
    train_embs_path = config.training_embeddings
    test_embs_path = config.test_embeddings
    # /!\ in fact all_labels (=train_val and test labels)
    train_lab_paths = config.data[0].subject_labels_file

    # if not specified, the embeddings the results are created from
    # are the ones used for training
    EoI_path = (config.embeddings_of_interest if config.embeddings_of_interest
                else train_embs_path)

    # if not specified, the outputs of the classifier will be stored next
    # to the embeddings used to generate them
    results_save_path = (config.results_save_path if config.results_save_path
                         else EoI_path)
    # remove the 'full_embeddings.csv if it is there
    if not os.path.isdir(results_save_path):
        results_save_path = os.path.dirname(results_save_path)
    # add a subfolder with the evaluated label as name
    results_save_path = results_save_path + "/" + config.label_names[0]
    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)

    embeddings, labels = load_embeddings(
        train_embs_path, train_lab_paths, config)
    names_col = 'ID' if 'ID' in embeddings.columns else 'Subject'
    X = embeddings.loc[:, embeddings.columns != names_col]

    Y = labels.label

    # debug regression
    # if np.unique(Y).shape[0] > 2:
    #     per_50 = np.percentile(Y, 50.)
    #     Z = Y.copy(deep=True)
    #     Z[Y <= per_50] = 0
    #     Z[Y > per_50] = 1
    #     Z = Z.astype(int)
    #     labels.label = Z.copy(deep=True)
    #     Y = labels.label

    if test_embs_path:
        test_embeddings, test_labels = load_embeddings(
            test_embs_path, train_lab_paths, config)
        names_col = 'ID' if 'ID' in test_embeddings.columns else 'Subject'
        X_test = test_embeddings.loc[:, test_embeddings.columns != names_col]
        Y_test = test_labels.label

    # Builds objects where the results are saved
    Curves = {'cross_val': []}
    aucs = {'cross_val': []}
    accuracies = {'cross_val': []}
    if config.permutations:
        pvalues_auc = {'cross_val': []}
        perm_scores_mean = {'cross_val': []}
        perm_scores_std = {'cross_val': []}
        perm_scores_95 = {'cross_val': []}
    proba_matrix = np.zeros((labels.shape[0], config.n_repeat))

    if test_embs_path:
        Curves['test'] = []
        aucs['test'] = []
        accuracies['test'] = []
        prediction_matrix_test = np.zeros(
            (test_labels.shape[0], config.n_repeat))

    # Configures loops

    repeats = range(config.n_repeat)

    inputs = {}
    inputs['X'] = X
    # rescale embeddings
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    inputs['Y'] = Y
    inputs['test_embs_path'] = test_embs_path
    if test_embs_path:
        inputs['X_test'] = X_test
        inputs['Y_test'] = Y_test

    # Actual loop done config.n_repeat times
    if _parallel:
        print(f"Computation done IN PARALLEL: {config.n_repeat} times")
        print(f"Number of subjects used by the SVM: {len(inputs['X'])}")
        func = partial(train_one_svm_classifier, config, inputs)
        outputs = pqdm(repeats, func, n_jobs=define_njobs())
    else:
        outputs = []
        print("Computation done SERIALLY")
        for i in repeats:
            print("model number", i)
            outputs.append(train_one_svm_classifier(config, inputs, i))

    # Put together the results
    for i, o in enumerate(outputs):
        probas_pred = o['proba_of_1']
        curves = o['curves']
        roc_auc = o['roc_auc']
        accuracy = o['accuracy']
        if config.permutations:
            pvalue_auc = o['pvalue_auc']
            perm_score_mean = o['perm_score_mean']
            perm_score_std = o['perm_score_std']
            perm_score_95 = o['perm_score_95']
        proba_matrix[:, i] = probas_pred
        Curves['cross_val'].append(curves)
        aucs['cross_val'].append(roc_auc)
        accuracies['cross_val'].append(accuracy)
        if config.permutations:
            pvalues_auc['cross_val'].append(pvalue_auc)
            perm_scores_mean['cross_val'].append(perm_score_mean)
            perm_scores_std['cross_val'].append(perm_score_std)
            perm_scores_95['cross_val'].append(perm_score_95)

        if test_embs_path:
            curves = o['curves_test']
            roc_auc = o['roc_auc_test']
            accuracy = o['accuracy_test']
            Curves['test'].append(curves)
            aucs['test'].append(roc_auc)
            accuracies['test'].append(accuracy)

    # add the predictions to the df where the true values are
    columns_names = ["svm_"+str(i) for i in range(config.n_repeat)]
    probas = pd.DataFrame(
        proba_matrix, columns=columns_names, index=labels.index)
    labels = pd.concat([labels, probas], axis=1)

    # post processing (mainly plotting graphs)
    values = {}
    mode = 'cross_val'
    if not config.permutations:
        pvalues_auc, perm_scores_mean, perm_scores_std, perm_scores_95 = None, None, None, None
    post_processing_results(labels, embeddings, Curves, aucs,
                            accuracies, values, columns_names,
                            mode, results_save_path, pvalues_auc,
                            perm_scores_mean, perm_scores_std, perm_scores_95)
    print(f"results_save_path = {results_save_path}")
    with open(results_save_path+"/values.json", 'w+') as file:
        json.dump(values, file)

    if test_embs_path:
        values = {}
        mode = 'test'
        post_processing_results(test_labels, test_embeddings, Curves, aucs,
                                accuracies, values, columns_names,
                                mode, results_save_path)

        with open(results_save_path+"/values_test.json", 'w+') as file:
            json.dump(values, file)

    # plt.show()
    plt.close('all')

    save_used_label(os.path.dirname(results_save_path), config)


@hydra.main(config_name='config_no_save', config_path="../configs")
def train_classifiers(config):
    """Train classifiers (either SVM or neural networks) to classify target embeddings
    with the given label.
    
    All the relevant information should be passed thanks to the input config.
    
    It saves txt files containg the acuuracies, the aucs and figures of the ROC curves."""

    config = process_config(config)

    set_root_logger_level(config.verbose)

    # runs the analysis with the chosen type of classifiers
    if config.classifier_name == 'neural_network':
        train_nn_classifiers(config)

    elif config.classifier_name == 'svm':
        train_svm_classifiers(config)

    else:
        raise ValueError(
            f"The classifer type {config.classifier_name} "
            "you are asking for is not implemented."
            "Please change the config.classifier used in the config file "
            "you are calling to solve the problem.")


if __name__ == "__main__":
    train_classifiers()
