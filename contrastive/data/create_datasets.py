#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  This software and supporting documentation are distributed by
#      Institut Federatif de Recherche 49
#      CEA/NeuroSpin, Batiment 145,
#      91191 Gif-sur-Yvette cedex
#      France
#
# This software is governed by the CeCILL license version 2 under
# French law and abiding by the rules of distribution of free software.
# You can  use, modify and/or redistribute the software under the
# terms of the CeCILL license version 2 as circulated by CEA, CNRS
# and INRIA at the following URL "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license version 2 and that you accept its terms.
"""
Tools to create datasets
"""
from contrastive.utils.logs import set_file_logger

from contrastive.data.datasets import ContrastiveDataset
from contrastive.data.datasets import ContrastiveDataset_Visualization
from contrastive.data.datasets import ContrastiveDataset_WithLabels
from contrastive.data.datasets import ContrastiveDataset_WithFoldLabels
from contrastive.data.datasets import \
    ContrastiveDataset_WithLabels_WithFoldLabels

from contrastive.data.utils import *

log = set_file_logger(__file__)


def create_sets_without_labels(config):
    """Creates train, validation and test sets

    Args:
        config (Omegaconf dict): contains configuration parameters
    Returns:
        train_dataset, val_dataset, test_datasetset, train_val_dataset (tuple)
    """

    # Loads and separates in train_val/test skeleton crops
    train_val_subjects, train_val_data, test_subjects, test_data = \
        extract_data(config.numpy_all, config)

    # Loads and separates in train_val/test set foldlabels if requested
    if config.foldlabel == True:
        train_val_foldlabel_subjects, train_val_foldlabel_data, \
            test_foldlabel_subjects, test_foldlabel_data = \
            extract_data(config.foldlabel_all, config)
        log.info("foldlabel data loaded")

        # Makes some sanity checks
        check_if_same_subjects(train_val_subjects,
                               train_val_foldlabel_subjects, "train_val")
        check_if_same_subjects(test_subjects,
                               test_foldlabel_subjects, "test")
        check_if_same_shape(train_val_data,
                            train_val_foldlabel_data, "train_val")
        check_if_same_shape(test_data,
                            test_foldlabel_data, "test")
    else:
        log.info("foldlabel data NOT requested. Foldlabel data NOT loaded")

    # Creates the dataset from these data by doing some preprocessing
    if config.mode == 'evaluation':
        test_dataset = ContrastiveDataset_Visualization(
            filenames=test_subjects,
            array=test_data,
            config=config)
        train_val_dataset = ContrastiveDataset_Visualization(
            filenames=train_val_subjects,
            array=train_val_data,
            config=config)
    else:
        if config.foldlabel == True:
            test_dataset = ContrastiveDataset_WithFoldLabels(
                filenames=test_subjects,
                array=test_data,
                foldlabel_array=test_foldlabel_data,
                config=config)
            train_val_dataset = ContrastiveDataset_WithFoldLabels(
                filenames=train_val_subjects,
                array=train_val_data,
                foldlabel_array=train_val_foldlabel_data,
                config=config)
        else:
            test_dataset = ContrastiveDataset(
                filenames=test_subjects,
                array=test_data,
                config=config)
            train_val_dataset = ContrastiveDataset(
                filenames=train_val_subjects,
                array=train_val_data,
                config=config)

    train_dataset, val_dataset = \
        extract_train_val_dataset(train_val_dataset,
                                  config.partition,
                                  config.seed)

    return train_dataset, val_dataset, test_dataset, train_val_dataset


def create_sets_with_labels(config):
    """Creates train, validation and test sets when there are labels

    Args:
        config (Omegaconf dict): contains configuration parameters
    Returns:
        train_dataset, val_dataset, test_datasetset, train_val_dataset (tuple)
    """

    # Gets labels for all subjects
    # Column subject_column_name is renamed 'Subject'
    subject_labels = read_labels(config.subject_labels_file,
                                 config.subject_column_name,
                                 config.label_names)

    # Loads and separates in train_val/test skeleton crops
    train_val_subjects, train_val_data, train_val_labels,\
    test_subjects, test_data, test_labels = \
        extract_data_with_labels(config.numpy_all, subject_labels, config)

    # Makes some sanity checks on ordering of label subjects
    check_if_same_subjects(train_val_subjects,
                           train_val_labels[['Subject']], "train_val labels")
    check_if_same_subjects(test_subjects,
                           test_labels[['Subject']], "test labels")

    # Loads and separates in train_val/test set foldlabels if requested
    if config.foldlabel == True:
        train_val_foldlabel_subjects, train_val_foldlabel_data, \
        train_val_labels, test_foldlabel_subjects, \
        test_foldlabel_data, test_labels = \
            extract_data_with_labels(config.foldlabel_all, subject_labels,
                                     config)
        log.info("foldlabel data loaded")

        # Makes some sanity checks
        check_if_same_subjects(train_val_subjects,
                               train_val_foldlabel_subjects, "train_val")
        check_if_same_subjects(test_subjects,
                               test_foldlabel_subjects, "test")
        check_if_same_shape(train_val_data,
                            train_val_foldlabel_data, "train_val")
        check_if_same_shape(test_data,
                            test_foldlabel_data, "test")
    else:
        log.info("foldlabel data NOT requested. Foldlabel data NOT loaded")

    # Creates the dataset from these data by doing some preprocessing
    if config.mode == 'evaluation':
        test_dataset = ContrastiveDataset_Visualization(
            filenames=test_subjects,
            array=test_data,
            config=config)
        train_val_dataset = ContrastiveDataset_Visualization(
            filenames=train_val_subjects,
            array=train_val_data,
            config=config)
    else:
        if config.foldlabel == True:
            test_dataset = ContrastiveDataset_WithLabels_WithFoldLabels(
                filenames=test_subjects,
                array=test_data,
                labels=test_labels,
                foldlabel_array=test_foldlabel_data,
                config=config)
            train_val_dataset = ContrastiveDataset_WithLabels_WithFoldLabels(
                filenames=train_val_subjects,
                array=train_val_data,
                labels=train_val_labels,
                foldlabel_array=train_val_foldlabel_data,
                config=config)
        else:
            test_dataset = ContrastiveDataset_WithLabels(
                filenames=test_subjects,
                array=test_data,
                labels=test_labels,
                config=config)
            train_val_dataset = ContrastiveDataset_WithLabels(
                filenames=train_val_subjects,
                array=train_val_data,
                labels=train_val_labels,
                config=config)

    train_dataset, val_dataset = \
        extract_train_val_dataset(train_val_dataset,
                                  config.partition,
                                  config.seed)

    return train_dataset, val_dataset, test_dataset, train_val_dataset
