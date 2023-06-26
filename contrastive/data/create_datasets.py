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

import pandas as pd
import numpy as np

# only if foldlabel == True
try:
    from deep_folding.brainvisa.utils.save_data import quality_checks
    from deep_folding.brainvisa.utils.save_data import compare_array_aims_files
except ImportError:
    print("INFO: you cannot use deep_folding in brainvisa. Probably OK.")

from contrastive.utils.logs import set_file_logger, set_root_logger_level

from contrastive.data.datasets import ContrastiveDatasetFusion

from contrastive.data.utils import \
    check_subject_consistency, extract_data, check_if_same_subjects,\
    check_if_same_shape, check_if_skeleton, extract_data_with_labels,\
    read_labels

import logging

log = set_file_logger(__file__)
root = logging.getLogger()


def sanity_checks_without_labels(config, skeleton_output, reg):
    # Loads and separates in train_val/test set foldlabels if requested
    check_subject_consistency(config.data[reg].subjects_all,
                              config.data[reg].subjects_foldlabel_all)
    # in order to avoid logging twice the same information
    if root.level == 20:  # root logger in INFO mode
        set_root_logger_level(0)
    # add all the other created objects in the next line
    foldlabel_output = extract_data(config.data[reg].foldlabel_all,
                                    config.data[reg].crop_dir,
                                    config, reg)
    if root.level == 10:  # root logger in WARNING mode
        set_root_logger_level(1)
    log.info("foldlabel data loaded")

    # Makes some sanity checks
    for subset_name in foldlabel_output.keys():
        log.debug("skeleton", skeleton_output[subset_name][1].shape)
        log.debug("foldlabel", foldlabel_output[subset_name][1].shape)
        check_if_same_subjects(skeleton_output[subset_name][0],
                               foldlabel_output[subset_name][0],
                               subset_name)
        check_if_same_shape(skeleton_output[subset_name][1],
                            foldlabel_output[subset_name][1],
                            subset_name)

    return foldlabel_output


def create_sets_without_labels(config):
    """Creates train, validation and test sets

    Args:
        config (Omegaconf dict): contains configuration parameters
    Returns:
        train_dataset, val_dataset, test_datasetset, train_val_dataset (tuple)
    """

    skeleton_all = []
    foldlabel_all = []
    
    # checks consistency among regions
    if len(config.data) > 1:
        for reg in range(len(config.data)-1):
            check_if_same_csv(config.data[0].subjects_all,
                              config.data[reg+1].subjects_all,
                              "subjects_all")
            if 'train_val_csv_file' in config.data[0].keys():
                check_if_same_csv(config.data[0].train_val_csv_file,
                                config.data[reg+1].train_val_csv_file,
                                "train_csv")
            else:
                check_if_same_csv(config.data[0].train_csv_file,
                                config.data[reg+1].train_csv_file,
                                "train_csv")
            check_if_numpy_same_length(config.data[0].numpy_all,
                                       config.data[1].numpy_all,
                                       "numpy_all")
            if config.foldlabel:
                check_if_numpy_same_length(config.data[0].foldlabel_all,
                                           config.data[1].foldlabel_all,
                                           "foldlabel_all")

    for reg in range(len(config.data)):
        # Loads and separates in train_val/test skeleton crops
        skeleton_output = extract_data(
            config.data[reg].numpy_all,
            config.data[reg].crop_dir, config, reg)
        skeleton_all.append(skeleton_output)

        # Loads and separates in train_val/test set foldlabels if requested
        if config.apply_augmentations and config.foldlabel:
            foldlabel_output = sanity_checks_without_labels(config,
                                                            skeleton_output,
                                                            reg)
        else:
            foldlabel_output = None
            log.info("foldlabel data NOT requested. Foldlabel data NOT loaded")
        
        foldlabel_all.append(foldlabel_output)
            

    # Creates the dataset from these data by doing some preprocessing
    datasets = {}
    for subset_name in skeleton_all[0].keys():
        log.debug(subset_name)
        # Concatenates filenames
        filenames = [skeleton_output[subset_name][0]
                     for skeleton_output in skeleton_all]
        # Concatenates arrays
        arrays = [skeleton_output[subset_name][1]
                  for skeleton_output in skeleton_all]

        # Concatenates foldabel arrays
        foldlabel_arrays = []
        for foldlabel_output in foldlabel_all:
            # select the augmentation method
            if config.apply_augmentations:
                if config.foldlabel:  # branch_clipping
                    foldlabel_array = foldlabel_output[subset_name][1]
                else:  # cutout
                    foldlabel_array = None  # no need of fold labels
            else:  # no augmentation
                foldlabel_array = None
            foldlabel_arrays.append(foldlabel_array)

        # Checks if equality of filenames and labels
        check_if_list_of_equal_dataframes(
            filenames,
            "filenames, " + subset_name)

        datasets[subset_name] = ContrastiveDatasetFusion(
            filenames=filenames,
            arrays=arrays,
            foldlabel_arrays=foldlabel_arrays,
            config=config,
            apply_transform=config.apply_augmentations)

    return datasets


def sanity_checks_with_labels(config, skeleton_output, subject_labels, reg):
    """Checks alignment of the generated objects."""
    # remove test_intra if not in config
    subsets = [key for key in skeleton_output.keys()]
    if 'test_intra_csv_file' not in config.keys():
        subsets.pop(3)
    log.debug(f"SANITY CHECKS {subsets}")

    for subset_name in subsets:
        check_if_skeleton(skeleton_output[subset_name][1], subset_name)

    if config.environment == "brainvisa" and config.checking:
        for subset_name in subsets:
            compare_array_aims_files(skeleton_output[subset_name][0],
                                     skeleton_output[subset_name][1],
                                     config.data[reg].crop_dir)

    # Makes some sanity checks on ordering of label subjects
    for subset_name in subsets:
        check_if_same_subjects(skeleton_output[subset_name][0][['Subject']],
                               skeleton_output[subset_name][2][['Subject']],
                               f"{subset_name} labels")

    # Loads and separates in train_val/test set foldlabels if requested
    if (
        ('foldlabel' in config.keys())
        and (config.foldlabel)
        and (config.mode != 'evaluation')
    ):
        check_subject_consistency(config.data[reg].subjects_all,
                                  config.data[reg].subjects_foldlabel_all)
        # in order to avoid logging twice the same information
        if root.level == 20:  # root logger in INFO mode
            set_root_logger_level(0)
        foldlabel_output = extract_data_with_labels(
            config.data[reg].foldlabel_all,
            subject_labels,
            config.data[reg].foldlabel_dir,
            config, reg)
        if root.level == 10:  # root logger in WARNING mode
            set_root_logger_level(1)
        log.info("foldlabel data loaded")

        # Makes some sanity checks
        for subset_name in subsets:
            check_if_same_subjects(skeleton_output[subset_name][0],
                                   foldlabel_output[subset_name][0],
                                   subset_name)
            check_if_same_shape(skeleton_output[subset_name][1],
                                foldlabel_output[subset_name][1],
                                subset_name)
            check_if_same_subjects(
                foldlabel_output[subset_name][0],
                skeleton_output[subset_name][2][['Subject']],
                f"{subset_name} labels")
            check_if_same_subjects(
                foldlabel_output[subset_name][2][['Subject']],
                skeleton_output[subset_name][2][['Subject']],
                f"{subset_name} labels")

        if config.environment == "brainvisa" and config.checking:
            for subset_name in foldlabel_output.keys():
                compare_array_aims_files(foldlabel_output[subset_name][0],
                                         foldlabel_output[subset_name][1],
                                         config.data[reg].foldlabel_dir)

    else:
        log.info("foldlabel data NOT requested. Foldlabel data NOT loaded")
        return None

    return foldlabel_output


def check_if_list_of_equal_dataframes(list_of_df, key):
    """Checks if it is a list of equal dataframes"""
    if len(list_of_df) > 1:
        df0 = list_of_df[0]
        for df in list_of_df[1:]:
            if not df0.equals(df):
                raise ValueError(
                    f"List of dataframes are not equal: {key}"
                    "First dataframe head:\n"
                    f"{df0.head()}\n"
                    "Other dataframe head:\n"
                    f"{df.head()}\n"    
                    f"length of first dataframe = {len(df0)}\n"
                    f"length of other dataframe = {len(df)}"               
                    )


def check_if_same_csv(csv_file_1, csv_file_2, key, header='infer'):
    """Checks if the two csv are identical"""
    csv1 = pd.read_csv(csv_file_1, header=header)
    csv2 = pd.read_csv(csv_file_2, header=header)
    if not csv1.equals(csv2):
        raise ValueError(
            f"Input {key} csv files are not equal"
            "First dataframe head:\n"
            f"{csv1.head()}\n"
            "Other dataframe head:\n"
            f"{csv2.head()}\n"
            f"length of first dataframe ({csv_file_1}) = {len(csv1)}\n"
            f"length of other dataframe ({csv_file_2}) = {len(csv2)}"
        )


def check_if_numpy_same_length(npy_file_1, npy_file_2, key):
    """Checks if the two numpy arrays have the same length"""
    arr1 = np.load(npy_file_1)
    arr2 = np.load(npy_file_2)
    if len(arr1) != len(arr2):
        raise ValueError(
            f"Input {key} numpy files don't have the same length"
        )


def create_sets_with_labels(config):
    """Creates train, validation and test sets when there are labels

    Args:
        config (Omegaconf dict): contains configuration parameters
        reg: region number
    Returns:
        train_dataset, val_dataset, test_datasetset, train_val_dataset (tuple)
    """

    skeleton_all = []
    foldlabel_all = []
    
    # checks consistency among regions
    if len(config.data) > 1:
        for reg in range(len(config.data)-1):
            check_if_same_csv(config.data[0].subject_labels_file,
                              config.data[reg+1].subject_labels_file,
                              "subject_labels")         
            check_if_same_csv(config.data[0].subjects_all,
                              config.data[reg+1].subjects_all,
                              "subjects_all")
            if 'train_val_csv_file' in config.data[0].keys():
                check_if_same_csv(config.data[0].train_val_csv_file,
                                config.data[reg+1].train_val_csv_file,
                                "train_val_csv", header=None)
            check_if_numpy_same_length(config.data[0].numpy_all,
                                       config.data[1].numpy_all,
                                       "numpy_all")
            if config.foldlabel:
                check_if_numpy_same_length(config.data[0].foldlabel_all,
                                           config.data[1].foldlabel_all,
                                           "foldlabel_all")

    for reg in range(len(config.data)):
        # Gets labels for all subjects
        # Column subject_column_name is renamed 'Subject'
        label_scaling = (None if 'label_scaling' not in config.keys()
                         else config.data[reg].label_scaling)
        #retrocompatibility 
        label_names = config.label_names if 'label_names' in config else config.data[0].label_names
        subject_labels = read_labels(
            config.data[reg].subject_labels_file,
            config.data[reg].subject_column_name,
            label_names,
            label_scaling)

        if config.environment == "brainvisa" and config.checking:
            quality_checks(config.data[reg].subjects_all,
                           config.data[reg].numpy_all,
                           config.data[reg].crop_dir, parallel=True)

        # Loads and separates in train_val/test skeleton crops
        skeleton_output = extract_data_with_labels(
            config.data[reg].numpy_all, subject_labels,
            config.data[reg].crop_dir, config, reg)

        foldlabel_output = sanity_checks_with_labels(
            config, skeleton_output, subject_labels, reg)

        skeleton_all.append(skeleton_output)
        foldlabel_all.append(foldlabel_output)

    # Creates the dataset from these data by doing some preprocessing
    datasets = {}
    for subset_name in skeleton_all[0].keys():
        log.debug(subset_name)
        # Concatenates filenames
        filenames = [skeleton_output[subset_name][0]
                     for skeleton_output in skeleton_all]
        # Concatenates arrays
        arrays = [skeleton_output[subset_name][1]
                  for skeleton_output in skeleton_all]

        # Concatenates foldabel arrays
        foldlabel_arrays = []
        for foldlabel_output in foldlabel_all:
            # select the augmentation method
            if config.apply_augmentations:
                if config.foldlabel:  # branch_clipping
                    foldlabel_array = foldlabel_output[subset_name][1]
                else:  # cutout
                    foldlabel_array = None  # no need of fold labels
            else:  # no augmentation
                foldlabel_array = None
            foldlabel_arrays.append(foldlabel_array)

        # Concatenates labels
        labels = [skeleton_output[subset_name][2]
                  for skeleton_output in skeleton_all]

        # Checks if equality of filenames and labels
        check_if_list_of_equal_dataframes(
            filenames,
            "filenames, " + subset_name)
        check_if_list_of_equal_dataframes(
            labels,
            "labels, " + subset_name)

        # Builds subset-name=train/val/test dataset
        datasets[subset_name] = ContrastiveDatasetFusion(
            filenames=filenames,
            arrays=arrays,
            foldlabel_arrays=foldlabel_arrays,
            labels=labels,
            config=config,
            apply_transform=config.apply_augmentations)

    return datasets
