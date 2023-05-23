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

# only if foldlabel == True
try:
    from deep_folding.brainvisa.utils.save_data import quality_checks
    from deep_folding.brainvisa.utils.save_data import compare_array_aims_files
except ImportError:
    print("INFO: you cannot use deep_folding in brainvisa. Probably OK.")

from contrastive.utils.logs import set_file_logger, set_root_logger_level

from contrastive.data.datasets_copy import ContrastiveDatasetFusion

from contrastive.data.utils import *

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
                                    config.data[reg].crop_dir, config)
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


def create_sets_without_labels(config, reg=0):
    """Creates train, validation and test sets

    Args:
        config (Omegaconf dict): contains configuration parameters
    Returns:
        train_dataset, val_dataset, test_datasetset, train_val_dataset (tuple)
    """

    # Loads and separates in train_val/test skeleton crops
    skeleton_output = extract_data(config.data[reg].numpy_all,
                                   config.data[reg].crop_dir, config)

    # Loads and separates in train_val/test set foldlabels if requested
    if config.apply_augmentations and config.foldlabel:
        foldlabel_output = sanity_checks_without_labels(config,
                                                        skeleton_output)
    else:
        log.info("foldlabel data NOT requested. Foldlabel data NOT loaded")

    # Creates the dataset from these data by doing some preprocessing
    datasets = {}
    for subset_name in skeleton_output.keys():
        # select the augmentation method
        if config.apply_augmentations:
            if config.foldlabel:  # branch_clipping
                foldlabel_array = foldlabel_output[subset_name][1]
            else:  # cutout
                foldlabel_array = None  # no nedd of fold labels
        else:  # no augmentation
            foldlabel_array = None

        datasets[subset_name] = ContrastiveDatasetFusion(
            filenames=skeleton_output[subset_name][0],
            array=skeleton_output[subset_name][1],
            foldlabel_array=foldlabel_array,
            config=config,
            apply_transform=config.apply_augmentations)

    # # just to have the same data format as train and val
    # test_dataset, _ = torch.utils.data.random_split(
    #     test_dataset,
    #     [len(test_dataset),0])

    return datasets


def sanity_checks_with_labels(config, skeleton_output, subject_labels, reg=0):
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
        check_if_same_subjects(skeleton_output[subset_name][0],
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
            config)
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


def create_sets_with_labels(config):
    """Creates train, validation and test sets when there are labels

    Args:
        config (Omegaconf dict): contains configuration parameters
        reg: region number
    Returns:
        train_dataset, val_dataset, test_datasetset, train_val_dataset (tuple)
    """

    for reg in range(len(config.data)):
        # Gets labels for all subjects
        # Column subject_column_name is renamed 'Subject'
        label_scaling = (None if 'label_scaling' not in config.data[reg].keys()
                         else config.data[reg].label_scaling)
        subject_labels = read_labels(
            config.data[reg].subject_labels_file,
            config.data[reg].subject_column_name,
            config.data[reg].label_names,
            label_scaling)

        if config.environment == "brainvisa" and config.checking:
            quality_checks(config.data[reg].subjects_all,
                           config.data[reg].numpy_all,
                           config.data[reg].crop_dir, parallel=True)

        # Loads and separates in train_val/test skeleton crops
        skeleton_output = extract_data_with_labels(
            config.data[reg].numpy_all, subject_labels,
            config.data[reg].crop_dir, config)

        foldlabel_output = sanity_checks_with_labels(
            config, skeleton_output, subject_labels, reg)

    # Creates the dataset from these data by doing some preprocessing
    datasets = {}
    for subset_name in skeleton_output.keys():
        # select the augmentation method
        if config.apply_augmentations:
            if config.foldlabel:  # branch_clipping
                foldlabel_array = foldlabel_output[subset_name][1]
            else:  # cutout
                foldlabel_array = None  # no need of fold labels
        else:  # no augmentation
            foldlabel_array = None

        log.debug(subset_name)
        datasets[subset_name] = ContrastiveDatasetFusion(
            filenames=skeleton_output[subset_name][0],
            array=skeleton_output[subset_name][1],
            foldlabel_array=foldlabel_array,
            labels=skeleton_output[subset_name][2],
            config=config,
            apply_transform=config.apply_augmentations)

    return datasets
