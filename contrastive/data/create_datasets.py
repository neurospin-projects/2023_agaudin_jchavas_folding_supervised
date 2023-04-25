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

from contrastive.utils.logs import set_file_logger

from contrastive.data.datasets import ContrastiveDataset
from contrastive.data.datasets import ContrastiveDataset_Visualization
from contrastive.data.datasets import ContrastiveDataset_WithLabels
from contrastive.data.datasets import ContrastiveDataset_WithFoldLabels
from contrastive.data.datasets import \
    ContrastiveDataset_WithLabels_WithFoldLabels

from contrastive.data.utils import *

log = set_file_logger(__file__)


def sanity_checks_without_labels(config, skeleton_output):
    # Loads and separates in train_val/test set foldlabels if requested
    check_subject_consistency(config.subjects_all,
                              config.subjects_foldlabel_all)
    # add all the other created objects in the next line
    foldlabel_output = extract_data(config.foldlabel_all, config)
    log.info("foldlabel data loaded")

    # Makes some sanity checks
    for subset_name in foldlabel_output.keys():
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

    # Loads and separates in train_val/test skeleton crops
    skeleton_output = extract_data(config.numpy_all, config)

    # Loads and separates in train_val/test set foldlabels if requested
    if (config.foldlabel == True) and (config.mode != 'evaluation'):
        foldlabel_output = sanity_checks_without_labels(config,
                                                        skeleton_output)
    else:
        log.info("foldlabel data NOT requested. Foldlabel data NOT loaded")

    # Creates the dataset from these data by doing some preprocessing
    datasets = {}
    if config.mode == 'evaluation':
        for subset_name in skeleton_output.keys():
            datasets[subset_name] = ContrastiveDataset_Visualization(
                filenames=skeleton_output[subset_name][0],
                array=skeleton_output[subset_name][1],
                config=config)
    else:
        if config.foldlabel == True:
            for subset_name in skeleton_output.keys():
                datasets[subset_name] = ContrastiveDataset_WithFoldLabels(
                    filenames=skeleton_output[subset_name][0],
                    array=skeleton_output[subset_name][1],
                    foldlabel_array=foldlabel_output[subset_name][1],
                    config=config)
        else:
            for subset_name in skeleton_output.keys():
                datasets[subset_name] = ContrastiveDataset(
                    filenames=skeleton_output[subset_name][0],
                    array=skeleton_output[subset_name][1],
                    config=config)
    
    # # just to have the same data format as train and val
    # test_dataset, _ = torch.utils.data.random_split(
    #     test_dataset,
    #     [len(test_dataset),0])

    return datasets


def sanity_checks_with_labels(config, skeleton_output, subject_labels):
    # remove test_intra if not in config
    subsets = [key for key in skeleton_output.keys()]
    if 'test_intra_csv_file' not in config.keys():
        subsets.pop(3)
    print("SANITY CHECKS", subsets)

    for subset_name in subsets:
        check_if_skeleton(skeleton_output[subset_name][1], subset_name)

    if config.environment == "brainvisa" and config.checking:
        for subset_name in subsets:
            compare_array_aims_files(skeleton_output[subset_name][0],
                                     skeleton_output[subset_name][1],
                                     config.crop_dir)
    

    # Makes some sanity checks on ordering of label subjects
    for subset_name in subsets:
        check_if_same_subjects(skeleton_output[subset_name][0],
                               skeleton_output[subset_name][2][['Subject']],
                               f"{subset_name} labels")

    # Loads and separates in train_val/test set foldlabels if requested
    if (config.foldlabel == True) and (config.mode != 'evaluation'):
        check_subject_consistency(config.subjects_all,
                                  config.subjects_foldlabel_all)
        foldlabel_output = extract_data_with_labels(config.foldlabel_all,
                                                    subject_labels,
                                                    config.foldlabel_dir,
                                                    config)
        log.info("foldlabel data loaded")

        # Makes some sanity checks
        for subset_name in subsets:
            check_if_same_subjects(skeleton_output[subset_name][0],
                                   foldlabel_output[subset_name][0],
                                   subset_name)
            check_if_same_shape(skeleton_output[subset_name][1],
                                foldlabel_output[subset_name][1],
                                subset_name)
            check_if_same_subjects(foldlabel_output[subset_name][0],
                                   skeleton_output[subset_name][2][['Subject']],
                                   f"{subset_name} labels")
            
        if config.environment == "brainvisa" and config.checking:
            for subset_name in foldlabel_output.keys():
                compare_array_aims_files(foldlabel_output[subset_name][0],
                                         foldlabel_output[subset_name][1],
                                         config.foldlabel_dir)

    else:
        log.info("foldlabel data NOT requested. Foldlabel data NOT loaded")
        return None

    return foldlabel_output


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

    if config.environment == "brainvisa" and config.checking:
        quality_checks(config.subjects_all, config.numpy_all, config.crop_dir, parallel=True)

    # Loads and separates in train_val/test skeleton crops
    skeleton_output = extract_data_with_labels(config.numpy_all, subject_labels, config.crop_dir, config)

    foldlabel_output = sanity_checks_with_labels(config, skeleton_output, subject_labels)

    # Creates the dataset from these data by doing some preprocessing
    datasets = {}
    if config.mode == 'evaluation':
        for subset_name in skeleton_output.keys():
            datasets[subset_name] = ContrastiveDataset_Visualization(
                filenames=skeleton_output[subset_name][0],
                array=skeleton_output[subset_name][1],
                config=config)
    else:
        if config.foldlabel == True:
            for subset_name in skeleton_output.keys():
                datasets[subset_name] = ContrastiveDataset_WithLabels_WithFoldLabels(
                    filenames=skeleton_output[subset_name][0],
                    array=skeleton_output[subset_name][1],
                    labels=skeleton_output[subset_name][2],
                    foldlabel_array=foldlabel_output[subset_name][1],
                    config=config)
        else:
            for subset_name in skeleton_output.keys():
                datasets[subset_name] = ContrastiveDataset_WithLabels(
                    filenames=skeleton_output[subset_name][0],
                    array=skeleton_output[subset_name][1],
                    labels=skeleton_output[subset_name][2],
                    config=config)

    return datasets
