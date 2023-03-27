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
Tools to create pytorch dataloaders
"""
import os

import numpy as np
import pandas as pd
import torch

from contrastive.utils.logs import set_file_logger
#from contrastive.data.transforms import transform_foldlabel
# only if foldlabel == True
try:
    from deep_folding.brainvisa.utils.save_data import compare_array_aims_files
except ImportError:
    print("INFO: you cannot use deep_folding in brainvisa. Probably OK.")

_ALL_SUBJECTS = -1

log = set_file_logger(__file__)


def read_npy_file(npy_file_path: str) -> np.ndarray:
    """Reads npy file containing all subjects and returns the numpy array."""
    # Loads crops from all subjects
    log.debug("Current directory = " + os.getcwd())
    arr = np.load(npy_file_path, mmap_mode='r')
    log.debug(f"shape of loaded numpy array = {arr.shape}")
    return arr


def read_subject_csv(csv_file_path: str) -> pd.DataFrame:
    """Reads csv subject file.
    It contains on a column named \'Subject\' all subject names"""
    subjects = pd.read_csv(csv_file_path)
    if 'Subject' in subjects.columns:
        return subjects
    else:
        raise ValueError(f"Column name of csv file {csv_file_path} must be "
                        f"\'Subject\'. Instead it is {subjects.columns}")


def length_object(object):
    """Returns object.shape[0] if numpy array else len(object)"""
    return object.shape[0] if type(object) == np.ndarray else len(object)


def is_equal_length(object_1, object_2):
    """Checks of the two objects have equal length"""
    len_1 = length_object(object_1)
    len_2 = length_object(object_2)
    return (len_1==len_2)


def read_numpy_data_and_subject_csv(npy_file_path, csv_file_path):
    npy_data = read_npy_file(npy_file_path)
    subjects = read_subject_csv(csv_file_path)
    if not is_equal_length(npy_data, subjects):
        raise ValueError(
            f"numpy array {npy_file_path} "
            f"and csv subject file {csv_file_path}" 
             "don't have the same length.")
    return npy_data, subjects


def check_subject_consistency(csv_file_path_1, csv_file_path_2):
    subjects_1 = read_subject_csv(csv_file_path_1)
    subjects_2 = read_subject_csv(csv_file_path_2)
    if not subjects_1.equals(subjects_2):
        raise ValueError("Both subject files (skel, foldlabel) are not equal:\n"
                         f"subjects_1 head = {subjects_1.head()}\n"
                         f"subjects_2 head = {subjects_2.head()}\n")


def check_if_skeleton(a: np.array, key: str):
    """Checks if values are compatible with skeletons"""
    is_skeleton = ((a == 0) +
                   (a == 10) +
                   (a == 20) +
                   (a == 11) +
                   (a == 30) +
                   (a == 35) +
                   (a == 40) +
                   (a == 50) +
                   (a == 60) +
                   (a == 70) +
                   (a == 80) +
                   (a == 90) +
                   (a == 100)+
                   (a == 110)+
                   (a == 120)).all()
    log.info(f"Values of {key} crops are in: {np.unique(a)}")
    if not is_skeleton:
        raise ValueError(
            f"Input array values of {key} are not compatible with skeletons"
            f"np.unique of input array = {np.unique(a)}"
        )


def read_train_val_csv(csv_file_path: str) -> pd.DataFrame:
    """Reads train_val csv.
    
    This csv has a unisque column.
    The resulting dataframe gives the name 'Subject' to this column
    """
    train_val_subjects = pd.read_csv(csv_file_path, names=['Subject'])
    log.debug(f"train_val_subjects = {train_val_subjects}")
    print("TRAIN_VAL_SUBJECTS",train_val_subjects.head())
    return train_val_subjects


def extract_test(normal_subjects, train_val_subjects, normal_data):
    """Extracts test subjects and test data from normal_data.
    
    Test subjects are all subjects from normal_subjects that are not listed
    in train_val_subjects.
    normal_data is a numpy array corresponding to normal_subjects."""

    test_subjects = normal_subjects[~normal_subjects.Subject.isin(
        train_val_subjects.Subject)]
    test_subjects_index = test_subjects.index
    len_test = len(test_subjects_index)
    log.debug(f"length of test = {len_test}")
    log.info(f"test_subjects = {test_subjects[:5]}")

    # /!\ copy the data to construct test_data
    test_data = normal_data[test_subjects_index]
    test_subjects = test_subjects.reset_index(drop=True)
    log.info(f"test set size: {test_data.shape}")

    return test_subjects, test_data


def restrict_length(subjects:pd.DataFrame, nb_subjects: int, is_random: bool=True, random_state: int=1) -> pd.DataFrame:
    """Restrict length by nb_subjects if requested"""
    if nb_subjects == _ALL_SUBJECTS:
        length = len(subjects)
    else:
        length = min(nb_subjects,
                     len(subjects))
        if is_random:
            subjects = subjects.sample(n=length, random_state=random_state)
        else:
            subjects = subjects[:length]
    return subjects


def extract_train_val(normal_subjects, train_val_subjects, normal_data):
    """Returns data corresponding to subjects listed in train_val_subjects"""

    log.info(f"Length of train/val dataframe = {len(train_val_subjects)}")
    # Determines train/val dataframe
    new_train_val_subjects = normal_subjects[normal_subjects.Subject.isin(
                                train_val_subjects.Subject)]
    new_train_val_subjects_index = new_train_val_subjects.index
    # /!\ copy the data to construct train_val_data
    train_val_data = normal_data[new_train_val_subjects_index]
    new_train_val_subjects = new_train_val_subjects.reset_index(drop=True)
    return new_train_val_subjects, train_val_data


def extract_labels(subject_labels, subjects):
    """Extracts subject_labels corresponding to test_subject
    
    For this, we compare the subjects listed in column 'Subject'
    """
    selected_subject_labels = subject_labels[subject_labels.Subject.isin(
                                subjects.Subject)]
    selected_subject_labels = \
        sort_labels_according_to_normal(selected_subject_labels, subjects)
    return selected_subject_labels


def extract_data(npy_file_path, config):
    """Extracts train_val and test data and subjects from npy and csv file

    Args:
        config (Omegaconf dict): contains configuration parameters
    Returns (subjects as dataframe, data as numpy array):
        train_val_subjects, train_val_data, test_subjects, test_data (tuple)
    """

    # Reads numpy data and subject list
    # normal_data corresponds to all data ('normal' != 'benchmark')
    normal_data, normal_subjects = \
        read_numpy_data_and_subject_csv(npy_file_path, config.subjects_all)

    if config.environment == "brainvisa" and config.checking:
        compare_array_aims_files(normal_subjects, normal_data, config.crop_dir)

    # Gets train_val subjects as dataframe from csv file
    train_val_subjects = read_train_val_csv(config.train_val_csv_file)

    # Extracts test subject names and corresponding data
    test_subjects, test_data = \
        extract_test(normal_subjects, train_val_subjects, normal_data)

    # Restricts train_val length
    random_state = None if not 'random_state' in config.keys() else config.random_state
    train_val_subjects = restrict_length(train_val_subjects, config.nb_subjects, random_state=random_state)

    # Extracts train_val from normal_data
    train_val_subjects, train_val_data = \
        extract_train_val(normal_subjects, train_val_subjects, normal_data)

    if config.environment == "brainvisa" and config.checking:
        compare_array_aims_files(train_val_subjects, train_val_data, config.crop_dir)
        compare_array_aims_files(test_subjects, test_data, config.crop_dir)

    return train_val_subjects, train_val_data, test_subjects, test_data


def extract_train_val_dataset(train_val_dataset, partition, seed):
    """Extracts traing and validation dataset from a train_val dataset"""
    # Split training/val set into train and validation set
    size_partitions = [round(i * (len(train_val_dataset))) for i in partition]
    # to be sure all the elements are actually taken
    size_partitions[-1] = len(train_val_dataset) - sum(size_partitions[:-1])

    log.info(f"size partitions = {size_partitions}")

    # Fixates seed if it is defined
    if seed:
        torch.manual_seed(seed)
        log.info(f"Seed for train/val split is {seed}")
    else:
        log.info("Train/val split has not fixed seed")

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_val_dataset,
        size_partitions)

    return train_dataset, val_dataset


def check_if_same_subjects(subjects_1, subjects_2, keyword):
    """Checks if the dataframes subjects_1 and subjects_2 are equal"""
    log.debug(f"Both heads (must be equal) of {keyword} subjects = \n"
              f"{subjects_1.head()}\n"
              f"and \n{subjects_2.head()}")
    if not subjects_1.reset_index(drop=True).equals(subjects_2.reset_index(drop=True)):
        log.error(f"subjects_1 head = {subjects_1.head()}")
        log.error(f"subjects_2 head = {subjects_2.head()}")
        raise ValueError(f"Both {keyword} subject dataframes are not equal")


def check_if_same_shape(arr1, arr2, keyword):
    """Checks if the two numpy arrays have the same shape"""
    if not (arr1.shape == arr2.shape):
        log.error(f"Shapes are {arr1.shape} and {arr2.shape}")
        raise ValueError(f"Both {keyword} numpy arrays "
                          "don't have the same shape")


def read_labels(subject_labels_file, subject_column_name, label_names):
    """Extracts labels from label file. Returns a dataframe with labels"""
    
    # Loads labels file
    subject_labels_file = subject_labels_file
    subject_labels = pd.read_csv(subject_labels_file)
    log.info(f"Subject_labels_file = {subject_labels_file}")
    log.debug(f"Subject_labels head just when loaded = {subject_labels.head()}")
    log.info(f"Labels to keep = {label_names} "
             f"of type {type(label_names)}")

    # Extracts only columns with subject name and labels
    subject_column_name = subject_column_name
    desired_columns = [subject_column_name,]
    desired_columns.extend(label_names)
    log.info(f"columns in subject_labels = {subject_labels.columns}")
    subject_labels = subject_labels[desired_columns]
    subject_labels = subject_labels.rename({subject_column_name: 'Subject'},
                                            axis = 'columns')

    # Factorizes the column if they are categories (strings for example)
    for col in label_names:
        if subject_labels[col].dtype.type == np.object_:
            subject_labels[col], uniques = \
                pd.factorize(subject_labels[col], sort=True)
            log.info(f"Column {col} sorted as categories. "
                     f"Categories are {uniques}")

    # Drops rows containing na and sets subject as index
    subject_labels = subject_labels.dropna()
    log.info(f"Head of subject_labels:\n{subject_labels.head()}")
    log.info(f"Number of non-NaN subjects with label = {len(subject_labels)}")

    return subject_labels


def sort_labels_according_to_normal(subject_labels, normal_subjects):
    """Sort subject labels according to normal_subjects order
    
    Returns reordered subject_labels
    """
    subject_labels = subject_labels.set_index('Subject')
    subject_labels = subject_labels.reindex(index=normal_subjects.Subject)
    subject_labels = subject_labels.reset_index('Subject')

    # Checks if label subject names and subjects names are the same
    log.info(f"Head of normal_subjects = \n{normal_subjects.head()}")
    log.info(f"Head of subject_labels = \n{subject_labels.head()}")
    if not normal_subjects.Subject.reset_index(drop=True).\
        equals(subject_labels.Subject):
        raise ValueError(\
            "Names of subject in subject labels are not included "
            "or are not in the same order as the csv file of the subjects")

    return subject_labels


def extract_data_with_labels(npy_file_path, subject_labels, sample_dir, config):
    """Extracts train_val and test data and subjects from npy and csv file

    Args:
        config (Omegaconf dict): contains configuration parameters
    Returns (subjects as dataframe, data as numpy array):
        train_val_subjects, train_val_data, test_subjects, test_data (tuple)
    """

    # Reads numpy data and subject list
    # normal_data corresponds to all data ('normal' != 'benchmark')
    normal_data, normal_subjects = \
        read_numpy_data_and_subject_csv(npy_file_path, config.subjects_all)

    # Selects subjects also present in subject_labels
    log.debug(f"Head of normal_subjects before label selection = \n"
              f"{normal_subjects.head()}")
    normal_subjects_index = normal_subjects[
        normal_subjects.Subject.isin(subject_labels.Subject)].index
    normal_subjects = normal_subjects.loc[normal_subjects_index]
    normal_data = normal_data[normal_subjects_index]
    normal_subjects = normal_subjects.reset_index(drop=True)

    if config.environment == "brainvisa" and config.checking:
        compare_array_aims_files(normal_subjects, normal_data, sample_dir)

    # Sort subject_labels according to normal_subjects
    subject_labels = \
        sort_labels_according_to_normal(subject_labels, normal_subjects)

    # Gets train_val subjects as dataframe from csv file
    train_val_subjects = read_train_val_csv(config.train_val_csv_file)

    # Extracts test subject names, corresponding data and labels
    test_subjects, test_data = \
        extract_test(normal_subjects, train_val_subjects, normal_data)
    test_labels = extract_labels(subject_labels, test_subjects)

    # Restricts train_val length
    train_val_subjects = restrict_length(train_val_subjects, config.nb_subjects, random_state=config.random_state)

    # Extracts train_val from normal_data
    train_val_subjects, train_val_data = \
        extract_train_val(normal_subjects, train_val_subjects, normal_data)
    train_val_labels = extract_labels(subject_labels, train_val_subjects)

    if config.environment == "brainvisa" and config.checking:
        compare_array_aims_files(train_val_subjects, train_val_data, sample_dir)
        compare_array_aims_files(test_subjects, test_data, sample_dir)


    return train_val_subjects, train_val_data, train_val_labels,\
           test_subjects, test_data, test_labels



# auxilary functions for ToPointnetTensor
def zero_padding(cloud, n_max, shuffle=False):
    return np.pad(cloud, ((0,0),(0,n_max-cloud.shape[1])))

def repeat_padding(cloud, n_max, replace=False):
    while n_max - cloud.shape[1] > 0: # loop in case len(cloud) < n_max/2
        n = min(n_max - cloud.shape[1], cloud.shape[1])
        if n < 0:
            raise ValueError("the vector is too long compared to the desired vector size")
        
        idx = np.random.choice(cloud.shape[1], size=n, replace=replace)
        padded_part = cloud[:, idx]

        cloud = np.concatenate([cloud, padded_part], axis=1)
    
    return cloud

def pad(clouds, padding_method=zero_padding, n_max=None):
    if not n_max:
        n_max = np.max([clouds[i].shape[1] for i in range(len(clouds))]) # max length of a sequence
    padded_clouds = np.array([padding_method(cloud, n_max) for cloud in clouds])
    return padded_clouds
