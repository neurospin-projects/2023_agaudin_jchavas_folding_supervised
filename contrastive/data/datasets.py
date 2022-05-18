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
import logging
import os
from re import sub

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms

from contrastive.augmentations import BinarizeTensor
from contrastive.augmentations import EndTensor
from contrastive.augmentations import PaddingTensor
from contrastive.augmentations import PartialCutOutTensor_Roll
from contrastive.augmentations import RotateTensor
from contrastive.augmentations import SimplifyTensor
from contrastive.augmentations import RemoveRandomBranchTensor
from contrastive.utils.logs import set_file_logger

_ALL_SUBJECTS = -1

log = set_file_logger(__file__)


class ContrastiveDataset():
    """Custom dataset that includes image file paths.

    Applies different transformations to data depending on the type of input.
    """

    def __init__(self, array, filenames, config):
        """
        Args:
            data_tensor (tensor): contains MRIs as numpy arrays
            filenames (list of strings): list of subjects' IDs
            config (Omegaconf dict): contains configuration information
        """
        self.arr = array
        self.transform = True
        self.nb_train = len(filenames)
        log.info(self.nb_train)
        self.filenames = filenames
        self.config = config

    def __len__(self):
        return (self.nb_train)

    def __getitem__(self, idx):
        """Returns the two views corresponding to index idx

        The two views are generated on the fly.

        Returns:
            tuple of (views, subject ID)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.arr[idx].astype('float32')
        log.debug(f"filenames[:5] = {self.filenames[:5]}")
        log.debug(f"len(filenames) = {len(self.filenames)}")
        log.debug(f"idx = {idx}")
        log.debug(f"{idx} in filename = {idx in self.filenames.index}")
        filename = self.filenames.ID[idx]

        self.transform1 = transforms.Compose([
            SimplifyTensor(),
            PaddingTensor(self.config.input_size,
                          fill_value=self.config.fill_value),
            PartialCutOutTensor_Roll(from_skeleton=True,
                                     keep_bottom=self.config.keep_bottom,
                                     patch_size=self.config.patch_size),
            RotateTensor(max_angle=self.config.max_angle),
            BinarizeTensor()
        ])

        # - padding
        # - + random rotation
        self.transform2 = transforms.Compose([
            SimplifyTensor(),
            PaddingTensor(self.config.input_size,
                          fill_value=self.config.fill_value),
            PartialCutOutTensor_Roll(from_skeleton=False,
                                     keep_bottom=self.config.keep_bottom,
                                     patch_size=self.config.patch_size),
            RotateTensor(max_angle=self.config.max_angle),
            BinarizeTensor()
        ])

        # - padding
        self.transform3 = transforms.Compose([
            SimplifyTensor(),
            PaddingTensor(self.config.input_size,
                          fill_value=self.config.fill_value),
            BinarizeTensor(),
            EndTensor()
        ])

        view1 = self.transform1(torch.from_numpy(sample))
        view2 = self.transform2(torch.from_numpy(sample))

        if self.config.mode == "decoder":
            view3 = self.transform3(sample)
            views = torch.stack((view1, view2, view3), dim=0)
        else:
            views = torch.stack((view1, view2), dim=0)

        tuple_with_path = (views, filename)
        return tuple_with_path


class ContrastiveDataset_WithLabels():
    """Custom dataset that includes images and labels

    Applies different transformations to data depending on the type of input.
    """

    def __init__(self, data, labels, filenames, config):
        """
        Args:
            data (dataframe): contains MRIs as numpy arrays
            labels (dataframe): contains labels as columns
            filenames (list of strings): list of subjects' IDs
            config (Omegaconf dict): contains configuration information
        """
        self.data = data
        self.labels = labels
        self.transform = True
        self.nb_train = len(filenames)
        log.info(self.nb_train)
        self.filenames = filenames
        self.config = config

    def __len__(self):
        return (self.nb_train)

    def __getitem__(self, idx):
        """Returns the two views corresponding to index idx

        The two views are generated on the fly.

        Returns:
            tuple of (views, subject ID)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data.loc[0].values[idx].astype('float32')
        labels = self.labels.values[idx]
        sample = torch.from_numpy(sample)
        labels = torch.from_numpy(labels)
        filenames = self.filenames[idx]

        self.transform1 = transforms.Compose([
            SimplifyTensor(),
            PaddingTensor(self.config.input_size,
                          fill_value=self.config.fill_value),
            PartialCutOutTensor_Roll(from_skeleton=True,
                                     keep_bottom=self.config.keep_bottom,
                                     patch_size=self.config.patch_size),
            RotateTensor(max_angle=self.config.max_angle),
            BinarizeTensor()
        ])

        # - padding
        # - + random rotation
        self.transform2 = transforms.Compose([
            SimplifyTensor(),
            PaddingTensor(self.config.input_size,
                          fill_value=self.config.fill_value),
            PartialCutOutTensor_Roll(from_skeleton=False,
                                     keep_bottom=self.config.keep_bottom,
                                     patch_size=self.config.patch_size),
            RotateTensor(max_angle=self.config.max_angle),
            BinarizeTensor()
        ])

        # - padding
        self.transform3 = transforms.Compose([
            SimplifyTensor(),
            PaddingTensor(self.config.input_size,
                          fill_value=self.config.fill_value),
            BinarizeTensor(),
            EndTensor()
        ])

        view1 = self.transform1(sample)
        view2 = self.transform2(sample)

        if self.config.mode == "decoder":
            view3 = self.transform3(sample)
            views = torch.stack((view1, view2, view3), dim=0)
        else:
            views = torch.stack((view1, view2), dim=0)

        tuple_with_path = (views, labels, filenames)
        return tuple_with_path



class ContrastiveDataset_WithLabels_WithFoldLabels():
    """Custom dataset that includes images and labels

    Applies different transformations to data depending on the type of input.
    """

    def __init__(self, data, foldlabel_data, labels, filenames, config):
        """
        Args:
            data (dataframe): contains MRIs as numpy arrays
            labels (dataframe): contains labels as columns
            filenames (list of strings): list of subjects' IDs
            config (Omegaconf dict): contains configuration information
        """
        self.data = data
        self.foldlabel_data = foldlabel_data
        self.labels = labels
        self.transform = True
        self.nb_train = len(filenames)
        log.info(self.nb_train)
        self.filenames = filenames
        self.config = config

    def __len__(self):
        return (self.nb_train)

    def __getitem__(self, idx):
        """Returns the two views corresponding to index idx

        The two views are generated on the fly.

        Returns:
            tuple of (views, subject ID)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        log.debug(f"index = {idx}")
        sample = self.data.loc[0].values[idx].astype('float32')
        sample_foldlabel = self.foldlabel_data.loc[0].values[idx].astype('int32')
        labels = self.labels.values[idx]
        sample = torch.from_numpy(sample)
        labels = torch.from_numpy(labels)
        filenames = self.filenames[idx]

        # Padd foldlabel
        sample_foldlabel = torch.from_numpy(sample_foldlabel)
        transform_foldlabel = PaddingTensor(
                                self.config.input_size,
                                fill_value=0)
        sample_foldlabel = transform_foldlabel(sample_foldlabel)

        self.transform1 = transforms.Compose([
            SimplifyTensor(),
            PaddingTensor(self.config.input_size,
                          fill_value=self.config.fill_value),
            RemoveRandomBranchTensor(sample_foldlabel=sample_foldlabel,
                                     percentage=self.config.percentage_1,
                                     input_size=self.config.input_size),
            RotateTensor(max_angle=self.config.max_angle),
            BinarizeTensor()
        ])

        # - padding
        # - + random rotation
        self.transform2 = transforms.Compose([
            SimplifyTensor(),
            PaddingTensor(self.config.input_size,
                          fill_value=self.config.fill_value),
            RemoveRandomBranchTensor(sample_foldlabel=sample_foldlabel,
                                     percentage=self.config.percentage_2,
                                     input_size=self.config.input_size),
            RotateTensor(max_angle=self.config.max_angle),
            BinarizeTensor()
        ])

        # - padding
        self.transform3 = transforms.Compose([
            SimplifyTensor(),
            PaddingTensor(self.config.input_size,
                          fill_value=self.config.fill_value),
            BinarizeTensor(),
            EndTensor()
        ])

        view1 = self.transform1(sample)
        view2 = self.transform2(sample)

        if self.config.mode == "decoder":
            view3 = self.transform3(sample)
            views = torch.stack((view1, view2, view3), dim=0)
        else:
            views = torch.stack((view1, view2), dim=0)

        tuple_with_path = (views, labels, filenames)
        return tuple_with_path


class ContrastiveDataset_Visualization():
    """Custom dataset that includes image file paths.

    Applies different transformations to data depending on the type of input.
    """

    def __init__(self, dataframe, filenames, config):
        """
        Args:
            data_tensor (tensor): contains MRIs as numpy arrays
            filenames (list of strings): list of subjects' IDs
            config (Omegaconf dict): contains configuration information
        """
        self.df = dataframe
        self.transform = True
        self.nb_train = len(filenames)
        log.info(self.nb_train)
        self.filenames = filenames
        self.config = config

    def __len__(self):
        return (self.nb_train)

    def __getitem__(self, idx):
        """Returns the two views corresponding to index idx

        The two views are generated on the fly.
        The second view is the reference view (only padding is applied)

        Returns:
            tuple of (views, subject ID)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.df.loc[0].values[idx].astype('float32')
        sample = torch.from_numpy(sample)
        filename = self.filenames[idx]

        self.transform1 = transforms.Compose([
            SimplifyTensor(),
            PaddingTensor(self.config.input_size,
                          fill_value=self.config.fill_value),
            BinarizeTensor(),
            EndTensor()
        ])

        self.transform2 = transforms.Compose([
            SimplifyTensor(),
            PaddingTensor(self.config.input_size,
                          fill_value=self.config.fill_value),
            # PartialCutOutTensor_Roll(from_skeleton=False,
            #                          patch_size=self.config.patch_size),
            # RotateTensor(max_angle=self.config.max_angle),
            BinarizeTensor(),
            EndTensor()
        ])

        view1 = self.transform1(sample)
        view2 = self.transform2(sample)

        if self.config.mode == "decoder":
            view3 = self.transform1(sample)
            views = torch.stack((view1, view2, view3), dim=0)
        else:
            views = torch.stack((view1, view2), dim=0)

        tuple_with_path = (views, filename)
        return tuple_with_path



def create_sets_with_labels_with_foldlabels(config, mode='training'):
    """Creates train, validation and test sets

    Args:
        config (Omegaconf dict): contains configuration parameters
        mode (str): either 'training' or 'visualization'
    Returns:
        train_set, val_set, test_set (tuple)
    """

    # Loads crops from all subjects
    pickle_file_path = config.pickle_normal
    log.info("Current directory = " + os.getcwd())
    normal_data = pd.read_pickle(pickle_file_path)
    log.info(f"normal_data head = {normal_data.head()}")

    # Loads foldlabel crops for all subjects
    pickle_foldlabel_file_path = config.pickle_foldlabel
    log.info("Current directory = " + os.getcwd())
    foldlabel_data = pd.read_pickle(pickle_foldlabel_file_path)
    log.info(f"normal_data head = {normal_data.head()}")

    # Gets subjects as list
    normal_subjects = normal_data.columns.tolist()

    # Gets labels for all subjects
    subject_labels_file = config.subject_labels_file
    subject_labels = pd.read_csv(subject_labels_file)
    log.debug(f"Subject_labels head = {subject_labels.head()}")
    log.info(f"Labels to keep = {config.label_names} of type {type(config.label_names)}")
    
    # subject_labels must have a column named 'Subject'
    # We here extract from subject_labels the column 'Subject'
    # and all columns identified by config.label_names
    desired_columns = ['Subject',]
    desired_columns.extend(config.label_names)
    subject_labels = subject_labels[desired_columns]
    subject_labels = subject_labels.replace(['M', 'F'], [0, 1])
    subject_labels = subject_labels.astype({'Subject': str})
    subject_labels = subject_labels.set_index('Subject')
    subject_labels = subject_labels.dropna()
    log.info(f"Head of subject_labels:\n{subject_labels.head()}")

    # Gets only normal_subjects that have numeric values for desired properties
    subject_labels_list = subject_labels.index.tolist()
    normal_subjects = list(set(normal_subjects).intersection(subject_labels_list))
    
    # Gets train_val subjects from csv file
    # It is a CSV file without column name
    # We add here a column name 'ID'
    train_val_subjects = pd.read_csv(config.train_val_csv_file, names=['ID']).T
    train_val_subjects = train_val_subjects.values[0].tolist()
    train_val_subjects = list(map(str, train_val_subjects))
    train_val_subjects = list(set(normal_subjects).intersection(train_val_subjects))
    log.info(f"train_val_subjects[:5] = {train_val_subjects[:5]}")
    log.debug(f"train_val_subjects = {train_val_subjects}")

    # Determines test dataframe
    test_subjects = list(set(normal_subjects).difference(train_val_subjects))
    len_test = len(test_subjects)
    log.info(f"test_subjects[:5] = {test_subjects[:5]}")
    log.debug(f"test_subjects = {test_subjects}")
    log.info(f"Number of test subjects = {len_test}")

    test_data = normal_data[normal_data.columns.intersection(test_subjects)]
    test_foldlabel_data = foldlabel_data[foldlabel_data.columns.intersection(test_subjects)]
    test_labels = subject_labels.loc[test_subjects]

    # Cuts train_val set to requested number
    if config.nb_subjects == _ALL_SUBJECTS:
        len_train_val = len(train_val_subjects)
    else:
        len_train_val = min(config.nb_subjects,
                            len(train_val_subjects))
        train_val_subjects = train_val_subjects[:len_train_val]

    log.info(f"length of train/val dataframe: {len_train_val}")

    # Determines train/val dataframe
    train_val_data = normal_data[normal_data.columns.intersection(
                                 train_val_subjects)]
    train_val_foldlabel_data = foldlabel_data[foldlabel_data.columns.intersection(
                                train_val_subjects)]
    train_val_labels = subject_labels.loc[train_val_subjects]

    # Creates the dataset from these tensors by doing some preprocessing
    if mode == 'visualization':
        test_dataset = ContrastiveDataset_Visualization(
            filenames=test_subjects,
            dataframe=test_data,
            config=config)
        train_val_dataset = ContrastiveDataset_Visualization(
            filenames=train_val_subjects,
            dataframe=train_val_data,
            config=config)
    else:
        test_dataset = ContrastiveDataset_WithLabels_WithFoldLabels(
            filenames=test_subjects,
            data=test_data,
            foldlabel_data=test_foldlabel_data,
            labels=test_labels,
            config=config)
        train_val_dataset = ContrastiveDataset_WithLabels_WithFoldLabels(
            filenames=train_val_subjects,
            data=train_val_data,
            foldlabel_data=train_val_foldlabel_data,
            labels=train_val_labels,
            config=config)

    log.info(f"Length of test data set: {len(test_dataset)}")
    log.info(
        f"Length of complete train/val data set: {len(train_val_dataset)}")

    # Split training/val set into train, val set
    partition = config.partition

    log.info([round(i * (len(train_val_dataset))) for i in partition])
    np.random.seed(config.seed)
    train_set, val_set = torch.utils.data.random_split(
        train_val_dataset,
        [round(i * (len(train_val_dataset))) for i in partition])

    return train_set, val_set, test_dataset, train_val_dataset


def create_sets_with_labels(config, mode='training'):
    """Creates train, validation and test sets

    Args:
        config (Omegaconf dict): contains configuration parameters
        mode (str): either 'training' or 'visualization'
    Returns:
        train_set, val_set, test_set (tuple)
    """

    # Loads crops from all subjects
    pickle_file_path = config.pickle_normal
    log.info("Current directory = " + os.getcwd())
    normal_data = pd.read_pickle(pickle_file_path)
    log.info(f"normal_data head = {normal_data.head()}")

    # Gets subjects as list
    normal_subjects = normal_data.columns.tolist()

    # Gets labels for all subjects
    subject_labels_file = config.subject_labels_file
    subject_labels = pd.read_csv(subject_labels_file)
    log.debug(f"Subject_labels head = {subject_labels.head()}")
    log.info(f"Labels to keep = {config.label_names} of type {type(config.label_names)}")
    
    # subject_labels must have a column named 'Subject'
    # We here extract from subject_labels the column 'Subject'
    # and all columns identified by config.label_names
    desired_columns = ['Subject',]
    desired_columns.extend(config.label_names)
    subject_labels = subject_labels[desired_columns]
    subject_labels = subject_labels.replace(['M', 'F'], [0, 1])
    subject_labels = subject_labels.astype({'Subject': str})
    subject_labels = subject_labels.set_index('Subject')
    subject_labels = subject_labels.dropna()
    log.info(f"Head of subject_labels:\n{subject_labels.head()}")

    # Gets only normal_subjects that have numeric values for desired properties
    subject_labels_list = subject_labels.index.tolist()
    normal_subjects = list(set(normal_subjects).intersection(subject_labels_list))
    
    # Gets train_val subjects from csv file
    # It is a CSV file without column name
    # We add here a column name 'ID'
    train_val_subjects = pd.read_csv(config.train_val_csv_file, names=['ID']).T
    train_val_subjects = train_val_subjects.values[0].tolist()
    train_val_subjects = list(map(str, train_val_subjects))
    train_val_subjects = list(set(normal_subjects).intersection(train_val_subjects))
    log.info(f"train_val_subjects[:5] = {train_val_subjects[:5]}")
    log.debug(f"train_val_subjects = {train_val_subjects}")

    # Determines test dataframe
    test_subjects = list(set(normal_subjects).difference(train_val_subjects))
    len_test = len(test_subjects)
    log.info(f"test_subjects[:5] = {test_subjects[:5]}")
    log.debug(f"test_subjects = {test_subjects}")
    log.info(f"Number of test subjects = {len_test}")

    test_data = normal_data[normal_data.columns.intersection(test_subjects)]
    test_labels = subject_labels.loc[test_subjects]

    # Cuts train_val set to requested number
    if config.nb_subjects == _ALL_SUBJECTS:
        len_train_val = len(train_val_subjects)
    else:
        len_train_val = min(config.nb_subjects,
                            len(train_val_subjects))
        train_val_subjects = train_val_subjects[:len_train_val]

    log.info(f"length of train/val dataframe: {len_train_val}")

    # Determines train/val dataframe
    train_val_data = normal_data[normal_data.columns.intersection(
                                 train_val_subjects)]
    train_val_labels = subject_labels.loc[train_val_subjects]

    # Creates the dataset from these tensors by doing some preprocessing
    if mode == 'visualization':
        test_dataset = ContrastiveDataset_Visualization(
            filenames=test_subjects,
            dataframe=test_data,
            config=config)
        train_val_dataset = ContrastiveDataset_Visualization(
            filenames=train_val_subjects,
            dataframe=train_val_data,
            config=config)
    else:
        test_dataset = ContrastiveDataset_WithLabels(
            filenames=test_subjects,
            data=test_data,
            labels=test_labels,
            config=config)
        train_val_dataset = ContrastiveDataset_WithLabels(
            filenames=train_val_subjects,
            data=train_val_data,
            labels=train_val_labels,
            config=config)

    log.info(f"Length of test data set: {len(test_dataset)}")
    log.info(
        f"Length of complete train/val data set: {len(train_val_dataset)}")

    # Split training/val set into train, val set
    partition = config.partition

    log.info([round(i * (len(train_val_dataset))) for i in partition])
    np.random.seed(config.seed)
    train_set, val_set = torch.utils.data.random_split(
        train_val_dataset,
        [round(i * (len(train_val_dataset))) for i in partition])

    return train_set, val_set, test_dataset, train_val_dataset


def read_npy_file(npy_file_path: str) -> np.ndarray:
    """Reads npy file containing all subjects and returns the numpy array."""
    # Loads crops from all subjects
    log.info("Current directory = " + os.getcwd())
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

def read_train_val_csv(csv_file_path: str) -> pd.DataFrame:
    """Reads train_val csv.
    
    This csv has a unisque column.
    The resulting dataframe gives the name 'ID' to this column
    """
    train_val_subjects = pd.read_csv(csv_file_path, names=['ID'])
    log.debug(f"train_val_subjects = {train_val_subjects}")
    return train_val_subjects

def extract_test(normal_subjects, train_val_subjects, normal_data):
    """Extracts test subjects and test data from normal_data.
    
    Test subjects are all subjects from normal_subjects that are not listed
    in train_val_subjects.
    normal_data is a numpy array corresponding to normal_subjects."""
    test_subjects_index = normal_subjects[~normal_subjects.Subject.isin(
        train_val_subjects.ID)].index
    test_subjects = normal_subjects.loc[test_subjects_index]
    len_test = len(test_subjects_index)
    log.debug(f"length of test = {len_test}")
    log.info(f"test_subjects = {test_subjects[:5]}")

    # /!\ copy the data to construct test_data
    test_data = normal_data[test_subjects_index]
    log.info(f"test set size: {test_data.shape}")

    return test_subjects, test_data

def restrict_length(subjects:pd.DataFrame, nb_subjects: int) -> pd.DataFrame:
    """Restrict length by nb_subjects if requested"""
    if nb_subjects == _ALL_SUBJECTS:
        length = len(subjects)
    else:
        length = min(nb_subjects,
                  len(subjects))
        subjects = subjects[:length]

    return subjects

def extract_train_val(normal_subjects, train_val_subjects, normal_data):
    """Returns data corresponding to subjects listed in train_val_subjects"""

    log.info(f"Length of train/val dataframe = {len(train_val_subjects)}")
    # Determines train/val dataframe
    train_val_subjects_index = normal_subjects[normal_subjects.Subject.isin(
                                train_val_subjects.ID)].index
    # /!\ copy the data to construct train_val_data
    train_val_data = normal_data[train_val_subjects_index]
    return train_val_data


def extract_data(config):
    """Extracts train_val and test data and subjects from npy and csv file

    Args:
        config (Omegaconf dict): contains configuration parameters
    Returns (subjects as dataframe, data as numpy array):
        train_val_subjects, train_val_data, test_subjects, test_data (tuple)
    """

    # Reads numpy data and subject list
    normal_data, normal_subjects = \
        read_numpy_data_and_subject_csv(config.numpy_all, config.subjects_all)

    # Gets train_val subjects as dataframe from csv file
    train_val_subjects = read_train_val_csv(config.train_val_csv_file)

    # Extracts test subject names and corresponding data
    test_subjects, test_data = \
        extract_test(normal_subjects, train_val_subjects, normal_data)

    # Restricts train_val length
    train_val_subjects = restrict_length(train_val_subjects, config.nb_subjects)

    # Extracts train_val from normal_data
    train_val_data = \
        extract_train_val(normal_subjects, train_val_subjects, normal_data)

    return train_val_subjects, train_val_data, test_subjects, test_data


def create_sets_pure_contrastive(config, mode='training'):
    """Creates train, validation and test sets

    Args:
        config (Omegaconf dict): contains configuration parameters
        mode (str): either 'training' or 'visualization'
    Returns:
        train_set, val_set, test_set (tuple)
    """

    train_val_subjects, train_val_data, test_subjects, test_data = \
        extract_data(config)

    # Creates the dataset from these tensors by doing some preprocessing
    if mode == 'evaluation':
        test_dataset = ContrastiveDataset_Visualization(
            filenames=test_subjects,
            array=test_data,
            config=config)
        train_val_dataset = ContrastiveDataset_Visualization(
            filenames=train_val_subjects,
            array=train_val_data,
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

    # Split training/val set into train, val set
    partition = config.partition

    size_partitions = [round(i * (len(train_val_dataset))) for i in partition]

    log.info(f"size partitions = {size_partitions}")

    # à vérifier comment le rendre random
    if config.seed:
        torch.manual_seed(config.seed)
        log.info(f"Seed for train/val split is {config.seed}")
    else:
        log.info("Train/val split has not fixed seed")

    train_set, val_set = torch.utils.data.random_split(
        train_val_dataset,
        size_partitions)

    return train_set, val_set, test_dataset, train_val_dataset
