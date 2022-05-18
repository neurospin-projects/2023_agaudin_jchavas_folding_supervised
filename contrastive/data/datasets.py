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


def create_sets_pure_contrastive(config, mode='training'):
    """Creates train, validation and test sets

    Args:
        config (Omegaconf dict): contains configuration parameters
        mode (str): either 'training' or 'visualization'
    Returns:
        train_set, val_set, test_set (tuple)
    """

    # Loads crops from all subjects
    numpy_all_path = config.numpy_all
    log.info("Current directory = " + os.getcwd())
    normal_data = np.load(numpy_all_path, mmap_mode='r')
    log.debug(f"shape of loaded numpy array = {normal_data.shape}")
    normal_subjects = pd.read_csv(config.subjects_all)

    # Gets train_val subjects from csv file
    train_val_subjects = pd.read_csv(config.train_val_csv_file, names=['ID'])
    log.debug(f"train_val_subjects = {train_val_subjects}")

    # Determines test dataframe
    test_subjects = normal_subjects[~normal_subjects.Subject.isin(
        train_val_subjects.ID)].index
    len_test = len(test_subjects)
    log.debug(f"length of test = {len_test}")
    log.debug(f"test_subjects = {test_subjects[:5]}")

    # /!\ copy the data to construct test_data
    test_data = normal_data[test_subjects]
    log.debug(f'test set size: {test_data.shape}')

    # Cuts train_val set to requested number
    if config.nb_subjects == _ALL_SUBJECTS:
        len_train_val = len(train_val_subjects)
    else:
        len_train_val = min(config.nb_subjects,
                            len(train_val_subjects))
        train_val_subjects = train_val_subjects[:len_train_val]

    log.info(f"length of train/val dataframe: {len_train_val}")

    # Determines train/val dataframe
    train_val_subjects_index = normal_subjects[normal_subjects.Subject.isin(
                                train_val_subjects.ID)].index
    # /!\ copy the data to construct train_val_data
    train_val_data = normal_data[train_val_subjects_index]

    # Creates the dataset from these tensors by doing some preprocessing
    if mode == 'visualization':
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

    log.info(f"Length of test data set: {len(test_dataset)}")
    log.info(
        f"Length of complete train/val data set: {len(train_val_dataset)}")

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
