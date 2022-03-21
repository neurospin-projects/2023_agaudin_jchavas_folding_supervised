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

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms

from SimCLR.augmentations import BinarizeTensor
from SimCLR.augmentations import EndTensor
from SimCLR.augmentations import PaddingTensor
from SimCLR.augmentations import PartialCutOutTensor_Roll
from SimCLR.augmentations import RotateTensor
from SimCLR.augmentations import SimplifyTensor

_ALL_SUBJECTS = -1

log = logging.getLogger(__name__)


class ContrastiveDataset():
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

        Returns:
            tuple of (views, subject ID)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.df.loc[0].values[idx].astype('float32')
        sample = torch.from_numpy(sample)
        filename = self.filenames[idx]

        # self.transform1 = transforms.Compose([
        #     SimplifyTensor(),
        #     PaddingTensor(self.config.input_size,
        #                   fill_value=self.config.fill_value),
        #     MixTensor(from_skeleton=True, patch_size=self.config.patch_size),
        #     RotateTensor(max_angle=self.config.max_angle)
        # ])

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

        tuple_with_path = (views, filename)
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
        # self.transform1 = transforms.Compose([
        #     SimplifyTensor(),
        #     PaddingTensor(self.config.input_size,
        #                   fill_value=self.config.fill_value),
        #     PartialCutOutTensor_Roll(from_skeleton=True,
        #                              patch_size=self.config.patch_size),
        #     RotateTensor(max_angle=self.config.max_angle),
        #     BinarizeTensor()
        # ])

        # - padding
        # - + random rotation
        self.transform2 = transforms.Compose([
            SimplifyTensor(),
            PaddingTensor(self.config.input_size,
                          fill_value=self.config.fill_value),
            PartialCutOutTensor_Roll(from_skeleton=False,
                                     patch_size=self.config.patch_size),
            RotateTensor(max_angle=self.config.max_angle),
            BinarizeTensor()
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


def create_sets(config, mode='training'):
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
    print(normal_data.head())
    normal_subjects = normal_data.columns.tolist()

    # Loads benchmarks (crops from another region) from all subjects
    if config.pickle_benchmark:
        pickle_benchmark_path = config.pickle_benchmark
        benchmark_data = pd.read_pickle(pickle_benchmark_path)

    # Gets train_val subjects from csv file
    train_val_subjects = pd.read_csv(config.train_val_csv_file, names=['ID']).T
    train_val_subjects = train_val_subjects.values[0].tolist()
    train_val_subjects = list(map(str, train_val_subjects))
    print(f"train_val_subjects = {train_val_subjects}")

    # Determines test dataframe
    test_subjects = list(set(normal_subjects).difference(train_val_subjects))
    len_test = len(test_subjects)
    print(f"test_subjects = {test_subjects}")

    if config.pickle_benchmark:
        normal_test_subjects = test_subjects[:round(len_test / 2)]
        normal_test_data = \
            normal_data[normal_data.columns.intersection(normal_test_subjects)]
        benchmark_test_subjects = test_subjects[round(len_test / 2):]
        benchmark_test_data = \
            benchmark_data[
                benchmark_data.columns.intersection(benchmark_test_subjects)]

        test_data = pd.concat(
            [normal_test_data, benchmark_test_data], axis=1, ignore_index=True)
    else:
        test_data = normal_data[normal_data.columns.intersection(
            test_subjects)]

    # Cuts train_val set to requested number
    if config.nb_subjects == _ALL_SUBJECTS:
        len_train_val = len(train_val_subjects)
    else:
        len_train_val = min(config.nb_subjects,
                            len(train_val_subjects))
        train_val_subjects = train_val_subjects[:len_train_val]

    log.info(f"length of train/val dataframe: {len_train_val}")

    # Determines train/val dataframe
    if config.pickle_benchmark:
        normal_train_val_subjects = train_val_subjects[:round(
            len(train_val_subjects) / 2)]
        normal_train_val_data = normal_data[
            normal_data.columns.intersection(normal_train_val_subjects)]
        benchmark_train_val_subjects = train_val_subjects[
            round(len(train_val_subjects) / 2):]
        benchmark_train_val_data = benchmark_data[
            benchmark_data.columns.intersection(benchmark_train_val_subjects)]
        train_val_data = pd.concat(
            [normal_train_val_data, benchmark_train_val_data],
            axis=1,
            ignore_index=True)
    else:
        train_val_data = normal_data[normal_data.columns.intersection(
            train_val_subjects)]

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
        test_dataset = ContrastiveDataset(
            filenames=test_subjects,
            dataframe=test_data,
            config=config)
        train_val_dataset = ContrastiveDataset(
            filenames=train_val_subjects,
            dataframe=train_val_data,
            config=config)
    log.info(f"Length of test data set: {len(test_dataset)}")
    log.info(
        f"Length of complete train/val data set: {len(train_val_dataset)}")

    # Split training/val set into train, val set
    partition = config.partition

    log.info([round(i * (len(train_val_dataset))) for i in partition])
    np.random.seed(1)
    train_set, val_set = torch.utils.data.random_split(
        train_val_dataset,
        [round(i * (len(train_val_dataset))) for i in partition])

    return train_set, val_set, test_dataset, train_val_dataset
