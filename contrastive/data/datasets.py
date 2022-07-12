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
from re import sub

import numpy as np
import pandas as pd
import torch

from contrastive.utils.logs import set_file_logger

from contrastive.data.transforms import *

from contrastive.augmentations import PaddingTensor

from contrastive.data.transforms import transform_nothing_done

log = set_file_logger(__file__)


def get_sample(arr, idx, type_el):
    """Returns sub-numpy torch tensors corresponding to array of indices idx.
    
    First axis of arr (numpy array) corresponds to subject numbers from 0 to N-1
    type_el is 'float32' for input, 'int32' for foldlabel
    """
    log.debug(f"idx (in get_sample) = {idx}")
    log.debug(f"shape of arr (in get_sample) = {arr.shape}")
    sample = arr[idx].astype(type_el)

    return torch.from_numpy(sample)


def get_filename(filenames, idx):
    """"Returns filenames corresponding to indices idx
    
    filenames: dataframe with column name 'ID'
    """
    filename = filenames.Subject[idx]
    log.debug(f"filenames[:5] = {filenames[:5]}")
    log.debug(f"len(filenames) = {len(filenames)}")
    log.debug(f"idx = {idx}, filename[idx] = {filename}")
    log.debug(f"{idx} in filename = {idx in filenames.index}")

    return filename


def get_label(labels, idx):
    """"Returns labels corresponding to indices idx
    
    labels: dataframe with column name 'Subject'
    """
    label = labels.drop(columns='Subject').values[idx]
    log.debug(f"idx = {idx}, labels[idx] = {label}")
    log.debug(f"{idx} in labels = {idx in labels.index}")

    return label


def check_consistency(filename, labels, idx):
    """Checks if filenames are identical"""
    filename_label = labels.Subject[idx]
    if filename_label != filename:
        raise ValueError("Filenames are not consitent between data and labels"
                         f"For idx = {idx}, filename = {filename}"
                         f"and filename_label = {filename_label}")


def padd_foldlabel(sample_foldlabel, input_size):
    """Padds foldlabel according to input_size"""
    transform_foldlabel = PaddingTensor(
                            input_size,
                            fill_value=0)
    sample_foldlabel = transform_foldlabel(sample_foldlabel)
    return sample_foldlabel


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

        # Gets data corresponding to idx
        sample   = get_sample(self.arr, idx, 'float32')
        filename = get_filename(self.filenames, idx)

        # Computes the transforms
        self.transform1 = transform_no_foldlabel(from_skeleton=True,
                                                 config=self.config)
        self.transform2 = transform_no_foldlabel(from_skeleton=False,
                                                 config=self.config)
        self.transform3 = transform_only_padding(self.config)

        # Computes the views
        view1 = self.transform1(sample)
        view2 = self.transform2(sample)

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

    def __init__(self, array, labels, filenames, config):
        """
        Args:
            array (np array): contains MRIs as numpy arrays
            labels (dataframe): contains labels as columns
            filenames (list of strings): list of subjects' IDs
            config (Omegaconf dict): contains configuration information
        """
        self.arr = array
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

        # Gets data corresponding to idx
        sample   = get_sample(self.arr, idx, 'float32')
        filename = get_filename(self.filenames, idx)
        check_consistency(filename, self.labels, idx)
        labels   = get_label(self.labels, idx)

        # Computes the transforms
        self.transform1 = transform_no_foldlabel(from_skeleton=True,
                                                 config=self.config)
        self.transform2 = transform_no_foldlabel(from_skeleton=False,
                                                 config=self.config)

        # Computes the views
        log.debug(f"sample shape = {sample.shape}")
        view1 = self.transform1(sample)
        view2 = self.transform2(sample)

        if self.config.mode == "decoder":
            self.transform3 = transform_only_padding(self.config)
            view3 = self.transform3(sample)
            views = torch.stack((view1, view2, view3), dim=0)
            tuple_with_path = (views, labels, filename)
            return tuple_with_path
        else:
            self.transform3 = transform_nothing_done()
            view3 = self.transform3(sample)
            views = torch.stack((view1, view2), dim=0)
            tuple_with_path = (views, labels, filename, view3)
            return tuple_with_path


class ContrastiveDataset_WithFoldLabels():
    """Custom dataset that includes images and foldlabels

    Applies different transformations to data depending on the type of input.
    """

    def __init__(self, array, foldlabel_array, filenames, config):
        """
        Args:
            data (numpy array): contains skeletonss as numpy arrays
            foldlabel_data (numpy array): contains foldlabels as numpy array
            filenames (list of strings): list of subjects' IDs
            config (Omegaconf dict): contains configuration information
        """
        self.arr = array
        self.foldlabel_arr = foldlabel_array
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

        # Gets data corresponding to idx
        log.info(f"length = {self.nb_train}")
        sample = get_sample(self.arr, idx, 'float32')
        sample_foldlabel = get_sample(self.foldlabel_arr, idx, 'int32')
        filename = get_filename(self.filenames, idx)

        # Padds foldlabel
        sample_foldlabel = padd_foldlabel(sample_foldlabel,
                                          self.config.input_size)

        # Computes the transforms
        self.transform1 = transform_foldlabel(sample_foldlabel,
                                              self.config.percentage,
                                              self.config)
        self.transform2 = transform_foldlabel(sample_foldlabel,
                                              self.config.percentage,
                                              self.config)
        self.transform3 = transform_only_padding(self.config)

        # Computes the views
        view1 = self.transform1(sample)
        view2 = self.transform2(sample)

        if self.config.mode == "decoder":
            view3 = self.transform3(sample)
            views = torch.stack((view1, view2, view3), dim=0)
        else:
            views = torch.stack((view1, view2), dim=0)

        tuple_with_path = (views, filename)
        return tuple_with_path


class ContrastiveDataset_WithLabels_WithFoldLabels():
    """Custom dataset that includes images and labels

    Applies different transformations to data depending on the type of input.
    """

    def __init__(self, array, foldlabel_array, labels, filenames, config):
        """
        Args:
            array (np array): contains MRIs as numpy arrays
            labels (dataframe): contains labels as columns
            filenames (list of strings): list of subjects' IDs
            config (Omegaconf dict): contains configuration information
        """
        self.arr = array
        self.foldlabel_arr = foldlabel_array
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

        # Gets the data corresponding to idx
        log.debug(f"length = {self.nb_train}")
        log.debug(f"filenames = {self.filenames}")
        sample = get_sample(self.arr, idx, 'float32')
        sample_foldlabel = get_sample(self.foldlabel_arr, idx, 'int32')
        labels = get_label(self.labels, idx)
        filename = get_filename(self.filenames, idx)

        # Padds foldlabel
        sample_foldlabel = padd_foldlabel(sample_foldlabel,
                                          self.config.input_size)

        # Computes the transforms
        self.transform1 = transform_foldlabel(sample_foldlabel,
                                              self.config.percentage,
                                              self.config)
        self.transform2 = transform_foldlabel(sample_foldlabel,
                                              self.config.percentage,
                                              self.config)

        # Computes the views
        view1 = self.transform1(sample)
        view2 = self.transform2(sample)

        if self.config.mode == "decoder":
            self.transform3 = transform_only_padding(self.config)
            view3 = self.transform3(sample)
            views = torch.stack((view1, view2, view3), dim=0)
            tuple_with_path = (views, labels, filename)
            return tuple_with_path
        else:
            self.transform3 = transform_nothing_done()
            view3 = self.transform3(sample)
            views = torch.stack((view1, view2), dim=0)
            tuple_with_path = (views, labels, filename, view3)
            return tuple_with_path
        



class ContrastiveDataset_Visualization():
    """Custom dataset that includes image file paths.

    Applies different transformations to data depending on the type of input.
    """

    def __init__(self, array, filenames, config):
        """
        Args:
            data (np array): contains MRIs as numpy arrays
            filenames (list of strings): list of subjects' IDs
            config (Omegaconf dict): contains configuration information
        """
        self.arr = array
        self.transform = True
        self.nb_train = len(filenames)
        log.info(f"Number of subjects = {self.nb_train}")
        self.filenames = filenames
        self.config = config

        # Some checks
        if 'Subject' in filenames.Subject.tolist():
            raise ValueError("\'Subject\' is not expected as subject name.\n"
                             "Probably the column name has been considered "
                             "as a subject. This is a bug.")

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
            idx = idx.tolist(self.nb_train)

        # Gets the data corresponding to idx
        log.info(f"length = {self.nb_train}")
        log.info(f"filenames = {self.filenames}")
        sample = get_sample(self.arr, idx, 'float32')
        filename = get_filename(self.filenames, idx)

        # Computes the transforms
        self.transform1 = transform_only_padding(self.config)
        self.transform2 = transform_only_padding(self.config)
        self.transform3 = transform_only_padding(self.config)

        # Computes the views
        view1 = self.transform1(sample)
        view2 = self.transform2(sample)

        if self.config.mode == "decoder":
            view3 = self.transform3(sample)
            views = torch.stack((view1, view2, view3), dim=0)
        else:
            views = torch.stack((view1, view2), dim=0)

        tuple_with_path = (views, filename)
        return tuple_with_path