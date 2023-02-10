# -*- coding: utf-8 -*-
# /usr/bin/env python3
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
#
# https://github.com/neurospin-projects/2021_jchavas_lguillon_deepcingulate/
#                   betaVAE/load_data.py

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms


class SkeletonDataset():
    """Custom dataset for skeleton images that includes image file paths.
    Args:
        dataframe: dataframe containing training and testing arrays
        filenames: optional, list of corresponding filenames
    Returns:
        tuple_with_path: tuple of type (sample, filename) with sample normalized
                         and padded
    """
    def __init__(self, dataframe, filenames=None):
        self.df = dataframe
        if filenames:
            self.filenames = filenames
        else:
            self.filenames = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.filenames:
            filename = self.filenames[idx]
            sample = np.expand_dims(np.squeeze(self.df.iloc[idx][0]), axis=0)
        else:
            filename = self.df.iloc[idx]['ID']
            sample = self.df.iloc[idx][0]

        fill_value = 0
        self.transform = transforms.Compose([NormalizeSkeleton(),
                         Padding([1, 20, 40, 40], fill_value=fill_value)
                         ])
        sample = self.transform(sample)
        tuple_with_path = (sample, filename)
        return tuple_with_path


class NormalizeSkeleton(object):
    """
    Class to normalize skeleton objects,
    black voxels: 0
    grey and white voxels: 1
    """
    def __init__(self, nb_cls=2):
        """ Initialize the instance"""
        self.nb_cls = nb_cls

    def __call__(self, arr):
        if self.nb_cls==2:
            arr[arr > 0] = 1
        else:
            arr[arr==40]=30
            arr[arr==70]=80
            arr[arr==30]=1
            arr[arr==60]=2
            arr[arr==80]=3
        return arr


class Padding(object):
    """ A class to pad an image.
    """
    def __init__(self, shape, nb_channels=1, fill_value=0):
        """ Initialize the instance.
        Parameters
        ----------
        shape: list of int
            the desired shape.
        nb_channels: int, default 1
            the number of channels.
        fill_value: int or list of int, default 0
            the value used to fill the array, if a list is given, use the
            specified value on each channel.
        """
        self.shape = shape
        self.nb_channels = nb_channels
        self.fill_value = fill_value
        if self.nb_channels > 1 and not isinstance(self.fill_value, list):
            self.fill_value = [self.fill_value] * self.nb_channels
        elif isinstance(self.fill_value, list):
            assert len(self.fill_value) == self.nb_channels()

    def __call__(self, arr):
        """ Fill an array to fit the desired shape.
        Parameters
        ----------
        arr: np.array
            an input array.
        Returns
        -------
        fill_arr: np.array
            the zero padded array.
        """
        if len(arr.shape) - len(self.shape) == 1:
            data = []
            for _arr, _fill_value in zip(arr, self.fill_value):
                data.append(self._apply_padding(_arr, _fill_value))
            return np.asarray(data)
        elif len(arr.shape) - len(self.shape) == 0:
            return self._apply_padding(arr, self.fill_value)
        else:
            raise ValueError("Wrong input shape specified!")

    def _apply_padding(self, arr, fill_value):
        """ See Padding.__call__().
        """
        orig_shape = arr.shape
        padding = []
        for orig_i, final_i in zip(orig_shape, self.shape):
            shape_i = final_i - orig_i
            half_shape_i = shape_i // 2
            if shape_i % 2 == 0:
                padding.append((half_shape_i, half_shape_i))
            else:
                padding.append((half_shape_i, half_shape_i + 1))
        for cnt in range(len(arr.shape) - len(padding)):
            padding.append((0, 0))

        fill_arr = np.pad(arr, padding, mode="constant",
                          constant_values=fill_value)
        return fill_arr
