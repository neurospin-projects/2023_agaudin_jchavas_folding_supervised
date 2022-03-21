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
import io
import logging

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from .visu_utils import buffer_to_image
from .visu_utils import prime_factors

logger = logging.getLogger(__name__)


def plot_img(img, buffer):
    """Plots one 2D slice of one of the 3D images of the batch

    Args:
        img: batch of images of size [N_batch, 1, size_X, size_Y, size_Z]
        buffer (boolean): True -> returns PNG image buffer
                          False -> plots the figure
    """
    plt.imshow(img[0, 0, img.shape[2] // 2, :, :])

    if buffer:
        return buffer_to_image(buffer=io.BytesIO())
    else:
        plt.show()


def plot_bucket(img, buffer):
    """Plots as 3D buckets the first 3D image of the batch

    Args:
        img: batch of images of size [size_batch, 1, size_X, size_Y, size_Z]
        buffer (boolean): True -> returns PNG image buffer
                          False -> plots the figure
    """

    arr = img[0, 0, :, :, :]
    logger.info(np.unique(arr, return_counts=True))
    logger.info(img.shape)
    logger.info(arr.shape)
    bucket = np.argwhere(arr)
    bucket_t = (bucket).T
    x = bucket_t[:, 0]
    y = bucket_t[:, 1]
    z = bucket_t[:, 2]

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim3d(0, 12)
    ax.set_ylim3d(0, 40)
    ax.set_zlim3d(0, 40)
    ax.scatter(x, y, z)

    if buffer:
        return buffer_to_image(buffer=io.BytesIO())
    else:
        plt.show()


def plot_output(img, buffer):

    arr = (img[0, :]).detach().numpy()
    # Reshapes the array into a 2D array
    primes = prime_factors(arr.size)
    row_size = np.prod(primes[:len(primes) // 2])
    arr = arr.reshape(row_size, -1)

    plt.imshow(arr)

    if buffer:
        return buffer_to_image(buffer=io.BytesIO())
    else:
        plt.show()


def plot_histogram(tensor, buffer):
    """Plots histogram of the values of a tensor"""
    arr = tensor.detach().cpu().numpy() * 100

    plt.hist(arr.flatten(), bins=50, range=[-100, 100])

    if buffer:
        return buffer_to_image(buffer=io.BytesIO())
    else:
        plt.show()
