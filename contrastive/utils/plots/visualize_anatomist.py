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
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
import dico_toolbox as dtx
import anatomist.headless as anatomist
from soma import aims
import io
import logging

import matplotlib.pyplot as plt
from numpy import int16

from .visu_utils import buffer_to_image

logger = logging.getLogger(__name__)


a = None
win = None


class Visu_Anatomist:

    def __init__(self, ):
        global a
        global win
        a = anatomist.Anatomist()
        win = a.createWindow('3D')
        win.setHasCursor(0)

    def plot_bucket(self, img, buffer):
        """Plots as 3D buckets the first 3D image of the batch

        Args:
            img: batch of images of size [size_batch, 1, size_X, size_Y, size_Z]
            buffer (boolean): True -> returns PNG image buffer
                            False -> plots the figure
        """
        global a
        global win
        arr = img[0, 0, :, :, :]
        vol = aims.Volume(arr.numpy().astype(int16))
        bucket_map = dtx.convert.volume_to_bucketMap_aims(vol)
        bucket_a = a.toAObject(bucket_map)
        bucket_a.addInWindows(win)
        view_quaternion = [0.4, 0.4, 0.5, 0.5]
        win.camera(view_quaternion=view_quaternion)
        win.imshow(show=False)

        if buffer:
            win.removeObjects(bucket_a)
            return buffer_to_image(buffer=io.BytesIO())
        else:
            plt.show()
