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
# knowledge of the CeCILL license version 2 and that you ac8
import numbers
from collections import namedtuple

import numpy as np
import torch
from scipy.ndimage import rotate, zoom
from sklearn.preprocessing import OneHotEncoder

from contrastive.utils import logs
from contrastive.utils.test_timeit import timeit
from contrastive.data.utils import zero_padding, repeat_padding, pad

log = logs.set_file_logger(__file__)


def rotate_list(l_list):
    "Rotates list by -1"
    return l_list[1:] + l_list[:1]


def checkerboard(shape, tile_size):
    return (np.indices(shape) // tile_size).sum(axis=0) % 2


class PaddingTensor(object):
    """A class to pad a tensor"""

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
        self.shape = rotate_list(shape)
        self.nb_channels = nb_channels
        self.fill_value = fill_value
        if self.nb_channels > 1 and not isinstance(self.fill_value, list):
            self.fill_value = [self.fill_value] * self.nb_channels
        elif isinstance(self.fill_value, list):
            assert len(self.fill_value) == self.nb_channels()

    def __call__(self, tensor):
        """ Fill a tensor to fit the desired shape.
        Parameters
        ----------
        tensor: torch.tensor
            an input tensor.
        Returns
        -------
        fill_tensor: torch.tensor
            the fill_value padded tensor.
        """
        if len(tensor.shape) - len(self.shape) == 1:
            data = []
            for _tensor, _fill_value in zip(tensor, self.fill_value):
                data.append(self._apply_padding(_tensor, _fill_value))
            return torch.from_numpy(np.asarray(data))
        elif len(tensor.shape) - len(self.shape) == 0:
            return self._apply_padding(tensor, self.fill_value)
        else:
            raise ValueError("Wrong input shape specified!")

    def _apply_padding(self, tensor, fill_value):
        """ See Padding.__call__().
        """
        arr = tensor.numpy()
        orig_shape = arr.shape
        padding = []
        # print(f"SHAPES: {orig_shape} - {self.shape}")
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

        # fill_arr = np.reshape(fill_arr, (1,) + fill_arr.shape[:-1])

        return torch.from_numpy(fill_arr)


class EndTensor(object):
    """Puts all internal and external values to background value 0
    """

    def __init__(self):
        None

    def __call__(self, tensor):
        arr = tensor.numpy()
        arr = np.reshape(arr, (1,) + arr.shape[:-1])
        return torch.from_numpy(arr)


class SimplifyTensor(object):
    """Puts all internal and external values to background value 0
    """

    def __init__(self):
        None

    def __call__(self, tensor):
        arr = tensor.numpy()
        arr[arr == 11] = 0
        return torch.from_numpy(arr)


class OnlyBottomTensor(object):
    """Keeps only bottom '30' values, puts everything else to '0'
    """

    def __init__(self):
        None

    def __call__(self, tensor):
        arr = tensor.numpy()
        arr = arr * (arr == 30)
        return torch.from_numpy(arr)


class BinarizeTensor(object):
    """Puts non-zero values to 1
    """

    def __init__(self):
        None

    def __call__(self, tensor):
        arr = tensor.numpy()
        arr[arr > 0] = 1
        return torch.from_numpy(arr)


def count_non_null(arr):
    return (arr != 0).sum()


def remove_branch(arr_foldlabel, arr_skel, selected_branch):
    """It masks the selected branch in arr_skel
    """
    mask = ((arr_foldlabel != 0) & (arr_foldlabel != selected_branch))
    mask = mask.astype(int)
    return arr_skel * mask


def intersection_skeleton_foldlabel(arr_foldlabel, arr_skel):
    """It returns the intersection between skeleton and foldlabel
    """
    mask = ((arr_foldlabel != 0)).astype(int)
    intersec = arr_skel*mask
    count_intersec = count_non_null(intersec)
    count_skel = count_non_null(arr_skel)
    if count_intersec != count_skel:
        raise ValueError("Probably misaligned skeleton and foldlabel\n"
                         f"Intersection between skeleton and foldlabel "
                         f"has {count_intersec} non-null elements.\n"
                         f"Skeleton has {count_skel} non-null elements")
    return count_intersec


def remove_bottom_branches(a):
    """Removes bottom branches from foldlabel.

    Bottom branches are numerated between 2000 and 2999"""
    return a*((a < 2000) | (a >= 3000)).astype(int)


def remove_branches_up_to_percent(arr_foldlabel, arr_skel,
                                  percentage, keep_bottom):
    """Removes from arr_skel random branches up to percentage of pixels
    If percentage==0, no pixel is deleted
    If percentage==100, all pixels are deleted
    """
    # We make some initial checks
    # of alignments between skeletons and foldlabels
    total_pixels_after = intersection_skeleton_foldlabel(arr_foldlabel,
                                                         arr_skel)

    if keep_bottom:
        arr_foldlabel_without_bottom = remove_bottom_branches(arr_foldlabel)
        branches, counts = np.unique(arr_foldlabel_without_bottom,
                                     return_counts=True)
    else:
        branches, counts = np.unique(arr_foldlabel,
                                     return_counts=True)
        
    total_pixels = count_non_null(arr_skel)
    # We take as index branches indexes that are not 0
    log.debug(f"Number of branches = {branches.size}")
    log.debug(f"Histogram of size of branches = {counts}")
    indexes = np.arange(branches.size-1) + 1

    log.debug(f"total_pixels = {total_pixels}")
    log.debug(f"skel shape = {arr_skel.shape}")
    log.debug(f"foldlabel shape = {arr_foldlabel.shape}")

    # We take random branches
    np.random.seed()
    np.random.shuffle(indexes)
    arr_skel_without_branches = arr_skel

    # We loop over shuffled indexes until enough branches have been removed
    for index in indexes:
        if total_pixels_after <= total_pixels*(100.-percentage)/100.:
            break
        arr_skel_without_branches = \
            remove_branch(arr_foldlabel,
                          arr_skel_without_branches,
                          branches[index])
        total_pixels_after = (arr_skel_without_branches != 0).sum()
        log.debug(f"total_pixels_after (iteration) = {total_pixels_after}")
    log.debug(f"total_pixels_after (final) = {total_pixels_after}")
    percent_pixels_removed = (total_pixels-total_pixels_after)/total_pixels*100
    log.debug(f"Minimum expected % removed pixels = {percentage}")
    log.debug(f"% removed pixels = {percent_pixels_removed}")
    assert (percent_pixels_removed >= percentage)

    return arr_skel_without_branches


class RemoveRandomBranchTensor(object):
    """Removes randomly branches up to percent
    """

    def __init__(self, sample_foldlabel,
                 percentage, input_size, keep_bottom, variable_percentage):
        self.sample_foldlabel = sample_foldlabel
        self.percentage = percentage
        self.variable_percentage = variable_percentage
        self.input_size = input_size
        self.keep_bottom = keep_bottom

    def __call__(self, tensor_skel):
        log.debug(f"Shape of tensor_skel = {tensor_skel.shape}")
        arr_skel = tensor_skel.numpy()
        arr_foldlabel = self.sample_foldlabel.numpy()

        # log.debug(f"arr_skel.shape = {arr_skel.shape}")
        # log.debug(f"arr_foldlabel.shape = {arr_foldlabel.shape}")
        assert (arr_skel.shape == arr_foldlabel.shape)

        # percentage can be a tuple or an int
        if type(self.percentage) != int:
            assert ((int(self.percentage[0]) >= 0) & (int(self.percentage[1]) >= 0))
            percentage = np.random.uniform(int(self.percentage[0]), int(self.percentage[1]))
        else:
            assert (self.percentage >= 0)
            if self.variable_percentage:
                percentage = np.random.uniform(0, self.percentage)
            else:
                percentage = self.percentage
        log.debug("expected percentage "
                  f"(RemoveRandomBranchTensor) = {percentage}")

        arr_skel_without_branches = np.zeros(arr_skel.shape)
        log.debug("Shape of arr_skel before calling transform: "
                  f"{arr_skel_without_branches.shape}")

        # Checks if it is only one image or a batch of images
        if len(arr_skel.shape) == len(self.input_size)+1:
            for num_img in np.arange(arr_skel.shape[0]):
                arr_skel_without_branches[num_img, ...] = \
                    remove_branches_up_to_percent(arr_foldlabel[num_img, ...],
                                                  arr_skel[num_img, ...],
                                                  percentage,
                                                  self.keep_bottom)
        elif len(arr_skel.shape) == len(self.input_size):
            arr_skel_without_branches = \
                remove_branches_up_to_percent(arr_foldlabel,
                                              arr_skel,
                                              percentage,
                                              self.keep_bottom)
        else:
            raise RuntimeError(
                f"Unexpected skeleton shape."
                f"Compare arr_skel shape {arr_skel.shape} "
                f"with input_size shape {self.input_size.shape}")

        arr_skel_without_branches = arr_skel_without_branches.astype('float32')

        return torch.from_numpy(arr_skel_without_branches)


class RotateTensor(object):
    """Apply a random rotation on the images
    """

    def __init__(self, max_angle):
        self.max_angle = max_angle

    def __call__(self, tensor):
        arr = tensor.numpy()
        log.debug("Shapes before rotation", tensor.shape, arr.shape)
        rot_array = np.copy(arr)

        for axes in (0, 1), (0, 2), (1, 2):
            np.random.seed()
            angle = np.random.uniform(-self.max_angle, self.max_angle)
            log.debug(axes, angle)
            rot_array = rotate(rot_array,
                               angle=angle,
                               axes=axes,
                               order=0,
                               reshape=False,
                               mode='constant',
                               cval=0)

        rot_array = np.expand_dims(rot_array[..., 0], axis=0)

        log.debug("Values in the array after rotation", np.unique(rot_array))

        return torch.from_numpy(rot_array)


class PartialCutOutTensor_Roll(object):
    """Apply a rolling cutout on the images and puts only bottom value
    inside the cutout
    cf. Improved Regularization of Convolutional Neural Networks with Cutout,
    arXiv, 2017
    We assume that the rectangle to be cut is inside the image.
    """

    def __init__(self, from_skeleton=True,
                 keep_bottom=True, patch_size=None,
                 random_size=False, localization=None):
        """[summary]
        If from_skeleton==True,
            takes skeleton image, cuts it out and fills with bottom_only image
        If from_skeleton==False,
            takes bottom_only image, cuts it out and fills with skeleton image
        Args:
            from_skeleton (bool, optional): Defaults to True.
            patch_size (either int or list of int): Defaults to None.
            random_size (bool, optional): Defaults to False.
            inplace (bool, optional): Defaults to False.
            localization ([type], optional): Defaults to None.
        """
        self.patch_size = rotate_list(patch_size)
        self.random_size = random_size
        self.localization = localization
        self.from_skeleton = from_skeleton
        self.keep_bottom = keep_bottom

    def __call__(self, tensor):

        arr = tensor.numpy()
        img_shape = np.array(arr.shape)
        if isinstance(self.patch_size, int):
            size = [self.patch_size for _ in range(len(img_shape))]
        else:
            size = np.copy(self.patch_size)
        assert len(size) == len(img_shape), "Incorrect patch dimension."
        start_cutout = []
        for ndim in range(len(img_shape)):
            if size[ndim] > img_shape[ndim] or size[ndim] < 0:
                size[ndim] = img_shape[ndim]
            if self.random_size:
                size[ndim] = np.random.randint(0, size[ndim])
            if self.localization is not None:
                delta_before = max(
                    self.localization[ndim] - size[ndim] // 2, 0)
            else:
                np.random.seed()
                delta_before = np.random.randint(0, img_shape[ndim])
            start_cutout.append(delta_before)

        # Creates rolling mask cutout
        mask_roll = np.zeros(img_shape).astype('float32')

        indexes = []
        for ndim in range(len(img_shape)):
            indexes.append(slice(0, int(size[ndim])))
        mask_roll[tuple(indexes)] = 1

        for ndim in range(len(img_shape)):
            mask_roll = np.roll(mask_roll, start_cutout[ndim], axis=ndim)

        # Determines part of the array inside and outside the cutout
        arr_inside = arr * mask_roll
        arr_outside = arr * (1 - mask_roll)

        # If self.from_skeleton == True:
        # This keeps the whole skeleton outside the cutout
        # and keeps only bottom value inside the cutout
        if self.from_skeleton:
            if self.keep_bottom:
                arr_inside = arr_inside * (arr_inside == 30)
            else:
                arr_inside = arr_inside * (arr_inside == 0)

        # If self.from_skeleton == False:
        # This keeps only bottom value outside the cutout
        # and keeps the whole skeleton inside the cutout
        else:
            if self.keep_bottom:
                arr_outside = arr_outside * (arr_outside == 30)
            else:
                arr_outside = arr_outside * (arr_outside == 0)

        return torch.from_numpy(arr_inside + arr_outside)


class CheckerboardTensor(object):
    """Apply a checkerboard noise
    """

    def __init__(self, checkerboard_size):
        """[summary]
        Args:
        """
        self.checkerboard_size = checkerboard_size

    def __call__(self, tensor):

        arr = tensor.numpy()
        img_shape = np.array(arr.shape)

        if isinstance(self.checkerboard_size, int):
            size = [self.checkerboard_size for _ in range(len(img_shape))]
        else:
            size = np.copy(self.checkerboard_size)
        assert len(size) == len(img_shape), "Incorrect patch dimension."

        start_cutout = []
        for ndim in range(len(img_shape)):
            if size[ndim] > img_shape[ndim] or size[ndim] < 0:
                size[ndim] = img_shape[ndim]
            np.random.seed()
            delta_before = np.random.randint(0, size[ndim])
            start_cutout.append(delta_before)

        # Creates checkerboard mask
        mask = checkerboard(
            img_shape,
            self.checkerboard_size).astype('float32')

        for ndim in range(len(img_shape)):
            mask = np.roll(mask, start_cutout[ndim], axis=ndim)

        return torch.from_numpy(arr * mask)


class PartialCutOutTensor(object):
    """Apply a cutout on the images and puts only bottom value inside
    cf. Improved Regularization of Convolutional Neural Networks with Cutout,
    arXiv, 2017
    We assume that the rectangle to be cut is inside the image.
    """

    def __init__(self, from_skeleton=True, patch_size=None, random_size=False,
                 inplace=False, localization=None):
        """[summary]
        If from_skeleton==True,
            takes skeleton image, cuts it out and fills with bottom_only image
        If from_skeleton==False,
            takes bottom_only image, cuts it out and fills with skeleton image
        Args:
            from_skeleton (bool, optional): Defaults to True.
            patch_size (either int or list of int): Defaults to None.
            random_size (bool, optional): Defaults to False.
            inplace (bool, optional): Defaults to False.
            localization ([type], optional): Defaults to None.
        """
        self.patch_size = rotate_list(patch_size)
        self.random_size = random_size
        self.inplace = inplace
        self.localization = localization
        self.from_skeleton = from_skeleton

    def __call__(self, tensor):

        arr = tensor.numpy()
        img_shape = np.array(arr.shape)
        if isinstance(self.patch_size, int):
            size = [self.patch_size for _ in range(len(img_shape))]
        else:
            size = np.copy(self.patch_size)
        assert len(size) == len(img_shape), "Incorrect patch dimension."
        indexes = []
        for ndim in range(len(img_shape)):
            if size[ndim] > img_shape[ndim] or size[ndim] < 0:
                size[ndim] = img_shape[ndim]
            if self.random_size:
                size[ndim] = np.random.randint(0, size[ndim])
            if self.localization is not None:
                delta_before = max(
                    self.localization[ndim] - size[ndim] // 2, 0)
            else:
                np.random.seed()
                delta_before = np.random.randint(
                    0, img_shape[ndim] - size[ndim] + 1)
            indexes.append(slice(int(delta_before),
                                 int(delta_before + size[ndim])))
        if self.from_skeleton:
            if self.inplace:
                arr_cut = arr[tuple(indexes)]
                arr[tuple(indexes)] = arr_cut * (arr_cut == 30)
                return torch.from_numpy(arr)
            else:
                arr_copy = np.copy(arr)
                arr_cut = arr_copy[tuple(indexes)]
                arr_copy[tuple(indexes)] = arr_cut * (arr_cut == 30)
                return torch.from_numpy(arr_copy)
        else:
            arr_bottom = arr * (arr == 30)
            arr_cut = arr[tuple(indexes)]
            arr_bottom[tuple(indexes)] = np.copy(arr_cut)
            return torch.from_numpy(arr_bottom)


class CutoutTensor(object):
    """Apply a cutout on the images
    cf. Improved Regularization of Convolutional Neural Networks with Cutout,
    arXiv, 2017
    We assume that the cube to be cut is inside the image.
    """

    def __init__(self, patch_size=None, value=0, random_size=False,
                 inplace=False, localization=None):
        self.patch_size = patch_size
        self.value = value
        self.random_size = random_size
        self.inplace = inplace
        self.localization = localization

    def __call__(self, arr):

        img_shape = np.array(arr.shape)
        if isinstance(self.patch_size, int):
            size = [self.patch_size for _ in range(len(img_shape))]
        else:
            size = np.copy(self.patch_size)
        assert len(size) == len(img_shape), "Incorrect patch dimension."
        indexes = []
        for ndim in range(len(img_shape)):
            if size[ndim] > img_shape[ndim] or size[ndim] < 0:
                size[ndim] = img_shape[ndim]
            if self.random_size:
                size[ndim] = np.random.randint(0, size[ndim])
            if self.localization is not None:
                delta_before = max(
                    self.localization[ndim] - size[ndim] // 2, 0)
            else:
                delta_before = np.random.randint(
                    0, img_shape[ndim] - size[ndim] + 1)
            indexes.append(slice(int(delta_before),
                                 int(delta_before + size[ndim])))

        if self.inplace:
            arr[tuple(indexes)] = self.value
            return torch.from_numpy(arr)
        else:
            arr_cut = np.copy(arr)
            arr_cut[tuple(indexes)] = self.value
            return torch.from_numpy(arr_cut)


class ToPointnetTensor(object):

    def __init__(self, padding_method=repeat_padding, n_max=None):
        self.padding_method = padding_method
        self.n_max = n_max

    def __call__(self, tensor):
        arr = tensor.numpy()

        clouds = []
        for i in range(arr.shape[0]):  # loop over batch elements
            point_cloud = np.array(arr[i].nonzero()[:3])
            clouds.append(point_cloud)

        padded_clouds = pad(clouds, padding_method=self.padding_method,
                            n_max=self.n_max)

        return torch.from_numpy(padded_clouds)


def interval(obj, lower=None):
    """ Listify an object.
    Parameters
o   ----------
    obj: 2-uplet or number
        the object used to build the interval.
    lower: number, default None
        the lower bound of the interval. If not specified, a symetric
        interval is generated.
    Returns
    -------
    interval: 2-uplet
        an interval.
    """
    if isinstance(obj, numbers.Number):
        if obj < 0:
            raise ValueError("Specified interval value must be positive.")
        if lower is None:
            lower = -obj
        return (lower, obj)
    if len(obj) != 2:
        raise ValueError("Interval must be specified with 2 values.")
    min_val, max_val = obj
    if min_val > max_val:
        raise ValueError("Wrong interval boudaries.")
    return tuple(obj)


class Transformer(object):
    """ Class that can be used to register a sequence of transformations.
    """
    Transform = namedtuple("Transform", ["transform", "probability"])

    def __init__(self):
        """ Initialize the class.
        """
        self.transforms = []

    def register(self, transform, probability=1):
        """ Register a new transformation.
        Parameters
        ----------
        transform: callable
            the transformation object.
        probability: float, default 1
            the transform is applied with the specified probability.
        """
        trf = self.Transform(transform=transform, probability=probability, )
        self.transforms.append(trf)

    def __call__(self, arr):
        """ Apply the registered transformations.
        """
        transformed = arr.copy()
        for trf in self.transforms:
            if np.random.rand() < trf.probability:
                transformed = trf.transform(transformed)
        return transformed

    def __str__(self):
        if len(self.transforms) == 0:
            return '(Empty Transformer)'
        s = 'Composition of:'
        for trf in self.transforms:
            s += '\n\t- ' + trf.__str__()
        return s


class ResizeTensor(object):
    """Apply resize to a 3D image
    """

    def __init__(self, resize_ratio):
        self.resize_ratio = resize_ratio

    def __call__(self, tensor):
        arr = tensor.numpy()
        resized_arr = np.copy(arr)
        log.debug(f"Resize Ratio: {self.resize_ratio}")
        resized_arr = zoom(resized_arr,
                           self.resize_ratio,
                           order=0)

        return torch.from_numpy(resized_arr)


class GaussianNoiseTensor(object):
    """Add gaussian noise to a 3D image."""

    def __init__(self, sigma):
        self.sigma = sigma
    
    def __call__(self, tensor):
        noise = torch.randn(tensor.shape)
        return tensor + self.sigma * noise