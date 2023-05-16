from numba import jit, cuda
import numpy as np
import torch
import scipy.ndimage
# to measure exec time
from timeit import default_timer as timer

from contrastive.augmentations import RotateTensor

rot_augm = RotateTensor(6)

# normal function to run on cpu


def func(skels):
    for i in range(skels.shape[0]):
        rot_skel = rot_augm(skels[i])

# function optimized to run on gpu


@jit(target_backend='cuda', forceobj=True)
def func2(skels):
    for i in range(skels.shape[0]):
        rot_skel = rot_augm(skels[i])


def plussain(a, n):
    for i in range(n):
        a[i] += 1


@jit(target_backend='cuda', forceobj=True)
def plussain2(a, n):
    for i in range(n):
        a[i] += 1


@jit(target_backend='cuda')
def plussain23(a, n):
    for i in range(n):
        a[i] += 1


def rotate(arr):
    rot_array = np.copy(arr)

    for axes in (0, 1), (0, 2), (1, 2):
        np.random.seed()
        angle = np.random.uniform(-6, 6)
        rot_array = scipy.ndimage.rotate(rot_array,
                                         angle=angle,
                                         axes=axes,
                                         order=0,
                                         reshape=False,
                                         mode='constant',
                                         cval=0)
    return rot_array


@jit(target_backend='cuda')
def rotate2(arr):
    rot_array = np.copy(arr)

    for axes in (0, 1), (0, 2), (1, 2):
        np.random.seed()
        angle = np.random.uniform(-6, 6)
        rot_array = scipy.ndimage.rotate(rot_array,
                                         angle=angle,
                                         axes=axes,
                                         order=0,
                                         reshape=False,
                                         mode='constant',
                                         cval=0)
    return rot_array


if __name__ == "__main__":
    skels = np.load(
        "/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/crops/2mm/CINGULATE/mask/subsets/Rskeleton_most_folded_551.npy")
    print(skels.shape)
    skels = torch.from_numpy(skels)
    start = timer()
    func(skels)
    print("without GPU:", timer()-start)

    skels = np.load(
        "/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/crops/2mm/CINGULATE/mask/subsets/Rskeleton_most_folded_551.npy")
    print(skels.shape)
    skels = torch.from_numpy(skels)
    start = timer()
    func2(skels)
    print("with GPU:", timer()-start)

    print("Plussains")

    n = 10000000
    a = np.ones(n, dtype=np.float64)
    start = timer()
    plussain(a, n)
    print("without GPU:", timer()-start)

    n = 10000000
    a = np.ones(n, dtype=np.float64)
    start = timer()
    plussain2(a, n)
    print("with GPU:", timer()-start)

    n = 10000000
    a = np.ones(n, dtype=np.float64)
    start = timer()
    plussain23(a, n)
    print("with GPU:", timer()-start)

    print("Simple rotates")

    skels = np.load(
        "/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/crops/2mm/CINGULATE/mask/subsets/Rskeleton_most_folded_551.npy")
    print(skels.shape)
    start = timer()
    for i in range(551):
        rotate(skels[i])
    print("without GPU:", timer()-start)

    skels = np.load(
        "/neurospin/dico/data/deep_folding/current/datasets/UkBioBank/crops/2mm/CINGULATE/mask/subsets/Rskeleton_most_folded_551.npy")
    print(skels.shape)
    start = timer()
    for i in range(551):
        rotate2(skels[i])
    print("with GPU:", timer()-start)
