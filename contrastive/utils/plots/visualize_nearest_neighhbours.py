import colorado as cld
import logging

import anatomist.api as anatomist
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from sklearn.neighbors import NearestNeighbors
from soma import aims

from .visualize_anatomist import Visu_Anatomist

"""Inspired from lightly
https://docs.lightly.ai/tutorials/package/tutorial_simclr_clothing.html
"""

log = logging.getLogger(__name__)
visu_anatomist = None


def get_image_as_np_array(filename: str):
    """Returns an image as an numpy array
    """
    img = PIL.Image.open(filename)
    return np.asarray(img)


def get_input(dataset, filenames, idx):
    """gets input numbered idx"""

    (views, filename) = dataset[idx // 2]
    if filenames:
        if filename != filenames[idx]:
            log.error(
                "filenames dont match: {} != {}".format(
                    filename, filenames[idx]))
    return views[idx % 2]


def plot_knn_examples(
        embeddings,
        dataset,
        filenames=None,
        n_neighbors=3,
        num_examples=6,
        savepath=None):
    """Plots multiple rows of random images with their nearest neighbors
    """
    # lets look at the nearest neighbors for some samples
    # we use the sklearn library
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # get 5 random samples
    samples_idx = np.random.choice(
        len(indices), size=num_examples, replace=False)

    # loop through our randomly picked samples
    for idx in samples_idx:
        fig = plt.figure()
        # loop through their nearest neighbors
        for plot_x_offset, neighbor_idx in enumerate(indices[idx]):
            # add the subplot
            ax = fig.add_subplot(1, len(indices[idx]), plot_x_offset + 1)
            # Recovers input
            view = get_input(dataset, filenames, neighbor_idx)
            # plot the image
            plt.imshow(view[0, view.shape[1] // 2, :, :].numpy())
            # set the title to the distance of the neighbor
            ax.set_title(f'd={distances[idx][plot_x_offset]:.3f}')
            # let's disable the axis
            plt.axis('off')
            if savepath:
                plt.savefig(f"{savepath}/nearest_neighbours_{idx}.png")

    if not savepath:
        plt.ion()
        plt.show()
        plt.pause(0.001)


def plot_knn_buckets(
        embeddings,
        dataset,
        filenames=None,
        n_neighbors=3,
        num_examples=6):
    """Plots multiple rows of random images with their nearest neighbors
    """
    # lets look at the nearest neighbors for some samples
    # we use the sklearn library
    global visu_anatomist
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # get 5 random samples
    view = get_input(dataset, filenames, 0)

    # Converts from tensor to aims volume
    arr = view.numpy()
    arr = np.reshape(arr, (1,) + arr.shape).astype(np.int16)
    if visu_anatomist is None:
        visu_anatomist = Visu_Anatomist()
    log.info(f"shape = {arr.shape}")
    visu_anatomist.plot_bucket(torch.from_numpy(arr),
                               buffer=False)

    # block = a.AWindowsBlock(a, n_neighbors)

    # # loop through our randomly picked samples
    # for idx in samples_idx:
    #     # loop through their nearest neighbors
    #     for plot_x_offset, neighbor_idx in enumerate(indices[idx]):
    #         # add the subplot
    #         ax = fig.add_subplot(1, len(indices[idx]), plot_x_offset + 1)
    #         # Recovers input
    #         view = get_input(dataset, filenames, neighbor_idx)
    #         # plot the image
    #         plt.imshow(view[0,view.shape[1]//2, :, :].numpy())
    #         # set the title to the distance of the neighbor
    #         ax.set_title(f'd={distances[idx][plot_x_offset]:.3f}')
    #         # let's disable the axis
    #         plt.axis('off')            win.imshow(show=False)


if __name__ == "__main__":
    n_samples = 20
    n_features = 10
    embeddings = np.random.rand(n_samples, n_features)
    plot_knn_examples(embeddings)
