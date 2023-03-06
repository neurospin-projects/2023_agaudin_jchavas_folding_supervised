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
"""
This program plots the evolution of the loss and the silhouette score
as a function of latent space size
"""
import argparse
import glob
import json
import sys

import six
from matplotlib import pyplot as plt


def parse_args(argv):
    """Parses command-line arguments

    Args:
        argv: a list containing command line arguments

    Returns:
        args
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog='plot_all.py',
        description='Reads json output files and plots them')
    parser.add_argument(
        "-s", "--src_dir", type=str, required=True,
        help='Source directory where json files lie.')

    args = parser.parse_args(argv)

    return args


def plot_loss_silhouette_score(src_dir):
    """Loops over deep learning directories
    """
    # Gets and creates all filenames
    dirnames = sorted(glob.glob(f"{src_dir}/*"))
    latent_space_size = []
    AffinityPropagation = []
    val_loss = []
    for deep_dir in dirnames:
        with open(f"{deep_dir}/result.json", 'r') as json_file:
            result_dict = json.load(json_file)
            print(result_dict)
            print(type(result_dict))
            latent_space_size.append(result_dict["latent_space_size"])
            val_loss.append(result_dict["val_loss"])
            af_dict = result_dict['AffinityPropagation']
            AffinityPropagation.append(list(af_dict.items())[0][1])

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(
        latent_space_size,
        val_loss,
        c=color,
        marker='o',
        label='Validation loss')
    ax1.set_xlabel('Latent space size')
    ax1.set_ylabel('Loss', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend()

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Silhouette score', color=color)
    ax2.plot(
        latent_space_size,
        AffinityPropagation,
        c=color,
        marker='o',
        label='Silhouette score')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc=0)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title("Convergence and clustering")
    plt.show()


def main(argv):
    """Reads argument line and plots silhouette score on directory

    Args:
        argv: a list containing command line arguments
    """

    # This code permits to catch SystemExit with exit code 0
    # such as the one raised when "--help" is given as argument
    try:
        # Parsing arguments
        args = parse_args(argv)
        plot_loss_silhouette_score(args.src_dir)
    except SystemExit as exc:
        if exc.code != 0:
            six.reraise(*sys.exc_info())


if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])

    # example of use
    # python3 plot_loss_silhouette_score.py -s ../../../Output/t-0.1
