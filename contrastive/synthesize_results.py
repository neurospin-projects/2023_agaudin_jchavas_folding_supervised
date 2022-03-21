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
""" Validates and clusterizes

"""
######################################################################
# Imports and global variables definitions
######################################################################
import sys
import argparse
import six
from os.path import abspath
from SimCLR.evaluation.loop_validate_and_clusterize import loop_over_directory
from SimCLR.evaluation.plot_loss_silhouette_score_vs_latent_dimension import plot_loss_silhouette_score

def parse_args(argv):
    """Parses command-line arguments

    Args:
        argv: a list containing command line arguments

    Returns:
        args
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog='synthesize_results.py',
        description='Analyzes all deep lerning subdirectories')
    parser.add_argument(
        "-s", "--src_dir", type=str, required=True,
        help='Source deep learning directory.')
    parser.add_argument(
        "-c", "--csv_file", type=str, required=True,
        help='csv file on which is done the evaluation.')

    args = parser.parse_args(argv)

    return args


def main(argv):
    """Reads argument line and launches postprocessing_results on each

    Args:
        argv: a list containing command line arguments
    """

    # This code permits to catch SystemExit with exit code 0
    # such as the one raised when "--help" is given as argument
    try:
        # Parsing arguments
        args = parse_args(argv)
        src_dir = abspath(args.src_dir)
        csv_file = abspath(args.csv_file)

        loop_over_directory(src_dir, csv_file)
        plot_loss_silhouette_score(src_dir)
    except SystemExit as exc:
        if exc.code != 0:
            six.reraise(*sys.exc_info())


if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])
