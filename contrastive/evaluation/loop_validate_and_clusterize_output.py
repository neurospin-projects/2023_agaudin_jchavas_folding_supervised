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
This program launches postprocessing_results for each training subfolder
"""
import argparse
import glob
import os
import sys
import inspect
import six

from SimCLR import evaluation


def parse_args(argv):
    """Parses command-line arguments

    Args:
        argv: a list containing command line arguments

    Returns:
        args
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog='loop_validate_and_clusterize.py',
        description='Analyses all output subfolders')
    parser.add_argument(
        "-s", "--src_dir", type=str, required=True,
        help='Source directory where deep learning results lie.')
    parser.add_argument(
        "-c", "--csv_file", type=str, required=True,
        help='csv file on which is done the evaluation.')

    args = parser.parse_args(argv)

    return args


def loop_over_directory(src_dir, csv_file):
    """Loops over deep learning directories
    """
    # Gets and creates all filenames
    dirnames = glob.glob(f"{src_dir}/*")
    for deep_dir in dirnames:
        deep_dir = os.path.abspath(deep_dir)
        analysis_path = f"{deep_dir}"
        checkpoint_file = glob.glob(
            f"{deep_dir}/logs/default/version_0/checkpoints/*.ckpt")
        checkpoint_file = os.path.abspath(checkpoint_file[0])
        checkpoint_path = f"'\"{checkpoint_file}\"'"
        config_path = f"{deep_dir}/.hydra"
        prog_path = os.path.dirname(inspect.getabsfile(evaluation))
        cmd = f"python3 {prog_path}/validate_and_clusterize_output.py " \
            f"+analysis_path={analysis_path} " \
            f"checkpoint_path={checkpoint_path} " \
            f"train_val_csv_file={csv_file} "\
            f"--config-path={config_path}"

        print(cmd)
        os.system(cmd)


def main(argv):
    """Reads argument line and launches validate_and_clusterize on each

    Args:
        argv: a list containing command line arguments
    """

    # This code permits to catch SystemExit with exit code 0
    # such as the one raised when "--help" is given as argument
    try:
        # Parsing arguments
        args = parse_args(argv)
        loop_over_directory(args.src_dir, args.csv_file)
    except SystemExit as exc:
        if exc.code != 0:
            six.reraise(*sys.exc_info())


if __name__ == '__main__':
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])

    # example of use
    # python3 loop_validate_and_clusterize_output.py -s ../../../Output/t-0.1
