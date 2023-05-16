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

import logging
import sys
from os import makedirs
from os.path import basename

# Default logging level of full_model
LOGGING_LEVEL = logging.INFO
LOG_FORMAT = "%(levelname)s:%(name)s: %(message)s"

# Sets up the default logger
logging.basicConfig(
    level=LOGGING_LEVEL,
    handlers=[
        logging.StreamHandler(sys.stderr)
    ])
formatter = logging.Formatter(LOG_FORMAT)
log_full_model = logging.getLogger('')
for hdlr in log_full_model.handlers[:]:
    hdlr.setFormatter(formatter)


def set_root_logger_level(verbose_level):
    """Sets root logger level

    if verbose_level is:
        - 0: logging.WARNING is selected
        - 1: logging.INFO is selected
        - >1: logging.DEBUG is selected

    Args:
        verbose_level: int giving verbose level"""
    levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(verbose_level, len(levels) - 1)]  # cap to last level
    root = logging.getLogger()
    root.setLevel(level)


def set_file_logger(path_file):
    """Returns specific file logger

    Args:
        path_file: string giving file name with path (__file__)

    Returns: file-specific logger
    """
    global log_full_model
    return log_full_model.getChild(basename(path_file))


def set_file_log_handler(file_dir, suffix):
    """Sets file handler for all logs.

    Args:
        file_dir: string with folder for file log
        suffix: string -> name of log file = log_{suffix}.log
    """
    global log_full_model
    global formatter

    # Creates filename
    if suffix:
        suffix = suffix.rstrip('.')
        file_name = f"{file_dir}/log_{suffix}.log"
    else:
        file_name = f"{file_dir}/log.log"

    # Creates handler
    makedirs(file_dir, exist_ok=True)
    filehandler = logging.FileHandler(file_name, mode='w')

    # Substitutes file handler in main logger
    for hdlr in log_full_model.handlers[:]:
        if isinstance(hdlr, logging.FileHandler):
            log_full_model.removeHandler(hdlr)
    log_full_model.addHandler(filehandler)
    for hdlr in log_full_model.handlers[:]:
        hdlr.setFormatter(formatter)

    # Logs name of log file
    simple_critical_log(log=log_full_model,
                        log_message=f"\nLog written to:\n{file_name}\n")


def simple_critical_log(log, log_message):
    """Prints simple log with only message printed out

    Args:
        log: logger
        log_message: string being log message to be printed
    """
    global log_full_model

    old_format = []
    for hdlr in log_full_model.handlers[:]:
        old_format.append(hdlr.formatter)
        new_formatter = logging.Formatter('%(message)s')
        hdlr.setFormatter(new_formatter)
    log.critical(log_message)
    for hdlr, form in zip(log_full_model.handlers[:], old_format):
        hdlr.setFormatter(form)
