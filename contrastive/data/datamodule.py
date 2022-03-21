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
# software by the user in light of its specific status of fproducing the
# software by the user in light of its specific status of  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license version 2 and that you accept its terms.
""" Data module
"""
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

from SimCLR.data.datasets import create_sets


class DataModule(pl.LightningDataModule):
    """Data module class
    """

    def __init__(self, config):
        super(DataModule, self).__init__()
        self.config = config

    def setup(self, stage=None, mode=None):
        self.dataset_train, self.dataset_val, self.dataset_test, _ = \
            create_sets(self.config)

    def train_dataloader(self):
        loader_train = DataLoader(self.dataset_train,
                                  batch_size=self.config.batch_size,
                                  sampler=RandomSampler(self.dataset_train),
                                  pin_memory=self.config.pin_mem,
                                  num_workers=self.config.num_cpu_workers
                                  )
        return loader_train

    def val_dataloader(self):
        loader_val = DataLoader(self.dataset_val,
                                batch_size=self.config.batch_size,
                                pin_memory=self.config.pin_mem,
                                num_workers=self.config.num_cpu_workers,
                                shuffle=False
                                )
        return loader_val

    def test_dataloader(self):
        loader_test = DataLoader(self.dataset_test,
                                 batch_size=self.config.batch_size,
                                 pin_memory=self.config.pin_mem,
                                 num_workers=self.config.num_cpu_workers,
                                 shuffle=False
                                 )
        return loader_test


class DataModule_Visualization(pl.LightningDataModule):
    """Data module class for visualization
    """

    def __init__(self, config):
        super(DataModule_Visualization, self).__init__()
        self.config = config

    def setup(self, stage, mode=None):
        self.dataset_train, self.dataset_val, self.dataset_test,\
            self.dataset_train_val = \
            create_sets(self.config, mode='visualization')

    def train_val_dataloader(self):
        loader_train = DataLoader(self.dataset_train_val,
                                  batch_size=self.config.batch_size,
                                  pin_memory=self.config.pin_mem,
                                  num_workers=self.config.num_cpu_workers,
                                  shuffle=False
                                  )
        return loader_train

    def train_dataloader(self):
        loader_train = DataLoader(self.dataset_train,
                                  batch_size=self.config.batch_size,
                                  pin_memory=self.config.pin_mem,
                                  num_workers=self.config.num_cpu_workers,
                                  shuffle=False
                                  )
        return loader_train

    def val_dataloader(self):
        loader_val = DataLoader(self.dataset_val,
                                batch_size=self.config.batch_size,
                                pin_memory=self.config.pin_mem,
                                num_workers=self.config.num_cpu_workers,
                                shuffle=False
                                )
        return loader_val

    def test_dataloader(self):
        loader_test = DataLoader(self.dataset_test,
                                 batch_size=self.config.batch_size,
                                 pin_memory=self.config.pin_mem,
                                 num_workers=self.config.num_cpu_workers,
                                 shuffle=False
                                 )
        return loader_test
