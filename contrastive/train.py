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
""" Training contrastive on skeleton images

"""
######################################################################
# Imports and global variables definitions
######################################################################
import os

import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from contrastive.data.datamodule import DataModule_Learning
from contrastive.data.datamodule import DataModule_Evaluation
from contrastive.models.contrastive_learner import ContrastiveLearner
from contrastive.models.contrastive_learner_with_labels import \
    ContrastiveLearner_WithLabels
from contrastive.models.contrastive_learner_visualization import \
    ContrastiveLearner_Visualization
from contrastive.utils.config import create_accessible_config, process_config,\
get_config_diff
from contrastive.utils.logs import set_root_logger_level
from contrastive.utils.logs import set_file_log_handler
from contrastive.utils.logs import set_file_logger

tb_logger = pl_loggers.TensorBoardLogger('logs')
writer = SummaryWriter()
log = set_file_logger(__file__)

"""
We use the following definitions:
- embedding or representation, the space before the projection head.
  The elements of the space are features
- output, the space after the projection head.
  The elements are called output vectors
"""

@hydra.main(config_name='config', config_path="configs")
def train(config):
    config = process_config(config)
    os.environ["NUMEXPR_MAX_THREADS"] = str(config.num_cpu_workers)

    set_root_logger_level(config.verbose)
    # Sets handler for logger
    set_file_log_handler(file_dir=os.getcwd(),
                         suffix='output')
    log.info(f"current directory = {os.getcwd()}")


    # copies some of the config parameters in a yaml file easily accessible
    keys_to_keep = ['dataset_name', 'nb_subjects', 'model', 'with_labels', 
    'input_size', 'temperature_initial', 'temperature', 'sigma', 'drop_rate', 'depth_decoder',
    'mode', 'foldlabel', 'fill_value', 'patch_size', 'max_angle', 'checkerboard_size', 'keep_bottom',
    'growth_rate', 'block_config', 'num_init_features', 'num_representation_features', 'num_outputs',
    'environment', 'batch_size', 'pin_mem', 'partition', 'lr', 'weight_decay', 'max_epochs',
    'early_stopping_patience', 'seed', 'backbone_name', 'n_max']

    create_accessible_config(keys_to_keep, os.getcwd()+"/.hydra/config.yaml")

    # create a csv file where the parameters changing between runs are stored
    get_config_diff(os.getcwd()+'/..', whole_config=False, save=True)    


    if config.mode == 'evaluation':
        data_module = DataModule_Evaluation(config)
    else:
        data_module = DataModule_Learning(config)

    if config.mode == 'evaluation':
        model = ContrastiveLearner_Visualization(config,
                               sample_data=data_module)   
    elif config.model == "SimCLR_supervised":
        model = ContrastiveLearner_WithLabels(config,
                               sample_data=data_module)
    elif config.model == 'SimCLR':
        model = ContrastiveLearner(config,
                               sample_data=data_module) 
    else:
        raise ValueError("Wrong combination of 'mode' and 'model'")


    if config.backbone_name != 'pointnet':
        summary(model, tuple(config.input_size), device="cpu")
    else:
        summary(model, device='cpu')

    early_stop_callback = EarlyStopping(monitor="val_loss",
          patience=config.early_stopping_patience)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=config.max_epochs,
        callbacks=[early_stop_callback],
        logger=tb_logger,
        flush_logs_every_n_steps=config.nb_steps_per_flush_logs,
        log_every_n_steps=config.log_every_n_steps)

    trainer.fit(model, data_module, ckpt_path=config.checkpoint_path)
    log.info("Fitting is done")
    log.info(f"Number of hooks: {len(model.save_output.outputs)} ; {len(model.hook_handles)}")

    save_path = './logs/trained_model.pt'
    torch.save(model, save_path)


if __name__ == "__main__":
    train()
