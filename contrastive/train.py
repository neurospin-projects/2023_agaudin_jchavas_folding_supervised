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
# os.environ['MPLCONFIGDIR'] = os.getcwd()+'/.config_mpl'

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
from contrastive.utils.logs import set_root_logger_level, \
    set_file_log_handler, set_file_logger

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


@hydra.main(config_name='config', version_base="1.1", config_path="configs")
def train(config):
    config = process_config(config)

    # set the number of working cpus
    available_cpus = len(os.sched_getaffinity(0))
    log.debug('Available working cpus:', available_cpus)
    n_cpus = min(available_cpus, config.num_cpu_workers)
    os.environ["NUMEXPR_MAX_THREADS"] = str(n_cpus)
    log.debug('NUMEXPR_MAX_THREADS', n_cpus)

    set_root_logger_level(config.verbose)
    # Sets handler for logger
    set_file_log_handler(file_dir=os.getcwd(),
                         suffix='output')
    log.debug(f"current directory = {os.getcwd()}")

    # copies some of the config parameters in a yaml file easily accessible
    keys_to_keep = ['datasets', 'nb_subjects', 'model', 'with_labels',
                    'input_size', 'temperature_initial', 'temperature',
                    'sigma', 'drop_rate', 'mode', 'both', 'foldlabel',
                    'resize', 'patch_size', 'max_angle',
                    'checkerboard_size', 'keep_bottom',
                    'growth_rate', 'block_config', 'num_init_features',
                    'num_representation_features', 'num_outputs',
                    'environment', 'batch_size', 'pin_mem', 'partition',
                    'lr', 'weight_decay', 'max_epochs',
                    'early_stopping_patience', 'random_state', 'seed',
                    'backbone_name', 'sigma_labels', 'label_names',
                    'proportion_pure_contrastive', 'percentage', 
                    'projection_head_name']
    if config.model == 'SimCLR_supervised':
        keys_to_keep.extend(
            ['temperature_supervised',
             'sigma_labels',
             'pretrained_model_path'])

    create_accessible_config(keys_to_keep, os.getcwd() + "/.hydra/config.yaml")

    # create a csv file where the parameters changing between runs are stored
    get_config_diff(os.getcwd() + '/..', whole_config=False, save=True)

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

    # load pretrained model's weights if in config
    if (
        'pretrained_model_path' in config.keys()
        and config.pretrained_model_path is not None
    ):
        log.info(f"Load weigths stored at {config.pretrained_model_path}")
        model.load_pretrained_model(config.pretrained_model_path,
                                    encoder_only=config.load_encoder_only)

    if config.backbone_name != 'pointnet':
        summary(model, None, device=config.device)
    else:
        summary(model, device='cpu')

    early_stop_callback = \
        EarlyStopping(monitor="val_loss",
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
    log.info("Number of hooks: "
             f"{len(model.save_output.outputs)} ; "
             f"{len(model.hook_handles)}")

    # save model with structure
    # save_path = './logs/trained_model.pt'
    # torch.save(model, save_path)
    # print(f"Full model successfully saved at {os.path.abspath(save_path)}.")

    print(f"End of training for model {os.path.abspath('./')}")


if __name__ == "__main__":
    train()
