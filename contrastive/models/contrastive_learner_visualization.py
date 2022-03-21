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
"""
A test to just analyze randomly generated input images
"""
import torch

from SimCLR.losses import NTXenLoss
from SimCLR.models.contrastive_learner import ContrastiveLearner
from SimCLR.utils.plots.visualize_anatomist import Visu_Anatomist


class ContrastiveLearner_Visualization(ContrastiveLearner):

    def __init__(self, config, sample_data):
        super(ContrastiveLearner_Visualization, self).__init__(
            config=config, sample_data=sample_data)
        self.config = config
        self.sample_data = sample_data
        self.sample_i = []
        self.sample_j = []
        self.val_sample_i = []
        self.val_sample_j = []
        self.recording_done = False
        self.visu_anatomist = Visu_Anatomist()

    def custom_histogram_adder(self):

        # iterating through all parameters
        for name, params in self.named_parameters():

            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.config.lr,
                                     weight_decay=self.config.weight_decay)
        return optimizer

    def nt_xen_loss(self, z_i, z_j):
        loss = NTXenLoss(temperature=self.config.temperature,
                         return_logits=True)
        return loss.forward(z_i, z_j)

    def training_step(self, train_batch, batch_idx):
        (inputs, filenames) = train_batch
        if batch_idx == 0:
            self.sample_i.append(inputs[:, 0, :].cpu())
            self.sample_j.append(inputs[:, 1, :].cpu())

    def training_epoch_end(self, outputs):
        image_input_i = self.visu_anatomist.plot_bucket(
            self.sample_i, buffer=True)
        self.logger.experiment.add_image(
            'input_test_i', image_input_i, self.current_epoch)
        image_input_j = self.visu_anatomist.plot_bucket(
            self.sample_j, buffer=True)
        self.logger.experiment.add_image(
            'input_test_j', image_input_j, self.current_epoch)
