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
Some helper functions are taken from:
https://learnopencv.com/tensorboard-with-pytorch-lightning

"""
import numpy as np
import torch
from sklearn.manifold import TSNE
from toolz.itertoolz import first

from contrastive.backbones.densenet import DenseNet
from contrastive.losses import GeneralizedSupervisedNTXenLoss
from contrastive.losses import CrossEntropyLoss
from contrastive.utils.plots.visualize_images import plot_bucket
from contrastive.utils.plots.visualize_images import plot_histogram
from contrastive.utils.plots.visualize_images import plot_histogram_weights
from contrastive.utils.plots.visualize_images import plot_scatter_matrix
from contrastive.utils.plots.visualize_images \
    import plot_scatter_matrix_with_labels
from contrastive.utils.plots.visualize_tsne import plot_tsne

try:
    from contrastive.utils.plots.visualize_anatomist import Visu_Anatomist
except ImportError:
    print("INFO: you are probably not in a brainvisa environment. Probably OK.")


class SaveOutput:
    def __init__(self):
        self.outputs = {}

    def __call__(self, module, module_in, module_out):
        self.outputs[module] = module_out.cpu()

    def clear(self):
        self.outputs = {}


class ContrastiveLearner_WithLabels(DenseNet):

    def __init__(self, config, sample_data):
        super(ContrastiveLearner_WithLabels, self).__init__(
            growth_rate=config.growth_rate,
            block_config=config.block_config,
            num_init_features=config.num_init_features,
            num_representation_features=config.num_representation_features,
            num_outputs=config.num_outputs,
            mode=config.mode,
            drop_rate=config.drop_rate,
            in_shape=config.input_size,
            depth=config.depth_decoder)
        self.config = config
        self.sample_data = sample_data
        self.sample_i = np.array([])
        self.sample_j = np.array([])
        self.save_output = SaveOutput()
        self.hook_handles = []
        self.get_layers()
        if self.config.environment == "brainvisa":
            self.visu_anatomist = Visu_Anatomist()

    def get_layers(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                handle = layer.register_forward_hook(self.save_output)
                self.hook_handles.append(handle)

    def custom_histogram_adder(self):
        """Builds histogram for each model parameter.
        """
        # iterating through all parameters
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name,
                params,
                self.current_epoch)

    def configure_optimizers(self):
        """Adam optimizer"""
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                     lr=self.config.lr,
                                     weight_decay=self.config.weight_decay)
        return optimizer

    def generalized_supervised_nt_xen_loss(self, z_i, z_j, labels):
        """Loss function for contrastive"""
        temperature = max(self.config.temperature,
                          self.config.temperature_initial
                          - self.current_epoch/50. *
                          (self.config.temperature_initial - self.config.temperature))

        loss = GeneralizedSupervisedNTXenLoss(
            temperature=temperature,
            sigma=self.config.sigma_labels,
            proportion_pure_contrastive=self.config.proportion_pure_contrastive,
            return_logits=True)
        return loss.forward(z_i, z_j, labels)

    def cross_entropy_loss(self, sample, output_i, output_j):
        """Loss function for decoder"""
        loss = CrossEntropyLoss(device=self.device)
        return loss.forward(sample, output_i, output_j)

    def training_step(self, train_batch, batch_idx):
        """Training step.
        """
        (inputs, labels, filenames) = train_batch
        input_i = inputs[:, 0, :]
        input_j = inputs[:, 1, :]
        z_i = self.forward(input_i)
        z_j = self.forward(input_j)

        if self.config.mode == "decoder":
            sample = inputs[:, 2, :]
            batch_loss = self.cross_entropy_loss(sample, z_i, z_j)
        else:
            batch_loss, sim_zij, sim_zii, sim_zjj, correct_pair, weights = \
                self.generalized_supervised_nt_xen_loss(z_i, z_j, labels)

        self.log('train_loss', float(batch_loss))

        # Only computes graph on first step
        if self.global_step == 1:
            self.logger.experiment.add_graph(self, inputs[:, 0, :])

        # Records sample for first batch of each epoch
        if batch_idx == 0:
            self.sample_i = inputs[:, 0, :].cpu()
            self.sample_j = inputs[:, 1, :].cpu()
            if self.config.mode != "decoder":
                self.sim_zij = sim_zij * self.config.temperature
                self.sim_zii = sim_zii * self.config.temperature
                self.sim_zjj = sim_zjj * self.config.temperature
                self.weights = weights

        # logs - a dictionary
        logs = {"train_loss": float(batch_loss)}

        batch_dictionary = {
            # REQUIRED: It is required for us to return "loss"
            "loss": batch_loss,
            # optional for batch logging purposes
            "log": logs,
        }

        return batch_dictionary

    def compute_outputs_skeletons(self, loader):
        """Computes the outputs of the model for each crop.

        This includes the projection head"""

        # Initialization
        X = torch.zeros([0, self.config.num_outputs]).cpu()
        labels_all = torch.zeros([0, 1]).cpu()
        filenames_list = []

        # Computes embeddings without computing gradient
        with torch.no_grad():
            for (inputs, labels, filenames) in loader:
                # First views of the whole batch
                inputs = inputs.cuda()
                model = self.cuda()
                X_i = model.forward(inputs[:, 0, :])
                # Second views of the whole batch
                X_j = model.forward(inputs[:, 1, :])

                # We now concatenate the embeddings

                # First views and second views are put side by side
                X_reordered = torch.cat([X_i, X_j], dim=-1)
                # X_i and X_j elements are put in successin by index
                # X_i[0], X_j[0], X_i[1], X_j[1],... X_i[N], X_j[N]
                # N being the number of samples in the batch
                X_reordered = X_reordered.view(-1, X_i.shape[-1])
                # At the end, it is concataneted with previous X
                X = torch.cat((X, X_reordered.cpu()), dim=0)

                # We now concatenate the labels

                labels_reordered = torch.cat([labels, labels], dim=-1)
                labels_reordered = labels_reordered.view(-1, labels.shape[-1])
                # At the end, labels are concatenated
                labels_all = torch.cat((labels_all, labels_reordered.cpu()),
                                       dim=0)

                filenames_duplicate = [
                    item for item in filenames
                    for repetitions in range(2)]
                filenames_list = filenames_list + filenames_duplicate
                del inputs

        return X, labels_all, filenames_list

    def compute_decoder_outputs_skeletons(self, loader):
        """Computes the outputs of the model for each crop.

        This includes the projection head"""

        # Initialization
        X = torch.zeros([0, 2, 20, 40, 40]).cpu()
        filenames_list = []

        # Computes embeddings without computing gradient
        with torch.no_grad():
            for (inputs, labels, filenames) in loader:
                # First views of the whole batch
                inputs = inputs.cuda()
                model = self.cuda()
                X_i = model.forward(inputs[:, 0, :])
                # First views re put side by side
                X = torch.cat((X, X_i.cpu()), dim=0)
                filenames_duplicate = [item
                                       for item in filenames]
                filenames_list = filenames_list + filenames_duplicate
                del inputs

        return X, filenames_list

    def compute_representations(self, loader):
        """Computes representations for each crop.

        Representation are before the projection head"""

        # Initialization
        X = torch.zeros([0, self.config.num_representation_features]).cpu()
        labels_all = torch.zeros([0, 1]).cpu()
        filenames_list = []

        # Computes representation (without gradient computation)
        with torch.no_grad():
            for (inputs, labels, filenames) in loader:
                # We first compute the embeddings
                # for the first views of the whole batch
                inputs = inputs.cuda()
                model = self.cuda()
                model.forward(inputs[:, 0, :])
                X_i = first(self.save_output.outputs.values())

                # We then compute the embeddings for the second views
                # of the whole batch
                model.forward(inputs[:, 1, :])
                X_j = first(self.save_output.outputs.values())

                # We now concatenate the embeddings

                # First views and second views are put side by side
                X_reordered = torch.cat([X_i, X_j], dim=-1)
                # X_i and X_j elements are put in successin by index
                # X_i[0], X_j[0], X_i[1], X_j[1],... X_i[N], X_j[N]
                # N being the number of samples in the batch
                X_reordered = X_reordered.view(-1, X_i.shape[-1])
                # At the end, it is concataneted with previous X
                X = torch.cat((X, X_reordered.cpu()), dim=0)

                # We now concatenate the labels

                labels_reordered = torch.cat([labels, labels], dim=-1)
                labels_reordered = labels_reordered.view(-1, labels.shape[-1])
                # At the end, labels are concatenated
                labels_all = torch.cat((labels_all, labels_reordered.cpu()),
                                       dim=0)

                filenames_duplicate = [
                    item for item in filenames
                    for repetitions in range(2)]
                filenames_list = filenames_list + filenames_duplicate
                del inputs

        return X, labels_all, filenames_list

    def compute_tsne(self, loader, register):
        """Computes t-SNE.

        It is computed either in the representation
        or in the output space"""

        if register == "output":
            X, _, _ = self.compute_outputs_skeletons(loader)
        elif register == "representation":
            X, _, _ = self.compute_representations(loader)
        else:
            raise ValueError(
                "Argument register must be either output or representation")

        tsne = TSNE(n_components=2, perplexity=25, init='pca', random_state=50)

        Y = X.detach().numpy()

        # Makes the t-SNE fit
        X_tsne = tsne.fit_transform(Y)

        # Returns tsne embeddings
        return X_tsne

    def training_epoch_end(self, outputs):
        """Computation done at the end of the epoch"""

        if self.config.mode == "encoder":
            # Computes t-SNE both in representation and output space
            if self.current_epoch % self.config.nb_epochs_per_tSNE == 0 \
                    or self.current_epoch >= self.config.max_epochs:
                X_tsne = self.compute_tsne(
                    self.sample_data.train_dataloader(), "output")
                image_TSNE = plot_tsne(X_tsne, buffer=True)
                self.logger.experiment.add_image(
                    'TSNE output image', image_TSNE, self.current_epoch)
                X_tsne = self.compute_tsne(
                    self.sample_data.train_dataloader(), "representation")
                image_TSNE = plot_tsne(X_tsne, buffer=True)
                self.logger.experiment.add_image(
                    'TSNE representation image', image_TSNE, self.current_epoch)

            # Makes scatter matrix of output space
            X, labels, _ = self.compute_outputs_skeletons(
                self.sample_data.train_dataloader())
            scatter_matrix_outputs = plot_scatter_matrix(X, buffer=True)
            self.logger.experiment.add_image(
                'scatter_matrix_outputs',
                scatter_matrix_outputs,
                self.current_epoch)

            # Makes scatter matrix of output space with label values
            scatter_matrix_outputs_with_labels = \
                plot_scatter_matrix_with_labels(X, labels, buffer=True)
            self.logger.experiment.add_image(
                'scatter_matrix_outputs_with_labels',
                scatter_matrix_outputs_with_labels,
                self.current_epoch)

            # Makes scatter matrix of representation space
            X, labels, _ = self.compute_representations(
                self.sample_data.train_dataloader())
            scatter_matrix_representations = plot_scatter_matrix(
                X, buffer=True)
            self.logger.experiment.add_image(
                'scatter_matrix_representations',
                scatter_matrix_representations,
                self.current_epoch)

            # Makes scatter matrix of representation space with label values
            scatter_matrix_representations_with_labels = \
                plot_scatter_matrix_with_labels(X, labels, buffer=True)
            self.logger.experiment.add_image(
                'scatter_matrix_representations_with_labels',
                scatter_matrix_representations_with_labels,
                self.current_epoch)

            # Computes histogram of sim_zii
            histogram_sim_zii = plot_histogram(self.sim_zii, buffer=True)
            self.logger.experiment.add_image(
                'histo_sim_zii', histogram_sim_zii, self.current_epoch)

            # Computes histogram of sim_zjj
            histogram_sim_zjj = plot_histogram(self.sim_zjj, buffer=True)
            self.logger.experiment.add_image(
                'histo_sim_zjj', histogram_sim_zjj, self.current_epoch)

            # Computes histogram of sim_zij
            histogram_sim_zij = plot_histogram(self.sim_zij, buffer=True)
            self.logger.experiment.add_image(
                'histo_sim_zij', histogram_sim_zij, self.current_epoch)

            # Computes histogram of weights
            histogram_weights = plot_histogram_weights(self.weights,
                                                       buffer=True)
            self.logger.experiment.add_image(
                'histo_weights', histogram_weights, self.current_epoch)

        # Plots views
        image_input_i = plot_bucket(self.sample_i, buffer=True)
        self.logger.experiment.add_image(
            'input_i', image_input_i, self.current_epoch)
        image_input_j = plot_bucket(self.sample_j, buffer=True)
        self.logger.experiment.add_image(
            'input_j', image_input_j, self.current_epoch)

        # Plots view using anatomist
        if self.config.environment == "brainvisa":
            image_input_i = self.visu_anatomist.plot_bucket(
                self.sample_i, buffer=True)
            self.logger.experiment.add_image(
                'input_ana_i', image_input_i, self.current_epoch)
            image_input_j = self.visu_anatomist.plot_bucket(
                self.sample_j, buffer=True)
            self.logger.experiment.add_image(
                'input_ana_j', image_input_j, self.current_epoch)

        # calculates average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # logs histograms
        # self.custom_histogram_adder()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar(
            "Loss/Train",
            avg_loss,
            self.current_epoch)

    def validation_step(self, val_batch, batch_idx):
        """Validation step"""

        (inputs, labels, filenames) = val_batch
        input_i = inputs[:, 0, :]
        input_j = inputs[:, 1, :]

        z_i = self.forward(input_i)
        z_j = self.forward(input_j)

        if self.config.mode == "decoder":
            sample = inputs[:, 2, :]
            batch_loss = self.cross_entropy_loss(sample, z_i, z_j)
        else:
            batch_loss, sim_zij, sim_sii, sim_sjj, correct_pairs, weights = \
                self.generalized_supervised_nt_xen_loss(z_i, z_j, labels)

        self.log('val_loss', float(batch_loss))

        # logs- a dictionary
        logs = {"val_loss": float(batch_loss)}

        batch_dictionary = {
            # REQUIRED: It is required for us to return "loss"
            "loss": batch_loss,

            # optional for batch logging purposes
            "log": logs,
        }

        return batch_dictionary

    def validation_epoch_end(self, outputs):
        """Computaion done at the end of each validation epoch"""

        # Computes t-SNE
        if self.config.mode == "encoder":
            if self.current_epoch % self.config.nb_epochs_per_tSNE == 0 \
                    or self.current_epoch >= self.config.max_epochs:
                X_tsne = self.compute_tsne(
                    self.sample_data.val_dataloader(), "output")
                image_TSNE = plot_tsne(X_tsne, buffer=True)
                self.logger.experiment.add_image(
                    'TSNE output validation image', image_TSNE, self.current_epoch)
                X_tsne = self.compute_tsne(
                    self.sample_data.val_dataloader(),
                    "representation")
                image_TSNE = plot_tsne(X_tsne, buffer=True)
                self.logger.experiment.add_image(
                    'TSNE representation validation image',
                    image_TSNE,
                    self.current_epoch)

            # Makes scatter matrix of representation space
            X, labels, _ = self.compute_representations(
                self.sample_data.val_dataloader())

            # Makes scatter matrix of representation space with label values
            scatter_matrix_representations_with_labels = \
                plot_scatter_matrix_with_labels(X, labels, buffer=True)
            self.logger.experiment.add_image(
                'scatter_matrix_representations_with_labels_validation',
                scatter_matrix_representations_with_labels,
                self.current_epoch)

        # calculates average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # logs losses using tensorboard logger
        self.logger.experiment.add_scalar(
            "Loss/Validation",
            avg_loss,
            self.current_epoch)
