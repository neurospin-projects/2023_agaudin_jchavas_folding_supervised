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
import json
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from toolz.itertoolz import first

from sklearn.metrics import r2_score

from contrastive.models.contrastive_learner import ContrastiveLearner
from contrastive.losses import GeneralizedSupervisedNTXenLoss,\
    CrossEntropyLoss_Classification
from contrastive.losses import MSELoss_Regression
from contrastive.utils.plots.visualize_images \
    import plot_scatter_matrix_with_labels
from contrastive.utils.plots.visualize_tsne import plot_tsne
from contrastive.evaluation.auc_score import regression_roc_auc_score

try:
    from soma import aims
except ImportError:
    print("INFO: you are not in a brainvisa environment. Probably OK.")

from contrastive.utils.logs import set_root_logger_level, set_file_logger
log = set_file_logger(__file__)


class SaveOutput:
    def __init__(self):
        self.outputs = {}

    def __call__(self, module, module_in, module_out):
        self.outputs[module] = module_out.cpu()

    def clear(self):
        self.outputs = {}


class ContrastiveLearner_WithLabels(ContrastiveLearner):

    def __init__(self, config, sample_data):
        super(ContrastiveLearner_WithLabels, self).__init__(
            config=config, sample_data=sample_data)

    def plot_scatter_matrices_with_labels(self, dataloader, key,
                                          mode="encoder"):
        """Plots scatter matrices with label values."""
        # Makes scatter matrix of output space
        r = self.compute_outputs_skeletons(dataloader)
        X = r[0]  # First element of tuple
        labels = r[1]  # Second element of tuple
        # Makes scatter matrix of output space with label values
        scatter_matrix_outputs_with_labels = \
            plot_scatter_matrix_with_labels(X, labels, buffer=True)
        self.logger.experiment.add_image(
            'scatter_matrix_outputs_with_labels_' + key,
            scatter_matrix_outputs_with_labels,
            self.current_epoch)
        if mode == "regresser":
            score = r2_score(labels, X)
        else:
            score = 0

        # Makes scatter matrix of representation space with label values
        r = self.compute_representations(dataloader)
        X = r[0]  # First element of tuple
        labels = r[1]  # Second element of tuple
        scatter_matrix_representations_with_labels = \
            plot_scatter_matrix_with_labels(X, labels, buffer=True)
        self.logger.experiment.add_image(
            'scatter_matrix_representations_with_labels_' + key,
            scatter_matrix_representations_with_labels,
            self.current_epoch)

        return score

    def generalized_supervised_nt_xen_loss(self, z_i, z_j, labels):
        """Loss function for contrastive"""
        # temperature = max(
        #     self.config.temperature,
        #     self.config.temperature_initial
        #     - self.current_epoch/50.
        #     * (self.config.temperature_initial - self.config.temperature))
        temperature = self.config.temperature
        temperature_supervised = self.config.temperature_supervised

        loss = GeneralizedSupervisedNTXenLoss(
            temperature=temperature,
            temperature_supervised=temperature_supervised,
            sigma=self.config.sigma_labels,
            proportion_pure_contrastive=self.config.proportion_pure_contrastive,
            return_logits=True)
        return loss.forward(z_i, z_j, labels)

    def cross_entropy_loss_classification(self, output_i, output_j, labels):
        """Loss function for decoder"""
        loss = CrossEntropyLoss_Classification(device=self.device)
        return loss.forward(output_i, output_j, labels)

    def mse_loss_regression(self, output_i, output_j, labels):
        """Loss function for decoder"""
        loss = MSELoss_Regression(device=self.device)
        return loss.forward(output_i, output_j, labels)

    def training_step(self, train_batch, batch_idx):
        """Training step.
        """
        (inputs, labels, filenames, view3) = train_batch
        input_i = inputs[:, 0, :]
        input_j = inputs[:, 1, :]
        z_i = self.forward(input_i)
        z_j = self.forward(input_j)

        if self.config.mode == "decoder":
            sample = inputs[:, 2, :]
            batch_loss = self.cross_entropy_loss(sample, z_i, z_j)
        elif self.config.mode == "classifier":
            batch_loss = self.cross_entropy_loss_classification(
                z_i, z_j, labels)
            batch_label_loss = torch.tensor(0.)
        elif self.config.mode == "regresser":
            batch_loss = self.mse_loss_regression(z_i, z_j, labels)
            batch_label_loss = torch.tensor(0.)
        else:
            batch_loss, batch_label_loss, \
                sim_zij, sim_zii, sim_zjj, correct_pair, weights = \
                self.generalized_supervised_nt_xen_loss(z_i, z_j, labels)

        self.log('train_loss', float(batch_loss))
        self.log('train_label_loss', float(batch_label_loss))

        # Only computes graph on first step
        if self.global_step == 1:
            self.logger.experiment.add_graph(self, inputs[:, 0, :])

        # Records sample for first batch of each epoch
        if batch_idx == 0:
            self.sample_i = input_i.cpu()
            self.sample_j = input_j.cpu()
            self.sample_k = view3.cpu()
            self.sample_filenames = filenames
            self.sample_labels = labels
            if self.config.environment == 'brainvisa' and self.config.checking:
                vol_file = \
                    f"{self.config.crop_dir}/{filenames[0]}" +\
                    f"{self.config.crop_file_suffix}"
                vol = aims.read(vol_file)
                self.sample_ref_0 = np.asarray(vol)
                if not np.array_equal(self.sample_ref_0[..., 0],
                                      self.sample_k[0, 0, ...]):
                    raise ValueError(
                        "Images files don't match!!!\n"
                        f"Subject name = {filenames[0]}\n"
                        f"Shape of reference file = "
                        f"{self.sample_ref_0[...,0].shape}\n"
                        f"Shape of file read from array = "
                        f"{self.sample_k[0,0,...].shape}\n"
                        f"Sum of reference file = "
                        f"{self.sample_ref_0.sum()}\n"
                        f"Sum of file read from array = "
                        f"{self.sample_k[0,...].sum()}")
            if (
                self.config.mode != "decoder"
                and self.config.mode != "classifier"
                and self.config.mode != "regresser"
            ):
                self.sim_zij = sim_zij * self.config.temperature
                self.sim_zii = sim_zii * self.config.temperature
                self.sim_zjj = sim_zjj * self.config.temperature
                self.weights = weights

        # logs - a dictionary
        logs = {"loss": float(batch_loss),
                "train_label_loss": float(batch_label_loss)}

        batch_dictionary = {
            # REQUIRED: It is required for us to return "loss"
            "loss": batch_loss,
            "label_loss": batch_label_loss,
            # optional for batch logging purposes
            "log": logs,
        }

        return batch_dictionary

    def compute_outputs_skeletons(self, loader):
        """Computes the outputs of the model for each crop.

        This includes the projection head"""

        # Initialization
        if self.config.mode == "regresser":
            num_outputs = 1
        elif self.config.mode == "classifier":
            num_outputs = 2
        else:
            num_outputs = self.config.num_representation_features
        X = torch.zeros([0, num_outputs]).cpu()
        labels_all = torch.zeros([0, len(self.config.label_names)]).cpu()
        filenames_list = []

        # Computes embeddings without computing gradient
        with torch.no_grad():
            for (inputs, labels, filenames, _) in loader:
                # First views of the whole batch
                inputs = inputs.cuda()
                model = self.cuda()
                log.debug("COMPUTE OUTPUTS SKELETONS")
                log.debug((inputs[:, 0, :] == inputs[:, 1, :]).all())
                X_i = model.forward(inputs[:, 0, :])
                # Second views of the whole batch
                X_j = model.forward(inputs[:, 1, :])

                # Reshape (necessary if num_outputs==1)
                X_i = X_i.reshape(X_i.shape[0], num_outputs)
                X_j = X_j.reshape(X_j.shape[0], num_outputs)

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

    def compute_output_probabilities(self, loader):
        if self.config.mode == 'classifier':
            X, labels_all, filenames_list = self.compute_output_skeletons(
                loader)
            # compute the mean of the two views' outputs
            X = (X[::2, ...] + X[1::2, ...]) / 2
            # remove the doubleing of labels
            labels = labels[::2]
            filenames_list = filenames_list[::2]
            X = nn.functional.softmax(X, dim=1)
            return X, labels_all, filenames_list
        else:
            raise ValueError(
                "The config.mode is not 'classifier'. "
                "You shouldn't compute probabilities with another mode.")

    def compute_output_auc(self, loader):
        X, labels, _ = self.compute_outputs_skeletons(loader)
        # compute the mean of the two views' outputs
        X = (X[::2, ...] + X[1::2, ...]) / 2
        # remove the doubleing of labels
        labels = labels[::2]
        if self.config.mode == "regresser":
            auc = regression_roc_auc_score(labels, X[:, 0])
        else:
            X = nn.functional.softmax(X, dim=1)
            auc = roc_auc_score(labels, X[:, 1])

        return auc

    def compute_decoder_outputs_skeletons(self, loader):
        """Computes the outputs of the model for each crop.

        This includes the projection head"""

        # Initialization
        X = torch.zeros([0, 2, 20, 40, 40]).cpu()
        filenames_list = []

        # Computes embeddings without computing gradient
        with torch.no_grad():
            for (inputs, labels, filenames, _) in loader:
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
        labels_all = torch.zeros([0, len(self.config.label_names)]).cpu()
        filenames_list = []

        # Computes representation (without gradient computation)
        with torch.no_grad():
            for (inputs, labels, filenames, _) in loader:
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
            X, labels, _ = self.compute_outputs_skeletons(loader)
        elif register == "representation":
            X, labels, _ = self.compute_representations(loader)
        else:
            raise ValueError(
                "Argument register must be either output or representation")

        tsne = TSNE(n_components=2, perplexity=25, init='pca', random_state=50)

        Y = X.detach().numpy()

        # Makes the t-SNE fit
        X_tsne = tsne.fit_transform(Y)

        # Returns tsne embeddings
        return X_tsne, labels

    def plotting_now(self):
        if self.config.nb_epochs_per_tSNE <= 0:
            return False
        elif self.current_epoch % self.config.nb_epochs_per_tSNE == 0 \
                or self.current_epoch >= self.config.max_epochs:
            return True
        else:
            return False

    def plotting_matrices_now(self):
        if self.config.nb_epochs_per_matrix_plot <= 0:
            return False
        elif (self.current_epoch % self.config.nb_epochs_per_matrix_plot == 0)\
                or (self.current_epoch >= self.config.max_epochs):
            return True
        else:
            return False

    def save_best_auc_model(self, current_auc, save_path='./logs/'):
        if self.current_epoch == 0:
            best_auc = 0
        elif self.current_epoch > 0:
            with open(save_path+"best_model_params.json", 'r') as file:
                best_model_params = json.load(file)
                best_auc = best_model_params['best_auc']

        if current_auc > best_auc:
            torch.save({'state_dict': self.state_dict()},
                       save_path+'best_model_weights.pt')
            best_model_params = {
                'epoch': self.current_epoch, 'best_auc': current_auc}
            with open(save_path+"best_model_params.json", 'w') as file:
                json.dump(best_model_params, file)

    def training_epoch_end(self, outputs):
        """Computation done at the end of the epoch"""

        score = 0
        if (
            (self.config.mode == "encoder")
            or (self.config.mode == "regresser")
        ):
            # Computes t-SNE both in representation and output space
            if self.plotting_now():
                X_tsne, labels = self.compute_tsne(
                    self.sample_data.train_dataloader(), "output")
                image_TSNE = plot_tsne(X_tsne, labels=labels, buffer=True)
                self.logger.experiment.add_image(
                    'TSNE output image', image_TSNE, self.current_epoch)
                X_tsne, labels = self.compute_tsne(
                    self.sample_data.train_dataloader(), "representation")
                image_TSNE = plot_tsne(X_tsne, labels=labels, buffer=True)
                self.logger.experiment.add_image(
                    'TSNE representation image', image_TSNE,
                    self.current_epoch)

            if self.plotting_matrices_now():
                # Plots scatter matrices
                # Plots zxx and weights histograms
                if (self.config.mode == "encoder"):
                    self.plot_histograms()

                # Plots scatter matrices
                self.plot_scatter_matrices()

                # Plots scatter matrices with label values
                score = self.plot_scatter_matrices_with_labels(
                    self.sample_data.train_dataloader(),
                    "train",
                    self.config.mode)

        if (
            (self.config.mode == 'classifier')
            or (self.config.mode == 'regresser')
        ):
            train_auc = self.compute_output_auc(
                self.sample_data.train_dataloader())
            self.logger.experiment.add_scalar(
                "AUC/Train",
                train_auc,
                self.current_epoch)

        if self.plotting_matrices_now():
            # logs histograms
            self.custom_histogram_adder()
            # Plots views
            self.plot_views()

        # calculates average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar(
            "Loss/Train",
            avg_loss,
            self.current_epoch)
        if score != 0:
            self.logger.experiment.add_scalar(
                "Score/Train",
                score,
                self.current_epoch)

    def validation_step(self, val_batch, batch_idx):
        """Validation step"""

        (inputs, labels, filenames, _) = val_batch
        input_i = inputs[:, 0, :]
        input_j = inputs[:, 1, :]

        z_i = self.forward(input_i)
        z_j = self.forward(input_j)

        if self.config.mode == "decoder":
            sample = inputs[:, 2, :]
            batch_loss = self.cross_entropy_loss(sample, z_i, z_j)
        elif self.config.mode == "classifier":
            batch_loss = self.cross_entropy_loss_classification(
                z_i, z_j, labels)
            batch_label_loss = torch.tensor(0.)
        elif self.config.mode == "regresser":
            batch_loss = self.mse_loss_regression(z_i, z_j, labels)
            batch_label_loss = torch.tensor(0.)
        else:
            batch_loss, batch_label_loss, \
                sim_zij, sim_sii, sim_sjj, correct_pairs, weights = \
                self.generalized_supervised_nt_xen_loss(z_i, z_j, labels)

        self.log('val_loss', float(batch_loss))
        self.log('val_label_loss', float(batch_label_loss))

        # logs- a dictionary
        logs = {"val_loss": float(batch_loss),
                "val_label_loss": float(batch_label_loss)}

        batch_dictionary = {
            # REQUIRED: It is required for us to return "loss"
            "val_loss": batch_loss,
            "label_loss": batch_label_loss,
            # optional for batch logging purposes
            "log": logs,
        }

        return batch_dictionary

    def validation_epoch_end(self, outputs):
        """Computaion done at the end of each validation epoch"""

        score = 0
        # Computes t-SNE
        if (
            (self.config.mode == "encoder")
            or (self.config.mode == "regresser")
        ):
            if self.plotting_now():
                X_tsne, labels = self.compute_tsne(
                    self.sample_data.val_dataloader(), "output")
                image_TSNE = plot_tsne(X_tsne, labels=labels, buffer=True)
                self.logger.experiment.add_image(
                    'TSNE output validation image', image_TSNE,
                    self.current_epoch)
                X_tsne, labels = self.compute_tsne(
                    self.sample_data.val_dataloader(),
                    "representation")
                image_TSNE = plot_tsne(X_tsne, labels=labels, buffer=True)
                self.logger.experiment.add_image(
                    'TSNE representation validation image',
                    image_TSNE,
                    self.current_epoch)

            if self.plotting_matrices_now():
                # Plots scatter matrices
                score = self.plot_scatter_matrices_with_labels(
                    self.sample_data.val_dataloader(),
                    "val",
                    self.config.mode)

        if (
            (self.config.mode == 'classifier')
            or (self.config.mode == 'regresser')
        ):
            val_auc = self.compute_output_auc(
                self.sample_data.val_dataloader())
            self.logger.experiment.add_scalar(
                "AUC/Val",
                val_auc,
                self.current_epoch)

            # save the model that has the best val auc during train
            self.save_best_auc_model(val_auc, save_path='./logs/')

        # calculates average loss
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        # logs losses using tensorboard logger
        self.logger.experiment.add_scalar(
            "Loss/Validation",
            avg_loss,
            self.current_epoch)
        if score != 0:
            self.logger.experiment.add_scalar(
                "score/Validation",
                score,
                self.current_epoch)
