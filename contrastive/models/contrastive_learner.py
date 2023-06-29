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
import json
import numpy as np
import torch
import pytorch_lightning as pl
from sklearn.manifold import TSNE
from toolz.itertoolz import first
from collections import OrderedDict

from contrastive.augmentations import ToPointnetTensor
from contrastive.backbones.densenet import DenseNet
from contrastive.backbones.convnet import ConvNet
#from contrastive.backbones.pointnet import PointNetCls
from contrastive.backbones.projection_heads import *
from contrastive.data.utils import change_list_device
from contrastive.losses import NTXenLoss
from contrastive.losses import CrossEntropyLoss
from contrastive.utils.plots.visualize_images import plot_bucket
from contrastive.utils.plots.visualize_images import plot_histogram
from contrastive.utils.plots.visualize_images import plot_histogram_weights
from contrastive.utils.plots.visualize_images import plot_histogram
from contrastive.utils.plots.visualize_images import plot_scatter_matrix
from contrastive.utils.plots.visualize_tsne import plot_tsne
from contrastive.utils.test_timeit import timeit

try:
    from contrastive.utils.plots.visualize_anatomist import Visu_Anatomist
except ImportError:
    print("INFO: you are probably not in a brainvisa env. Probably OK.")

from contrastive.utils.logs import set_root_logger_level, set_file_logger
log = set_file_logger(__file__)


class SaveOutput:
    def __init__(self):
        self.outputs = {}

    def __call__(self, module, module_in, module_out):
        self.outputs[module] = module_out.cpu()

    def clear(self):
        self.outputs = {}


class ContrastiveLearner(pl.LightningModule):

    def __init__(self, config, sample_data):
        super(ContrastiveLearner, self).__init__()

        self.automatic_optimization = True
        self.validation_step_outputs = []
        self.training_step_outputs = []

        n_datasets = len(config.data)
        log.info(f"n_datasets {n_datasets}")

        self.backbones = nn.ModuleList()
        if config.backbone_name == 'densenet':
            for i in range(n_datasets):
                self.backbones.append(DenseNet(
                    growth_rate=config.growth_rate,
                    block_config=config.block_config,
                    num_init_features=config.num_init_features,
                    num_representation_features=config.num_representation_features,
                    drop_rate=config.drop_rate,
                    in_shape=config.data[i].input_size))
        elif config.backbone_name == "convnet":
            for i in range(n_datasets):
                self.backbones.append(ConvNet(
                    encoder_depth=config.encoder_depth,
                    num_representation_features=config.num_representation_features,
                    drop_rate=config.drop_rate,
                    in_shape=config.data[i].input_size))
        # elif config.backbone_name == 'pointnet':
        #     self.backbone = PointNetCls(
        #         k=config.num_representation_features,
        #         num_outputs=config.num_representation_features,
        #         projection_head_hidden_layers=config.projection_head_hidden_layers,
        #         drop_rate=config.drop_rate,
        #         feature_transform=False)
        else:
            raise ValueError(f"No underlying backbone with backbone name {config.backbone_name}")
        
        num_representation_features_total = config.num_representation_features * n_datasets

        # define the shape of the projection head
        # prioritize the shapes explicitely specified in config
        if config.proj_layers_shapes is not None:
            layers_shapes = config.proj_layers_shapes
        else:
            # else, construct it in a standardized way
            if config.mode == 'encoder':
                output_shape = num_representation_features_total
            elif config.mode == 'classifier':
                output_shape = 2
            elif config.mode == 'regresser':
                output_shape = 1
            else:
                raise ValueError(f"Mode {config.mode} doesn't exist.")
            layers_shapes = [num_representation_features_total] * (config.length_projection_head - 1) + [output_shape]

        activation = config.projection_head_name
        log.info(f"activation = {activation}")
        self.projection_head = ProjectionHead(
            num_representation_features=num_representation_features_total,
            layers_shapes=layers_shapes,
            activation=activation)

        self.config = config
        self.n_datasets = n_datasets
        self.sample_data = sample_data
        self.sample_i = np.array([])
        self.sample_j = np.array([])
        self.sample_k = np.array([])
        self.sample_filenames = []
        self.save_output = SaveOutput()
        self.output_shape = output_shape
        self.hook_handles = []
        self.get_layers()
        if self.config.environment == "brainvisa":
            self.visu_anatomist = Visu_Anatomist()

    def forward(self, x):
        embeddings = []
        for i in range(self.n_datasets):
            embedding = self.backbones[i].forward(x[i])
            embeddings.append(embedding)
        embeddings = torch.cat(embeddings, dim=1)
        out = self.projection_head.forward(embeddings)
        return out

    def get_full_inputs_from_batch(self, batch):
        full_inputs = []
        for (inputs, filenames) in batch:
            if self.config.backbone_name == 'pointnet':
                inputs = torch.squeeze(inputs).to(torch.float)
            full_inputs.append(inputs)
        
        inputs = full_inputs
        #inputs = torch.stack(full_inputs, dim=0)
        return (inputs, filenames)

    def get_layers(self):
        i = 0
        for layer in self.modules():
            if self.config.backbone_name in ['densenet', 'convnet']:
                if isinstance(layer, torch.nn.Linear):
                    handle = layer.register_forward_hook(self.save_output)
                    self.hook_handles.append(handle)
            elif self.config.backbone_name == 'pointnet':
                # for the moment, keep the same method
                # need to pass the wanted representation layer to the 1st place
                # => remove the first five layers
                if isinstance(layer, torch.nn.Linear):
                    if i >= 5:
                        handle = layer.register_forward_hook(self.save_output)
                        self.hook_handles.append(handle)
                    i += 1

    def load_pretrained_model(self, pretrained_model_path, encoder_only=False):
        """Load weights stored in a state_dict at pretrained_model_path
        """

        pretrained_state_dict = torch.load(pretrained_model_path)['state_dict']
        if encoder_only:
            pretrained_state_dict = OrderedDict(
                {k: v for k, v in pretrained_state_dict.items()
                 if 'encoder' in k})

        model_dict = self.state_dict()

        loaded_layers = []
        for n, p in pretrained_state_dict.items():
            if n in model_dict:
                loaded_layers.append(n)
                model_dict[n] = p

        self.load_state_dict(model_dict)

        not_loaded_layers = [
            key for key in model_dict.keys() if key not in loaded_layers]
        # print(f"Loaded layers = {loaded_layers}")
        log.info(f"Layers not loaded = {not_loaded_layers}")

    def custom_histogram_adder(self):
        """Builds histogram for each model parameter.
        """
        # iterating through all parameters
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name,
                params,
                self.current_epoch)

    def plot_histograms(self):
        """Plots all zii, zjj, zij and weights histograms"""

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

    def plot_scatter_matrices(self):
        """Plots scatter matrices of output and representations spaces"""
        # Makes scatter matrix of output space
        r = self.compute_outputs_skeletons(
            self.sample_data.train_dataloader())
        X = r[0]  # First element of tuple
        scatter_matrix_outputs = plot_scatter_matrix(X, buffer=True)
        self.logger.experiment.add_image(
            'scatter_matrix_outputs',
            scatter_matrix_outputs,
            self.current_epoch)

        # Makes scatter matrix of representation space
        r = self.compute_representations(
            self.sample_data.train_dataloader())
        X = r[0]  # First element of tuple
        scatter_matrix_representations = plot_scatter_matrix(
            X, buffer=True)
        self.logger.experiment.add_image(
            'scatter_matrix_representations',
            scatter_matrix_representations,
            self.current_epoch)

    def plot_views(self):
        """Plots different 3D views"""
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
                'input_ana_i: ',
                image_input_i, self.current_epoch)
            # self.logger.experiment.add_text(
            #     'filename: ',self.sample_filenames[0], self.current_epoch)
            image_input_j = self.visu_anatomist.plot_bucket(
                self.sample_j, buffer=True)
            self.logger.experiment.add_image(
                'input_ana_j: ',
                image_input_j, self.current_epoch)
            if len(self.sample_k) != 0:
                image_input_k = self.visu_anatomist.plot_bucket(
                    self.sample_k, buffer=True)
                self.logger.experiment.add_image(
                    'input_ana_k: ',
                    image_input_k, self.current_epoch)

    def configure_optimizers(self):
        """Adam optimizer"""
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer=optimizer,
        #     T_max=100,
        #     eta_min=0.0001)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }


    def nt_xen_loss(self, z_i, z_j):
        """Loss function for contrastive"""
        loss = NTXenLoss(temperature=self.config.temperature,
                         return_logits=True)
        return loss.forward(z_i, z_j)

    def cross_entropy_loss(self, sample, output_i, output_j):
        """Loss function for decoder"""
        loss = CrossEntropyLoss(device=self.device)
        return loss.forward(sample, output_i, output_j)

    def training_step(self, train_batch, batch_idx):
        """Training step.
        """
        inputs, filenames = self.get_full_inputs_from_batch(train_batch)

        # print("TRAINING STEP", inputs.shape)
        input_i = [inputs[i][:, 0, ...] for i in range(self.n_datasets)]
        input_j = [inputs[i][:, 1, ...] for i in range(self.n_datasets)]
        z_i = self.forward(input_i)
        z_j = self.forward(input_j)

        if self.config.mode == "decoder":
            sample = inputs[:, 2, :]
            batch_loss = self.cross_entropy_loss(sample, z_i, z_j)
        else:
            batch_loss, sim_zij, sim_zii, sim_zjj = self.nt_xen_loss(z_i, z_j)

        self.log('train_loss', float(batch_loss))

        # Only computes graph on first step
        if self.global_step == 1:
            self.logger.experiment.add_graph(self, [input_i])

        # Records sample for first batch of each epoch
        if batch_idx == 0:
            self.sample_i = change_list_device(input_i, 'cpu')
            self.sample_j = change_list_device(input_j, 'cpu')
            self.sample_filenames = filenames
            if self.config.mode != "decoder":
                self.sim_zij = sim_zij * self.config.temperature
                self.sim_zii = sim_zii * self.config.temperature
                self.sim_zjj = sim_zjj * self.config.temperature

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
        X = torch.zeros([0, self.output_shape]).cpu()
        filenames_list = []
        transform = ToPointnetTensor()

        # Computes embeddings without computing gradient
        with torch.no_grad():
            for batch in loader:
                inputs, filenames = self.get_full_inputs_from_batch(batch)
                # First views of the whole batch
                inputs = change_list_device(inputs, 'cuda')
                # model = self.cuda()
                input_i = [inputs[i][:, 0, ...] for i in range(self.n_datasets)]
                input_j = [inputs[i][:, 1, ...] for i in range(self.n_datasets)]
                if self.config.backbone_name == 'pointnet':
                    input_i = transform(input_i.cpu()).cuda().to(torch.float)
                    input_j = transform(input_j.cpu()).cuda().to(torch.float)
                X_i = self.forward(input_i)
                # Second views of the whole batch
                X_j = self.forward(input_j)
                # First views and second views
                # are put side by side
                X_reordered = torch.cat([X_i, X_j], dim=-1)
                X_reordered = X_reordered.view(-1, X_i.shape[-1])
                X = torch.cat((X, X_reordered.cpu()), dim=0)
                filenames_duplicate = [item
                                       for item in filenames
                                       for repetitions in range(2)]
                filenames_list = filenames_list + filenames_duplicate
                del inputs

        return X, filenames_list

    def compute_decoder_outputs_skeletons(self, loader):
        """Computes the outputs of the model for each crop.

        This includes the projection head"""

        # Initialization
        X = torch.zeros([0, 2, 20, 40, 40]).cpu()
        filenames_list = []

        # Computes embeddings without computing gradient
        with torch.no_grad():
            for (inputs, filenames) in loader:
                # First views of the whole batch
                inputs = change_list_device(inputs, 'cuda')
                model = self.cuda()
                X_i = model.forward(inputs[:, 0, :])
                print(f"shape X and X_i: {X.shape}, {X_i.shape}")
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
        X = torch.zeros([0, self.config.num_representation_features * self.n_datasets]).cpu()
        filenames_list = []

        # Computes representation (without gradient computation)
        with torch.no_grad():
            for batch in loader:
                inputs, filenames = self.get_full_inputs_from_batch(batch)
                # First views of the whole batch
                if self.config.device != 'cpu':
                    inputs = change_list_device(inputs, 'cuda')
                else:
                    inputs = change_list_device(inputs, 'cpu')
                if self.config.backbone_name == 'pointnet':
                    inputs = torch.squeeze(inputs).to(torch.float)
                if self.config.device != 'cpu':
                    model = self.cuda()
                else:
                    model = self.cpu()
                input_i = [inputs[k][:, 0, ...] for k in range(self.n_datasets)]
                input_j = [inputs[k][:, 1, ...] for k in range(self.n_datasets)]
                
                X_i = []
                for k in range(self.n_datasets):
                    embedding = model.backbones[k].forward(input_i[k])
                    X_i.append(embedding)
                X_i = torch.cat(X_i, dim=1)

                # Second views of the whole batch
                X_j = []
                for k in range(self.n_datasets):
                    embedding = model.backbones[k].forward(input_j[k])
                    X_j.append(embedding)
                X_j = torch.cat(X_j, dim=1)

                # print("representations", X_i.shape, X_j.shape)
                # First views and second views are put side by side
                X_reordered = torch.cat([X_i, X_j], dim=-1)
                X_reordered = X_reordered.view(-1, X_i.shape[-1])
                X = torch.cat((X, X_reordered.cpu()), dim=0)
                # print(f"filenames = {filenames}")
                filenames_duplicate = [
                    item for item in filenames
                    for repetitions in range(2)]
                filenames_list = filenames_list + filenames_duplicate
                del inputs

        return X, filenames_list

    def plotting_now(self):
        """Tells if it is the right epoch to plot the tSNE."""
        if self.config.nb_epochs_per_tSNE <= 0:
            return False
        elif self.current_epoch % self.config.nb_epochs_per_tSNE == 0 \
                or self.current_epoch >= self.config.max_epochs:
            return True
        else:
            return False

    def compute_tsne(self, loader, register):
        """Computes t-SNE.

        It is computed either in the representation
        or in the output space"""

        if register == "output":
            X, _ = self.compute_outputs_skeletons(loader)
        elif register == "representation":
            X, _ = self.compute_representations(loader)
        else:
            raise ValueError(
                "Argument register must be either output or representation")

        tsne = TSNE(n_components=2, perplexity=5, init='pca', random_state=50)

        Y = X.detach().numpy()

        # Makes the t-SNE fit
        X_tsne = tsne.fit_transform(Y)

        # Returns tsne embeddings
        return X_tsne

    def on_train_epoch_end(self, outputs):
        """Computation done at the end of the epoch"""

        if self.config.mode == "encoder":
            # Computes t-SNE both in representation and output space
            if self.plotting_now():
                print("Computing tsne\n")
                X_tsne = self.compute_tsne(
                    self.sample_data.train_dataloader(), "output")
                image_TSNE = plot_tsne(X_tsne, buffer=True)
                self.logger.experiment.add_image(
                    'TSNE output image', image_TSNE, self.current_epoch)
                X_tsne = self.compute_tsne(
                    self.sample_data.train_dataloader(), "representation")
                image_TSNE = plot_tsne(X_tsne, buffer=True)
                self.logger.experiment.add_image(
                    'TSNE representation image',
                    image_TSNE, self.current_epoch)

                # Computes histogram of sim_zij
                histogram_sim_zij = plot_histogram(self.sim_zij, buffer=True)
                self.logger.experiment.add_image(
                    'histo_sim_zij', histogram_sim_zij, self.current_epoch)

        # Plots views
        # if self.config.backbone_name != 'pointnet':
        #     self.plot_views() # far too slow and I don't get what it is doing

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
        inputs, filenames = self.get_full_inputs_from_batch(val_batch)
        
        input_i = [inputs[i][:, 0, ...] for i in range(self.n_datasets)]
        input_j = [inputs[i][:, 1, ...] for i in range(self.n_datasets)]
        z_i = self.forward(input_i)
        z_j = self.forward(input_j)

        if self.config.mode == "decoder":
            sample = inputs[:, 2, :]
            batch_loss = self.cross_entropy_loss(sample, z_i, z_j)
        else:
            batch_loss, sim_zij, sim_zii, sim_zjj = self.nt_xen_loss(z_i, z_j)
        self.log('val_loss', float(batch_loss))

        # logs- a dictionary
        logs = {"val_loss": float(batch_loss)}

        batch_dictionary = {
            # REQUIRED: It ie required for us to return "loss"
            "loss": batch_loss,

            # optional for batch logging purposes
            "log": logs,
        }

        return batch_dictionary

    def on_validation_epoch_end(self, outputs):
        """Computation done at the end of each validation epoch"""

        # Computes t-SNE
        if self.config.mode == "encoder":
            if self.plotting_now():
                print("Computing tsne\n")
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

        # calculates average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # logs losses using tensorboard logger
        self.logger.experiment.add_scalar(
            "Loss/Validation",
            avg_loss,
            self.current_epoch)

        # save model if best
        save_path = './logs/'
        if self.current_epoch == 0:
            best_loss = np.inf
        elif self.current_epoch > 0:
            with open(save_path+"best_model_params.json", 'r') as file:
                best_model_params = json.load(file)
                best_loss = best_model_params['best_loss']

        avg_loss = avg_loss.cpu().item()
        if avg_loss < best_loss:
            torch.save({'state_dict': self.state_dict()},
                       save_path+'best_model_weights.pt')
            best_model_params = {
                'epoch': self.current_epoch, 'best_loss': avg_loss}
            with open(save_path+"best_model_params.json", 'w') as file:
                json.dump(best_model_params, file)
