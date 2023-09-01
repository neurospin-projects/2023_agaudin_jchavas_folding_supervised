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
import os
import json
import numpy as np
import torch
import pytorch_lightning as pl
from sklearn.manifold import TSNE
from collections import OrderedDict

from sklearn.metrics import roc_auc_score, r2_score

from contrastive.augmentations import ToPointnetTensor
from contrastive.backbones.densenet import DenseNet
from contrastive.backbones.convnet import ConvNet
#from contrastive.backbones.pointnet import PointNetCls
from contrastive.backbones.projection_heads import *
from contrastive.data.utils import change_list_device
from contrastive.evaluation.auc_score import regression_roc_auc_score
from contrastive.models.models_utils import *
from contrastive.losses import *
from contrastive.utils.plots.visualize_images import plot_bucket, \
    plot_histogram, plot_histogram_weights, plot_scatter_matrix, \
    plot_scatter_matrix_with_labels
from contrastive.utils.plots.visualize_tsne import plot_tsne
from contrastive.utils.test_timeit import timeit

try:
    from contrastive.utils.plots.visualize_anatomist import Visu_Anatomist
except ImportError:
    print("INFO: you are probably not in a brainvisa env. Probably OK.")

from contrastive.utils.logs import set_root_logger_level, set_file_logger
log = set_file_logger(__file__)



class ContrastiveLearnerFusion(pl.LightningModule):

    def __init__(self, config, sample_data, with_labels=False):
        super(ContrastiveLearnerFusion, self).__init__()

        n_datasets = len(config.data)
        log.info(f"n_datasets {n_datasets}")

        # define the encoder structure
        self.backbones = nn.ModuleList()
        if config.backbone_name == 'densenet':
            for i in range(n_datasets):
                self.backbones.append(DenseNet(
                    growth_rate=config.growth_rate,
                    block_config=config.block_config,
                    num_init_features=config.num_init_features,
                    num_representation_features=config.backbone_output_size,
                    drop_rate=config.drop_rate,
                    in_shape=config.data[i].input_size))
        elif config.backbone_name == "convnet":
            for i in range(n_datasets):
                self.backbones.append(ConvNet(
                    encoder_depth=config.encoder_depth,
                    num_representation_features=config.backbone_output_size,
                    drop_rate=config.drop_rate,
                    in_shape=config.data[i].input_size))
        # elif config.backbone_name == 'pointnet':
        #     self.backbone = PointNetCls(
        #         k=config.num_representation_features,
        #         num_outputs=config.backbone_output_size,
        #         projection_head_hidden_layers=config.projection_head_hidden_layers,
        #         drop_rate=config.drop_rate,
        #         feature_transform=False)
        else:
            raise ValueError(f"No underlying backbone with backbone name {config.backbone_name}")
        
        # rename variables
        concat_latent_spaces_size = config.backbone_output_size * n_datasets

        # build converter (if required) and set the latent space size according to it
        converter, num_representation_features = build_converter(config, concat_latent_spaces_size)
        self.converter = converter

        # set up the projection head layers shapes
        layers_shapes = get_projection_head_shape(config, num_representation_features)
        output_shape = layers_shapes[-1]

        # set projection head activation
        activation = config.projection_head_name
        log.info(f"activation = {activation}")
        self.projection_head = ProjectionHead(
            num_representation_features=num_representation_features,
            layers_shapes=layers_shapes,
            activation=activation)

        # set up class keywords
        self.config = config
        self.with_labels = with_labels
        self.n_datasets = n_datasets
        self.sample_data = sample_data
        self.sample_i = np.array([])
        self.sample_j = np.array([])
        self.sample_k = np.array([])
        self.sample_filenames = []
        self.num_representation_features = num_representation_features
        self.output_shape = output_shape
        self.lr = self.config.lr
        if self.config.environment == "brainvisa":
            self.visu_anatomist = Visu_Anatomist()

        if 'class_weights' in config.keys():
            self.class_weights = torch.Tensor(config.class_weights).to(device=config.device)
        else:
            self.class_weights = None

    def forward(self, x):
        embeddings = []
        for i in range(self.n_datasets):
            embedding = self.backbones[i].forward(x[i])
            embeddings.append(embedding)
        embeddings = torch.cat(embeddings, dim=1)
        embeddings = self.converter.forward(embeddings)
        out = self.projection_head.forward(embeddings)
        return out


    def get_full_inputs_from_batch(self, batch):
        full_inputs = []
        for (inputs, filenames) in batch:  # loop over datasets
            if self.config.backbone_name == 'pointnet':
                inputs = torch.squeeze(inputs).to(torch.float)
            full_inputs.append(inputs)
        
        inputs = full_inputs
        return (inputs, filenames)
    
    def get_full_inputs_from_batch_with_labels(self, batch):
        #print("A-T-ON ENCORE BESOIN DE LA VIEW3 ?")
        full_inputs = []
        full_view3 = []
        for (inputs, filenames, labels, view3) in batch: # loop over datasets
            if self.config.backbone_name == 'pointnet':
                inputs = torch.squeeze(inputs).to(torch.float)
            full_inputs.append(inputs)
            full_view3.append(view3)
        
        inputs = full_inputs
        view3 = full_view3
        return (inputs, filenames, labels, view3)


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


        # freeze all layers except last
        #for name, para in self.named_parameters():
        #    if ("2a" not in name) & ("projection_head" not in name):
        #        para.requires_grad = False
        #    print("-"*20)
        #    print(f"name: {name}")
        #    print("values")
        #    print(para)

        # freeze whole encoder
        if encoder_only:
            for name, para in self.named_parameters():
                if "encoder" in name:
                    para.requires_grad = False

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


    def plot_scatter_matrices(self, dataloader, key):
        """Plots scatter matrices of output and representations spaces"""
        
        funcs = {'outputs': self.compute_outputs_skeletons,
                 'representations': self.compute_outputs_skeletons}
        
        for name, func in funcs.items():
            r = func(dataloader)
            X = r[0]  # get inputs

            if self.with_labels:
                labels = r[2]
                scatter_matrix_with_labels = \
                    plot_scatter_matrix_with_labels(X, labels, buffer=True)
                self.logger.experiment.add_image(
                    f'scatter_matrix_{name}_with_labels_' + key,
                    scatter_matrix_with_labels,
                    self.current_epoch)
            else:
                scatter_matrix = plot_scatter_matrix(X, buffer=True)
                self.logger.experiment.add_image(
                    f'scatter_matrix_{name}',
                    scatter_matrix,
                    self.current_epoch)
            
            if (self.config.mode == "regresser") and (name =='output'):
                score = r2_score(labels, X)
            else:
                score = 0

        return score


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
            lr=self.lr,
            weight_decay=self.config.weight_decay)
        # steps = 140
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

        return optimizer


    def nt_xen_loss(self, z_i, z_j):
        """Loss function for contrastive"""
        loss = NTXenLoss(temperature=self.config.temperature,
                         return_logits=True)
        return loss.forward(z_i, z_j)

    def generalized_supervised_nt_xen_loss(self, z_i, z_j, labels):
        """Loss function for supervised contrastive"""
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
        loss = CrossEntropyLoss_Classification(device=self.device,
                                               class_weights=self.class_weights)
        return loss.forward(output_i, output_j, labels)

    def mse_loss_regression(self, output_i, output_j, labels):
        """Loss function for decoder"""
        loss = MSELoss_Regression(device=self.device)
        return loss.forward(output_i, output_j, labels)


    def training_step(self, train_batch, batch_idx):
        """Training step.
        """
        if self.config.with_labels:
            inputs, filenames, labels, view3 = \
                self.get_full_inputs_from_batch_with_labels(train_batch)
        else:
            inputs, filenames = self.get_full_inputs_from_batch(train_batch)

        # print("TRAINING STEP", inputs.shape)
        input_i = [inputs[i][:, 0, ...] for i in range(self.n_datasets)]
        input_j = [inputs[i][:, 1, ...] for i in range(self.n_datasets)]
        z_i = self.forward(input_i)
        z_j = self.forward(input_j)

        # compute the right loss depending on the learning mode
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
        elif self.config.proportion_pure_contrastive != 1:
            batch_loss, batch_label_loss, \
                sim_zij, sim_zii, sim_zjj, correct_pair, weights = \
                self.generalized_supervised_nt_xen_loss(z_i, z_j, labels)
        else:
            batch_loss, sim_zij, sim_zii, sim_zjj = self.nt_xen_loss(z_i, z_j)

        # Only computes graph on first step
        if self.global_step == 1:
            self.logger.experiment.add_graph(self, [input_i])

        # Records sample for first batch of each epoch
        if batch_idx == 0:
            self.sample_i = change_list_device(input_i, 'cpu')
            self.sample_j = change_list_device(input_j, 'cpu')
            self.sample_filenames = filenames
            if self.config.with_labels:
                self.sample_k = change_list_device(view3, 'cpu')
                self.sample_labels = labels
            if self.config.mode == "encoder":
                self.sim_zij = sim_zij * self.config.temperature
                self.sim_zii = sim_zii * self.config.temperature
                self.sim_zjj = sim_zjj * self.config.temperature
            if self.config.environment == 'brainvisa' and self.config.checking:
                bv_checks(self, filenames)  # untested
        
        # logs - a dictionary
        self.log('train_loss', float(batch_loss))
        logs = {"train_loss": float(batch_loss)}

        batch_dictionary = {
            # REQUIRED: It is required for us to return "loss"
            "loss": batch_loss,
            # optional for batch logging purposes
            "log": logs}

        if self.config.with_labels:
            # add label_loss (a part of the loss) to log
            self.log('train_label_loss', float(batch_label_loss))
            logs['train_label_loss'] = float(batch_label_loss)
            batch_dictionary['label_loss'] = batch_label_loss

        return batch_dictionary

    def compute_outputs_skeletons(self, loader):
        """Computes the outputs of the model for each crop.

        This includes the projection head"""

        # Initialization
        X = torch.zeros([0, self.output_shape]).cuda()
        labels_all = torch.zeros(
            [0, len(self.config.label_names)]).cuda()
        filenames_list = []
        transform = ToPointnetTensor()

        # Computes embeddings without computing gradient
        with torch.no_grad():
            for batch in loader:
                if self.config.with_labels:
                    inputs, filenames, labels, _ = \
                        self.get_full_inputs_from_batch_with_labels(batch)
                else:
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

                # First views and second views are put side by side
                X_reordered = torch.cat([X_i, X_j], dim=-1)
                X_reordered = X_reordered.view(-1, X_i.shape[-1])
                X = torch.cat((X, X_reordered.cuda()), dim=0)

                # concat filenames
                filenames_duplicate = [item
                                       for item in filenames
                                       for repetitions in range(2)]
                filenames_list = filenames_list + filenames_duplicate
                del inputs

                # deal with labels if required
                if self.config.with_labels:
                    # We now concatenate the labels
                    labels_reordered = torch.cat([labels, labels], dim=-1)
                    labels_reordered = labels_reordered.view(-1, labels.shape[-1])
                    # At the end, labels are concatenated
                    labels_all = torch.cat((labels_all, labels_reordered.cuda()),
                                        dim=0)
        
        if self.config.with_labels:
            return X.cpu(), filenames_list, labels_all.cpu()
        else:
            return X.cpu(), filenames_list
    
    def compute_output_probabilities(self, loader):
        """Only available in classifier mode.
        Gets the output of the model and convert it to probabilities thanks to softmax."""
        if self.config.mode == 'classifier':
            X, filenames_list, labels_all = self.compute_output_skeletons(
                loader)
            # compute the mean of the two views' outputs
            X = (X[::2, ...] + X[1::2, ...]) / 2
            # remove the doubleing of labels
            labels_all = labels_all[::2]
            filenames_list = filenames_list[::2]
            X = nn.functional.softmax(X, dim=1)
            return X, labels_all, filenames_list
        else:
            raise ValueError(
                "The config.mode is not 'classifier'. "
                "You shouldn't compute probabilities with another mode.")

    def compute_output_auc(self, loader):
        """Only available in classifier and regresser modes.
        Computes the auc from the outputs of the model and the associated labels."""
        # we don't apply transforms for the AUC computation
        loader.dataset.transform = False

        X, filenames_list, labels = self.compute_outputs_skeletons(loader)
        # compute the mean of the two views' outputs
        X = (X[::2, ...] + X[1::2, ...]) / 2
        # remove the doubleing of labels
        labels = labels[::2]
        # and filenames
        filenames_list = filenames_list[::2]
        if self.config.mode == "regresser":
            X = X[:, 0]
            auc = regression_roc_auc_score(labels, X)
        else:
            X = nn.functional.softmax(X, dim=1)
            X = X[:, 1]
            auc = roc_auc_score(labels, X)
        
        # put augmentations back to normal
        loader.dataset.transform = self.config.apply_augmentations

        return auc, filenames_list, labels.numpy()[:, 0], X.numpy()


    def compute_decoder_outputs_skeletons(self, loader):
        """Computes the outputs of the model for each crop,
        but for decoder mode.

        This includes the projection head"""

        # Initialization
        X = torch.zeros([0, 2, 20, 40, 40]).cpu()
        filenames_list = []

        # Computes embeddings without computing gradient
        with torch.no_grad():
            for batch in loader:
                if self.config.with_labels:
                    (inputs, filenames, _, _) = self.get_full_inputs_from_batch_with_labels(batch)
                else:
                    (inputs, filenames) = self.get_full_inputs_from_batch(batch)
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
        X = torch.zeros(
            [0, self.num_representation_features]).cuda()
        labels_all = torch.zeros(
            [0, len(self.config.label_names)]).cuda()
        filenames_list = []

        # Computes representation (without gradient computation)
        with torch.no_grad():
            for batch in loader:
                if self.config.with_labels:
                    (inputs, filenames, labels, _) = self.get_full_inputs_from_batch_with_labels(batch)
                else:
                    inputs, filenames = self.get_full_inputs_from_batch(batch)
                
               # deal with devices
                if self.config.device != 'cpu':
                    inputs = change_list_device(inputs, 'cuda')
                else:
                    inputs = change_list_device(inputs, 'cpu')
                if self.config.device != 'cpu':
                    model = self.cuda()
                else:
                    model = self.cpu()
                # deal with pointnet
                if self.config.backbone_name == 'pointnet':
                    inputs = torch.squeeze(inputs).to(torch.float)
                
                input_i = [inputs[k][:, 0, ...] for k in range(self.n_datasets)]
                input_j = [inputs[k][:, 1, ...] for k in range(self.n_datasets)]

                # First views of the whole batch               
                X_i = []
                for k in range(self.n_datasets):
                    embedding = model.backbones[k].forward(input_i[k])
                    X_i.append(embedding)
                X_i = torch.cat(X_i, dim=1)
                X_i = self.converter.forward(X_i)

                # Second views of the whole batch
                X_j = []
                for k in range(self.n_datasets):
                    embedding = model.backbones[k].forward(input_j[k])
                    X_j.append(embedding)
                X_j = torch.cat(X_j, dim=1)
                X_j = self.converter.forward(X_j)

                # First views and second views are put side by side
                X_reordered = torch.cat([X_i, X_j], dim=-1)
                X_reordered = X_reordered.view(-1, X_i.shape[-1])
                X = torch.cat((X, X_reordered.cuda()), dim=0)
                # print(f"filenames = {filenames}")
                filenames_duplicate = [
                    item for item in filenames
                    for repetitions in range(2)]
                filenames_list = filenames_list + filenames_duplicate
                del inputs

                # deal with labels if required
                if self.config.with_labels:
                    # We now concatenate the labels
                    labels_reordered = torch.cat([labels, labels], dim=-1)
                    labels_reordered = labels_reordered.view(-1, labels.shape[-1])
                    # At the end, labels are concatenated
                    labels_all = torch.cat((labels_all, labels_reordered.cuda()),
                                        dim=0)
        
        if self.config.with_labels:
            return X.cpu(), filenames_list, labels_all.cpu()
        else:
            return X.cpu(), filenames_list


    def plotting_now(self):
        """Tells if it is the right epoch to plot the tSNE."""
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


    def compute_tsne(self, loader, register):
        """Computes t-SNE.

        It is computed either in the representation
        or in the output space"""

        if register == "output":
            func = self.compute_outputs_skeletons
        elif register == "representation":
            func = self.compute_representations
        else:
            raise ValueError(
                "Argument register must be either 'output' or 'representation'")

        if self.config.with_labels:
            X, _, labels = func(loader)
        else:
            X, _ = func(loader)

        tsne = TSNE(n_components=2, perplexity=5, init='pca', random_state=50)

        Y = X.detach().numpy()

        # Makes the t-SNE fit
        X_tsne = tsne.fit_transform(Y)

        # Returns tsne embeddings
        if self.config.with_labels:
            return X_tsne, labels
        else:
            return X_tsne


    def save_best_auc_model(self, current_auc, save_path='./logs/'):
        if self.current_epoch == 0:
            best_auc = 0
        elif self.current_epoch > 0:
            with open(save_path + "best_model_params.json", 'r') as file:
                best_model_params = json.load(file)
                best_auc = best_model_params['best_auc']

        if current_auc > best_auc:
            torch.save({'state_dict': self.state_dict()},
                       save_path + 'best_model_weights.pt')
            best_model_params = {
                'epoch': self.current_epoch, 'best_auc': current_auc}
            with open(save_path + "best_model_params.json", 'w') as file:
                json.dump(best_model_params, file)


    def training_epoch_end(self, outputs):
        """Computation done at the end of the epoch"""

        # score = 0
        if self.config.mode in ["encoder", "regresser"]:
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
            
            # Plots scatter matrices
            if self.plotting_matrices_now():
                # Plots zxx and weights histograms
                if (self.config.mode == "encoder"):
                    self.plot_histograms()

                # Plots scatter matrices
                self.plot_scatter_matrices()

                # # Plots scatter matrices with label values
                # score = self.plot_scatter_matrices_with_labels(
                #     self.sample_data.train_dataloader(),
                #     "train",
                #     self.config.mode)
                # # Computes histogram of sim_zij
                # histogram_sim_zij = plot_histogram(self.sim_zij, buffer=True)
                # self.logger.experiment.add_image(
                #     'histo_sim_zij', histogram_sim_zij, self.current_epoch)

        if self.config.mode in ['classifier', 'regresser']:
            train_auc,_,_,_ = self.compute_output_auc(
                self.sample_data.train_dataloader())
            self.logger.experiment.add_scalar(
                "AUC/Train",
                train_auc,
                self.current_epoch)
            # save train_auc to use it during validation end step
            auc_dict = {'train_auc': train_auc}
            save_path = './' + self.logger.experiment.log_dir + '/train_auc.json'
            with open(save_path, 'w') as file:
                json.dump(auc_dict, file)


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
        # if score != 0:
        #     self.logger.experiment.add_scalar(
        #         "Score/Train",
        #         score,
        #         self.current_epoch)


    def validation_step(self, val_batch, batch_idx):
        """Validation step"""
        if self.config.with_labels:
            (inputs, _, labels, _) = \
                self.get_full_inputs_from_batch_with_labels(val_batch)
        else:
            inputs, _ = self.get_full_inputs_from_batch(val_batch)
        
        input_i = [inputs[i][:, 0, ...] for i in range(self.n_datasets)]
        input_j = [inputs[i][:, 1, ...] for i in range(self.n_datasets)]
        z_i = self.forward(input_i)
        z_j = self.forward(input_j)

        # compute the right loss depending on the learning mode
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
        elif self.config.proportion_pure_contrastive != 1:
            batch_loss, batch_label_loss, _ = \
                self.generalized_supervised_nt_xen_loss(z_i, z_j, labels)
        else:
            batch_loss, sim_zij, sim_zii, sim_zjj = self.nt_xen_loss(z_i, z_j)
        
        self.log('val_loss', float(batch_loss))
        self.log('diff_auc', float(0))  # line to be able to use early stopping
        # logs- a dictionary
        logs = {"val_loss": float(batch_loss)}
        batch_dictionary = {
            # REQUIRED: It ie required for us to return "loss"
            "val_loss": batch_loss,
            # optional for batch logging purposes
            "log": logs}

        if self.config.with_labels:
            # add label_loss (a part of the loss) to log
            self.log('val_label_loss', float(batch_label_loss))
            logs['val_label_loss'] = float(batch_label_loss)
            batch_dictionary['val_label_loss'] = batch_label_loss

        return batch_dictionary


    def validation_epoch_end(self, outputs):
        """Computation done at the end of each validation epoch"""

        # score = 0
        # Computes t-SNE
        if self.config.mode in ["encoder", "regresser"]:
            if self.plotting_now():
                log.info("Computing tsne\n")
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
        
            # # Plots scatter matrices
            # if self.plotting_matrices_now():
            #     score = self.plot_scatter_matrices_with_labels(
            #         self.sample_data.val_dataloader(),
            #         "val",
            #         self.config.mode)
        
        # compute val auc
        if self.config.mode in ['classifier', 'regresser']:
            val_auc,_,_,_ = self.compute_output_auc(
                self.sample_data.val_dataloader())
            self.logger.experiment.add_scalar(
                "AUC/Val",
                val_auc,
                self.current_epoch)
            # compute overfitting early stopping relevant value
            if self.current_epoch > 0:
                # load train_auc
                save_path = './' + self.logger.experiment.log_dir + '/train_auc.json'
                with open(save_path, 'r') as file:
                    train_auc = json.load(file)['train_auc']
                self.log('diff_auc', float(train_auc - val_auc))

            # save the model that has the best val auc during train
            self.save_best_auc_model(val_auc, save_path='./logs/')

        # calculates average loss
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        # logs losses using tensorboard logger
        self.logger.experiment.add_scalar(
            "Loss/Validation",
            avg_loss,
            self.current_epoch)
        # if score != 0:
        #     self.logger.experiment.add_scalar(
        #         "score/Validation",
        #         score,
        #         self.current_epoch)

        # save best model by loss if no auc to do so
        if not self.config.with_labels:
            # save model if best validation loss
            save_path = './logs/'
            if self.current_epoch == 0:
                best_loss = np.inf
            elif self.current_epoch > 0:
                # load the current best loss
                with open(save_path+"best_model_params.json", 'r') as file:
                    best_model_params = json.load(file)
                    best_loss = best_model_params['best_loss']

            # compare to the current loss and replace the best if necessary
            avg_loss = avg_loss.cpu().item()
            if avg_loss < best_loss:
                torch.save({'state_dict': self.state_dict()},
                        save_path+'best_model_weights.pt')
                best_model_params = {
                    'epoch': self.current_epoch, 'best_loss': avg_loss}
                with open(save_path+"best_model_params.json", 'w') as file:
                    json.dump(best_model_params, file)
