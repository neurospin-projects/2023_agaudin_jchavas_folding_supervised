from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearProjectionHead(pl.LightningModule):

    def __init__(self, num_representation_features=256,
                 layers_shapes=[256,10]):
        super(LinearProjectionHead, self).__init__()

        self.num_representation_features = num_representation_features

        # define layers
        layers = []
        input_size = self.num_representation_features
        for i, dim_i in enumerate(layers_shapes):
            output_size = dim_i
            layers.append(
                ('Linear%s' % i, nn.Linear(input_size, output_size)))
            input_size = output_size
        
        self.layers = nn.Sequential(OrderedDict(layers))
    
    def forward(self, x):
        out = self.layers(x)
        return out


class ReluProjectionHead(pl.LightningModule):

    def __init__(self, num_representation_features=256,
                 layers_shapes=[256,10]):
        super(ReluProjectionHead, self).__init__()

        self.num_representation_features = num_representation_features

        # define layers
        layers = []
        input_size = self.num_representation_features
        for i, dim_i in enumerate(layers_shapes):
            output_size = dim_i
            layers.append(
                ('Linear%s' % i, nn.Linear(input_size, output_size)))
            layers.append((f'LeakyReLU{i}', nn.LeakyReLU()))
            input_size = output_size
        
        self.layers = nn.Sequential(OrderedDict(layers))
    
    def forward(self, x):
        out = self.layers(x)
        return out