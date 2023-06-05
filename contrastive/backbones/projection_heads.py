from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearProjectionHead(pl.LightningModule):

    def __init__(self, num_representation_features=256,
                 layers_shapes=[256, 10]):
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
                 layers_shapes=[256, 10]):
        super(ReluProjectionHead, self).__init__()

        self.num_representation_features = num_representation_features

        # define layers
        layers = []
        input_size = self.num_representation_features
        for i, dim_i in enumerate(layers_shapes):
            output_size = dim_i
            layers.append(
                ('Linear%s' % i, nn.Linear(input_size, output_size)))
            if i < (len(layers_shapes)-1):
                layers.append(
                    ('norm%s' % i, nn.BatchNorm1d(output_size)))
                layers.append((f'LeakyReLU{i}', nn.LeakyReLU()))
            input_size = output_size

        self.layers = nn.Sequential(OrderedDict(layers))

        for m in self.layers:
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.5)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.layers(x)
        return out
