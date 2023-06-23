from collections import OrderedDict

import pytorch_lightning as pl
import torch.nn as nn


class ProjectionHead(pl.LightningModule):

    def __init__(self, num_representation_features=256,
                 layers_shapes=[256,10],
                 activation='linear'):
        super(ProjectionHead, self).__init__()
        self.num_representation_features = num_representation_features

        # define layers
        layers = []
        input_size = self.num_representation_features

        for i, dim_i in enumerate(layers_shapes):
            output_size = dim_i
            layers.append(
                ('Linear%s' % i, nn.Linear(input_size, output_size)))
            
            # add activation after each layer
            if activation == 'linear':
                pass
            elif activation == 'relu':
                layers.append((f'LeakyReLU{i}', nn.LeakyReLU()))
            elif activation == 'sigmoid':
                layers.append((f'Sigmoid{i}', nn.Sigmoid()))
            else:
                raise ValueError(f"The given activation '{activation}' is not \
handled. Choose between 'linear', 'relu' or 'sigmoid'.")
            
            input_size = output_size
        
        self.layers = nn.Sequential(OrderedDict(layers))
    
    def forward(self, x):
        out = self.layers(x)
        return out