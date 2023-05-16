import torch
import torch.nn as nn
import pytorch_lightning as pl


class BinaryClassifier(pl.LightningModule):
    def __init__(self, layers_sizes, activation=None, loss='MSE',
                 keep_log=False):
        super().__init__()
        self.keep_log = keep_log
        self.layers = nn.Sequential()
        for i in range(len(layers_sizes)-1):
            self.layers.add_module('layer%d' % (i), nn.Linear(
                layers_sizes[i], layers_sizes[i+1]))

        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = None

        if loss == 'BCE':
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.MSELoss()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        for layer in self.layers:
            x = layer(x)
        if self.activation:
            x = self.activation(x)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        output = self.forward(x)
        y = y.view(y.size(0), 1)
        loss = self.loss(output, y)
        if self.keep_log:
            print("KEEP LOG")
            # Logging to TensorBoard by default
            self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
