import torch
import torch.nn as nn
import pytorch_lightning as pl


class BinaryClassifier(pl.LightningModule):
    def __init__(self, input_size, output_size, activation=None, loss='MSE'):
        super().__init__()
        self.layer0 = nn.Linear(input_size, output_size)

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
        output = self.layer0(x)
        if self.activation:
            output = self.activation(output)
        return output
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        #x = x.view(x.size(0), -1)
        output = self.forward(x)
        loss = self.loss(output, y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer