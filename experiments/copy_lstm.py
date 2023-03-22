import pytorch_lightning as pl
import torch
from torch import nn


class LitLSTM(pl.LightningModule):
    def __init__(self, args):
        self.args = args
        input_size, hidden_size, output_size = self.args.input_size, self.args.hidden_size, self.args.output_size

        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, bias=True, batch_first=True, bidirectional=False,
                            proj_size=output_size)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # [B,S] -> [B, S, I=1]
        x = x.view(x.shape[0], x.shape[1], 1)
        # [B,S,H_out=9]
        output, _ = self.lstm(x)
        return output  # [B,S,O]

    def step(self, batch):
        x, y = batch
        y_hat = self(x)
        # Reshape from [B,S,O] -> [B*S,O]
        # TODO - view/reshape - which is more efficient?
        y_hat = y_hat.contiguous().view(y_hat.shape[0] * y_hat.shape[1], y_hat.shape[2])
        # Reshape from [B,S] -> [B*S]
        y = y.reshape(y.shape[0] * y.shape[1])

        loss = self.criterion(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        loss = self.step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return optimizer
