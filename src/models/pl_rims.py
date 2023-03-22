import pytorch_lightning as pl
import torch
from torch import nn

from .layers.rims import RIMs
from .layers.vanilla_lstm import VanillaLSTM
from .projections.decoder import Decoder
from .projections.encoder import Encoder


class LitRIM(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.all_epochs_cell_fp_time = 0
        self.epoch_cell_fp_count = 0

        # log hyperparameters
        self.save_hyperparameters()

        self.args = args
        d_enc_inp = args.d_enc_inp
        num_tokens = args.num_tokens
        dropout = args.dropout
        d_out = args.d_out

        # decoder input dim is the same size as the last Cell layer's hidden dim
        d_dec_in, d_dec_out = args.d_hid_all_rims_per_layer[-1], d_out

        self.encoder = Encoder(num_tokens=num_tokens, d_out=d_enc_inp, dropout=dropout)

        # select the type of layer. Either lstm or rim. It's possible to use a lstm cell in a rim layer.
        if args.cell == 'VanillaLSTMCell' and args.use_vanilla_lstm_layer:
            recurrent_layer = VanillaLSTM
        else:
            recurrent_layer = RIMs

        self.rim = recurrent_layer(args.cell,
                        d_enc_inp=d_enc_inp, d_hid_all_rims_per_layer=args.d_hid_all_rims_per_layer,
                        num_rims_per_layer=args.num_rims_per_layer,
                        num_active_rims_per_layer=args.num_active_rims_per_layer,
                        use_comm_attn=args.use_comm_attn,
                        num_modules_read_input=args.num_modules_read_input,
                        in_num_head=args.in_num_head, comm_num_head=args.comm_num_head,
                        in_d_k=args.in_d_k, in_d_v=args.in_d_v, comm_d_k=args.comm_d_k, comm_d_v=args.comm_d_v,
                        in_residual=args.in_residual, comm_residual=args.comm_residual,
                        in_learn_head_weights=False, comm_learn_head_weights=True,
                        in_sparsifier=None, comm_sparsifier=None,
                        batch_first=args.batch_first, dropout=args.dropout,
                        use_inactive_rims=args.use_inactive_rims, block_grads=args.block_grads,
                        device=args.device)

        self.decoder = Decoder(d_dec_in, d_dec_out)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # dataloader will by default give batch first data, so it needs to be transposed to become timestep first.
        x = x.t()                          # [B,S] -> [S,B]
        x = self.encoder(x)                # [S,B] -> [S, B, d_enc]
        output = self.rim(x)               # [S, B, n_rims, d_rim]
        output = self.decoder(output)      # [S*B, d_out]
        return output  # [S*B,O]

    def step(self, batch, log_last_n=0):
        """
        Wrapper for an epoch step for the train, val and test
        :param batch: batch [B,S]
        :param log_last_n: Logs loss over last n timesteps of a sequence. If None then ignore.
        :return: loss
        """
        last_n_ts_loss = None
        x, y = batch

        bsz = x.shape[0]
        d_seq = x.shape[1]

        y_hat = self(x)     # [S*B, d_out]
        # dataloader will by default give batch first data, so it needs to be transposed to match y_hat
        y = y.t()           # [B,S] -> [S,B]
        # Reshape from [S, B] -> [S * B]
        y = y.reshape(y.shape[0] * y.shape[1])

        if log_last_n > 0:
            last_n_ts_loss = self.last_n_ts_loss(y_hat, y, d_seq=d_seq, bsz=bsz, n=log_last_n)

        loss = self.criterion(y_hat, y)

        return loss, last_n_ts_loss

    def last_n_ts_loss(self, y_hat, y, d_seq, bsz, n=0):
        """
        Calc the loss over the last n timesteps f.e. sequence
        :param y_hat: [S*B, d_out]
        :param y: [S * B]
        :param n: no. of timesteps to look back on
        :return: loss
        """
        loss = self.criterion(
            y_hat.reshape(d_seq, bsz, -1)[-n:, :, :].reshape(n*bsz, -1),
            y.reshape(d_seq, bsz)[-n:, ].reshape(n*bsz)
        )
        return loss

    def training_step(self, batch, batch_idx):
        # todo - cmmt out when running real exps
        # if self.current_epoch == 0:
        #     seq_len = self.args.train_copy_len + self.args.train_zeros_len + self.args.train_pad_len
        #     print(batch[0].shape, torch.Tensor((self.args.batch_size, seq_len)).shape)
        #     assert batch[0].shape == torch.Tensor(batch[0].shape[0], seq_len).shape, \
        #         f"X: got: {batch[0].shape}, should be:{torch.Tensor(batch[0].shape[0], seq_len).shape}"
        #     assert batch[1].shape == torch.Tensor(batch[1].shape[0], seq_len).shape, \
        #         f"Y: got: {batch[1].shape}, should be:{torch.Tensor(batch[1].shape[0], seq_len).shape}"

        # training_step defines the train loop. It is independent of forward
        loss, last_n_ts_loss = self.step(batch, log_last_n=self.args.log_last_n)

        # print loss f.e. batch
        # if batch_idx % 1 == 0:
        #     print(f" loss {loss.item()} | loss (last 10) {last_n_ts_loss.item()} |")

        # average loss over epoch
        self.log("metric/train/loss/all", loss.item(), on_step=False, on_epoch=True)
        if self.args.log_last_n > 0:
            self.log(f"metric/train/loss/last-{self.args.log_last_n}", last_n_ts_loss.item(), on_step=False, on_epoch=True)

        # can't return .item must be Tensor
        return loss

    # def training_epoch_end(self, outputs):
    #     # print the last 10 timesteps CE loss for the validation (interpolation) and test (extrapolation) datasets
    #     train_loss = sum(map(lambda x: x['loss'].item(), outputs))/len(outputs)
    #
    #     print(f"\nTrain (all): {train_loss}")

    def validation_step(self, batch, batch_idx, dataloader_nb=None):
        state = 'valid' if dataloader_nb == 0 else 'test'
        loss, last_n_ts_loss = self.step(batch, log_last_n=self.args.log_last_n)
        loss, last_n_ts_loss = loss.item(), last_n_ts_loss.item()

        self.log(f'metric/{state}/loss/all', loss, on_step=False, on_epoch=True)
        if self.args.log_last_n > 0:
            self.log(f"metric/{state}/loss/last-{self.args.log_last_n}", last_n_ts_loss, on_step=False, on_epoch=True)

        return loss, last_n_ts_loss, state

    # def validation_epoch_end(self, outputs):
    #     # print the last 10 timesteps CE loss for the validation (interpolation) and test (extrapolation) datasets
    #     valid_last_n_ts_loss = sum(map(lambda x: x[1], outputs[0])) / len(outputs[0])
    #     test_last_n_ts_loss = sum(map(lambda x: x[1], outputs[1])) / len(outputs[1])
    #
    #     print(f"\nValid (last 10): {valid_last_n_ts_loss}")
    #     print(f"Test (last 10): {test_last_n_ts_loss}")

    def test_step(self, batch, batch_idx):
        loss, last_n_ts_loss = self.step(batch, log_last_n=self.args.log_last_n)
        loss, last_n_ts_loss = loss.item(), last_n_ts_loss.item()

        self.log('metric/test/loss/all', loss, on_step=False, on_epoch=True)
        if self.args.log_last_n > 0:
            self.log(f"metric/test/loss/last-{self.args.log_last_n}", last_n_ts_loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return optimizer
