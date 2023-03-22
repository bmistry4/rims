import torch
import torch.nn as nn
from torchtyping import TensorType

from ..cells.blocked_cell import BlockCell
from ..cells.rims_cell import RIMsCell
from ..utils.blocked_grad import BlockedGradients
from ..utils.helper import str_to_class


class RIMs(nn.Module):
    """
    Goes through each layer and each time step and applies the RIMsCell
    """

    def __init__(self, cell_name: str, d_enc_inp=600, d_hid_all_rims_per_layer=[600], num_rims_per_layer=[6],
                 num_active_rims_per_layer=[4], use_comm_attn=False, num_modules_read_input=2, in_num_head=1, comm_num_head=4,
                 in_d_k=64, in_d_v=400, comm_d_k=32, comm_d_v=32,
                 in_residual=False, comm_residual=True,
                 in_learn_head_weights=False, comm_learn_head_weights=True,
                 in_sparsifier=None, comm_sparsifier=None,
                 batch_first=False,
                 dropout=0.5,
                 use_inactive_rims=False, block_grads=False,
                 device='cpu'
                 ):
        super(RIMs, self).__init__()
        """
        NOTE: see rims_cell.py for any unexplained arguments

        :param cell_name: RIIMCell to use
        :param d_enc_inp: dimension of the input feature once encoded
        :param d_hid_all_rims_per_layer: total dim for all RIMs f.e. layer
        :param num_rims_per_layer: number of RIMs that each layer can contain
        :param num_active_rims_per_layer: number of RIMs to not mask out (set to 0) during input attn
        :param use_comm_attn: controls if communication attention is applied
        :param num_modules_read_input: todo
        :param in_num_head: number of heads for the input attention
        :param comm_num_head: number of heads for the communication attention
        :param in_d_k: input attention projection size for the keys
        :param in_d_v: input attention projection size for the values
        :param comm_d_k: communication attention projection size for the keys
        :param comm_d_v: communication attention projection size for the values
        :param in_residual: use a residual connection in the input attention
        :param comm_residual: use a residual connection in the communication attention
        :param in_learn_head_weights: if the input attention should learn the weighting of each head (or just use the mean)
        :param comm_learn_head_weights: if the communication attention should learn the weighting of each head (or just use the mean)
        :param in_sparsifier: input attention's sparsification function to apply to the attention tensor
        :param comm_sparsifier: communication attention's sparsification function to apply to the attention tensor
        :param batch_first: if the input should have the batch is the first dim (True) or second dim (False)
        :param dropout: dropout value to use (will be the same f.e. layer)
        :param use_inactive_rims: lets the inactive rims be used as part of the output 
        :param block_grads: blocks the gradients of the inactive rims contributing in the back pass
        """
        self.device = device
        self.batch_first = batch_first
        self.drop = nn.Dropout(dropout)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(len(num_rims_per_layer))])
        self.rims_cells = []
        self.use_inactive_rims = use_inactive_rims
        self.block_grads = block_grads
        self.in_d_v = in_d_v

        # populate a list of rims_cells (which are made of multiple rims). 1 rim cell for each layer.
        for i in range(len(num_rims_per_layer)):
            curr_layer_d_hid_single_rim = d_hid_all_rims_per_layer[i] // num_rims_per_layer[i]
            d_cell_inp = d_enc_inp if i == 0 else d_hid_all_rims_per_layer[i - 1]
            in_num_k_or_v = num_modules_read_input   # +1 to consider the null input k/v

            # create cell for the layer
            d_rim_cell_inp = self.in_d_v    # following dido1998's way
            cell = str_to_class(cell_name)(num_rims=num_rims_per_layer[i], d_input=d_rim_cell_inp,
                                           d_hidden=curr_layer_d_hid_single_rim, device=self.device)
            print(f"Cell type used: {cell._get_name()}\n")

            self.rims_cells.append(RIMsCell(cell, d_cell_inp, d_hid_all_rims_per_layer[i],
                                            num_rims_per_layer[i], num_active_rims_per_layer[i],
                                            use_comm_attn, in_num_k_or_v, in_num_head, comm_num_head,
                                            in_d_k, in_d_v,
                                            comm_d_k, comm_d_v,
                                            in_residual, comm_residual,
                                            in_learn_head_weights, comm_learn_head_weights,
                                            in_sparsifier, comm_sparsifier,
                                            device))
            self.rims_cells = nn.ModuleList(self.rims_cells)

    def forward(self, input: TensorType["timestep", "batch", "d_enc_inp"]) -> \
            TensorType["timestep", "batch", "num_rims", "d_rim"]:

        # apply droupout
        input = self.drop(input)

        # analogous to the LSTM flag which allows the batch be in dim 0.
        if self.batch_first:
            input = input.transpose(0, 1)

        num_layers = len(self.rims_cells)
        layer_input = input

        # loop through layers
        for l_idx in range(num_layers):
            states = None

            output = []                         # will collect the cell's hidden state f.e. timestep
            rims_cell = self.rims_cells[l_idx]

            # apply blockify to the cell params if you're using a BlockCell
            rims_cell.cell.blockify_params()

            # loop through timesteps
            for ts_idx in range(input.shape[0]):
                # mask = [B, num_rims, d_rim]
                states, mask = rims_cell(layer_input[ts_idx], states)
                # [B, num_rims, d_rim]
                hx = rims_cell.cell.get_hidden_state(states)

                # last layer
                if l_idx == len(self.rims_cells) - 1:
                    # let the inactive rims be used
                    if self.use_inactive_rims:
                        # block grads of inactive rims in backwards
                        if self.block_grads:
                            bg = BlockedGradients()
                            output.append(bg(hx, mask))
                        else:
                            output.append(hx)
                    # do not allow inactive rims to contribution to the output
                    else:
                        if self.block_grads:
                            bg = BlockedGradients()
                            output.append(mask * bg(hx, mask))
                        else:
                            output.append(mask * hx)
                # layers that aren't the last layer won't allow the inactive rims to be used for the output
                else:
                    output.append(mask * hx)

            # list[B, n_rims, d_rim] -> [timesteps, B, n_rims, d_rim]
            output = torch.stack(output)

            # apply dropout to all layers except the last
            if l_idx < num_layers - 1:
                layer_input = self.dropouts[l_idx](output)

        # apply dropout to last layer. Use same dropout instance as the one used for the input
        output = self.drop(output)

        # set the batch dimension back to being first
        if self.batch_first:
            output = output.transpose(1, 0)

        # return the stacked output for the last layer; shape = [timesteps, B, n_rims, d_rim]
        return output
