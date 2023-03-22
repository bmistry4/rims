from typing import Sequence, Union, Tuple

import torch
import torch.nn as nn
from torch.nn import GRUCell, LSTMCell, Module
from torchtyping import TensorType

from ..cells._cell import Cell


class BlockCell(Cell):
    def __init__(self, recurrent_cell: Module, num_gates: int, d_input: int, d_hidden: int, num_rims: int, device: str):
        """
        A Cell where only parts of the cell parameters are required for representing RIMs. The cell can be split up into
        blocks where only some blocks are represent RIMs. All other blocks are zeroed out resulting in
        sparse parameters.

        :param recurrent_cell: Recurrent cell module
        :param num_gates: Number of internal gates which are part of the cell. E.g. the LSTM has 4 gates -
            input, forget, reset and output.
        :param d_input: Input dimension size for the Cell for a single block (RIM)
        :param d_hidden: Hidden dimension size for the Cell for a single block (RIM)
        :param num_rims: Number of RIMs a cell can represent
        """
        super(BlockCell, self).__init__(d_hidden, num_rims, device)

        self.d_in = d_input
        self.recurrent_cell = recurrent_cell
        self.num_gates = num_gates

        self.d_hid_all_rims = self.d_hid * self.num_rims
        self.d_in_all_rims = self.d_in * self.num_rims

    def set_hidden_state(self, states, hx_new):
        """
        Takes in states in their canonical shape and sets the hidden state in the cell specific shape
        :param states:
        :param hx_new:
        :return:
        """
        hx_new = hx_new.reshape(-1, self.num_rims * self.d_hid)
        if isinstance(states, Sequence):
            return type(states)((hx_new, *states[1:]))
        return hx_new

    def zero_matrix_elements(self, matrix: TensorType["d", "d"], k: int) -> None:
        """
        Given a matrix, split it into blocks with k rows and columns where all the blocks not part of the diagonal are
        set to zero.
        :param matrix:
        :param k: number of diagonal blocks you want. (Each diagonal block will represent a single RIM).
        :return:
        """
        assert matrix.shape[0] % k == 0
        assert matrix.shape[1] % k == 0
        g1 = matrix.shape[0] // k
        g2 = matrix.shape[1] // k
        new_mat = torch.zeros_like(matrix)

        # retain the values for the blocks on the diagonal
        for b in range(0, k):
            new_mat[b * g1: (b + 1) * g1, b * g2: (b + 1) * g2] += matrix[b * g1: (b + 1) * g1, b * g2: (b + 1) * g2]

        matrix *= 0.0
        matrix += new_mat

    def blockify_params(self) -> None:
        """
        Iterate through the Cell's learnable parameters and for the parameters referring to the weight matrices applied
        during gating apply two steps:
        1) Blockify the matrix such that the matrix is partitioned into k rows and columns
        2) Zero the non-diagonal blocks of the matrix (representing the irrelevant parameters)
        Assume the diagonal blocks correspond to different RIMs.
        :return: None
        """
        # d_hid_all_rims = self.d_hid * self.num_rims
        # d_in_all_rims = self.d_in * self.num_rims

        pl = self.recurrent_cell.parameters()
        for p in pl:
            p = p.data
            # ignore the parameters referring to the biases
            if p.shape == torch.Size([self.d_hid_all_rims * self.num_gates]):
                pass
            # Matrices which get matrix multiplied with either the input or the hidden state
            if p.shape == torch.Size([self.d_hid_all_rims * self.num_gates, self.d_hid_all_rims]) or \
                p.shape == torch.Size([self.d_hid_all_rims * self.num_gates, self.d_in_all_rims]):
                for e in range(0, self.num_gates):
                    # select a single gate's worth of rows (d_hid/d_in) and all the columns (d_hid/d_din) and
                    # zero out the non-diagonal blocks
                    self.zero_matrix_elements(p[self.d_hid_all_rims * e: self.d_hid_all_rims * (e + 1)], k=self.num_rims)

    def forward(self, input: TensorType["batch", "num_rims", "d_input"],
                states: Union[Sequence[torch.Tensor], torch.Tensor]
                ) -> Union[Sequence[torch.Tensor], torch.Tensor]:

        # [b, num_rim, d_in] -> [b, num_rim * d_in]
        input = input.reshape((input.shape[0], self.num_rims * self.d_in))

        # independent dynamics (assumes blockify_params has already been called)
        states = self.recurrent_cell(input, states)
        return states


class BlockCellLSTM(BlockCell):
    """
    An LSTM based Cell which uses blocks of parameters to represent RIMs.
    """
    def __init__(self, num_rims, d_input, d_hidden, device):
        """
        :param num_rims: Number of RIMs a cell can represent
        :param d_input: Input dimension size for a single RIM
        :param d_hidden: Hidden dimension size for a single RIM
        """
        super(BlockCellLSTM, self).__init__(recurrent_cell=LSTMCell(num_rims * d_input, num_rims * d_hidden),
                                            num_gates=4, d_input=d_input, d_hidden=d_hidden, num_rims=num_rims,
                                            device=device)

    def apply_mask(self,
                   old_states: Tuple[TensorType["batch", "num_rims * d_hidden"],
                                     TensorType["batch", "num_rims * d_hidden"]],
                   new_states: Tuple[TensorType["batch", "num_rims * d_hidden"],
                                     TensorType["batch", "num_rims * d_hidden"]],
                   mask: TensorType["batch", "num_rims", "d_hidden"]
                   ) -> Tuple[TensorType["batch", "num_rims * d_hidden"],
                              TensorType["batch", "num_rims * d_hidden"]]:
        """
        Apply masking to each state resulting in the final states for the timestep.
        States will use the new_state values for important RIMs (mask value of 1) and retain the old_state
        values for the unimportant RIMS (those with a mask value of 0).
        :param old_states: states before independent dynamics
        :param new_states: states after independent dynamics and communication attention
        :param mask: represents which RIMs should be ignored
        :return: final states for a timestep
        """
        hx_old, cx_old = old_states
        hx_new, cx_new = new_states
        bsz = hx_old.shape[0]

        # [b, num_rims, d_hid] -> [b, num_rims * d_hid]
        mask = mask.reshape(bsz, self.num_rims * self.d_hid)

        hx = mask * hx_new + (1 - mask) * hx_old
        cx = mask * cx_new + (1 - mask) * cx_old

        return hx, cx

    def init_states(self, bsz: int) -> \
            Tuple[TensorType["batch", "num_rims * d_hidden"], TensorType["batch", "num_rims * d_hidden"]]:

        # init hx and cx
        return torch.zeros(bsz, self.num_rims * self.d_hid, device=self.device), \
               torch.zeros(bsz, self.num_rims * self.d_hid, device=self.device)

    def reshape_states(self,
                       states: Tuple[TensorType["batch", "num_rims * d_hidden"],
                                     TensorType["batch", "num_rims * d_hidden"]]
                       ) -> Tuple[TensorType["batch", "num_rims", "d_hidden"],
                                  TensorType["batch", "num_rims", "d_hidden"]]:
        """
        Reshape states (hx,cx) to have its own RIM dimension of size num_rims.
        :param states: hidden state, cell state
        :return: reshaped hidden state, cell state
        """
        hx, cx = states
        return hx.view(hx.shape[0], self.num_rims, self.d_hid), \
               cx.view(cx.shape[0], self.num_rims, self.d_hid)


class BlockCellGRU(BlockCell):
    """
    An GRU based Cell which uses blocks of parameters to represent RIMs.
    """

    def __init__(self, num_rims, d_input, d_hidden, device):
        """
        :param num_rims: Number of RIMs a cell can represent
        :param d_input: Input dimension size for a single RIM
        :param d_hidden: Hidden dimension size for a single RIM
        """
        super(BlockCellGRU, self).__init__(recurrent_cell=GRUCell(num_rims * d_input, num_rims * d_hidden),
                                            num_gates=3, d_input=d_input, d_hidden=d_hidden, num_rims=num_rims,
                                           device=device)

    def apply_mask(self,
                   old_states: TensorType["batch", "num_rims * d_hidden"],
                   new_states: TensorType["batch", "num_rims * d_hidden"],
                   mask: TensorType["batch", "num_rims", "d_hidden"]
                   ) -> TensorType["batch", "num_rims * d_hidden"]:
        """
        Apply masking to each state resulting in the final states for the timestep.
        States will use the new_state values for important RIMs (mask value of 1) and retain the old_state
        values for the unimportant RIMS (those with a mask value of 0).
        :param old_states: states before independent dynamics
        :param new_states: states after independent dynamics and communication attention
        :param mask: represents which RIMs should be ignored
        :return: final states for a timestep
        """
        hx_old = old_states
        hx_new = new_states
        bsz = hx_old.shape[0]

        # [b, num_rims, d_hid] -> [b, num_rims * d_hid]
        mask = mask.reshape(bsz, self.num_rims * self.d_hid)
        hx = mask * hx_new + (1 - mask) * hx_old

        return hx

    def init_states(self, bsz: int) \
            -> TensorType["batch", "num_rims * d_hidden"]:
        # init hx
        return torch.zeros(bsz, self.num_rims * self.d_hid, device=self.device)

    def reshape_states(self, states: TensorType["batch", "num_rims * d_hidden"]) \
            -> TensorType["batch", "num_rims", "d_hidden"]:
        """
        Reshape states (=hx) to have its own RIM dimension of size num_rims.
        :param states: hidden state
        :return: reshaped hidden state
        """
        return states.view(states.shape[0], self.num_rims, self.d_hid)


if __name__ == '__main__':
    num_rims = 6
    d_input = 10
    d_hidden = 100
    bsz = 64
    ###################################################################################################################
    block_gru = BlockCellGRU(num_rims, d_input, d_hidden)
    input = torch.zeros(bsz, d_input)
    hx = block_gru.init_states(bsz)
    hx = block_gru(input, hx)
    assert hx.shape == torch.zeros(bsz, num_rims * d_hidden).shape, "Invalid states shape"
    reshaped_hx = block_gru.reshape_states(hx)

    ####################################################################################################################

    block_lstm = BlockCellLSTM(num_rims, d_input, d_hidden)
    input = torch.zeros(bsz, d_input)
    states = block_lstm.init_states(bsz)
    states = block_lstm(input, states)
    hx, cx = states
    assert hx.shape == torch.zeros(bsz, num_rims * d_hidden).shape, "Invalid states shape"
    assert cx.shape == torch.zeros(bsz, num_rims * d_hidden).shape, "Invalid states shape"
    reshaped_states = block_lstm.reshape_states(states)