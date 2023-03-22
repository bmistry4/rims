from abc import ABC
from typing import Sequence, Union, Tuple

import torch
from torch.nn import GRUCell, LSTMCell, Module
from torchtyping import TensorType

from ..cells._cell import Cell


class BatchCell(Cell, ABC):
    def __init__(self, recurrent_cell: Module, d_input: int, d_hidden: int, num_rims: int, device: str):
        """
        A Cell where RIMs are represented as part of the

        :param recurrent_cell: Recurrent cell module
        :param d_input: Input dimension size for a single RIM
        :param d_hidden: Hidden dimension size for a single RIM
        :param num_rims: Number of RIMs a cell can represent
        """
        super(BatchCell, self).__init__(d_hidden, num_rims, device)
        self.d_in = d_input
        self.recurrent_cell = recurrent_cell

    def set_hidden_state(self, states, hx_new):
        """
        Takes in states in their canonical shape and sets the hidden state in the cell specific shape
        :param states:
        :param hx_new:
        :return:
        """
        bsz = hx_new.shape[0]
        hx_new = hx_new.reshape(bsz * self.num_rims, self.d_hid)
        if isinstance(states, Sequence):
            return type(states)((hx_new, *states[1:]))
        return hx_new

    def forward(self, input: TensorType["batch", "num_rims", "d_input"],
                states: Union[Sequence[torch.Tensor], torch.Tensor]
                ) -> Union[Sequence[torch.Tensor], torch.Tensor]:

        # [b, num_rim, d_in] -> [b * num_rim, d_in]
        input = input.reshape(input.shape[0] * self.num_rims, self.d_in)

        # independent dynamics
        states = self.recurrent_cell(input, states)
        return states


class BatchCellLSTM(BatchCell):
    """
    An LSTM based Cell which uses batches of parameters to represent RIMs.
    """

    def __init__(self, num_rims, d_input, d_hidden, device):
        """
        :param num_rims: Number of RIMs a cell can represent
        :param d_input: Input dimension size for for a single RIM.
        :param d_hidden: Hidden dimension size for a single RIM.
        """
        super(BatchCellLSTM, self).__init__(recurrent_cell=LSTMCell(d_input, d_hidden),
                                            d_input=d_input, d_hidden=d_hidden, num_rims=num_rims, device=device)

    def apply_mask(self,
                   old_states: Tuple[TensorType["batch * num_rims", "d_hidden"],
                                     TensorType["batch * num_rims", "d_hidden"]],
                   new_states: Tuple[TensorType["batch * num_rims", "d_hidden"],
                                     TensorType["batch * num_rims", "d_hidden"]],
                   mask: TensorType["batch", "num_rims", "d_hidden"]
                   ) -> Tuple[TensorType["batch * num_rims", "d_hidden"],
                              TensorType["batch * num_rims", "d_hidden"]]:
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
        # bsz = hx_old.shape[0] // self.num_rims      # bsz = batch * num_rims // num_rims

        # [b, num_rims, d_hid] -> [b * num_rims, d_hid]
        mask = mask.reshape(-1, self.d_hid)

        hx = mask * hx_new + (1 - mask) * hx_old
        cx = mask * cx_new + (1 - mask) * cx_old

        return hx, cx

    def init_states(self, bsz: int) -> Tuple[TensorType["batch * num_rims", "d_hidden"],
                                             TensorType["batch * num_rims", "d_hidden"]]:
        # init hx and cx
        return torch.zeros(bsz * self.num_rims, self.d_hid, device=self.device), \
               torch.zeros(bsz * self.num_rims, self.d_hid, device=self.device)

    def reshape_states(self, states: Tuple[TensorType["batch * num_rims", "d_hidden"],
                                           TensorType["batch * num_rims", "d_hidden"]]
                       ) -> Tuple[TensorType["batch", "num_rims", "d_hidden"],
                                  TensorType["batch", "num_rims", "d_hidden"]]:
        """
        Reshape states (hx,cx) to have its own RIM dimension of size num_rims.
        :param states: hidden state, cell state
        :return: reshaped states
        """
        hx, cx = states
        bsz = hx.shape[0] // self.num_rims      # fixme: for speed, cmmt out this line and use -1 in view
        return hx.view(bsz, self.num_rims, self.d_hid), \
               cx.view(bsz, self.num_rims, self.d_hid)


class BatchCellGRU(BatchCell):
    """
    An LSTM based Cell which uses batches of parameters to represent RIMs.
    """

    def __init__(self, num_rims, d_input, d_hidden, device):
        """
        :param num_rims: Number of RIMs a cell can represent
        :param d_input: Input dimension size for a single RIM
        :param d_hidden: Hidden dimension size for a single RIM
        """
        super(BatchCellGRU, self).__init__(recurrent_cell=GRUCell(d_input, d_hidden),
                                           d_input=d_input, d_hidden=d_hidden, num_rims=num_rims, device=device)

    def apply_mask(self,
                   old_states: TensorType["batch * num_rims", "d_hidden"],
                   new_states: TensorType["batch * num_rims", "d_hidden"],
                   mask: TensorType["batch", "num_rims", "d_hidden"]
                   ) -> TensorType["batch * num_rims", "d_hidden"]:
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
        # bsz = hx_old.shape[0] // self.num_rims  # bsz = batch * num_rims // num_rims

        # [b, num_rims, d_hid] -> [b * num_rims, d_hid]
        mask = mask.reshape(-1, self.d_hid)

        hx = mask * hx_new + (1 - mask) * hx_old
        return hx

    def init_states(self, bsz: int) -> TensorType["batch * num_rims", "d_hidden"]:
        # init hx
        return torch.zeros(bsz * self.num_rims, self.d_hid, device=self.device)

    # fixme - state types should be Sequence not just single TensorType. Need to do this for all cell impls
    def reshape_states(self, states: TensorType["batch * num_rims", "d_hidden"]) \
            -> TensorType["batch", "num_rims", "d_hidden"]:
        """
        Reshape states (=hx) to have its own RIM dimension of size num_rims.
        :param states: hidden state
        :return: reshaped hidden state
        """
        bsz = states.shape[0] // self.num_rims
        return states.view(bsz, self.num_rims, self.d_hid)

if __name__ == '__main__':
    num_rims = 6
    d_input = 10
    d_hidden = 100
    bsz = 64
    ###################################################################################################################
    block_gru = BatchCellGRU(num_rims, d_input, d_hidden)
    input = torch.zeros(bsz, d_input)
    hx = block_gru.init_states(bsz)
    hx = block_gru(input, hx)
    assert hx.shape == torch.zeros(bsz * num_rims, d_hidden).shape, "Invalid states shape"
    reshaped_hx = block_gru.reshape_states(hx)

    ####################################################################################################################

    block_lstm = BatchCellLSTM(num_rims, d_input, d_hidden)
    input = torch.zeros(bsz, d_input)
    states = block_lstm.init_states(bsz)
    states = block_lstm(input, states)
    hx, cx = states
    assert hx.shape == torch.zeros(bsz * num_rims, d_hidden).shape, "Invalid states shape"
    assert cx.shape == torch.zeros(bsz * num_rims, d_hidden).shape, "Invalid states shape"
    reshaped_states = block_lstm.reshape_states(states)