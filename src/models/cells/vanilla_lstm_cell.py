from abc import ABC
from typing import Sequence, Union, Tuple

import torch
from torch.nn import LSTMCell
from torchtyping import TensorType

from ..cells._cell import Cell


class VanillaLSTMCell(Cell, ABC):
    """
    Highest level abstraction for a container representing a collection of recurrent cells
    """
    def __init__(self, d_input: int, d_hidden: int, num_rims: int = 1, device='cpu'):
        """

        :param d_hidden: size of a single rim
        """
        super(VanillaLSTMCell, self).__init__(d_hidden, num_rims, device)
        self.recurrent_cell = LSTMCell(d_input, d_hidden)
        self.d_in = d_input
        self.d_hid = d_hidden
        assert num_rims == 1, "Cannot have multiple RIMs in a Vanilla LSTM"

    def forward(self, input: TensorType["batch", "d_input"],
                states: TensorType["batch", "d_hidden"]
                ) -> Union[Sequence[torch.Tensor], torch.Tensor]:

        # independent dynamics
        states = self.recurrent_cell(input, states)

        return states

    def init_states(self, bsz: int) -> Tuple[TensorType["batch", "d_hidden"],
                                             TensorType["batch", "d_hidden"]]:
        # init hx and cx
        return torch.zeros(bsz, self.d_hid, device=self.device), torch.zeros(bsz, self.d_hid, device=self.device)

    def apply_mask(self,
                   old_states: Tuple[TensorType["batch", "d_hidden"],
                                     TensorType["batch", "d_hidden"]],
                   new_states: Tuple[TensorType["batch", "d_hidden"],
                                     TensorType["batch", "d_hidden"]],
                   mask: TensorType["batch", "num_rims", "d_hidden"]
                   ) -> Tuple[TensorType["batch", "d_hidden"],
                              TensorType["batch ", "d_hidden"]]:
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
        bsz = hx_old.shape[0] // self.num_rims  # bsz = batch * num_rims // num_rims

        # [b, num_rims, d_hid] -> [b * num_rims, d_hid]
        mask = mask.reshape(bsz * self.num_rims, self.d_hid)

        hx = mask * hx_new + (1 - mask) * hx_old
        cx = mask * cx_new + (1 - mask) * cx_old

        return hx, cx

    # TODO - is reshape_states actually used anywhere? If not remove
    def reshape_states(self, states):
        """
        Reshape states (hx,cx) to have its own RIM dimension of size num_rims.
        :param states: hidden state, cell state
        :return: reshaped hidden state, cell state
        """
        # pass  # fixme - don't need to impl unless VanillaLSTM is used as a cell in the rims_cell class
        return states

    def set_hidden_state(self, states, hx_new):
        """
        Takes in states in their canonical shape and sets the hidden state in the cell specific shape
        :param states:
        :param hx_new:
        :return:
        """
        bsz = hx_new.shape[0]
        # [B,num_rim=1, d_hid] -> [B, d_hid]
        hx_new = hx_new.reshape(bsz, self.d_hid)
        if isinstance(states, Sequence):
            return type(states)((hx_new, *states[1:]))
        return hx_new
