from abc import ABC, abstractmethod
from typing import Sequence, Union

import torch
import torch.nn as nn


class Cell(ABC, nn.Module):
    """
    Highest level abstraction for a container representing a collection of recurrent cells
    """
    def __init__(self, d_hidden: int, num_rims: int, device):
        """

        :param d_hidden: size of a single rim
        :param num_rims: Number of RIMs a cell contains
        """
        super(Cell, self).__init__()

        self.d_hid = d_hidden
        self.num_rims = num_rims
        self.device = device

    @abstractmethod
    def forward(self, input: torch.Tensor, states: Union[Sequence[torch.Tensor], torch.Tensor]) \
            -> Union[Sequence[torch.Tensor], torch.Tensor]:
        pass

    @abstractmethod
    def apply_mask(self, old_states, new_states, mask):
        """
        Applies the given mask to the required states.
        :param old_states: States of the previous time step of the Cell e.g. hidden state, cell state etc.
        :param new_states: States of the current time step of the Cell e.g. hidden state, cell state etc.
        :param mask: applied to the different states to mask out unnecessary components.
        :return: states
        """
        pass

    @abstractmethod
    def init_states(self, bsz):
        """
        Initialise the states
        :param bsz: batch size
        :return: new states
        """
        pass

    @abstractmethod
    def reshape_states(self, states):
        pass

    def get_hidden_state(self, states):
        """
        Takes in cell specific sizes and return the hidden state in a canonical form
        :param states:
        :return:
        """
        if isinstance(states, Sequence):
            hx = states[0]
        else:
            hx = states
        return hx.reshape(-1, self.num_rims, self.d_hid)

    @abstractmethod
    def set_hidden_state(self, states, hx_new):
        """
        Updates state's current hidden state with a new hidden state. The new hidden state will be converted from the
        cell specific form to the canonical form
        :param states: in cell specific form
        :param hx_new: in canonical shape [B,num_rims, d_hid]
        :return:
        """
        pass

    def blockify_params(self) -> None:
        """
        Blockifies the weight matrices of the cell params.
        :return: None
        """
        pass
