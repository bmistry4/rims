import unittest

import torch

from ..blocked_cell import BlockCellGRU, BlockCellLSTM


class TestBlockedCellGRU(unittest.TestCase):
    def setUp(self) -> None:
        self.num_rims = 4  # 6
        self.bsz = 3  # 64
        self.d_hidden = 5  # 100
        self.d_input = 2  # 10

        self.block_gru = BlockCellGRU(self.num_rims, self.d_input, self.d_hidden)
        self.input = torch.zeros(self.bsz, self.d_input * self.num_rims)
        self.hx = self.block_gru.init_states(self.bsz)

    def test_new_state_sizes(self):
        hx = self.block_gru.init_states(self.bsz)
        self.assertEqual(hx.shape, torch.zeros(self.bsz, self.num_rims * self.d_hidden).shape)

    def test_in_out_state_sizes(self):
        new_hx = self.block_gru(self.input, self.hx)
        self.assertEqual(self.hx.shape, new_hx.shape)

    def test_reshape_states(self):
        # check output of reshape states method
        reshaped_hx = self.block_gru.reshape_states(self.hx)
        self.assertEqual(reshaped_hx.shape, torch.zeros(self.bsz, self.num_rims, self.d_hidden).shape)

    def test_blockify_params(self):
        #  (3*hidden_size*rims, input_size*rims)
        self.block_gru.blockify_params()
        p = self.block_gru.recurrent_cell.weight_ih.data
        rows_dim, cols_dim = p.shape
        single_gate_row_dim = rows_dim // self.block_gru.num_gates
        single_block_col_dim = cols_dim // self.num_rims
        for g in range(self.block_gru.num_gates):
            # get a gate's worth of weights
            gate = p[g * single_gate_row_dim: (g + 1) * single_gate_row_dim, :]
            for r_idx in range(self.num_rims):
                # get a diagonal block (which reps a RIM)
                rim_block = gate[r_idx * self.d_hidden: (r_idx + 1) * self.d_hidden,
                            r_idx * single_block_col_dim: (r_idx + 1) * single_block_col_dim]
                # assume RIMs won't be exactly 0
                self.assertEqual(rim_block.count_nonzero().sum(), rim_block.shape[0] * rim_block.shape[1])

                # check everything else is 0 by checking that the number of non-0 elems is equal to the total number of
                # RIM block elems
                self.assertEqual(gate.count_nonzero().sum(), self.num_rims * self.d_hidden * self.d_input)


class TestBlockedCellLSTM(unittest.TestCase):
    def setUp(self) -> None:
        self.num_rims = 6
        self.bsz = 64
        self.d_hidden = 100
        self.d_input = 10

        self.block_lstm = BlockCellLSTM(self.num_rims, self.d_input, self.d_hidden)
        self.input = torch.zeros(self.bsz, self.d_input * self.num_rims)
        self.states = self.block_lstm.init_states(self.bsz)

    def test_new_state_sizes(self):
        hx, cx = self.block_lstm.init_states(self.bsz)
        self.assertEqual(hx.shape, cx.shape)
        self.assertEqual(hx.shape, torch.zeros(self.bsz, self.num_rims * self.d_hidden).shape)
        self.assertEqual(cx.shape, torch.zeros(self.bsz, self.num_rims * self.d_hidden).shape)

    def test_in_out_state_sizes(self):
        new_states = self.block_lstm(self.input, self.states)
        self.assertEqual(self.states[0].shape, new_states[0].shape)
        self.assertEqual(self.states[1].shape, new_states[1].shape)

    def test_reshape_states(self):
        # check output of reshape states method
        reshaped_states = self.block_lstm.reshape_states(self.states)
        self.assertEqual(reshaped_states[0].shape, torch.zeros(self.bsz, self.num_rims, self.d_hidden).shape)
        self.assertEqual(reshaped_states[1].shape, torch.zeros(self.bsz, self.num_rims, self.d_hidden).shape)

    def test_blockify_params(self):
        self.block_lstm.blockify_params()
        p = self.block_lstm.recurrent_cell.weight_ih.data
        rows_dim, cols_dim = p.shape
        single_gate_row_dim = rows_dim // self.block_lstm.num_gates
        single_block_col_dim = cols_dim // self.num_rims
        for g in range(self.block_lstm.num_gates):
            # get a gate's worth of weights
            gate = p[g * single_gate_row_dim: (g + 1) * single_gate_row_dim, :]
            for r_idx in range(self.num_rims):
                # get a diagonal block (which reps a RIM)
                rim_block = gate[r_idx * self.d_hidden: (r_idx + 1) * self.d_hidden,
                            r_idx * single_block_col_dim: (r_idx + 1) * single_block_col_dim]
                # assume RIMs won't be exactly 0
                self.assertEqual(rim_block.count_nonzero().sum(), rim_block.shape[0] * rim_block.shape[1])

                # check everything else is 0 by checking that the number of non-0 elems is equal to the total number of
                # RIM block elems
                self.assertEqual(gate.count_nonzero().sum(), self.num_rims * self.d_hidden * self.d_input)


if __name__ == '__main__':
    unittest.main()
