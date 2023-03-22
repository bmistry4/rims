import unittest

import torch

from ..batched_cell import BatchCellGRU, BatchCellLSTM


class TestBatchedCellGRU(unittest.TestCase):
    def setUp(self) -> None:
        self.num_rims = 6
        self.bsz = 64
        self.d_hidden = 100
        d_input = 10

        self.batched_gru = BatchCellGRU(self.num_rims, d_input, self.d_hidden)
        self.input = torch.zeros(self.bsz, d_input * self.num_rims)
        self.hx = self.batched_gru.init_states(self.bsz)

    def test_new_state_sizes(self):
        hx = self.batched_gru.init_states(self.bsz)
        self.assertEqual(hx.shape, torch.zeros(self.bsz * self.num_rims, self.d_hidden).shape)

    def test_in_out_state_sizes(self):
        new_hx = self.batched_gru(self.input, self.hx)
        self.assertEqual(self.hx.shape, new_hx.shape)

    def test_reshape_states(self):
        # check output of reshape states method
        reshaped_hx = self.batched_gru.reshape_states(self.hx)
        self.assertEqual(reshaped_hx.shape, torch.zeros(self.bsz, self.num_rims, self.d_hidden).shape)


class TestBatchedCellLSTM(unittest.TestCase):
    def setUp(self) -> None:
        self.num_rims = 6
        self.bsz = 64
        self.d_hidden = 100
        d_input = 10

        self.batched_lstm = BatchCellLSTM(self.num_rims, d_input, self.d_hidden)
        self.input = torch.zeros(self.bsz, d_input * self.num_rims)
        self.states = self.batched_lstm.init_states(self.bsz)

    def test_new_state_sizes(self):
        hx, cx = self.batched_lstm.init_states(self.bsz)
        self.assertEqual(hx.shape, cx.shape)
        self.assertEqual(hx.shape, torch.zeros(self.bsz * self.num_rims, self.d_hidden).shape)
        self.assertEqual(cx.shape, torch.zeros(self.bsz * self.num_rims, self.d_hidden).shape)

    def test_in_out_state_sizes(self):
        new_states = self.batched_lstm(self.input, self.states)
        self.assertEqual(self.states[0].shape, new_states[0].shape)
        self.assertEqual(self.states[1].shape, new_states[1].shape)

    def test_reshape_states(self):
        # check output of reshape states method
        reshaped_states = self.batched_lstm.reshape_states(self.states)
        self.assertEqual(reshaped_states[0].shape, torch.zeros(self.bsz, self.num_rims, self.d_hidden).shape)
        self.assertEqual(reshaped_states[1].shape, torch.zeros(self.bsz, self.num_rims, self.d_hidden).shape)


if __name__ == '__main__':
    unittest.main()
