import unittest

import torch

from ..rims import RIMs
from src.parser_args import create_base_parser


def create_parser(args):
    # Parse arguments
    parser = create_base_parser()
    args = parser.parse_args(args)
    # print(' '.join(f'{k}: {v}\n' for k, v in vars(args).items()))
    return args


class TestRIMs(unittest.TestCase):

    def setUp(self):
        self.num_timesteps = 3
        self.args = create_parser([])
        self.cell_names_list = ['BlockCellGRU', 'BlockCellLSTM', 'BatchCellGRU', 'BatchCellLSTM']

    def create_RIMs(self, cell_name, args):
        return RIMs(cell_name,
                    d_enc_inp=args.d_enc_inp, d_hid_all_rims_per_layer=args.d_hid_all_rims_per_layer,
                    num_rims_per_layer=args.num_rims_per_layer,
                    num_masked_rims_per_layer=args.num_masked_rims_per_layer,
                    num_modules_read_input=args.num_modules_read_input,
                    in_num_head=args.in_num_head, comm_num_head=args.comm_num_head,
                    in_d_k=args.in_d_k, comm_d_k=args.comm_d_k, comm_d_v=args.comm_d_v,
                    in_residual=False, comm_residual=True,
                    in_learn_head_weights=False, comm_learn_head_weights=True,
                    in_sparsifier=None, comm_sparsifier=None,
                    batch_first=args.batch_first, dropout=args.dropout,
                    use_inactive_rims=args.use_inactive_rims, block_grads=args.block_grads)

    def test_single_layer_all_cells_output_shape(self):
        """
        Go through each different cell type and check if the resulting output of calling the cell is as expected.
        Assume a single layer RIM is used.
        :return:
        """
        # change args to use multilayer RIMs.
        self.args.d_enc_inp = 100
        self.args.d_hid_all_rims_per_layer = [600]
        self.args.num_rims_per_layer = [6]
        self.args.num_masked_rims_per_layer = [4]
        print(self.args)

        for cell_name in self.cell_names_list:
            rims = self.create_RIMs(cell_name, self.args)

            layer_idx = 0
            num_rims = self.args.num_rims_per_layer[layer_idx]
            d_rim = self.args.d_hid_all_rims_per_layer[layer_idx] // num_rims

            inp = torch.rand(self.num_timesteps, self.args.batch_size, self.args.d_enc_inp)
            output = rims(inp)
            self.assertEqual(output.shape, torch.rand(self.num_timesteps, self.args.batch_size,
                                                      num_rims, d_rim).shape, "Wrong output shape")

    def test_multilayer_all_cells_output_shape(self):
        # change args to use multilayer RIMs.
        self.args.d_enc_inp = 100
        self.args.d_hid_all_rims_per_layer = [600, 600, 600]
        self.args.num_rims_per_layer = [6, 6, 6]
        self.args.num_masked_rims_per_layer = [4, 4, 4]
        print(self.args)

        layer_idx = len(self.args.d_hid_all_rims_per_layer) - 1

        for cell_name in self.cell_names_list:
            rims = self.create_RIMs(cell_name, self.args)

            num_rims = self.args.num_rims_per_layer[layer_idx]
            d_rim = self.args.d_hid_all_rims_per_layer[layer_idx] // num_rims

            inp = torch.rand(self.num_timesteps, self.args.batch_size, self.args.d_enc_inp)
            output = rims(inp)
            self.assertEqual(output.shape, torch.rand(self.num_timesteps, self.args.batch_size,
                                                      num_rims, d_rim).shape, "Wrong output shape")

    # fixme
    def test_different_layer_dims_all_cells_output_shape(self):
        # change args to use multilayer RIMs.
        self.args.d_enc_inp = 100
        self.args.d_hid_all_rims_per_layer = [600, 400]
        self.args.num_rims_per_layer = [6, 4]
        self.args.num_masked_rims_per_layer = [4, 2]
        print(self.args)

        layer_idx = len(self.args.d_hid_all_rims_per_layer) - 1

        for cell_name in self.cell_names_list:
            rims = self.create_RIMs(cell_name, self.args)

            num_rims = self.args.num_rims_per_layer[layer_idx]
            d_rim = self.args.d_hid_all_rims_per_layer[layer_idx] // num_rims

            inp = torch.rand(self.num_timesteps, self.args.batch_size, self.args.d_enc_inp)
            output = rims(inp)
            self.assertEqual(output.shape, torch.rand(self.num_timesteps, self.args.batch_size,
                                                      num_rims, d_rim).shape, "Wrong output shape")


if __name__ == '__main__':
    unittest.main()
