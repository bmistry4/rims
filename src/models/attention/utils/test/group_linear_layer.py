import unittest

import torch

from ..group_linear_layer import GroupLinearLayer


class TestGroupLinearLayer(unittest.TestCase):

    def setUp(self):
        self.din, self.dout, self.num_rims = 100, 64, 6

    def test_out_shape(self):
        GLN = GroupLinearLayer(self.din, self.dout, self.num_rims)
        batch_size = 128
        x = torch.randn(batch_size, self.num_rims, self.din)
        self.assertEqual(GLN(x).shape, torch.Tensor(batch_size, self.num_rims, self.dout).shape,
                         "Incorrect output shape")


if __name__ == '__main__':
    unittest.main()
