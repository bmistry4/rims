import unittest

import numpy as np
import torch

from ..attention import MultiHeadAttention, ScaledDotProductAttention, TopkSparsification


class TestScaledDotProductAttention(unittest.TestCase):

    def setUp(self):
        self.d_k = 64
        self.temp = np.power(self.d_k, 0.5)
        topk = 2
        sparsifier = TopkSparsification(topk=topk) if topk > 0 else None
        self.sdpa = ScaledDotProductAttention(temperature=np.power(self.d_k, 0.5), sparsifier=sparsifier)

    def setUp_input_attn(self):
        hxb = 64  # hxb = |heads| x batch size
        self.hxb = hxb

        num_q, num_k, num_v = 6, 2, 2  # assume k=v
        d_k, d_v = 64, 400  # dimension size can be anything as long as d_q = d_k
        d_q = d_k

        self.num_q, self.num_k, self.d_v = num_q, num_k, d_v
        self.q = torch.randn((self.hxb, num_q, d_q))
        self.k = torch.randn((self.hxb, num_k, d_k))
        self.v = torch.randn((self.hxb, num_v, d_v))

    def test_input_attn_returned_output_shape(self):
        self.setUp_input_attn()
        output, _ = self.sdpa(self.q, self.k, self.v)
        self.assertEqual(output.shape, torch.randn(self.hxb, self.num_q, self.d_v).shape, "Wrong output shape")

    def test_input_attn_returned_attn_shape(self):
        self.setUp_input_attn()
        _, attn = self.sdpa(self.q, self.k, self.v)
        self.assertEqual(attn.shape, torch.randn(self.hxb, self.num_q, self.num_k).shape, "Wrong attn shape")

    def setUp_comm_attn(self):
        hxb = 256  # hxb = |heads| x batch size
        self.hxb = hxb
        num_q, num_k, num_v = 6, 6, 6  # Number of k=v
        d_k, d_v = 16, 16  # dimension size can be anything as long as d_q = d_k
        d_q = d_k

        self.num_q, self.num_k, self.d_v = num_q, num_k, d_v
        self.q = torch.randn((self.hxb, num_q, d_q))
        self.k = torch.randn((self.hxb, num_k, d_k))
        self.v = torch.randn((self.hxb, num_v, d_v))

    def test_comm_attn_returned_output_shape(self):
        self.setUp_comm_attn()
        output, _ = self.sdpa(self.q, self.k, self.v)
        self.assertEqual(output.shape, torch.randn(self.hxb, self.num_q, self.d_v).shape, "Wrong output shape")

    def test_comm_attn_returned_attn_shape(self):
        self.setUp_comm_attn()
        _, attn = self.sdpa(self.q, self.k, self.v)
        self.assertEqual(attn.shape, torch.randn(self.hxb, self.num_q, self.num_k).shape, "Wrong attn shape")


class TestMultiHeadAttention(unittest.TestCase):

    def setUp(self):
        self.bsz = 64
        self.ninp = 600

    def setUp_input_attn(self):
        n_head = 1
        d_model_read = 100
        d_model_write = 600
        d_model_out = 400  # = att_out
        d_k = 64
        d_v = 400
        num_blocks_read = 6
        num_blocks_write = 2
        residual = False
        topk = 2  # num_blocks_in (=1) +1
        # grad_sparse = False
        learn_weighting_heads = False
        sparsifier = TopkSparsification(topk=topk) if topk > 0 else None

        self.n_head = n_head
        self.d_model_out = d_model_out
        self.d_v = d_v
        self.len_q, self.len_k = 6, 2
        d_q_in, d_k_in, d_v_in = d_model_read, d_model_write, d_model_write

        self.q = torch.randn((self.bsz, self.len_q, d_q_in))
        self.k, self.v = torch.randn((self.bsz, self.len_k, d_k_in)), torch.randn((self.bsz, self.len_k, d_v_in))
        self.mha = MultiHeadAttention(n_head=n_head, d_q_in=d_model_read, d_k_in=d_model_write, d_v_in=d_model_write,
                                      d_mha_out=d_model_out, d_k=d_k, d_v=d_v, num_q=num_blocks_read,
                                      num_k_or_v=num_blocks_write, residual=residual, learn_weighting_heads=learn_weighting_heads,
                                      sparsifier=sparsifier)

    def test_input_attn_returned_output_shape(self):
        self.setUp_input_attn()
        output, _ = self.mha(self.q, self.k, self.v)
        self.assertEqual(output.shape, torch.randn(self.bsz, self.len_q, self.d_model_out).shape,
                         "Wrong output shape")

    def test_input_attn_returned_attn_shape(self):
        self.setUp_input_attn()
        _, attn = self.mha(self.q, self.k, self.v)
        self.assertEqual(attn.shape, torch.randn(self.bsz, self.n_head, self.len_q, self.len_k).shape,
                         "Wrong output shape")

    def setUp_comm_attn(self):
        n_head = 4
        d_model_read = 100
        d_model_write = 100
        d_model_out = 100
        d_k = 16
        d_v = 16
        num_blocks_read = 6
        num_blocks_write = 6
        residual = True
        topk = 2
        # grad_sparse = False
        learn_weighting_heads = True
        sparsifier = TopkSparsification(topk=topk) if topk > 0 else None

        self.n_head = n_head
        self.d_model_out = d_model_out
        self.d_v = d_v
        self.len_q, self.len_k = 6, 6
        d_q_in, d_k_in, d_v_in = d_model_read, d_model_write, d_model_write
        self.q, self.k, self.v = torch.randn((self.bsz, self.len_q, d_q_in)), torch.randn(
            (self.bsz, self.len_k, d_k_in)), torch.randn((self.bsz, self.len_k, d_v_in))

        self.mha = MultiHeadAttention(n_head=n_head, d_q_in=d_model_read, d_k_in=d_model_write, d_v_in=d_model_write,
                                      d_mha_out=d_model_out, d_k=d_k, d_v=d_v, num_q=num_blocks_read,
                                      num_k_or_v=num_blocks_write, residual=residual, learn_weighting_heads=learn_weighting_heads,
                                      sparsifier=sparsifier)

    def test_comm_attn_returned_output_shape(self):
        self.setUp_comm_attn()
        output, _ = self.mha(self.q, self.k, self.v)
        self.assertEqual(output.shape, torch.randn(self.bsz, self.len_q, self.d_model_out).shape,
                         "Wrong output shape")

    def test_comm_attn_returned_attn_shape(self):
        self.setUp_input_attn()
        _, attn = self.mha(self.q, self.k, self.v)
        self.assertEqual(attn.shape, torch.randn(self.bsz, self.n_head, self.len_q, self.len_k).shape,
                         "Wrong output shape")


if __name__ == '__main__':
    unittest.main()
