from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torchtyping import TensorType

from .utils.group_linear_layer import GroupLinearLayer
from .utils.sparse_attention import TopkSparseAttention


class TopkSparsification(nn.Module):
    """
    Select top-k queries given an attention tensor returning an attention tensor where only the keys relating to the
    top-k queries will have non-0 values
    """

    def __init__(self, topk: int):
        """
        :param topk: Number of 'rims' we want to select
        """
        super(TopkSparsification, self).__init__()

        self.topk = topk
        self.sa = TopkSparseAttention(topk=topk)

    def forward(self, attn: TensorType['batch', 'num_q', 'num_k']) -> TensorType['batch', 'num_q', 'num_k']:
        mb, num_q, num_k = attn.shape[0], attn.shape[1], attn.shape[2]
        sparse_attn = attn.reshape((mb * num_q, num_k))                  # [b, num_q, num_k] -> [b*num_q, num_k]
        sparse_attn = self.sa(sparse_attn)                               # [b*num_q, num_k]  -> [b*num_q, num_k]
        sparse_attn = sparse_attn.reshape((mb, num_q, num_k))            # [b*num_q, num_k]  -> [b, num_q, num_k]
        return sparse_attn


class ScaledDotProductAttention(nn.Module):
    """
    Applies scale dot product attention with the additional feature of (optional) sparification of the resulting
    attention matrix.
    """

    def __init__(self, temperature: float,
                 sparsifier: Callable[[TensorType['batch', 'num_q', 'num_k']],
                                      TensorType['batch', 'num_q', 'num_k']] = None
                 ):
        """

        :param temperature: Acts as the sqrt(d_e) from equation 2 in https://openreview.net/forum?id=mLcmdlEUxy-.
        :param sparsifier: Method to make the attention matrix sparse. Default: None results in no sparsification.
        """
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)  # over the key dimension (d_k)
        self.sparsifier = sparsifier

    def forward(self,
                q: TensorType["n_heads * batch", "num_q", "d_q"],
                k: TensorType["n_heads * batch", "num_q", "d_q"],
                v: TensorType["n_heads * batch", "num_q", "d_q"]) -> \
            [TensorType["n_heads * batch", "num_q", "d_v"],
             TensorType["n_heads * batch", "num_q", "num_k"]
             ]:

        # q     = [n_heads * batch, num_q, d_q]
        # k     = [n_heads * batch, num_k, d_k]; assume d_q = d_k
        # attn  = [n_heads * batch, num_q, num_k]
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        attn = self.softmax(attn)

        if self.sparsifier is not None:
            # [n_heads * batch, num_q, num_k]
            attn = self.sparsifier(attn)

        # assume num_k = num_v
        # output = [n_heads * batch, num_q, d_v] -> [n_heads * batch, num_q, num_k] bmm [n_heads * batch, num_v, d_v]
        output = torch.bmm(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """
    Applied multiheaded ((sparse) scaled dot product) attention given Tensors representing the queries, keys and values.
    Utilised for the input and communication attentions in the RIMs architecture.
    """

    def __init__(self, n_head: int, d_q_in: int, d_k_in: int, d_v_in: int, d_mha_out: int, d_k: int, d_v: int,
                 num_q: int, num_k_or_v: int, residual: bool = False, dropout: float = 0.1,
                 learn_weighting_heads: bool = True, sparsifier: Callable[[torch.Tensor], torch.Tensor] = None) -> None:
        """
        :param n_head: number of heads
        :param d_q_in: dim of input query vectors
        :param d_k_in: dim of input key vectors. NOTE: d_k_in and d_v_in don't have to be the same!
        :param d_v_in: dim of input value vectors. NOTE: d_k_in and d_v_in don't have to be the same!
        :param d_mha_out: dim of output of MHA; can be arbitrary, and in original MHA implementation of self-attention
                (Attention is all you need), it is equal to d_model_write (d_v) AND d_model_read (d_q_in) (which is not
                 a constraint in this implementation).
        :param d_k: dim of projected key vectors.
        :param d_v: dim of projected value vectors.
        :param num_q: number of queries
        :param num_k_or_v: number of keys/values
        :param residual: if to use a residual connection
        :param dropout: dropout probability (i.e. probability of an element to be zeroed)
        :param learn_weighting_heads: if to learn weights for averaging heads (weighted average) or to use a fixed
                weighting (i.e. average)
        :param sparsifier: sparsification function to apply to the attention tensor. Default to None (no sparsifier).
        """
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.num_q = num_q
        self.num_k = num_k_or_v
        self.num_v = num_k_or_v

        self.GLN_qs = GroupLinearLayer(d_q_in, n_head * d_k, num_q)
        self.GLN_ks = GroupLinearLayer(d_k_in, n_head * d_k, num_k_or_v)
        self.GLN_vs = GroupLinearLayer(d_v_in, n_head * d_v, num_k_or_v)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), sparsifier=sparsifier)

        # if case is for the comm attention; else case is for the inp attention
        # (fc is analogous to applying the W^O from attention is all you need to combine the heads)
        self.fc = nn.Linear(n_head * d_v, d_mha_out) if learn_weighting_heads else \
            (lambda a: a.reshape(a.shape[0], a.shape[1], self.n_head, -1).mean(dim=2))

        self.residual = residual
        self.dropout = nn.Dropout(dropout)
        self.gate_fc = nn.Linear(n_head * d_v, d_mha_out)       # todo - remove if unused


    def forward(self,
                q: TensorType["batch", "num_q", "d_q_in"],
                k: TensorType["batch", "num_k", "d_k_in"],
                v: TensorType["batch", "num_k", "d_v_in"]
                ) -> \
            (
                TensorType["batch", "num_q", "d_v/d_mha_out"],
                TensorType["batch", "n_heads", "num_q", "num_k"]
            ):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        num_q, num_k, num_v = self.num_q, self.num_k, self.num_v
        bsz = q.shape[0]


        # todo q: if inp_attn then [B, num_blocks_out=6, block_size_out=100]; if comm_attn hx_new_grad_mask=[B, 6, 100]
        # todo k: if inp_attn then inp_use = [B,2,1]; if comm_attn hx_new_grad_mask=[B, 6, 100]
        # todo v: if inp_attn then inp_use = [B,2,1] ; if comm_attn hx_new_grad_mask=[B, 6, 100]
        # project q,k and v into the multi-head representation space
        q = self.GLN_qs(q).view(bsz, num_q, n_head, d_k)  # [b, n_q, h*d_k] -> [b, n_q, h, d_k]
        k = self.GLN_ks(k).view(bsz, num_k, n_head, d_k)
        v = self.GLN_vs(v).view(bsz, num_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, num_q, d_k)  # [(h*b), num_q, d_k]
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, num_k, d_k)  # [(h*b), num_k, d_k]
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, num_v, d_v)  # [(h*b), num_v, d_v]

        # scaled dp attn -> softmax(QK.t()/sqrt(d_k))V
        output, attn = self.attention(q, k, v)
        attn = attn.reshape(n_head, bsz, num_q, -1).permute(1, 0, 2, 3)

        # [h*b, num_q, d_v] -> [h, b, num_q, d_v]
        output = output.view(n_head, bsz, num_q, d_v)
        # [h, b, num_q, d_v] -> [b, num_q, h, d_v] -> [b, num_q, h*d_v]; last dim is BLOCKs of heads
        output = output.permute(1, 2, 0, 3).contiguous().view(bsz, num_q, -1)
        # [b, num_q, h*d_v]
        output_init = output * 1.0
        # [b, num_q, d_v] (for input attn) or [b, num_q, d_mha_out] (for comm attn)
        output = self.dropout(self.fc(output_init))

        # residual connection used during the communication phase
        # fixme - where mentioned in paper???
        if self.residual:
            # [b, num_q, h*d_v]@[h*d_v, d_o] = [b, n_q, d_o]
            gate = torch.sigmoid(self.gate_fc(output_init))

            # [b, num_q, d_mha_out]
            output = gate * torch.tanh(output)

        # output is a sdpa with head reduction (and residual). attn is sdpa without the V applied.
        return output, attn


if __name__ == "__main__":
    bsz = 64
    ninp = 600

    # d_model_read = block_size_out = nhid // num_blocks_out = 600 // 6 = 100
    # d_model_write = 600 ninp (inp_attn) or 100 block_size_out (comm_attn)
    # n_head = 1 (inp_attn), 4 (comm_attn)
    # d_k = 64 (inp_attn), 16 (comm_attn)
    # num_blocks_read = num_blocks_out = 6
    # num_blocks_write = 2 num_modules_read_input (inp_attn) , or 6 num_blocks_out (comm_attn)

    print('input attn')
    # input attn
    n_head = 1
    d_model_read = 100
    d_model_write = 600
    d_model_out = 400  # = att_out
    d_k = 64
    d_v = 400
    num_blocks_read = 6
    num_blocks_write = 2
    residual = False
    learn_weighting_heads = False
    topk = 2  # num_blocks_in (=1) +1
    sparsifier = TopkSparsification(topk=topk) if topk > 0 else None

    # input attn (assuming 1 layer (unstacked) LSTM)
    # # x is of shape [hx.reshape((hx.shape[0], self.num_blocks_out, self.block_size_out)), inp_use, inp_use]
    # -> [ [B, 6, 100] , [B,2,1] , [B, 2(x_i and null inp)*num_blocks_in (1) = 2,ninp=1] ]
    hx = torch.randn((bsz, 6, 100))
    inp_use = torch.randn((bsz, 2, ninp))
    inp_attn = MultiHeadAttention(n_head=n_head, d_q_in=d_model_read, d_k_in=d_model_write, d_v_in=d_model_write,
                                  d_mha_out=d_model_out, d_k=d_k, d_v=d_v, num_q=num_blocks_read,
                                  num_k_or_v=num_blocks_write, residual=residual,
                                  learn_weighting_heads=learn_weighting_heads, sparsifier=sparsifier)
    out, attn = inp_attn(hx, inp_use, inp_use)
    print('out (qkv) shape', out.shape)
    print('qk attn shape', attn.shape)

    #################################################################################################################
    print('\ncomm attn')
    # comm attn
    n_head = 4
    d_model_read = 100
    d_model_write = 100
    d_model_out = 100
    d_k = 16
    d_v = 16
    num_blocks_read = 6
    num_blocks_write = 6
    residual = True
    learn_weighting_heads = True
    topk = 2
    sparsifier = TopkSparsification(topk=topk) if topk > 0 else None

    # for comm_attn we want x to be the same shape as hx_new -> [B, num_blocks_out=6, block_size_out=100]
    x = torch.randn((bsz, 6, 100))
    comm_attn = MultiHeadAttention(n_head=n_head, d_q_in=d_model_read, d_k_in=d_model_write, d_v_in=d_model_write,
                                   d_mha_out=d_model_out, d_k=d_k, d_v=d_v, num_q=num_blocks_read,
                                   num_k_or_v=num_blocks_write, residual=residual,
                                   learn_weighting_heads=learn_weighting_heads, sparsifier=sparsifier)
    out, attn = comm_attn(x, x, x)
    print('out (qkv) shape', out.shape)
    print('qk attn shape', attn.shape)
