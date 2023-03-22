import numpy
import torch
import torch.nn as nn
from torchtyping import TensorType


class TopkSparseAttention(nn.Module):
    """
    Apply top-k selection to the given attention Tensor returning a sparse normalised version of the Tensor with only
    the items relating to the top-k elements of the query dimension (num_q) having non-zero values.
    """

    def __init__(self, topk: int):
        """
        :param topk: number of queries to select
        """
        super(TopkSparseAttention, self).__init__()
        # increment because we will use the topk + 1 value to find the min thr value to be a top-k element
        topk += 1
        self.topk = topk

    def forward(self, attn: TensorType["B", "num_q", "num_k"]) -> TensorType["B", "num_q", "num_k"]:
        # used to avoid throwing away the second largest value
        eps = torch.finfo().eps
        # [num_q]
        dim_size_to_filter = attn.size()[1]

        # if there is less items than the amount to select then no sparsification is required.
        if dim_size_to_filter < self.topk:
            return attn

        # value of the top k elements over the rows of the matrix representing a minibatch item
        # get the values of the topk elements (row-wise)
        # [B, k, num_k]
        min_topk_val = torch.topk(attn, self.topk, dim=1).values
        # get the smallest of the returned top-k values (f.e. row) and add eps to it. Doing so means we know the
        # smallest delta we can subtract from the attention so all values not part of the topk will become negative
        # [B, 1, num_k]
        min_topk_val = min_topk_val[:, -1] + eps
        # [B] -> [B,1]
        min_topk_val = min_topk_val.reshape((min_topk_val.shape[0], 1))

        # creates sign mask out of attention values (-ve = not in top-k, +ve = in top-k)
        # [B, num_q, num_k] = [B, num_q, num_k] - [B, num_q]
        attn_sign_mask = attn - min_topk_val.repeat(1, dim_size_to_filter)
        # mask out the negative values (i.e. the non top-k items)
        attn_sparse = torch.clamp(attn_sign_mask, min=0)
        # [B, 1, num_k]
        attn_sparse_sum = torch.sum(attn_sparse, dim=1, keepdim=True)
        # cancel the effect of the eps used in the delta
        attn_sparse_sum = attn_sparse_sum + eps
        # normalise the values for the top-k items
        attn_sparse_normalized = attn_sparse / attn_sparse_sum.repeat(1, dim_size_to_filter)
        # [B, num_q, num_k]
        return attn_sparse_normalized


if __name__ == "__main__":
    k = 2
    print('take top k', k)
    sa = TopkSparseAttention(topk=k)

    # batch x time
    x = torch.from_numpy(numpy.array([[[0.1, 0.0, 0.3, 0.2, 0.4], [0.5, 0.4, 0.1, 0.0, 0.0]]]))
    x = x.reshape((2, 5))
    print('x shape', x.shape)
    print('x', x)

    o = sa(x)
    print('o', o)
