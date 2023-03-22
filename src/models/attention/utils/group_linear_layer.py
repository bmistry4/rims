import torch
import torch.nn as nn
from torchtyping import TensorType


class GroupLinearLayer(nn.Module):
    """
    Given a set of queries, keys, or values vectors as a matrix (i.e., Q, K, or V), a GroupLinearLayer provides a way
    to project the vectors into another dimension. The resulting output dimension will contain the representations for
    multiple heads. The resulting operation is similar to the projection used in the Attention is all you need paper,
    e.g., QW but with the projection representing multiple heads rather than just a single head.

    A BMM operation enables applying the projection to all the vectors in the set (e.g. all the queries) in parallel.
    Due to this, a permutation is applied to swap the batch dimension with the vector set dimension.
    """

    def __init__(self, d_in: int, d_concat_out: int, num_in: int):
        """

        :param d_in: Size (/dimensionality) of the queries/keys/values prior to projection
        :param d_concat_out: Size (/dimensionality) of the queries/keys/values after projection
        :param num_in: number of queries/keys/values
        """
        super(GroupLinearLayer, self).__init__()
        # let W represent the projection
        self.W = nn.Parameter(0.01 * torch.randn(num_in, d_in, d_concat_out), requires_grad=True)

    def forward(self, x: TensorType["B", "num_in", "d_in"]) -> TensorType["B", "num_in", "d_concat_out"]:
        x = x.permute(1, 0, 2)      # [B, num_in, d_in] -> [num_in, B, d_in]
        x = torch.bmm(x, self.W)    # [num_in, B, d_in] bmm [num_in, d_in, d_concat_out] = [num_in, B, d_concat_out]
        return x.permute(1, 0, 2)   # [num_in, B, d_concat_out] -> [B, num_in, d_concat_out]


if __name__ == "__main__":
    din, d_concat_out, num_in = 100, 64, 6
    GLN = GroupLinearLayer(din, d_concat_out, num_in)

    x = torch.randn(128, num_in, din)

    print("x shape: ", GLN(x).shape)

    for n, p in GLN.named_parameters():
        print(n, p.shape)
