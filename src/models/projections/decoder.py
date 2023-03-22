import torch.nn as nn
from torchtyping import TensorType


class Decoder(nn.Module):
    """
    Decodes the features to predict the output
    """

    def __init__(self, d_in: int, d_out: int):
        super(Decoder, self).__init__()
        self.decoder = nn.Linear(d_in, d_out)
        self.init_weights()

    def init_weights(self):
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-0.1, 0.1)

    def forward(self, input: TensorType["timestep", "batch", "num_rims", "d_rim"]) -> \
            TensorType["timestep", "batch", "d_out"]:

        num_timesteps, bsz, num_rims, d_rim = input.shape
        input = input.contiguous().view(num_timesteps * bsz, num_rims, d_rim)
        input = input.view(-1, num_rims * d_rim)                    # [ts*b, n_rims, d_rim]
        output = self.decoder(input)                                # [ts*b, d_out]
        return output
