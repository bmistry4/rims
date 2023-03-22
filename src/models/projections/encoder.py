import torch.nn as nn
from torchtyping import TensorType


class Encoder(nn.Module):
    """
    Encode the input to features
    """

    def __init__(self, num_tokens: int, d_out: int, dropout: float = 0.5):
        super(Encoder, self).__init__()
        self.encoder = nn.Embedding(num_tokens, d_out)
        self.init_weights()

    def init_weights(self):
        self.encoder.weight.data.uniform_(-0.1, 0.1)

    def forward(self, input: TensorType["timestep", "batch", "d_in"]) -> \
            TensorType["timestep", "batch", "d_out"]:
        return self.encoder(input)
