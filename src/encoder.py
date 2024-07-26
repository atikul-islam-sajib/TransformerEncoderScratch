import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src")

from utils import config
from feed_forward import PointWiseNeuralNetwork
from multihead_attention import MultiHeadAttenion
from layer_normalization import LayerNormalization


class EncoderBlock(nn.Module):
    def __init__(
        self,
        dimension: int = 512,
        heads: int = 8,
        fead_foward: int = 2048,
        dropout: float = 0.1,
        epsilon: float = 1e-5,
        mask=None,
    ):
        super(EncoderBlock, self).__init__()

        self.dimension = dimension
        self.haeds = heads
        self.feed_forward = fead_foward
        self.dropout = dropout
        self.epsilon = epsilon
        self.mask = mask

        self.multihead_attention = MultiHeadAttenion(
            dimension=self.dimension, heads=self.haeds, mask=self.mask
        )

        self.layer_norm = LayerNormalization(
            normalized_shape=self.dimension, epsilon=self.epsilon
        )

        self.feedfoward = PointWiseNeuralNetwork(
            in_features=self.dimension,
            out_features=self.feed_forward,
            dropout=self.dropout,
        )

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            residual = x

            x = self.multihead_attention(x)
            x = torch.dropout(input=x, p=self.dropout, train=self.training)
            x = torch.add(x, residual)
            x = self.layer_norm(x)

            residual = x

            x = self.feedfoward(residual)
            x = torch.dropout(input=x, p=self.dropout, train=self.training)
            x = torch.add(x, residual)
            x = self.layer_norm(x)

            return x

        else:
            raise TypeError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encoder Block for transformers".capitalize()
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=config()["transformer"]["dimension"],
        help="Dimension of the input tensor".capitalize(),
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=config()["transformer"]["heads"],
        help="Number of heads in the multihead attention".capitalize(),
    )
    parser.add_argument(
        "--feedfoward",
        type=int,
        default=config()["transformer"]["feed_forward"],
        help="Dimension of the feedforward layer".capitalize(),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=config()["transformer"]["dropout"],
        help="Dropout rate".capitalize(),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=config()["transformer"]["eps"],
        help="Epsilon for layer norm".capitalize(),
    )
    parser.add_argument(
        "--mask",
        type=torch.Tensor,
        default=None,
        help="Mask for attention".capitalize(),
    )

    args = parser.parse_args()

    dimension = args.d_model
    heads = args.heads
    feed_forward = args.feedfoward
    dropout = args.dropout
    epsilon = args.epsilon
    mask = args.mask

    batch_size = config()["embedding"]["batch_size"]
    sequence_length = config()["embedding"]["sequence_length"]

    encoder = EncoderBlock(
        dimension=dimension,
        heads=heads,
        fead_foward=feed_forward,
        dropout=dropout,
        epsilon=epsilon,
        mask=mask,
    )

    assert encoder(torch.randn(batch_size, sequence_length, dimension)).size() == (
        batch_size,
        sequence_length,
        dimension,
    ), "Dimension mismatch in the EncoderBlock".capitalize()
