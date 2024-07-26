import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src")

from utils import config


class LayerNormalization(nn.Module):
    def __init__(self, normalized_shape: int = 512, epsilon: float = 1e-05):
        super(LayerNormalization, self).__init__()

        self.normalized_shape = normalized_shape
        self.epsilon = epsilon

        self.gamma = torch.ones((self.normalized_shape,))
        self.betas = torch.zeros((self.normalized_shape,))

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            self.mean = x.mean(dim=-1)
            self.variance = x.var(dim=-1, unbiased=False)

            self.mean = self.mean.unsqueeze(-1)
            self.variance = self.variance.unsqueeze(-1)

            return (
                self.gamma
                * ((x - self.mean) / (torch.sqrt(self.variance + self.epsilon)))
                + self.betas
            )

        else:
            raise TypeError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Layer Normalization for Transformer encoder".title()
    )
    parser.add_argument(
        "--normalized_shape",
        type=int,
        default=config()["transformer"]["dimension"],
        help="The normalized shape of the input tensor".capitalize(),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=config()["transformer"]["eps"],
        help="Epsilon value".capitalize(),
    )
    args = parser.parse_args()

    normalization = LayerNormalization(
        normalized_shape=args.normalized_shape, epsilon=args.epsilon
    )

    assert normalization(
        torch.randn(
            config()["embedding"]["batch_size"],
            config()["embedding"]["sequence_length"],
            args.normalized_shape,
        )
    ).size() == (
        config()["embedding"]["batch_size"],
        config()["embedding"]["sequence_length"],
        args.normalized_shape,
    ), "Dimension mismatch in the layer normalization layer".capitalize()
