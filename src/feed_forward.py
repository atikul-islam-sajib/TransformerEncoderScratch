import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src")

from utils import config


class PointWiseNeuralNetwork(nn.Module):
    def __init__(
        self, in_features: int = 512, out_features: int = 512, dropout: float = 0.1
    ):
        super(PointWiseNeuralNetwork, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout

        self.layers = []

        for index in range(2):
            self.layers.append(
                nn.Linear(in_features=self.in_features, out_features=self.out_features),
            )

            self.in_features = self.out_features
            self.out_features = in_features

            if index % 2 == 0:
                self.layers.append(nn.ReLU(inplace=True))
                self.layers.append(nn.Dropout(p=self.dropout))

        self.layer = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            return self.layer(x)

        else:
            raise TypeError("Input must be a torch.Tensor".capitalize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pointwise neural network for transformer".title()
    )
    parser.add_argument(
        "--in_features",
        type=int,
        default=config()["transformer"]["dimension"],
        help="Input features".capitalize(),
    )
    parser.add_argument(
        "--out_features",
        type=int,
        default=config()["transformer"]["feed_forward"],
        help="Output features".capitalize(),
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout rate".capitalize()
    )

    args = parser.parse_args()

    net = PointWiseNeuralNetwork(
        in_features=args.in_features,
        out_features=args.out_features,
        dropout=args.dropout,
    )

    batch_size = config()["embedding"]["batch_size"]
    sequence_length = config()["embedding"]["sequence_length"]

    assert (
        args.in_features % config()["transformer"]["heads"] == 0
    ), "Input features must be divisible by the number of heads".capitalize()

    assert net(torch.rand((batch_size, sequence_length, args.in_features))).size() == (
        batch_size,
        sequence_length,
        args.in_features,
    ), "Dimensions do not match in the poinytwise neural network".capitalize()
