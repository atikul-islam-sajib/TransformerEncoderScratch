import sys
import torch
import argparse
import torch.nn as nn

sys.path.append("./src")

from utils import config
from scaled_dot_product import scaled_dot_product


class MultiHeadAttenion(nn.Module):
    def __init__(self, dimension: int = 512, heads: int = 8, mask=None):
        super(MultiHeadAttenion, self).__init__()

        self.dimension = dimension
        self.heads = heads
        self.mask = mask

        assert (
            self.dimension % self.heads == 0
        ), "dimension should be divisible by heads".capitalize()

        self.QKV = nn.Linear(
            in_features=self.dimension, out_features=3 * self.dimension, bias=False
        )
        self.layer = nn.Linear(
            in_features=self.dimension, out_features=self.dimension, bias=False
        )

    def forward(self, x: torch.Tensor, mask=None):
        if isinstance(x, torch.Tensor):
            self.QueryKeyValues = self.QKV(x)

            self.query, self.key, self.values = torch.chunk(
                self.QueryKeyValues, 3, dim=-1
            )

            self.query = self.query.view(
                self.query.size(0),
                self.query.size(1),
                self.heads,
                self.dimension // self.heads,
            )
            self.key = self.key.view(
                self.key.size(0),
                self.key.size(1),
                self.heads,
                self.dimension // self.heads,
            )
            self.values = self.values.view(
                self.values.size(0),
                self.values.size(1),
                self.heads,
                self.dimension // self.heads,
            )

            self.query = self.query.permute(0, 2, 1, 3)
            self.key = self.key.permute(0, 2, 1, 3)
            self.values = self.values.permute(0, 2, 1, 3)

            _, attention = scaled_dot_product(
                query=self.query, key=self.key, values=self.values, mask=self.mask
            )

            assert (
                attention.size() == self.query.size()
                and self.key.size()
                and self.values.size()
            ), "attention size is not equal to query, key and values size".capitalize()

            self.attention = attention.view(
                attention.size(0),
                attention.size(2),
                attention.size(1),
                attention.size(3),
            )

            self.attention = self.attention.view(
                self.attention.size(0), self.attention.size(1), -1
            )

            assert (
                self.attention.size(-1) == self.dimension
            ), "attention size is not equal to dimension".capitalize()

            return self.layer(self.attention)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MultiHeadAttention for Transformer Encoder".title()
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=config()["transformer"]["dimension"],
        help="dimension of the input".capitalize(),
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=config()["transformer"]["heads"],
        help="number of heads".capitalize(),
    )
    parser.add_argument(
        "--mask",
        type=torch.Tensor,
        default=None,
        help="mask for attention".capitalize(),
    )

    args = parser.parse_args()

    batch_size = config()["embedding"]["batch_size"]
    sequence_length = config()["embedding"]["sequence_length"]

    attention = MultiHeadAttenion(
        dimension=args.dimension, heads=args.heads, mask=args.mask
    )

    assert attention(
        torch.randn(batch_size, sequence_length, args.dimension)
    ).size() == (
        batch_size,
        sequence_length,
        args.dimension,
    ), "MultiHeadAttention output size is not equal to input size".capitalize()
