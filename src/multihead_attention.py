import sys
import torch
import torch.nn as nn

sys.path.append("./src")

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

            return attention


if __name__ == "__main__":
    attention = MultiHeadAttenion(dimension=512, heads=8, mask=None)

    print(attention(torch.randn(40, 200, 512))[3].size())
