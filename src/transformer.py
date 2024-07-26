import os
import sys
import torch
import argparse
import torch.nn as nn
from torchsummary import summary
from torchview import draw_graph

sys.path.append("./src")

from utils import config
from encoder import EncoderBlock


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        dimension: int = 512,
        heads: int = 8,
        feed_forward: int = 2048,
        dropout: float = 0.1,
        epsilon: float = 1e-5,
        mask=None,
    ):
        super(TransformerEncoder, self).__init__()

        self.dimension = dimension
        self.heads = heads
        self.feed_forward = feed_forward
        self.dropout = dropout
        self.epsilon = epsilon
        self.mask = mask

        self.model = nn.Sequential(
            *[
                EncoderBlock(
                    dimension=self.dimension,
                    heads=self.heads,
                    feed_forward=self.feed_forward,
                    dropout=self.dropout,
                    epsilon=self.epsilon,
                    mask=self.mask,
                )
                for _ in range(self.heads)
            ]
        )

    def forward(self, x: torch.Tensor):
        if isinstance(x, torch.Tensor):
            return self.model(x)

        else:
            raise TypeError("Input must be a tensor".capitalize())

    @staticmethod
    def total_params(model=None):
        if isinstance(model, TransformerEncoder):
            return sum(params.numel() for params in model.parameters())

        else:
            raise TypeError("Input must be a transformer encoder".capitalize())


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
        "--feedforward",
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

    parser.add_argument(
        "--display", action="store_true", help="Display the arguments".capitalize()
    )

    args = parser.parse_args()

    dimension = args.d_model
    heads = args.heads
    feed_forward = args.feedforward
    dropout = args.dropout
    epsilon = args.epsilon
    mask = args.mask

    batch_size = config()["embedding"]["batch_size"]
    sequence_length = config()["embedding"]["sequence_length"]

    netTransfomer = TransformerEncoder(
        dimension=dimension,
        heads=heads,
        feed_forward=feed_forward,
        dropout=dropout,
        epsilon=epsilon,
        mask=mask,
    )
    assert netTransfomer(
        torch.randn(batch_size, sequence_length, dimension)
    ).size() == (
        batch_size,
        sequence_length,
        dimension,
    ), "Dimension mismatch in the EncoderBlock".capitalize()

    if args.display:

        print(
            "Total parameters of the model is # {}".format(
                TransformerEncoder.total_params(model=netTransfomer)
            )
        )

        print(summary(model=netTransfomer, input_size=(sequence_length, dimension)))

        draw_graph(
            model=netTransfomer,
            input_data=torch.randn(batch_size, sequence_length, dimension),
        ).visual_graph.render(
            filename=os.path.join(config()["path"]["FILES_PATH"], "transfomerEncoder"),
            format="png",
        )
