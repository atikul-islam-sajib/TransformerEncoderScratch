import sys
import math
import torch
import argparse
import torch.nn as nn

sys.path.append("./src")


def scaled_dot_product(
    query: torch.Tensor,
    key: torch.Tensor,
    values: torch.Tensor,
    dimension: int = 512,
    mask=None,
):
    if (
        (isinstance(query, torch.Tensor))
        and (isinstance(key, torch.Tensor))
        and isinstance(values, torch.Tensor)
    ):
        result = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(dimension)

        if (mask is not None) and isinstance(mask, torch.Tensor):
            mask = mask.unsqueeze(1).unsqueeze(2)

            result = torch.add(result, mask)

        result = torch.softmax(input=result, dim=-1)

        attention = torch.matmul(result, values)

        return result, attention

    else:
        raise TypeError(
            "All inputs, for instance, query, key, and values must be of type torch.Tensor".capitalize()
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scaled Dot Product Attention for Transformers".title()
    )
    parser.add_argument(
        "--dimension", type=int, default=512, help="Dimension of the input".capitalize()
    )
    parser.add_argument(
        "--query",
        type=torch.Tensor,
        default=torch.randn((40, 8, 200, 64)),
        help="Query tensor".capitalize(),
    )
    parser.add_argument(
        "--key",
        type=torch.Tensor,
        default=torch.randn((40, 8, 200, 64)),
        help="Key tensor".capitalize(),
    )
    parser.add_argument(
        "--values",
        type=torch.Tensor,
        default=torch.randn((40, 8, 200, 64)),
        help="Values tensor".capitalize(),
    )
    parser.add_argument(
        "--mask",
        type=torch.Tensor,
        default=torch.randn((40, 200)),
        help="Mask tensor for padding".capitalize(),
    )

    args = parser.parse_args()

    values, attention = scaled_dot_product(
        query=args.query,
        key=args.key,
        values=args.values,
        mask=args.mask,
    )

    assert args.query.size() == attention.size()
