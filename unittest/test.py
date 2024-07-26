import os
import sys
import torch
import torch.nn as nn
import unittest

sys.path.append("./src")

from scaled_dot_product import scaled_dot_product
from multihead_attention import MultiHeadAttenion
from layer_normalization import LayerNormalization
from feed_forward import PointWiseNeuralNetwork
from encoder import EncoderBlock
from transformer import TransformerEncoder


class UnitTest(unittest.TestCase):
    def setUp(self):
        self.query = torch.randn((40, 8, 200, 64))
        self.values = torch.randn((40, 8, 200, 64))
        self.key = torch.randn((40, 8, 200, 64))
        self.dimension = 512
        self.feed_forward = 2048
        self.heads = 8
        self.mask = None

        self.values, self.attention = scaled_dot_product(
            query=self.query,
            key=self.key,
            values=self.values,
            mask=self.mask,
        )

        self.multihead = MultiHeadAttenion(
            dimension=self.dimension, heads=self.heads, mask=self.mask
        )

        self.normalization = LayerNormalization(normalized_shape=self.dimension)

        self.net = PointWiseNeuralNetwork(
            in_features=self.dimension,
            out_features=self.feed_forward,
        )

        self.encoder = EncoderBlock(
            dimension=self.dimension,
            heads=self.heads,
            feed_forward=self.feed_forward,
            mask=self.mask,
        )

        self.transformer = TransformerEncoder(
            dimension=self.dimension,
            heads=self.heads,
            feed_forward=self.feed_forward,
            mask=self.mask,
        )

    def tearDown(self):
        self.query = self.key = self.values = self.attention = None

    def test_scaled_dot_product(self):
        self.assertEqual(self.query.size(), self.attention.size())
        self.assertNotEqual(self.values.size(), self.attention.size())

    def test_multihead_attention(self):
        self.assertEqual(
            self.query.view(self.query.size(0), self.query.size(2), -1).size(),
            self.multihead(torch.randn((40, 200, 512))).size(),
        )

    def test_layer_normalization(self):
        self.assertEqual(
            self.normalization(torch.randn((40, 200, 512))).size(), (40, 200, 512)
        )

    def test_feed_forward_network(self):
        self.assertEqual(self.net(torch.randn((40, 200, 512))).size(), (40, 200, 512))

    def test_encoder_block(self):
        self.assertEqual(
            self.encoder(torch.randn((40, 200, 512))).size(), (40, 200, 512)
        )

    def test_transformer_encoder(self):
        self.assertEqual(
            self.transformer(torch.randn((40, 200, 512))).size(), (40, 200, 512)
        )


if __name__ == "__main__":
    unittest.main()
