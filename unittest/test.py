import os
import sys
import torch
import torch.nn as nn
import unittest

sys.path.append("./src")

from scaled_dot_product import scaled_dot_product


class UnitTest(unittest.TestCase):
    def setUp(self):
        self.query = torch.randn((40, 8, 200, 64))
        self.values = torch.randn((40, 8, 200, 64))
        self.key = torch.randn((40, 8, 200, 64))
        self.mask = None

        self.values, self.attention = scaled_dot_product(
            query=self.query,
            key=self.key,
            values=self.values,
            mask=self.mask,
        )

    def test_scaled_dot_product(self):
        self.assertEqual(self.query.size(), self.attention.size())


if __name__ == "__main__":
    unittest.main()
