import torch
from .transformer import TransformerEncoder

"""
This script initializes a Transformer Encoder with specified parameters, 
creates a random embedding tensor, and prints the shapes of the embedding 
and the output tensors.

Attributes:
    batch_size (int): The batch size for the input tensor.
    sequence_length (int): The sequence length for the input tensor.
    model_dimension (int): The dimension of the model.
    feed_forward (int): The dimension of the feed forward network.
    number_heads (int): The number of attention heads.
    dropout (float): The dropout rate.
    epsilon (float): The epsilon value for numerical stability.
"""

batch_size = 64
sequence_length = 512
model_dimension = 768
feed_forward = 2048
number_heads = 12
dropout = 0.1
epsilon = 1e-6

# Create a random embedding tensor with the specified shape
embedding = torch.randn((batch_size, sequence_length, model_dimension))

# Create a random padding mask tensor
padding_masked = torch.randn((batch_size, sequence_length))

# Initialize the Transformer Encoder with the specified parameters
netTransformer = TransformerEncoder(
    dimension=model_dimension,
    heads=number_heads,
    feed_forward=feed_forward,
    dropout=dropout,
    epsilon=epsilon,
    mask=padding_masked,
)

# Print the divider line
print("|", "-" * 100, "|")

# Print the shape of the embedding tensor
print("|", "\tThe embedding shape is: ", embedding.size())

# Pass the embedding through the Transformer Encoder and print the output shape
print(
    "|",
    "\tThe output shape is: ",
    netTransformer(embedding).size(),
)  # (batch_size, sequence_length, model_dimension)

# Print the closing divider line
print("|", "-" * 100, "|")
