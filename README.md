# TransformerEncoderScratch

This repository contains an implementation of a Transformer Encoder from scratch using PyTorch. The Transformer Encoder is a key component in modern natural language processing (NLP) tasks, enabling efficient and scalable attention mechanisms.

The Transformer Encoder is a neural network architecture introduced in the paper "Attention is All You Need" by Vaswani et al. It has become the foundation for many state-of-the-art models in NLP, including BERT, GPT, and T5. This repository provides a clean and simple implementation of the Transformer Encoder to help you understand its inner workings and experiment with its components.

<img src="https://miro.medium.com/v2/resize:fit:1030/1*tb9TT-mwFn1WPzkkbjoMCQ.png" alt="Transformer Architecture">

## Installation

To get started, clone the repository and install the required dependencies. It is recommended to use a virtual environment to manage dependencies.

### Clone the Repository

```bash
git clone https://github.com/atikul-islam-sajib/TransformerEncoderScratch.git
cd TransformerEncoderScratch
```

### Set Up the Environment

```bash
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
pip install -r requirements.txt
```

## Usage

The repository provides scripts and notebooks to help you understand and experiment with the Transformer Encoder. Below is an example script that demonstrates how to initialize the Transformer Encoder, create random input tensors, and print the shapes of the embedding and output tensors.

## Project Structure
```

├── Dockerfile
├── LICENSE
├── README.md
├── artifacts
│   ├── files
│   │   ├── __init__.py
│   │   ├── encoder
│   │   ├── encoder.png
│   │   ├── transfomerEncoder
│   │   └── transfomerEncoder.png
│   └── metrics
│       └── __init__.py
├── config.yml
├── mypy.ini
├── notebook
│   ├── Model-Inference.ipynb
│   └── Model-Prototype-Exp.ipynb
├── requirements.txt
├── setup.py
└── src
    ├── __init__.py
    ├── encoder.py
    ├── feed_forward.py
    ├── inference.py
    ├── layer_normalization.py
    ├── multihead_attention.py
    ├── scaled_dot_product.py
    ├── transformer.py
    └── utils.py
```

### User Guide Notebook (Tutorial for inferencing)

For detailed documentation on the implementation and usage, visit the -> [Transformer Encoder Tutorial Notebook](https://github.com/atikul-islam-sajib/TransformerEncoderScratch/blob/main/notebook/Model_Inference.ipynb)

## Example Script

```python
import torch
from transformer import TransformerEncoder

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
```

## Configuration File

You can use a configuration file to manage the parameters for your Transformer Encoder setup. Below is an example of a configuration file named `config.yaml`.

### `config.yaml`

```yaml
path:
  FILES_PATH: "./artifacts/files/"  # Path to store artifact files
  
embedding:
  batch_size: 40                    # Number of samples per batch
  sequence_length: 200              # Length of each input sequence

transformer:
  dimension: 512                    # Model dimension (size of the embeddings)
  heads: 8                          # Number of attention heads
  feed_forward: 2048                # Dimension of the feed-forward network
  dropout: 0.1                      # Dropout rate
  eps: 1e-6                         # Epsilon value for numerical stability in layer normalization

```

## Additional Resources

- [Attention is All You Need (Vaswani et al.)](https://arxiv.org/abs/1706.03762)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
