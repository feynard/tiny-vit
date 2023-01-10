import torch

from attention import Attention


class Transformer(torch.nn.Module):
    """
    Implementation of a simple transformer model
    """    

    class Layer(torch.nn.Module):
        """
        Basic transformer layer (ViT scheme)
        """

        def __init__(self, dim: int, num_heads: int, mlp_dim: int, dropout: float):
            super().__init__()

            self.norm_1 = torch.nn.LayerNorm(dim)

            self.attention = Attention(dim, num_heads)

            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(dim, mlp_dim),
                torch.nn.Dropout(dropout),
                torch.nn.ReLU(),
                torch.nn.Linear(mlp_dim, dim),
                torch.nn.Dropout(dropout)
            )

            self.norm_2 = torch.nn.LayerNorm(dim)

        def forward(self, x):
            """
            Input: tensor of shape [Batch, Length, Embedding]
            """

            y = self.norm_1(x + self.attention(x))
            z = self.norm_2(y + self.mlp(y))

            return z


    def __init__(self, num_layers: int, dim: int, num_heads: int, mlp_dim: int, dropout: float):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            [Transformer.Layer(dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
