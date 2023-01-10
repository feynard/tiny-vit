import torch

from typing import Tuple, Literal

from transformer import Transformer


class VisionTransformer(torch.nn.Module):

    class ConfigurationError(Exception):
        def __init__(self, message):
            self.message = message

    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        patch_size: int,
        num_layers: int,
        transformer_dim: int,
        attention_heads: int,
        num_classes: int,
        dropout: float,
        pooling_type: Literal['average', 'class_token'] = 'average'):

        super().__init__()

        c, h, w = image_shape

        if h % patch_size != 0 or w % patch_size != 0:
            raise VisionTransformer.ConfigurationError("Image dimensions must be divisible by the patch size")

        if pooling_type not in { 'average', 'class_token' }:
            raise VisionTransformer.ConfigurationError("Pooling type must be either `average` or `class_token`")

        self.unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)

        self.transformer = Transformer(num_layers, transformer_dim, attention_heads, transformer_dim * 4, dropout)

        self.positional_embedding = torch.nn.Parameter(
            torch.rand(1, (h * w) // (patch_size ** 2), transformer_dim),
            requires_grad=True
        )

        self.projection = torch.nn.Linear(c * (patch_size ** 2), transformer_dim)
        
        self.pooling_type = pooling_type

        # BERT-like `class` token for classification problems
        if pooling_type == 'class_token':
            self.class_token = torch.nn.Parameter(torch.rand(1, 1, transformer_dim), requires_grad=True)

        self.main_mlp = torch.nn.Sequential(
            torch.nn.Linear(transformer_dim, transformer_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(transformer_dim, num_classes)
        )

    def forward(self, x):
        """
        Input: tensor of shape [Batch, 3, H, W]
        """

        x = self.unfold(x).permute(0, 2, 1) # to [Batch, Len, Dimension] format
        x = self.projection(x)

        x = x + self.positional_embedding

        if self.pooling_type == 'class_token':
            x = torch.cat([self.class_token.repeat(len(x), 1, 1), x], dim=1) # [Batch, Len + 1, Dimension]

        y = self.transformer(x)

        if self.pooling_type == 'average':
            y = y.mean(1)
        else:
            y = y[:, 0, :]

        return self.main_mlp(y)
