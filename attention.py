import math

import torch


class SelfAttention(torch.nn.Module):
    """
    This is a simple implementation of a multi-head self attention layer
    Convention for the input: [Batch, Length, Embedding]

    Notation for `einsum`:
        b -- batch
        d -- input dimension
        n -- number of heads
        e -- head dimension equal to `d // n`
        l, m -- length
    """

    class DimensionError(Exception):
        pass

    def __init__(self, dim: int, n_heads: int):
        super().__init__()

        if dim % n_heads != 0:
            raise SelfAttention.DimensionError("Dimension must be divisible by the number of attention heads!")

        self.head_dim = dim // n_heads

        self.k = torch.nn.Parameter(torch.rand(n_heads, dim, self.head_dim))
        self.q = torch.nn.Parameter(torch.rand(n_heads, dim, self.head_dim))
        self.v = torch.nn.Parameter(torch.rand(n_heads, dim, self.head_dim))

    @staticmethod
    def batch_head_product(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Batch matrix multiplication of the input `x` with the weight `w` for each attention head.
        Input shape: [Batch, Length, Embedding]
        """

        return torch.einsum('bld, nde -> bnle', x, w)

    @staticmethod
    def attention_operation(k: torch.Tensor, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Vanilla Self-Attention operation.
        Input shapes: [Batch, Heads, Length, Head Dimension]
        """

        head_dim = k.shape[-1]
        s = torch.nn.functional.softmax(torch.einsum('bnle, bnme -> bnlm', q, k) / math.sqrt(head_dim), dim=-1)
        r = torch.einsum('bnlm, bnme -> blne', s, v)

        return r.flatten(2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape: [Batch, Length, Embedding]
        """

        k = SelfAttention.batch_head_product(x, self.k)
        q = SelfAttention.batch_head_product(x, self.q)
        v = SelfAttention.batch_head_product(x, self.v)

        return SelfAttention.attention_operation(k, q, v)
