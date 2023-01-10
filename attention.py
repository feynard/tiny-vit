import torch


class Attention(torch.nn.Module):
    """
    This is a simple implementation of a multi-head self attention layer
    Convention for the input: [Batch, Length, Embedding]
    """

    class DimensionError(Exception):
        def __init__(self, message):
            self.message = message

    def __init__(self, input_dim: int, num_heads: int):
        super().__init__()

        if input_dim % num_heads != 0:
            raise Attention.DimensionError("Dimension must be divisible by the number of attention heads!")

        self.input_dim = input_dim
        self.num_heads = num_heads

        self.k = torch.nn.Parameter(torch.rand(num_heads, input_dim, input_dim))
        self.q = torch.nn.Parameter(torch.rand(num_heads, input_dim, input_dim))
        self.v = torch.nn.Parameter(torch.rand(num_heads, input_dim // num_heads, input_dim))

        self.softmax = torch.nn.Softmax(dim=3)

    @staticmethod
    def batch_head_product(x, y):
        return torch.einsum('bdl, ncd -> bncl', x, y)

    def forward(self, x):
        """
        Input: tensor of shape [Batch, Length, Embedding]
        """

        x = x.permute(0, 2, 1)

        k = Attention.batch_head_product(x, self.k)
        q = Attention.batch_head_product(x, self.q)
        v = Attention.batch_head_product(x, self.v)

        s = self.softmax(torch.einsum('bndl, bndm -> bnlm', q, k))

        r = torch.einsum('bnkl, bndl -> bndk', s, v)
        r = r.reshape(len(x), self.input_dim, -1)

        return r.permute(0, 2, 1)
