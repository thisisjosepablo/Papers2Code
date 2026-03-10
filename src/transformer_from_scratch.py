import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        """
            - embed_size: The total dimensionality of the input vector
            - heads: The number of parallel attention heads

            The multi-head attention works by splitting the large embeddings
            into smaller chunks. So if the embed_size is 512 and we have 8 heads,
            each head will operate on a vector of size 512/8=64.
        """
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

        # Linear projections for Q, K and V
        self.values = nn.Linear(self.head_dim, self.head_dim, bias = False)     # Y = xW instead of Y = xW + b
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias = False)

        # An output layer to bring the heads back together
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries, mask):
        """
            The dimensions of values, keys, and queries tensors are:
                - (Batch_Size, Sequence_Length, Embedding_Size)
                - Ex: (32, 10, 512)
        """
        N = queries.shape[0] # How many samples are we sending at the same time? Batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim) # New dimensions (32, 10, 8, 64)
                                                                         # This grid format is what allows the GPU to process all 8
                                                                         # attention heads at the exact time
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, key_len, self.heads, self.head_dim)

        energy = torch.einsum()
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape: (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)
