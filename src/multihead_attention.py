import torch
import torch.nn as nn
from src.attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attn = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projection
        Q = self.w_q(query)  # (B, L, d_model)
        K = self.w_k(key)
        V = self.w_v(value)

        # Reshape into (B, n_heads, L, head_dim)
        def split_heads(x):
            return x.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        Q, K, V = map(split_heads, (Q, K, V))

        # Apply scaled dot-product attention
        attn_output, attn_weights = self.attn(Q, K, V, mask=mask)

        # Concatenate heads back
        def combine_heads(x):
            x = x.transpose(1, 2).contiguous()
            return x.view(batch_size, -1, self.d_model)

        out = combine_heads(attn_output)

        # Final linear layer
        out = self.w_o(out)
        out = self.dropout(out)
        return out, attn_weights
