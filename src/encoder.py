import torch
import torch.nn as nn
from src.encoder_block import EncoderBlock
from src.positional_encoding import PositionalEncoding


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_ff: int,
                 num_layers: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Encoder blocks
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x: input tensor (batch_size, seq_len)
        """
        # Token embedding + scale
        x = self.embedding(x) * (self.embedding.embedding_dim ** 0.5)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Pass through stacked encoder blocks
        for layer in self.layers:
            x = layer(x, mask)

        # Final layer norm
        return self.norm(x)