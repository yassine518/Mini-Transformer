import torch
import torch.nn as nn
from src.decoder_block import DecoderBlock
from src.positional_encoding import PositionalEncoding


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_ff: int,
                 num_layers: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # Stack of decoder blocks
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        # Embed + positional encoding
        x = self.embedding(x) * (self.embedding.embedding_dim ** 0.5)
        x = self.pos_encoding(x)

        # Pass through stacked decoder blocks
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask=tgt_mask, memory_mask=memory_mask)

        # Final layer norm
        return self.norm(x)