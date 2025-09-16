import torch
import torch.nn as nn
from src.multihead_attention import MultiHeadAttention
from src.encoder_block import PositionwiseFeedForward


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Self-attention (masked)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Encoder-decoder attention
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        # 1. Masked self-attention
        _x, _ = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(_x))

        # 2. Encoder-decoder attention
        _x, _ = self.cross_attn(x, enc_output, enc_output, mask=memory_mask)
        x = self.norm2(x + self.dropout(_x))

        # 3. Feed forward
        _x = self.ffn(x)
        x = self.norm3(x + self.dropout(_x))

        return x
