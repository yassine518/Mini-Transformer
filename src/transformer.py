import torch
import torch.nn as nn
from src.encoder import Encoder
from src.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int,
                 n_heads: int, d_ff: int, num_encoder_layers: int,
                 num_decoder_layers: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()

        # Encoder
        self.encoder = Encoder(src_vocab_size, d_model, n_heads, d_ff,
                               num_encoder_layers, max_len, dropout)

        # Decoder
        self.decoder = Decoder(tgt_vocab_size, d_model, n_heads, d_ff,
                               num_decoder_layers, max_len, dropout)

        # Final linear layer to project to vocab size
        self.out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        # Encode source
        enc_output = self.encoder(src, mask=src_mask)

        # Decode target
        dec_output = self.decoder(tgt, enc_output, tgt_mask=tgt_mask, memory_mask=memory_mask)

        # Project to vocab
        logits = self.out(dec_output)
        return logits