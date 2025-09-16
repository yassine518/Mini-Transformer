import torch
from src.transformer import Transformer


def test_transformer_shapes():
    batch, src_len, tgt_len = 2, 10, 8
    src_vocab_size, tgt_vocab_size = 50, 50
    d_model, n_heads, d_ff = 16, 4, 64
    num_encoder_layers, num_decoder_layers = 2, 2

    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_heads, d_ff,
                        num_encoder_layers, num_decoder_layers)

    src = torch.randint(0, src_vocab_size, (batch, src_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch, tgt_len))

    logits = model(src, tgt)

    assert logits.shape == (batch, tgt_len, tgt_vocab_size)
    print("Transformer test passed ✅")


if __name__ == "__main__":
    test_transformer_shapes()
