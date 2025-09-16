import torch
from src.decoder import Decoder


def test_decoder_shapes():
    batch, seq_len, vocab_size, d_model, n_heads, d_ff, num_layers = 2, 12, 100, 16, 4, 64, 2
    decoder = Decoder(vocab_size, d_model, n_heads, d_ff, num_layers)

    x = torch.randint(0, vocab_size, (batch, seq_len))
    enc_output = torch.rand(batch, seq_len, d_model)

    out = decoder(x, enc_output)

    assert out.shape == (batch, seq_len, d_model)
    print("Decoder test passed ✅")


if __name__ == "__main__":
    test_decoder_shapes()
