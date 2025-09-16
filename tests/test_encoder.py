import torch
from src.encoder import Encoder


def test_encoder_shapes():
    batch, seq_len, vocab_size, d_model, n_heads, d_ff, num_layers = 2, 12, 100, 16, 4, 64, 2
    encoder = Encoder(vocab_size, d_model, n_heads, d_ff, num_layers)

    x = torch.randint(0, vocab_size, (batch, seq_len))
    out = encoder(x)

    assert out.shape == (batch, seq_len, d_model)
    print("Encoder test passed ✅")


if __name__ == "__main__":
    test_encoder_shapes()
